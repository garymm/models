package sim

import (
	"math/rand"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/estats"
	"github.com/emer/etable/etable"
)

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// TODO Net maybe shouldn't be in Sim because it won't always be an axon.Network
	Net *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	// TODO This should be moved to the environment or the Sim extension
	Params  emer.Params   `view:"inline" desc:"all parameter management"`
	Tag     string        `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Pats    *etable.Table `view:"no-inline" desc:"the training patterns to use"`
	Stats   estats.Stats  `desc:"contains computed statistic values"`
	Logs    elog.Logs     `desc:"Contains all the logs and information about the logs.'"`
	GUI     egui.GUI      `view:"-" desc:"manages all the gui elements"`
	CmdArgs CmdArgs       `view:"-" desc:"Arguments passed in through the command line"`

	Run          env.Ctr `desc:"run number"`
	TestInterval int     `desc:"how often (in epochs) to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	PCAInterval  int     `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	NZeroStop    int     `desc:"if a positive number, training will stop after this many epochs with zero UnitErr"`

	TrainEnv Environment `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv  Environment `desc:"Testing environment -- manages iterating over testing"`

	Time      axon.Time       `view:"-" desc:"axon timing parameters and state"`
	ViewOn    bool            `desc:"whether to update the network view while running"`
	TrainUpdt axon.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt  axon.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`

	TrialStatsFunc func(ss *Sim, accum bool) `view:"-" desc:"a function that calculates trial stats"`

	HipStopNow   bool `view:"-" desc:"flag to stop running"`
	PreTrainEpcs int  `desc:"number of epochs to run for pretraining"` //TODO where would this be most appropriate
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Pats = &etable.Table{}
	ss.Stats.Init()
	ss.Run.Scale = env.Run
	ss.CmdArgs.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.CmdArgs.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = axon.AlphaCycle
	ss.TestUpdt = axon.Cycle
	ss.TestInterval = 500 // TODO this should be a value we update or save, seems to log every epoch
	ss.PCAInterval = 10
	ss.Time.Defaults()
}

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.InitRndSeed()
	//TODO: need to modify such that you can load and update environment without calling
	//ss.ConfigEnv()  // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetMsg = ss.CmdArgs.LogSetParams
	ss.Params.SetAll()
	// NOTE uncomment following to see the compiled hyper params
	// fmt.Println(ss.Params.NetHypers.JSONString())
	ss.NewRun()
	ss.GUI.UpdateNetView()
	ss.Stats.ResetTimer("PerTrlMSec")
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Run.Cur
	rand.Seed(ss.CmdArgs.RndSeeds[run])
}

// NewRndSeed gets a new set of random seeds based on current time -- otherwise uses
// the same random seeds for every run
func (ss *Sim) NewRndSeed() {
	rs := time.Now().UnixNano()
	for i := 0; i < 100; i++ {
		ss.CmdArgs.RndSeeds[i] = rs + int64(i)
	}
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Targ values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations
// TODO This was copied
func (ss *Sim) MemStats(train bool) {
	ecout := ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon() // note: must be input b/c ECin can be active
	nn := ecout.Shape().Len()
	actThr := float32(0.2)
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	actMi, _ := ecout.UnitVarIdx("ActM")
	targi, _ := ecout.UnitVarIdx("Targ")
	actQ1i, _ := ecout.UnitVarIdx("ActSt1")
	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitVal1D(actMi, ni)
		trg := ecout.UnitVal1D(targi, ni) // full pattern target
		inact := inp.UnitVal1D(actQ1i, ni)
		if trg < actThr { // trgOff
			trgOffN += 1
			if actm > actThr {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < actThr { // missing in ECin -- completion target
				cmpN += 1
				if actm < actThr {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < actThr {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	trgOnWasOffAll /= trgOnN
	trgOffWasOn /= trgOffN
	if train { // no cmp
		if trgOnWasOffAll < ss.Stats.Float("MemThr") && trgOffWasOn < ss.Stats.Float("MemThr") {
			ss.Stats.SetFloat("Mem", 1)
		} else {
			ss.Stats.SetFloat("Mem", 0)
		}
	} else { // test
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < ss.Stats.Float("MemThr") && trgOffWasOn < ss.Stats.Float("MemThr") {
				ss.Stats.SetFloat("Mem", 1)
			} else {
				ss.Stats.SetFloat("Mem", 0)
			}
		}
	}
	ss.Stats.SetFloat("TrgOnWasOffAll", trgOnWasOffAll)
	ss.Stats.SetFloat("TrgOnWasOffCmp", trgOnWasOffCmp)
	ss.Stats.SetFloat("TrgOffWasOn", trgOffWasOn)
}
