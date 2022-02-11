package sim

import (
	"math/rand"
	"time"

	"github.com/Astera-org/models/library/egui"
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
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
	Pats *etable.Table `view:"no-inline" desc:"the training patterns to use"`

	Logs   elog.Logs   `desc:"Contains all the logs and information about the logs.'"`
	Params emer.Params `view:"inline" desc:"all parameter management"`

	GUI egui.GUI

	TrialStatsFunc func(ss *Sim, accum bool) `view:"inline" desc:"a function that calculates trial stats"`

	Tag string  `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Run env.Ctr `desc:"Information about the current run."`

	// TODO This refactor will have to happen later
	NZeroStop int `desc:"if a positive number, training will stop after this many epochs with zero UnitErr"`

	TrainEnv Environment `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv  Environment `desc:"Testing environment -- manages iterating over testing"`

	Time         axon.Time       `desc:"axon timing parameters and state"`
	ViewOn       bool            `desc:"whether to update the network view while running"`
	TrainUpdt    axon.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     axon.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int             `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`

	// TODO These maybe don't need to be stored on Sim at all
	LayStatNms   []string `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	SpikeRecLays []string `desc:"names of layers to record spikes of during testing"`

	// statistics: note use float64 as that is best for etable.Table
	// TODO Maybe put this on a Stats object - moved to map
	TrlErr     float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on UnitErr = 0 (subject to .5 unit-wise tolerance)"`
	TrlClosest string  `inactive:"+" desc:"Name of the pattern with the closest output"`
	TrlCorrel  float64 `inactive:"+" desc:"Correlation with closest output"`
	TrlUnitErr float64 `inactive:"+" desc:"current trial's unit-level pct error"`
	TrlCosDiff float64 `inactive:"+" desc:"current trial's cosine difference"`

	// TODO Move these to a newly created func EpochStats
	// State about how long there's been zero error.
	FirstZero   int       `inactive:"+" desc:"epoch at when all TrlErr first went to zero"`
	NZero       int       `inactive:"+" desc:"number of epochs in a row with no TrlErr"`
	LastEpcTime time.Time `view:"-" desc:"timer for last epoch"`

	// internal state - view:"-"
	SumErr float64 `view:"-" inactive:"+" desc:"Sum of errors throughout epoch. This way we can know when an epoch is error free, for early stopping."`

	CmdArgs CmdArgs `desc:"Arguments passed in through the command line"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Pats = &etable.Table{}
	ss.CmdArgs.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.CmdArgs.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = axon.AlphaCycle
	ss.TestUpdt = axon.Cycle
	ss.TestInterval = 500                                               // TODO this should be a value we update or save, seems to log every epoch
	ss.LayStatNms = []string{"Hidden1", "Hidden2", "Output"}            // TODO randy is gonna refactor out
	ss.SpikeRecLays = []string{"Input", "Hidden1", "Hidden2", "Output"} //TODO randy is gonna refactor out
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
	ss.UpdateView(true)
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
