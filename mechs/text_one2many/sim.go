package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/goki/gi/gi"
	"math/rand"
	"os"
	"time"
)

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net            *axon.Network                 `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Pats           *etable.Table                 `view:"no-inline" desc:"the training patterns to use"`
	NInputs        int                           `desc:"Number of input/output pattern pairs"`
	NOutputs       int                           `desc:"The number of output patterns potentially associated with each input pattern."`
	LogSpec        LogSpec                       `desc:"Specifies which details are to be logged"`
	TrnEpcLog      *etable.Table                 `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog      *etable.Table                 `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog      *etable.Table                 `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog      *etable.Table                 `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats    *etable.Table                 `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog      *etable.Table                 `view:"no-inline" desc:"testing cycle-level log data"`
	SpikeRasters   map[string]*etensor.Float32   `desc:"spike raster data for different layers"`
	SpikeRastGrids map[string]*etview.TensorGrid `desc:"spike raster plots for different layers"`
	RunLog         *etable.Table                 `view:"no-inline" desc:"summary log of each run"`
	RunStats       *etable.Table                 `view:"no-inline" desc:"aggregate stats on all runs"`
	ErrLrMod       axon.LrateMod                 `view:"inline" desc:"learning rate modulation as function of error"`

	Params    params.Sets `view:"no-inline" desc:"full collection of param sets"`
	ParamSet  string      `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Tag       string      `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	StartRun  int         `desc:"starting run number -- typically 0 but can be set in command args for parallel runs on a cluster"`
	MaxRuns   int         `desc:"maximum number of model runs to perform (starting from StartRun)"`
	MaxEpcs   int         `desc:"maximum number of epochs to run per model run"`
	NZeroStop int         `desc:"if a positive number, training will stop after this many epochs with zero UnitErr"`

	TrainEnv     CorpusEnv       `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      CorpusEnv       `desc:"Testing environment -- manages iterating over testing"`
	Time         axon.Time       `desc:"axon timing parameters and state"`
	ViewOn       bool            `desc:"whether to update the network view while running"`
	TrainUpdt    axon.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     axon.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int             `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	LayStatNms   []string        `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	SpikeRecLays []string        `desc:"names of layers to record spikes of during testing"`

	// statistics: note use float64 as that is best for etable.Table
	TrlErr        float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on UnitErr = 0 (subject to .5 unit-wise tolerance)"`
	TrlClosest    string  `inactive:"+" desc:"Name of the pattern with the closest output"`
	TrlCorrel     float64 `inactive:"+" desc:"Correlation with closest output"`
	TrlUnitErr    float64 `inactive:"+" desc:"current trial's unit-level pct error"`
	TrlCosDiff    float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcUnitErr    float64 `inactive:"+" desc:"last epoch's total unit-level pct error"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcCorrel     float64 `inactive:"+" desc:"last epoch's average correlation for output layer"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when all TrlErr first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with no TrlErr"`

	// internal state - view:"-"
	SumErr       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumUnitErr   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCorrel    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D               `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds     []int64                     `view:"-" desc:"a list of random seeds to use for each run"`
	NetData      *netview.NetData            `view:"-" desc:"net data for recording in nogui mode"`
	LastEpcTime  time.Time                   `view:"-" desc:"timer for last epoch"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Pats = &etable.Table{}
	ss.NInputs = 25
	ss.NOutputs = 2
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.ErrLrMod.Defaults()
	ss.ErrLrMod.Base = 0.5 // 0.5 > 0.2 -- not very useful in this model, but key in larger nets
	ss.ErrLrMod.Range.Set(0, 0.5)
	ss.Params = ParamSetsMin         // ParamSetsAll
	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = axon.AlphaCycle
	ss.TestUpdt = axon.Cycle
	ss.TestInterval = 500
	ss.LayStatNms = []string{"Hidden1", "Hidden2", "Output"}
	ss.SpikeRecLays = []string{"Input", "Hidden1", "Hidden2", "Output"}
	ss.Time.Defaults()
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {

	ss.ConfigEnv()

	ss.ConfigPatsFromEnv()

	ss.ConfigNet(ss.Net)
	ss.ConfigLogSpecEpoch()
	fmt.Println("What the hell")
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigSpikeRasts()
	ss.ConfigRunLog(ss.RunLog)
}

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.InitRndSeed()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.NewRun()
	ss.UpdateView(true)
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.TrainEnv.Run.Cur
	rand.Seed(ss.RndSeeds[run])
}

// NewRndSeed gets a new set of random seeds based on current time -- otherwise uses
// the same random seeds for every run
func (ss *Sim) NewRndSeed() {
	rs := time.Now().UnixNano()
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = rs + int64(i)
	}
}
