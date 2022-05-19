// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// emery2 is a simulated virtual rat / cat, using axon spiking model
package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/pca"
	"math/rand"
)

func main() {

	var sim Sim
	sim.DefineSimVariables()
	//sim.WorldEnv = sim.ConfigEnv()
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()

	userInterface := egui.UserInterface{
		StructForView:             &sim,
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
		AppName:                   "Attempted refactor of Fworld",
		AppTitle:                  "Fworld with refactored code",
		AppAbout:                  `Learn to mimic patterns coming from a teacher signal in a flat grid world.`,
		AddNetworkLoggingCallback: axon.AddCommonLogItemsForOutputLayers,
	}
	userInterface.AddDefaultLogging()
	userInterface.CreateAndRunGui() // CreateAndRunGui blocks, so don't put any code after this.
}

// see params_def.go for default params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct { // TODO(refactor): Remove a lot of this stuff
	Net      *deep.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Loops    *looper.Manager `view:"no-inline" desc:"contains looper control loops for running sim"`
	Time     axon.Time       `desc:"axon timing parameters and state"`
	LoopTime string          `desc:"Printout of the current time."`
	GUI      egui.GUI        `view:"-" desc:"manages all the gui elements"`

	ActionMapping      map[string]agent.SpaceSpec `view:"-" desc:"shape and structure of actions agent can take"`
	ObservationMapping map[string]agent.SpaceSpec `view:"-" desc:"shape and structure of observations agent can take"`

	PctCortex        float64        `desc:"proportion of action driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax     float64        `desc:"maximum PctCortex, when running on the schedule"`
	TrnErrStats      *etable.Table  `view:"no-inline" desc:"stats on train trials where errors were made"`
	TrnAggStats      *etable.Table  `view:"no-inline" desc:"stats on all train trials"`
	MinusCycles      int            `desc:"number of minus-phase cycles"`
	PlusCycles       int            `desc:"number of plus-phase cycles"`
	ErrLrMod         axon.LrateMod  `view:"inline" desc:"learning rate modulation as function of error"`
	Params           params.Sets    `view:"no-inline" desc:"full collection of param sets"`
	ParamSet         string         `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag              string         `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Prjn4x4Skp2      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x4Skp2Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn4x3Skp2      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn"`
	Prjn4x3Skp2Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 2 topo prjn, recip"`
	Prjn3x3Skp1      *prjn.PoolTile `view:"no-inline" desc:"feedforward 3x3 skip 1 topo prjn"`
	Prjn4x4Skp4      *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn"`
	Prjn4x4Skp4Recip *prjn.PoolTile `view:"no-inline" desc:"feedforward 4x4 skip 4 topo prjn, recip"`
	MaxRuns          int            `desc:"maximum number of model runs to perform"`
	MaxEpcs          int            `desc:"maximum number of epochs to run per model run"`
	TestEpcs         int            `desc:"number of epochs of testing to run, cumulative after MaxEpcs of training"`
	OnlyEnv          ExampleWorld   `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestInterval     int            `desc:"how often to run through all the test patterns, in terms of training epochs"`

	// statistics: note use float64 as that is best for etable.Table
	RFMaps        map[string]*etensor.Float32 `view:"no-inline" desc:"maps for plotting activation-based receptive fields"`
	PulvLays      []string                    `view:"-" desc:"pulvinar layers -- for stats"`
	HidLays       []string                    `view:"-" desc:"hidden layers: super and CT -- for hogging stats"`
	SuperLays     []string                    `view:"-" desc:"superficial layers"`
	InputLays     []string                    `view:"-" desc:"input layers"`
	NetAction     string                      `inactive:"+" desc:"action activated by the cortical network"`
	GenAction     string                      `inactive:"+" desc:"action generated by subcortical code"`
	ActAction     string                      `inactive:"+" desc:"actual action taken"`
	ActMatch      float64                     `inactive:"+" desc:"1 if net action matches gen action, 0 otherwise"`
	TrlCosDiff    float64                     `inactive:"+" desc:"current trial's overall cosine difference"`
	TrlCosDiffTRC []float64                   `inactive:"+" desc:"current trial's cosine difference for pulvinar (TRC) layers"`
	EpcActMatch   float64                     `inactive:"+" desc:"last epoch's average act match"`
	EpcCosDiff    float64                     `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	NumTrlStats   int                         `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumActMatch   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	PCA           pca.PCA                     `view:"-" desc:"pca obj"`

	// internal state - view:"-"
	PopVals      []float32                   `view:"-" desc:"tmp pop code values"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	SaveARFs     bool                        `view:"-" desc:"for command-line run only, auto-save receptive field data"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	RndSeed      int64                       `view:"-" desc:"the current random seed"`
	UseMPI       bool                        `view:"-" desc:"if true, use MPI to distribute computation across nodes"`
	SaveProcLog  bool                        `view:"-" desc:"if true, save logs per processor"`
	Comm         *mpi.Comm                   `view:"-" desc:"mpi communicator"`
	AllDWts      []float32                   `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts      []float32                   `view:"-" desc:"buffer of MPI summed dwt weight changes"`

	// Characteristics of the environment interface.
	FoveaSize  int        `desc:"number of items on each size of the fovea, in addition to center (0 or more)"`
	DepthSize  int        `inactive:"+" desc:"number of units in depth population codes"`
	DepthPools int        `inactive:"+" desc:"number of pools to divide DepthSize into"`
	PatSize    evec.Vec2i `desc:"size of patterns for mats, acts"`
	Inters     []string   `desc:"list of interoceptive body states, represented as pop codes"`
	PopSize    int        `inactive:"+" desc:"number of units in population codes"`
	NFOVRays   int        `inactive:"+" desc:"total number of FOV rays that are traced"`
}

// DefineSimVariables creates new blank elements and initializes defaults
func (ss *Sim) DefineSimVariables() { // TODO(refactor): Remove a lot
	ss.Params = ParamSets

	ss.Time.Defaults()
	ss.MinusCycles = 150
	ss.PlusCycles = 50

	ss.ErrLrMod.Defaults()
	ss.ErrLrMod.Base = 0.05 // 0.05 >= .01, .1 -- hard to tell
	ss.ErrLrMod.Range.Set(0.2, 0.8)
	ss.Params = ParamSets
	ss.RndSeed = 1

	// Default values
	ss.PctCortexMax = 0.9 // 0.5 before
	ss.TestInterval = 50000

	// These relate to the shape of the network/environment interface.
	ss.DepthPools = 8
	ss.DepthSize = 32
	ss.NFOVRays = 13
	ss.FoveaSize = 1
	ss.PopSize = 16
	ss.Inters = []string{"Energy", "Hydra", "BumpPain", "FoodRew", "WaterRew"}
	ss.PatSize = evec.Vec2i{X: 5, Y: 5}

	actionobsMapping := (&NetCharacteristics{}).Init()
	ss.ActionMapping = actionobsMapping.ActionMapping
	ss.ObservationMapping = actionobsMapping.ObservationMapping
	// This has to be called after those variables are defined, because they're used in here.
	DefineNetworkCharacteristics(ss) //todo this sohuld be removed and made locally in config network

}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet()
}

func (ss *Sim) ConfigEnv() {
	ss.OnlyEnv.InitWorld(nil)
}

func (ss *Sim) ConfigNet() *deep.Network {
	net := &deep.Network{}
	DefineNetworkStructure(ss, net)
	return net
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() { // TODO(refactor): this should be broken up
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	SetParams("", ss.LogSetParams, ss.Net, &ss.Params, ss.ParamSet, ss) // all sheets
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigLoops() *looper.Manager {
	manager := looper.Manager{}.Init()
	manager.Stacks[etime.Train] = &looper.Stack{}
	manager.Stacks[etime.Train].Init().AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 1).AddTime(etime.Cycle, 200)
	axon.AddPlusAndMinusPhases(manager, &ss.Time, ss.Net.AsAxon())

	plusPhase := &manager.GetLoop(etime.Train, etime.Cycle).Events[1]
	plusPhase.OnEvent.Add("Sim:PlusPhase:SendActionsThenStep", func() {
		axon.SendActionAndStep(ss.Net.AsAxon(), &ss.OnlyEnv)
	})

	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Sim:ResetState", func() {
		ss.Net.NewState()
		ss.Time.NewState(mode.String())
	})

	stack.Loops[etime.Trial].OnStart.Add("Sim:Trial:Observe", func() {
		for name, _ := range ss.ObservationMapping {
			axon.ApplyInputs(ss.Net.AsAxon(), &ss.OnlyEnv, name, func(spec agent.SpaceSpec) etensor.Tensor {
				return ss.OnlyEnv.Observe(name)
			})
		}

	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)
	//manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewPatterns", func() { ss.WorldEnv.InitWorld(nil) })
	axon.AddDefaultLoopSimLogic(manager, &ss.Time, ss.Net.AsAxon())

	// Initialize and print loop structure, then add to Sim
	manager.Init()
	fmt.Println(manager.DocString())

	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Sim:Trial:QuickScore", func() {
		loss := ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon().PctUnitErr()
		s := fmt.Sprintf("%f", loss)
		fmt.Println("the pctuniterror is " + s)
		//simple error calculation
	})

	return manager
}

// NewRun intializes a new run of the model, using the OnlyEnv.GetCounter(etime.Run) counter
// for the new run value
func (ss *Sim) NewRun() { // TODO(refactor): looper call
	//run := ss.OnlyEnv.GetCounter(etime.Run).Cur
	ss.PctCortex = 0
	ss.OnlyEnv.InitWorld(nil)
	//ss.OnlyEnv.Init("DefineSimVariables Run") //TODO: meaningful init info that should be passed

	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() { // TODO(refactor): use Stats
	// accumulators
	ss.NumTrlStats = 0
	ss.SumActMatch = 0
	ss.SumCosDiff = 0
	// clear rest just to make Sim look initialized
	ss.EpcActMatch = 0
	ss.EpcCosDiff = 0
}

// TrialStatsTRC computes the trial-level statistics for TRC layers
func (ss *Sim) TrialStatsTRC(accum bool) { // TODO(refactor): looper stats?
	nt := len(ss.PulvLays)
	if len(ss.TrlCosDiffTRC) != nt {
		ss.TrlCosDiffTRC = make([]float64, nt)
	}
	acd := 0.0
	for i, ln := range ss.PulvLays {
		ly := ss.Net.LayerByName(ln).(axon.AxonLayer).AsAxon()
		cd := float64(ly.CosDiff.Cos)
		acd += cd
		ss.TrlCosDiffTRC[i] = cd
	}
	ss.TrlCosDiff = acd / float64(len(ss.PulvLays))
	if accum {
		ss.SumCosDiff += ss.TrlCosDiff
	}
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) { // TODO(refactor): looper call
	ss.TrialStatsTRC(accum)
	if accum {
		ss.SumActMatch += ss.ActMatch
		ss.NumTrlStats++
	} else {
		ss.UpdtARFs() // only in testing
	}
	return
}
