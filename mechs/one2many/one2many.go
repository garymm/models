// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// one2many is a copy of ra25, but instead of a single output
// associated with each input, there are multiple. The Correl
// metric that's reported is computed based on correlation with
// the closest found pattern.

package main

import (
	"fmt"
	"log"

	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gimain"
)

var ProgramName = "One2Many"
var TestEnv = EnvOne2Many{}
var TrainEnv = EnvOne2Many{}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func TrialStats(ss *sim.Sim, accum bool) {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCosDiff", float64(out.CosDiff.Cos))

	_, cor, cnm := ss.Stats.ClosestPat(ss.Net, "Output", "ActM", ss.Pats, "Output", "Name")

	ss.Stats.SetString("TrlClosest", cnm)
	ss.Stats.SetFloat("TrlCorrel", float64(cor))
	tnm := ""
	if accum { // really train
		tnm = ss.TrainEnv.TrialName().Cur
	} else {
		tnm = ss.TestEnv.TrialName().Cur
	}
	if cnm == tnm {
		ss.Stats.SetFloat("TrlErr", 0)
	} else {
		ss.Stats.SetFloat("TrlErr", 1)
	}

}

type One2Sim struct {
	sim.Sim
	// Specific to the one2many module
	NInputs  int `desc:"Number of input/output pattern pairs"`
	NOutputs int `desc:"The number of output patterns potentially associated with each input pattern."`
}

func main() {
	// TheSim is the overall state for this simulation
	var TheSim One2Sim
	TheSim.New()
	TheSim.NInputs = 25
	TheSim.NOutputs = 2

	Config(&TheSim)

	if TheSim.CmdArgs.NoGui {
		TheSim.RunFromArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			sim.GuiRun(&TheSim.Sim, ProgramName, "One to Many", `This demonstrates a basic Axon model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
		})
	}

}

// Config configures all the elements using the standard functions
func Config(ss *One2Sim) {
	ConfigPats(ss)
	OpenPats(&ss.Sim)
	ConfigParams(&ss.Sim)
	// Parse arguments before configuring the network and env, in case parameters are set.
	ss.ParseArgs()
	ConfigEnv(&ss.Sim)
	ConfigNet(&ss.Sim, ss.Net)
	ss.ConfigLogs()
	ss.ConfigSpikeRasts()
}

// ConfigParams configure the parameters
func ConfigParams(ss *sim.Sim) {
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()

	// ParamSetsMin sets the minimal non-default params
	// Base is always applied, and others can be optionally selected to apply on top of that
	ss.Params.Params = params.Sets{
		{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
			"NetSize": &params.Sheet{
				{Sel: ".Hidden", Desc: "all hidden layers",
					Params: params.Params{
						"Layer.X": "8",
						"Layer.Y": "8",
					}},
			},
			"Network": &params.Sheet{
				{Sel: "Layer", Desc: "all defaults",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi":    "1.2",  // 1.2 > 1.1
						"Layer.Inhib.ActAvg.Init": "0.04", // 0.04 for 1.2, 0.08 for 1.1
						"Layer.Inhib.Layer.Bg":    "0.3",  // 0.3 > 0.0
						"Layer.Act.Decay.Glong":   "0.6",  // 0.6
						"Layer.Act.Dend.GbarExp":  "0.2",  // 0.2 > 0.1 > 0
						"Layer.Act.Dend.GbarR":    "3",    // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
						"Layer.Act.Dt.VmDendTau":  "5",    // 5 > 2.81 here but small effect
						"Layer.Act.Dt.VmSteps":    "2",    // 2 > 3 -- somehow works better
						"Layer.Act.Dt.GeTau":      "5",
						"Layer.Act.NMDA.Gbar":     "0.15", //
						"Layer.Act.GABAB.Gbar":    "0.2",  // 0.2 > 0.15
					}, Hypers: params.Hypers{
						"Layer.Inhib.ActAvg.Init": {"StdDev": "0.01", "Min": "0.01"},
					}},
				{Sel: "#Input", Desc: "critical now to specify the activity level",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi": "0.9", // 0.9 > 1.0
						"Layer.Act.Clamp.Ge":   "1.0", // 1.0 > 0.6 >= 0.7 == 0.5
						// This should only be 0.04 in one-hot encoding
						"Layer.Inhib.ActAvg.Init": "0.04", // .24 nominal, lower to give higher excitation
					},
					Hypers: params.Hypers{
						"Layer.Inhib.Layer.Gi": {"StdDev": ".1", "Min": "0", "Priority": "2", "Scale": "LogLinear"},
						"Layer.Act.Clamp.Ge":   {"StdDev": ".2"},
					}},
				{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 0.8 > 1.0 > 0.7 even with adapt -- not beneficial to start low
						"Layer.Inhib.ActAvg.Init": "0.04", // this has to be exact for adapt
						"Layer.Act.Spike.Tr":      "1",    // 1 is new minimum.
						"Layer.Act.Clamp.Ge":      "0.6",  // .6 > .5 v94
						// "Layer.Act.NMDA.Gbar":     "0.3",  // higher not better
					}},
				{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base": "0.2", // 0.04 no rlr, 0.2 rlr; .3, WtSig.Gain = 1 is pretty close
						"Prjn.SWt.Adapt.Lrate":  "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
						"Prjn.SWt.Init.SPct":    "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					}},
				{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
					},
					Hypers: params.Hypers{
						"Prjn.PrjnScale.Rel": {"StdDev": ".05"},
					}},
			},
			"Sim": &params.Sheet{ // sim params apply to sim object
				{Sel: "Sim", Desc: "best params always finish in this time",
					Params: params.Params{
						"Sim.CmdArgs.MaxEpcs": "100",
					}},
			},
		}},
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//// 		Configs

func ConfigEnv(ss *sim.Sim) {

	ss.TestEnv = &TestEnv
	ss.TrainEnv = &TrainEnv

	ss.TrialStatsFunc = TrialStats

	ss.NZeroStop = 5

	TrainEnv.Nm = "TrainEnv"
	TrainEnv.Dsc = "training params and state"
	TrainEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.Run.Max = ss.CmdArgs.MaxRuns // note: we are not setting epoch max -- do that manually
	TrainEnv.Epoch().Max = ss.CmdArgs.MaxEpcs

	TestEnv.Nm = "TestEnv"
	TestEnv.Dsc = "testing params and state"
	TestEnv.Table = etable.NewIdxView(ss.Pats)
	TestEnv.SetSequential(true)
	ss.TestEnv.Validate()
	TestEnv.Epoch().Max = ss.CmdArgs.MaxEpcs
	TestEnv.Run().Max = ss.CmdArgs.MaxRuns

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

//ConfigPats used to configure patterns
func ConfigPats(ss *One2Sim) {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, ss.NInputs*ss.NOutputs)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	for i := 0; i < ss.NInputs; i++ {
		for j := 0; j < ss.NOutputs; j++ {
			dt.SetCellTensor("Input", i*ss.NOutputs+j, dt.CellTensor("Input", i*ss.NOutputs))
			dt.SetCellString("Name", i*ss.NOutputs+j, fmt.Sprintf("%d", i))
		}
	}
	dt.SaveCSV("random_5x5_25_gen.tsv", etable.Tab, etable.Headers)
}

func OpenPats(ss *sim.Sim) {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

func ConfigNet(ss *sim.Sim, net *axon.Network) {
	ss.Params.AddLayers([]string{"Hidden1", "Hidden2"}, "Hidden")
	ss.Params.SetObject("NetSize")

	net.InitName(net, ProgramName) // TODO this should have a name that corresponds to project, leaving for now as it will cause a problem in optimize

	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", ss.Params.LayY("Hidden1", 10), ss.Params.LayX("Hidden1", 10), emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", ss.Params.LayY("Hidden2", 10), ss.Params.LayX("Hidden2", 10), emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	// hid2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern
	full := prjn.NewFull()

	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	// net.LateralConnectLayerPrjn(hid1, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	// if Thread {
	// 	hid2.SetThread(1)
	// 	out.SetThread(1)
	// }

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}
