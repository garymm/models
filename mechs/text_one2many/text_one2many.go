// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// one2many is a copy of ra25, but instead of a single output
// associated with each input, there are multiple. The Correl
// metric that's reported is computed based on correlation with
// the closest found pattern.
package main

import (
	"github.com/Astera-org/models/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/goki/gi/gimain"
	"log"
	"os"
)

func main() {
	// TheSim is the overall state for this simulation
	var TheSim sim.Sim
	TheSim.New()

	Config(&TheSim)

	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			sim.GuiRun(&TheSim)
		})
	}

}

// Config configures all the elements using the standard functions
func Config(ss *sim.Sim) {
	ConfigParams(ss)
	ConfigEnv(ss)
	ConfigPats(ss)
	ConfigNet(ss, ss.Net)
	// LogSpec needs to be configured after Net
	ss.ConfigLogSpec()
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigSpikeRasts()
	ss.ConfigRunLog(ss.RunLog)
}

func ConfigParams(ss *sim.Sim) {

	// ParamSetsMin sets the minimal non-default params
	// Base is always applied, and others can be optionally selected to apply on top of that
	ss.Params = params.Sets{
		{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
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
					}},
				{Sel: "#Input", Desc: "critical now to specify the activity level",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi": "0.9", // 0.9 > 1.0
						"Layer.Act.Clamp.Ge":   "1.0", // 1.0 > 0.6 >= 0.7 == 0.5
						// TODO This should vary based on n-hot
						"Layer.Inhib.ActAvg.Init": "0.04", // .24 nominal, lower to give higher excitation
					},
					Hypers: params.Hypers{
						// TODO Set these numbers to be less random
						"Layer.Inhib.Layer.Gi": {"Val": "0.9", "Min": "1", "Max": "3", "Sigma": ".45", "Priority": "5"},
						"Layer.Act.Clamp.Ge":   {"Val": "1.0"},
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
					}},
			},
			"Sim": &params.Sheet{ // sim params apply to sim object
				{Sel: "Sim", Desc: "best params always finish in this time",
					Params: params.Params{
						"Sim.MaxEpcs": "100",
					}},
			},
		}},
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

func ConfigEnv(ss *sim.Sim) {
	//ss = *sim.Sim
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 5
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 100
		ss.NZeroStop = 5
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	//ss.TrainEnv.Table = etable.NewIdxView(ss.Pats)
	//ss.TrainEnv.Con
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Config("mechs/text_one2many/data/cbt_train_filt.json", evec.Vec2i{5, 5}, false, 1, 3, 10)
	ss.TrainEnv.Trial.Max = len(ss.TrainEnv.NGrams)
	ss.TrainEnv.Epoch.Max = ss.MaxEpcs

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Validate()
	ss.TestEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TestEnv.Config("mechs/text_one2many/data/cbt_train_filt.json", evec.Vec2i{5, 5}, false, 1, 3, 10)
	ss.TestEnv.Trial.Max = len(ss.TestEnv.NGrams)
	ss.TestEnv.Epoch.Max = ss.MaxEpcs
	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func ConfigPats(ss *sim.Sim) {
	dt := ss.Pats
	dt.SetMetaData("name", "SuccessorPatterns")
	dt.SetMetaData("desc", "SuccessorPatterns")
	sch := etable.Schema{
		{"Word", etensor.STRING, nil, nil},
		{"Pattern", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, ss.NInputs*ss.NOutputs)

	i := 0
	for _, word := range ss.TrainEnv.Words {
		idx := ss.TrainEnv.WordMap[word]
		mytensor := ss.TrainEnv.WordReps.SubSpace([]int{idx})
		dt.SetCellString("Word", i, word)
		dt.SetCellTensor("Pattern", i, mytensor)
		i++

	}

	dt.SaveCSV("random_5x5_25_gen.tsv", etable.Tab, etable.Headers)
}

func ConfigNet(ss *sim.Sim, net *axon.Network) {
	net.InitName(net, "One2Many")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", 10, 10, emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", 10, 10, emer.Hidden)
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
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}
