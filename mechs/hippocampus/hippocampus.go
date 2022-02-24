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
	sim "github.com/Astera-org/models/mechs/hippocampus/hipsim"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/hip"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/etable"
	"github.com/goki/gi/gimain"
	"log"
)

var ProgramName = "Hippocampus"

var TestEnv = EnvHipBench{}
var TrainEnv = EnvHipBench{}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func TrialStats(ss *sim.Sim, accum bool) {
	outLay := ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
	ss.Stats.SetFloat("TrlCosDiff", float64(outLay.CosDiff.Cos))
	ss.Stats.SetFloat("TrlUnitErr", outLay.PctUnitErr())
	if accum {
		ss.Stats.SetFloat("SumUnitErr", ss.Stats.Float("SumUnitErr")+ss.Stats.Float("TrlUnitErr"))
		ss.Stats.SetFloat("SumCosDiff", ss.Stats.Float("SumCosDiff")+ss.Stats.Float("TrlCosDiff"))
		if ss.Stats.Float("TrlCosDiff") != 0 {
			ss.Stats.SetInt("CntErr", ss.Stats.Int("CntErr")+1)
		}
	}
}

type HipSim struct {
	sim.Sim
	// Specific to the one2many module
	Hip       HipParams    `desc:"hippocampus sizing parameters"`
	PoolVocab patgen.Vocab `view:"no-inline" desc:"pool patterns vocabulary"`
	Pat       PatParams
}

func (ss *HipSim) New() {
	ss.Sim.New()
	ss.PoolVocab = patgen.Vocab{}
	ss.Hip = HipParams{}
	ss.Hip.Defaults()
	ss.Pat.Defaults()
	ss.Hip.Defaults()
	ss.Pat.Defaults()
	ss.Run.Cur = 0 // for initializing envs if using Gui // TODO Makesure this works right. It was StartRun.
	ss.Update()
}

func (ss *HipSim) Update() {
	ss.Hip.Update()
}

func main() {
	// TheSim is the overall state for this simulation
	var TheSim HipSim
	TheSim.New()
	TrainEnv.InitTables(TrainAB, TrainAC, PretrainLure, TrainAll)
	TestEnv.InitTables(TestAB, TestAC, TestLure)
	Config(&TheSim)

	if TheSim.CmdArgs.NoGui {
		TheSim.RunFromArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want

	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			sim.GuiRun(&TheSim.Sim, ProgramName, "Hippocampus AB-AC", `This demonstrates a basic Hippocampus model in Axon. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
		})
	}

}

// Config configures all the elements using the standard functions
func Config(ss *HipSim) {
	ConfigPats(ss)
	//OpenPats(&ss.Sim)
	ConfigParams(&ss.Sim)
	// Parse arguments before configuring the network and env, in case parameters are set.
	ss.ParseArgs()
	ConfigEnv(ss)
	ConfigNet(ss, ss.Net)
	ss.ConfigLogs()
}

// ConfigParams configure the parameters
func ConfigParams(ss *sim.Sim) {
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Params.Params = params.Sets{
		{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
			"Network": &params.Sheet{
				{Sel: "Layer", Desc: "generic layer params",
					Params: params.Params{
						"Layer.Act.KNa.On":         "false", // false > true
						"Layer.Learn.TrgAvgAct.On": "false", // true > false?
						"Layer.Learn.RLrate.On":    "false", // no diff..
						"Layer.Act.Gbar.L":         "0.2",   // .2 > .1
						"Layer.Act.Decay.Act":      "1.0",   // 1.0 both is best by far!
						"Layer.Act.Decay.Glong":    "1.0",
						"Layer.Inhib.Pool.Bg":      "0.0",
					}},
				{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level",
					Params: params.Params{
						"Layer.Learn.TrgAvgAct.On": "false", // def true, not rel?
						"Layer.Learn.RLrate.On":    "false", // def true, too slow?
						"Layer.Inhib.ActAvg.Init":  "0.15",
						"Layer.Inhib.Layer.On":     "false",
						"Layer.Inhib.Layer.Gi":     "0.2", // weak just to keep it from blowing up
						"Layer.Inhib.Pool.Gi":      "1.1",
						"Layer.Inhib.Pool.On":      "true",
					}},
				{Sel: "#ECout", Desc: "all EC layers: only pools, no layer-level",
					Params: params.Params{
						"Layer.Inhib.Pool.Gi": "1.1",
						"Layer.Act.Clamp.Ge":  "0.6",
					}},
				{Sel: "#CA1", Desc: "CA1 only Pools",
					Params: params.Params{
						"Layer.Learn.TrgAvgAct.On": "true",  // actually a bit better
						"Layer.Learn.RLrate.On":    "false", // def true, too slow?
						"Layer.Inhib.ActAvg.Init":  "0.02",
						"Layer.Inhib.Layer.On":     "false",
						"Layer.Inhib.Pool.Gi":      "1.3", // 1.3 > 1.2 > 1.1
						"Layer.Inhib.Pool.On":      "true",
						"Layer.Inhib.Pool.FFEx0":   "1.0", // blowup protection
						"Layer.Inhib.Pool.FFEx":    "0.0",
					}},
				{Sel: "#DG", Desc: "very sparse = high inibhition",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.005", // actual .002-3
						"Layer.Inhib.Layer.Gi":    "2.2",   // 2.2 > 2.0 on larger
					}},
				{Sel: "#CA3", Desc: "sparse = high inibhition",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.02",
						"Layer.Inhib.Layer.Gi":    "1.8", // 1.8 > 1.6 > 2.0
					}},
				{Sel: "Prjn", Desc: "keeping default params for generic prjns",
					Params: params.Params{
						"Prjn.SWt.Init.SPct": "0.5", // 0.5 == 1.0 > 0.0
					}},
				{Sel: ".EcCa1Prjn", Desc: "encoder projections",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base": "0.04", // 0.04 for Axon -- 0.01 for EcCa1
					}},
				{Sel: ".HippoCHL", Desc: "hippo CHL projections",
					Params: params.Params{
						"Prjn.CHL.Hebb":         "0.05",
						"Prjn.Learn.Lrate.Base": "0.02", // .2 def
					}},
				{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Prjn prjns",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base": "0.1", // .1 > .04 -- makes a diff
						// moss=4, delta=4, lr=0.2, test = 3 are best
					}},
				{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
					Params: params.Params{
						"Prjn.PrjnScale.Abs": "2.0", // 2.0 > 3.0 for larger
					}},
				{Sel: "#ECinToCA3", Desc: "stronger",
					Params: params.Params{
						"Prjn.PrjnScale.Abs": "3.0", // 4.0 > 3.0
					}},
				{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
					Params: params.Params{
						"Prjn.Learn.Learn":      "true", // absolutely essential to have on!
						"Prjn.CHL.Hebb":         "0.5",  // .5 > 1 overall
						"Prjn.CHL.SAvgCor":      "0.1",  // .1 > .2 > .3 > .4 ?
						"Prjn.CHL.MinusQ1":      "true", // dg self err?
						"Prjn.Learn.Lrate.Base": "0.01", // 0.01 > 0.04 maybe
					}},
				{Sel: "#InputToECin", Desc: "one-to-one input to EC",
					Params: params.Params{
						"Prjn.Learn.Learn":   "false",
						"Prjn.SWt.Init.Mean": "0.9",
						"Prjn.SWt.Init.Var":  "0.0",
						"Prjn.PrjnScale.Abs": "1.0",
					}},
				{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
					Params: params.Params{
						"Prjn.Learn.Learn":   "false",
						"Prjn.SWt.Init.Mean": "0.9",
						"Prjn.SWt.Init.Var":  "0.01",
						"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 1 (sig worse)
					}},
				{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
					Params: params.Params{
						"Prjn.Learn.Learn":   "false",
						"Prjn.SWt.Init.Mean": "0.9",
						"Prjn.SWt.Init.Var":  "0.01",
						"Prjn.PrjnScale.Rel": "3", // 4 def
					}},
				{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons",
					Params: params.Params{
						"Prjn.PrjnScale.Rel":    "0.1",  // 0.1 > 0.2 == 0
						"Prjn.Learn.Lrate.Base": "0.04", // 0.1 v.s .04 not much diff
					}},
				{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
					Params: params.Params{
						// "Prjn.CHL.Hebb":         "0.01",
						// "Prjn.CHL.SAvgCor":      "0.4",
						"Prjn.Learn.Lrate.Base": "0.1", // 0.1 > 0.04
						"Prjn.PrjnScale.Rel":    "2",   // 2 > 1
					}},
				{Sel: "#ECoutToCA1", Desc: "weaker",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "1.0", // 1.0 -- try 0.5
					}},
			},
			// NOTE: it is essential not to put Pat / Hip params here, as we have to use Base
			// to initialize the network every time, even if it is a different size.
		}},
		{Name: "List010", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "10",
					}},
			},
		}},
		{Name: "List020", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "20",
					}},
			},
		}},
		{Name: "List030", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "30",
					}},
			},
		}},
		{Name: "List040", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "40",
					}},
			},
		}},
		{Name: "List050", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "50",
					}},
			},
		}},
		{Name: "List060", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "60",
					}},
			},
		}},
		{Name: "List070", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "70",
					}},
			},
		}},
		{Name: "List080", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "80",
					}},
			},
		}},
		{Name: "List090", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "90",
					}},
			},
		}},
		{Name: "List100", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "100",
					}},
			},
		}},
		{Name: "List125", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "125",
					}},
			},
		}},
		{Name: "List150", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "150",
					}},
			},
		}},
		{Name: "List200", Desc: "list size", Sheets: params.Sheets{
			"Pat": &params.Sheet{
				{Sel: "Pat", Desc: "pattern params",
					Params: params.Params{
						"Pat.ListSize": "200",
					}},
			},
		}},
		{Name: "SmallHip", Desc: "hippo size", Sheets: params.Sheets{
			"Hip": &params.Sheet{
				{Sel: "HipParams", Desc: "hip sizes",
					Params: params.Params{
						"HipParams.ECPool.Y":  "7",
						"HipParams.ECPool.X":  "7",
						"HipParams.CA1Pool.Y": "10",
						"HipParams.CA1Pool.X": "10",
						"HipParams.CA3Size.Y": "20",
						"HipParams.CA3Size.X": "20",
						"HipParams.DGRatio":   "2.236", // 1.5 before, sqrt(5) aligns with Ketz et al. 2013
					}},
			},
		}},
		{Name: "MedHip", Desc: "hippo size", Sheets: params.Sheets{
			"Hip": &params.Sheet{
				{Sel: "HipParams", Desc: "hip sizes",
					Params: params.Params{
						"HipParams.ECPool.Y":  "7",
						"HipParams.ECPool.X":  "7",
						"HipParams.CA1Pool.Y": "15",
						"HipParams.CA1Pool.X": "15",
						"HipParams.CA3Size.Y": "30",
						"HipParams.CA3Size.X": "30",
						"HipParams.DGRatio":   "2.236", // 1.5 before
					}},
			},
		}},
		{Name: "BigHip", Desc: "hippo size", Sheets: params.Sheets{
			"Hip": &params.Sheet{
				{Sel: "HipParams", Desc: "hip sizes",
					Params: params.Params{
						"HipParams.ECPool.Y":  "7",
						"HipParams.ECPool.X":  "7",
						"HipParams.CA1Pool.Y": "20",
						"HipParams.CA1Pool.X": "20",
						"HipParams.CA3Size.Y": "40",
						"HipParams.CA3Size.X": "40",
						"HipParams.DGRatio":   "2.236", // 1.5 before
					}},
			},
		}},
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//// 		Configs

func ConfigEnv(ss *HipSim) {
	ss.TestEnv = &TestEnv
	ss.TrainEnv = &TrainEnv
	ss.PreTrainEpcs = 10 //from hip sim
	ss.TrialStatsFunc = TrialStats
	ss.Stop()
	//Todo Delete these variables and get from CmdArgs instead
	PholderMaxruns := 1
	PholderMaxepochs := 1
	//PHOLDERPreTrainEpcs := -1
	PholderTrainAB := TrainEnv.EvalTables[TrainAB]
	PHolderTestAB := TestEnv.EvalTables[TestAB]
	PholderStartRun := 0

	if PholderMaxruns == 0 { // allow user override
		PholderMaxruns = 1
	}
	if PholderMaxepochs == 0 { // allow user override
		PholderMaxepochs = 30
		ss.NZeroStop = 1
		//PHOLDERPreTrainEpcs = 10 // 10 > 20 perf wise
	}

	TrainEnv.Nm = "TrainEnv"
	TrainEnv.Dsc = "training params and state"

	TrainEnv.Table = etable.NewIdxView(PholderTrainAB)
	// to simulate training items in order, uncomment this line:
	// ss.TrainEnv.Sequential = true
	TrainEnv.Validate()
	TrainEnv.Run().Max = PholderMaxruns // note: we are not setting epoch max -- do that manually
	TrainEnv.Epoch().Max = PholderMaxepochs

	TestEnv.Nm = "TestEnv"
	TestEnv.Dsc = "testing params and state"
	TestEnv.Table = etable.NewIdxView(PHolderTestAB)
	TestEnv.SetSequential(true)
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(PholderStartRun) //What should this be?
	ss.TestEnv.Init(PholderStartRun)  //what should this be
}

//ConfigPats used to configure patterns
func ConfigPats(ss *HipSim) {

	trainEnv := &TrainEnv
	testEnv := &TestEnv
	hp := &ss.Hip
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.ECPool.Y // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X // makes much more readable
	npats := ss.Pat.ListSize
	pctAct := hp.ECPctAct
	minDiff := ss.Pat.MinDiffPct
	nOn := patgen.NFmPct(pctAct, plY*plX)
	ctxtflip := patgen.NFmPct(ss.Pat.CtxtFlipPct, nOn)
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt", 3, plY, plX, pctAct, minDiff) // totally diff

	for i := 0; i < (ecY-1)*ecX*3; i++ { // 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(ss.PoolVocab, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
		//todo: also support drifting
		//solution 2: drift based on last trial (will require sequential learning)
		//patgen.VocabDrift(ss.PoolVocab, ss.NFlipBits, "ctxt"+strconv.Itoa(i+1))
	}

	TrainAB, TestAB := trainEnv.EvalTables[TrainAB], testEnv.EvalTables[TestAB]
	TrainAC, TestAC := trainEnv.EvalTables[TrainAC], testEnv.EvalTables[TestAC]
	PreTrainLure, TestLure := trainEnv.EvalTables[PretrainLure], testEnv.EvalTables[TestLure]
	TrainALL := trainEnv.EvalTables[TrainAll]

	patgen.InitPats(TrainAB, "TrainAB", "TrainAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(TrainAB, ss.PoolVocab, "Input", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(TrainAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(TestAB, "TestAB", "TestAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(TestAB, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(TestAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(TrainAC, "TrainAC", "TrainAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(TrainAC, ss.PoolVocab, "Input", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(TrainAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(TestAC, "TestAC", "TestAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(TestAC, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(TestAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(PreTrainLure, "PreTrainLure", "PreTrainLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(PreTrainLure, ss.PoolVocab, "Input", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(PreTrainLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here

	patgen.InitPats(TestLure, "TestLure", "TestLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(TestLure, ss.PoolVocab, "Input", []string{"lA", "empty", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(TestLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})    // arbitrary ctxt here

	//shuold potentially go in Environments
	TrainALL = TrainAB.Clone()
	TrainALL.AppendRows(TrainAC)
	TrainALL.AppendRows(PreTrainLure)
	trainEnv.EvalTables[TrainAll] = TrainALL
}

func OpenPats(ss *sim.Sim) {
	// TODO
}

func ConfigNet(ss *HipSim, net *axon.Network) {
	net.InitName(net, ProgramName)
	hp := &ss.Hip

	in := net.AddLayer4D("Input", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Input)
	ecin := net.AddLayer4D("ECin", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Hidden)
	ecout := net.AddLayer4D("ECout", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Target) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", hp.ECSize.Y, hp.ECSize.X, hp.CA1Pool.Y, hp.CA1Pool.X, emer.Hidden)
	dg := net.AddLayer2D("DG", hp.DGSize.Y, hp.DGSize.X, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", hp.CA3Size.Y, hp.CA3Size.X, emer.Hidden)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	ecin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	ecout.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ECin", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "DG", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 2})

	onetoone := prjn.NewOneToOne()
	pool1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(in, ecin, onetoone, emer.Forward)
	net.ConnectLayers(ecout, ecin, onetoone, emer.Back)

	// EC <-> CA1 encoder pathways
	if false { // false = actually works better to use the regular projections here
		pj := net.ConnectLayersPrjn(ecin, ca1, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayersPrjn(ca1, ecout, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayersPrjn(ecout, ca1, pool1to1, emer.Back, &hip.EcCa1Prjn{})
		pj.SetClass("EcCa1Prjn")
	} else {
		pj := net.ConnectLayers(ecin, ca1, pool1to1, emer.Forward)
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayers(ca1, ecout, pool1to1, emer.Forward)
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayers(ecout, ca1, pool1to1, emer.Back)
		pj.SetClass("EcCa1Prjn")
	}

	// Perforant pathway
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hp.DGPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hp.CA3PCon

	pj := net.ConnectLayersPrjn(ecin, dg, ppathDG, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")

	if true { // toggle for bcm vs. ppath, zycyc: must use false for orig_param, true for def_param
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
	} else {
		// so far, this is sig worse, even with error-driven MinusQ1 case (which is better than off)
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
	}

	// always use this for now:
	if false { // todo: was true
		pj = net.ConnectLayersPrjn(ca3, ca1, full, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
	} else {
		// note: this requires lrate = 1.0 or maybe 1.2, doesn't work *nearly* as well
		pj = net.ConnectLayers(ca3, ca1, full, emer.Forward) // default con
		// pj.SetClass("HippoCHL")
	}

	// Mossy fibers
	mossy := prjn.NewUnifRnd()
	mossy.PCon = hp.MossyPCon
	pj = net.ConnectLayersPrjn(dg, ca3, mossy, emer.Forward, &hip.CHLPrjn{}) // no learning
	pj.SetClass("HippoCHL")

	// using 4 threads total (rest on 0)
	dg.SetThread(1)
	ca3.SetThread(2)
	ca1.SetThread(3) // this has the most

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()

	ss.Params.SetObject("Network")
	//ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func SetParamsSet(ss *HipSim, setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}

	if sheet == "" || sheet == "Hip" {
		simp, ok := pset.Sheets["Hip"]
		if ok {
			simp.Apply(&ss.Hip, setMsg)
		}
	}

	if sheet == "" || sheet == "Pat" {
		simp, ok := pset.Sheets["Pat"]
		if ok {
			simp.Apply(&ss.Pat, setMsg)
		}
	}

	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

// zycyc
// ModelSizeParams are the parameters to run for outer crossed factor testing
//var ModelSizeParams = []string{"BigHip"}
// var ModelSizeParams = []string{"MedHip", "BigHip"}
var ModelSizeParams = []string{"MedHip"}

// ListSizeParams are the parameters to run for inner crossed factor testing
// var ListSizeParams = []string{"List010"}

var ListSizeParams = []string{"List040", "List060"}

// var ListSizeParams = []string{"List020", "List040", "List060", "List080", "List100"}

// TwoFactorRun runs outer-loop crossed with inner-loop params
// TODO This needs to be called
func TwoFactorRun(ss *HipSim) {
	tag := ss.Tag
	usetag := tag
	if usetag != "" {
		usetag += "_"
	}
	for _, modelSize := range ModelSizeParams {
		for _, listSize := range ListSizeParams {
			ss.Tag = usetag + modelSize + "_" + listSize
			ss.InitRndSeed()
			SetParamsSet(ss, modelSize, "", ss.CmdArgs.LogSetParams)
			SetParamsSet(ss, listSize, "", ss.CmdArgs.LogSetParams)
			// TODO Need to uncomment this
			//ss.ReConfigNet() // note: this applies Base params to Network
			//ConfigEnv(ss)
			ss.GUI.StopNow = false
			// TODO Need to uncomment this
			//ss.PreTrain() // zycyc
			ss.NewRun()
			ss.Train()
		}
	}
	ss.Tag = tag
}
