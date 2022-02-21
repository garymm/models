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
			sim.GuiRun(&TheSim.Sim, ProgramName, "Hippocampus", `This demonstrates a hippocampus Axon model.`)
		})
	}

}

// Config configures all the elements using the standard functions
func Config(ss *HipSim) {
	ConfigPats(ss)
	OpenPats(&ss.Sim)
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
			"NetSize": &params.Sheet{
				{Sel: ".Hidden", Desc: "all hidden layers",
					Params: params.Params{
						"Layer.X": "10", //todo layer size correspondence between areas that are connected upstream parameter - get there when we get there
						"Layer.Y": "10",
					},
					Hypers: params.Hypers{
						"Layer.X": {"StdDev": "0.3", "Min": "2"},
						"Layer.Y": {"StdDev": "0.3", "Min": "2"},
					},
				},
			},
			"Network": &params.Sheet{
				{Sel: "Layer", Desc: "all defaults",
					Params: params.Params{
						// All params with importance >=5 have hypers
						"Layer.Inhib.Layer.Gi": "1.2", // 1.2 > 1.1     importance: 10
						// TODO This param should vary with Gi it looks like
						"Layer.Inhib.ActAvg.Init": "0.04", // 0.04 for 1.2, 0.08 for 1.1  importance: 10
						"Layer.Inhib.Layer.Bg":    "0.3",  // 0.3 > 0.0   importance: 2
						"Layer.Act.Decay.Glong":   "0.6",  // 0.6   importance: 2
						"Layer.Act.Dend.GbarExp":  "0.5",  // 0.2 > 0.1 > 0   importance: 5
						"Layer.Act.Dend.GbarR":    "6",    // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels importance: 5
						"Layer.Act.Dt.VmDendTau":  "5",    // 5 > 2.81 here but small effect importance: 1
						"Layer.Act.Dend.VGCCCa":   "20",
						"Layer.Act.Dend.CaMax":    "90",
						"Layer.Act.Dt.VmSteps":    "2",    // 2 > 3 -- somehow works better importance: 1
						"Layer.Act.Dt.GeTau":      "5",    // importance: 1
						"Layer.Act.NMDA.Gbar":     "0.15", //  importance: 7
						"Layer.Act.NMDA.MgC":      "1.4",
						"Layer.Act.NMDA.Voff":     "5",
						"Layer.Act.GABAB.Gbar":    "0.2", // 0.2 > 0.15  importance: 7
					}, Hypers: params.Hypers{
						"Layer.Inhib.Layer.Gi":    {"StdDev": "0.2"},
						"Layer.Inhib.ActAvg.Init": {"StdDev": "0.01", "Min": "0.01"},
						"Layer.Act.Dend.GbarExp":  {"StdDev": "0.05"},
						"Layer.Act.Dend.GbarR":    {"StdDev": "1"},
						"Layer.Act.NMDA.Gbar":     {"StdDev": "0.04"},
						"Layer.Act.GABAB.Gbar":    {"StdDev": "0.05"},
					}},
				{Sel: "#Input", Desc: "critical now to specify the activity level",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0
						"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > 0.6 >= 0.7 == 0.5
						"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
					},
					Hypers: params.Hypers{
						"Layer.Inhib.Layer.Gi": {"StdDev": ".1", "Min": "0", "Priority": "2", "Scale": "LogLinear"},
						"Layer.Act.Clamp.Ge":   {"StdDev": ".2"},
					}},
				{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 0.8 > 1.0 > 0.7 even with adapt -- not beneficial to start low
						"Layer.Inhib.ActAvg.Init": "0.24", // this has to be exact for adapt
						"Layer.Act.Spike.Tr":      "1",    // 1 is new minimum.
						"Layer.Act.Clamp.Ge":      "0.6",  // .6 > .5 v94
						// "Layer.Act.NMDA.Gbar":     "0.3",  // higher not better
					}},
				{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base": "0.2", // 0.04 no rlr, 0.2 rlr; .3, WtSig.Gain = 1 is pretty close  //importance: 10
						"Prjn.SWt.Adapt.Lrate":  "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint //importance: 5
						"Prjn.SWt.Init.SPct":    "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..  //importance: 7
					},
					Hypers: params.Hypers{
						"Prjn.Learn.Lrate.Base": {"StdDev": "0.05"},
						"Prjn.SWt.Adapt.Lrate":  {"StdDev": "0.025"},
						"Prjn.SWt.Init.SPct":    {"StdDev": "0.1"},
					}},
				{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5 //importance: 9
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

func ConfigEnv(ss *HipSim) {
	ss.TestEnv = &TestEnv
	ss.TrainEnv = &TrainEnv

	//Todo create the appropriate names
	PholderMaxruns := -1
	PholderMaxepochs := -1
	//PHOLDERPreTrainEpcs := -1
	PholderTrainAB := etable.Table{}
	PHolderTestAB := etable.Table{}
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

	TrainEnv.Table = etable.NewIdxView(&PholderTrainAB)
	// to simulate training items in order, uncomment this line:
	// ss.TrainEnv.Sequential = true
	TrainEnv.Validate()
	TrainEnv.Run().Max = PholderMaxruns // note: we are not setting epoch max -- do that manually

	TestEnv.Nm = "TestEnv"
	TestEnv.Dsc = "testing params and state"
	TestEnv.Table = etable.NewIdxView(&PHolderTestAB)
	TestEnv.SetSequential(true)
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(PholderStartRun) //What should this be?
	ss.TestEnv.Init(PholderStartRun)  //what should this be
}

//ConfigPats used to configure patterns
func ConfigPats(ss *HipSim) {

	trainEnv := &TrainEnv
	testEnv := &TrainEnv
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
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", etable.Tab)
	if err != nil {
		log.Println(err)
	}
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

// TODO Move everything after this point out of here into libraries
