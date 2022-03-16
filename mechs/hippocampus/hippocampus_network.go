package main

import (
	"bytes"

	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
)

func SetDgCa3Off(ss *sim.Sim, net *axon.Network, off bool) {
	ca3 := net.LayerByName("CA3").(axon.AxonLayer).AsAxon()
	dg := net.LayerByName("DG").(axon.AxonLayer).AsAxon()
	ca3.Off = off
	dg.Off = off
}

// PreTrain runs pre-training, saves weights to PreTrainWts
func PreTrain(ss *sim.Sim) {
	SetDgCa3Off(ss, ss.Net, true)

	ss.TrainEnv.AssignTable("TrainAll")
	ss.GUI.StopNow = false

	curRun := ss.TrainEnv.Run().Cur
	ss.TrainEnv.Init(curRun) // need this after changing num of rows in tables
	done := false
	for {
		done = PreTrainTrial(ss)
		if ss.GUI.StopNow || done {
			break
		}
	}
	if done {
		b := &bytes.Buffer{}
		ss.Net.WriteWtsJSON(b)
		ss.PreTrainWts = b.Bytes()
		ss.TrainEnv.AssignTable("TrainAB")
		ss.TrainEnv.Init(0)
		SetDgCa3Off(ss, ss.Net, false)
	}
	ss.GUI.Stopped()
}

// PreTrainTrial runs one trial of pretraining using TrainEnv
// returns true if done with pretraining
func PreTrainTrial(ss *sim.Sim) bool {
	//if ss.NeedsNewRun {
	//	ss.NewRun()
	//}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.Logs.Log(elog.Train, elog.Epoch)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc >= ss.CmdArgs.PreTrainEpcs { // done with training..
			ss.GUI.StopNow = true
			return true
		}
	}

	ss.ApplyInputs(ss.TrainEnv)
	PreThetaCyc(ss, true)       // special!
	ss.TrialStatsFunc(ss, true) // accumulate
	ss.Logs.Log(elog.Train, elog.Trial)
	return false
}

// PreThetaCyc runs one theta cycle (200 msec) of processing.
// This one is for pretraining: no connection switching.
func PreThetaCyc(ss *sim.Sim, train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt(&ss.Time)
	}

	ca1 := ss.Net.LayerByName("CA1").(axon.AxonLayer).AsAxon()
	// ca3 := ss.Net.LayerByName("CA3").(axon.AxonLayer).AsAxon()
	// ecin := ss.Net.LayerByName("ECin").(axon.AxonLayer).AsAxon()
	ecout := ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(axon.AxonPrjn).AsAxon()
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(axon.AxonPrjn).AsAxon()

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.PrjnScale.Abs = 1
	ca1FmCa3.PrjnScale.Abs = 0

	if train {
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		ecout.SetType(emer.Compare) // don't clamp
	}
	ecout.UpdateExtFlags() // call this after updating type

	ss.Net.InitGScale() // update computed scaling factors

	cycPerQtr := []int{100, 50, 50, 50} // 100, 50, 50, 50 notably better

	ss.Net.NewState()
	ss.Time.NewState(train)
	for qtr := 0; qtr < 4; qtr++ {
		maxCyc := cycPerQtr[qtr]
		for cyc := 0; cyc < maxCyc; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.Logs.Log(elog.Test, elog.Cycle)
			}
			ss.Time.CycleInc()

			if ss.ViewOn {
				ss.UpdateViewTime(viewUpdt)
			}
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ss.Net.ActSt1(&ss.Time)
		case 2:
			ss.Net.ActSt2(&ss.Time)
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ss.Net.MinusPhase(&ss.Time)
			ss.MemStats(train) // must come after QuarterFinal
		case 4:
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(viewUpdt)
		}
	}

	if train {
		ss.Net.DWt(&ss.Time)
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.UpdateView(train)
	}

	ss.GUI.Plot(elog.Test, elog.Cycle)
}

// HipThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope
// TODO This is hippocampus specific and needs to be refactored.
func HipThetaCyc(ss *sim.Sim) {

	train := ss.Trainer.EvalMode == elog.Train
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt(&ss.Time)
	}

	ca1 := ss.Net.LayerByName("CA1").(axon.AxonLayer).AsAxon()
	ca3 := ss.Net.LayerByName("CA3").(axon.AxonLayer).AsAxon()
	// ecin := ss.Net.LayerByName("ECin").(axon.AxonLayer).AsAxon()
	ecout := ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(axon.AxonPrjn).AsAxon()
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(axon.AxonPrjn).AsAxon()
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(axon.AxonPrjn).AsAxon()

	absGain := float32(2)

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.PrjnScale.Abs = absGain
	ca1FmCa3.PrjnScale.Abs = 0

	dgwtscale := ca3FmDg.PrjnScale.Rel

	//ca3FmDg.PrjnScale.Rel = dgwtscale - ss.Hip.MossyDel
	ca3FmDg.PrjnScale.Rel = dgwtscale - 3 // turn off DG input to CA3 in first quarter // TODO 3 Should be replaced with HipSim.MossyDel, and that brings up doubts about our overall approach to HipSim

	if train {
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		ecout.SetType(emer.Compare) // don't clamp
	}
	ecout.UpdateExtFlags() // call this after updating type

	ss.Net.InitGScale() // update computed scaling factors

	// cycPerQtr := []int{100, 100, 100, 100}
	cycPerQtr := []int{50, 50, 50, 50} // 100, 25, 25, 50 best so far, vs 75,50 at start, 50,50 instead of 25..
	// cycPerQtr := []int{100, 1, 1, 50} // 150, 1, 1, 50 works for EcCa1Prjn, but 100, 1, 1, 50 does not

	ss.Net.NewState()
	ss.Time.NewState(train)
	for qtr := 0; qtr < 4; qtr++ {
		maxCyc := cycPerQtr[qtr]
		for cyc := 0; cyc < maxCyc; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.Log(elog.Test, elog.Cycle)
			}
			ss.Time.CycleInc()

			if ss.ViewOn {
				ss.UpdateViewTime(viewUpdt) //TOdo in original version train is a variable with true, ask randy why this is removed
			}
		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ss.Net.ActSt1(&ss.Time)
			ca1FmECin.PrjnScale.Abs = 0
			ca1FmCa3.PrjnScale.Abs = absGain
			if train {
				ca3FmDg.PrjnScale.Rel = dgwtscale // restore after 1st quarter
			} else {
				ca3FmDg.PrjnScale.Rel = dgwtscale - 0 //TODO 3 Should be replaced with HipSim.MossyDel, and that brings up doubts about our overall approach to HipSim
				//ca3FmDg.PrjnScale.Rel = dgwtscale - ss.Hip.MossyDelTest // testing
			}
			ss.Net.InitGScale() // update computed scaling factors
		case 2:
			ss.Net.ActSt2(&ss.Time)
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			if train { // clamp ECout from ECin
				ca1FmECin.PrjnScale.Abs = absGain
				ca1FmCa3.PrjnScale.Abs = 0
				ss.Net.InitGScale() // update computed scaling factors
				// ecin.UnitVals(&ss.TmpVals, "Act")
				// ecout.ApplyExt1D32(ss.TmpVals)
			}
			ss.Net.MinusPhase(&ss.Time)

			ss.MemStats(train) // must come after QuarterFinal
		case 4:
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(viewUpdt)
		}
	}

	ca3FmDg.PrjnScale.Rel = dgwtscale // restore
	ca1FmCa3.PrjnScale.Abs = absGain

	if train {
		ss.Net.DWt(&ss.Time)
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.GUI.UpdateNetView()
	}
	if !train {
		ss.GUI.UpdatePlot(elog.Test, elog.Cycle) // make sure always updated at end
	}
}
