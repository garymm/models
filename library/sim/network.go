package sim

import (
	"bytes"
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	_ "github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
	"log"
)

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// ThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
func (ss *Sim) ThetaCyc(train bool) {
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
		ss.Net.WtFmDWt()
	}

	minusCyc := 150 // 150
	plusCyc := 50   // 50

	ss.Net.NewState()
	ss.Time.NewState()
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		ss.StatCounters(train)
		if !train {
			ss.Log(elog.Test, elog.Cycle)
		}
		if ss.GUI.Active {
			ss.RasterRec(ss.Time.Cycle)
		}
		ss.Time.CycleInc()
		switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
		case 75:
			ss.Net.ActSt1(&ss.Time)
		case 100:
			ss.Net.ActSt2(&ss.Time)
		}

		if cyc == minusCyc-1 { // do before view update
			ss.Net.MinusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(viewUpdt)
		}
	}
	ss.Time.NewPhase()
	ss.StatCounters(train)
	if viewUpdt == axon.Phase {
		ss.GUI.UpdateNetView()
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		ss.StatCounters(train)
		if !train {
			ss.Log(elog.Test, elog.Cycle)
		}
		if ss.GUI.Active {
			ss.RasterRec(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(viewUpdt)
		}
	}
	ss.TrialStatsFunc(ss, train)
	ss.StatCounters(train)

	if train {
		ss.Net.DWt()
	}

	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.GUI.UpdateNetView()
	}
	if !train {
		ss.GUI.UpdatePlot(elog.Test, elog.Cycle) // make sure always updated at end
	}
}

// HipThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope
// TODO This is hippocampus specific and needs to be refactored.
func (ss *Sim) HipThetaCyc(train bool) {
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
		ss.Net.WtFmDWt()
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
	ss.Time.NewState()
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
		ss.Net.DWt()
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.GUI.UpdateNetView()
	}
	if !train {
		ss.GUI.UpdatePlot(elog.Test, elog.Cycle) // make sure always updated at end
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
// This function should maybe be moved to Environment.
func (ss *Sim) ApplyInputs(env Environment) {
	// TODO This was not being done in RA25; is it ok to do?
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := env.InputAndOutputLayers()
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := (env).State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.CmdArgs.NeedsNewRun {
		ss.NewRun()
	}

	TrainEnv := ss.TrainEnv

	TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := TrainEnv.Counter(env.Epoch)
	if chg {
		if (ss.PCAInterval > 0) && ((epc-1)%ss.PCAInterval == 0) { // -1 so runs on first epc
			ss.PCAStats()
		}
		ss.Log(elog.Train, elog.Epoch)
		ss.LrateSched(epc)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.GUI.UpdateNetView()
		}
		if (ss.TestInterval > 0) && (epc%ss.TestInterval == 0) {
			ss.TestAll()
		}
		// TODO Early stopping logic should be on the environment
		if epc == 0 || (ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.Run.Incr() { // we are done!
				ss.GUI.StopNow = true
				return
			} else {
				ss.CmdArgs.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(TrainEnv)
	if ss.UseHipTheta {
		ss.HipThetaCyc(true)
	} else {
		ss.ThetaCyc(true) // !train
	}
	ss.Log(elog.Train, elog.Trial)
	if (ss.PCAInterval > 0) && (epc%ss.PCAInterval == 0) {
		ss.Log(elog.Analyze, elog.Trial)
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(elog.Train, elog.Run)

	if ss.CmdArgs.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %s\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	TrainEnv := ss.TrainEnv
	TestEnv := ss.TestEnv

	run := ss.Run.Cur
	TrainEnv.Init(run)
	TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.LoadPretrainedWts()
	ss.InitStats()
	ss.StatCounters(true)

	ss.Logs.ResetLog(elog.Train, elog.Epoch)
	ss.Logs.ResetLog(elog.Test, elog.Epoch)
	ss.CmdArgs.NeedsNewRun = false
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	TrainEnv := ss.TrainEnv
	ss.GUI.StopNow = false
	curEpc := TrainEnv.Epoch().Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || TrainEnv.Epoch().Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.GUI.StopNow = false
	curRun := ss.Run.Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || ss.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

func (ss *Sim) TrueTrain() {

	for ; ss.Run.Cur < ss.Run.Max; ss.Run.Incr() {
		if ss.CmdArgs.NeedsNewRun {
			ss.NewRun()
		}
		for ; ss.TrainEnv.Epoch().Cur < ss.TrainEnv.Epoch().Max; ss.TrainEnv.Epoch().Incr() {
			for ; ss.TrainEnv.Trial().Cur < ss.TrainEnv.Trial().Max; ss.TrainEnv.Trial().Incr() {
				ss.TrueTrainTrial()
				if ss.GUI.StopNow == true {
					ss.Stopped()
					return
				}
			}

			epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
			if chg {
				//epc := ss.TrainEnv.Epoch().Cur
				if (ss.PCAInterval > 0) && ((epc-1)%ss.PCAInterval == 0) { // -1 so runs on first epc
					ss.PCAStats()
				}
				ss.Log(elog.Train, elog.Epoch)
				ss.LrateSched(epc)
				if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
					ss.GUI.UpdateNetView()
				}
				if (ss.TestInterval > 0) && (epc%ss.TestInterval == 0) {
					ss.TestAll()
				}
				if epc == 0 || (ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop) {
					// done with training..
					ss.RunEnd()
					if ss.Run.Incr() { // we are done!
						ss.GUI.StopNow = true
						return
					} else {
						ss.CmdArgs.NeedsNewRun = true
						return
					}
				}
			}
		}
	}
}

func (ss *Sim) TrueTrainEpoch() {
	//Do the epoch logic, this should be in a separate function

}

func (ss *Sim) TrueTrainTrial() {
	epc := ss.TrainEnv.Epoch().Cur
	ss.TrainEnv.Step()
	ss.ApplyInputs(ss.TrainEnv)
	if ss.UseHipTheta {
		ss.HipThetaCyc(true)
	} else {
		ss.ThetaCyc(true) // !train
	}
	ss.Log(elog.Train, elog.Trial)
	if (ss.PCAInterval > 0) && (epc%ss.PCAInterval == 0) {
		ss.Log(elog.Analyze, elog.Trial)
	}

}

// Train runs the full training from this point onward
// TODO This should loop through Runs, Epochs, and Trials with explicit nested for loops
func (ss *Sim) Train() {
	ss.GUI.StopNow = false
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.GUI.Stopped()
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// LrateSched implements the learning rate schedule
func (ss *Sim) LrateSched(epc int) {
	switch epc {
	case 40:
		ss.Net.LrateMod(0.5)
		fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	TestEnv := ss.TestEnv
	TestEnv.Step()

	// Query counters FIRST
	_, _, chg := TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > axon.AlphaCycle {
			ss.GUI.UpdateNetView()
		}
		ss.Log(elog.Test, elog.Epoch)
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(ss.TestEnv)
	if ss.UseHipTheta {
		ss.HipThetaCyc(false)
	} else {
		ss.ThetaCyc(false) // !train
	}
	ss.Log(elog.Test, elog.Trial)
	if ss.CmdArgs.NetData != nil { // offline record net data from testing, just final state
		ss.CmdArgs.NetData.Record(ss.GUI.NetViewText)
	}
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	TestEnv := ss.TestEnv
	cur := TestEnv.Trial().Cur
	TestEnv.Trial().Cur = idx
	ss.ApplyInputs(ss.TestEnv)
	if ss.UseHipTheta {
		ss.HipThetaCyc(false)
	} else {
		ss.ThetaCyc(false) // !train
	}
	TestEnv.Trial().Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	TestEnv := ss.TestEnv
	TestEnv.Init(ss.Run.Cur)
	for {
		ss.TestTrial(true) // return on change -- don't wrap
		_, _, chg := TestEnv.Counter(env.Epoch)
		if chg || ss.GUI.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

func (ss *Sim) LoadPretrainedWts() bool {
	if ss.PreTrainWts == nil {
		return false
	}
	b := bytes.NewReader(ss.PreTrainWts)
	err := ss.Net.ReadWtsJSON(b)
	if err != nil {
		log.Println(err)
	} else {
		fmt.Printf("loaded pretrained wts\n")
	}
	return true
}
