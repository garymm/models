package sim

import (
	"fmt"

	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/env"
	"github.com/goki/gi/gi"
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
		if !train {
			ss.Log(elog.Test, elog.Cycle)
			if ss.GUI.CycleUpdateRate > 0 && (ss.Time.Cycle%ss.GUI.CycleUpdateRate) == 0 {
				ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Test, elog.Cycle))
			}
		}
		if !ss.CmdArgs.NoGui {
			ss.RecSpikes(ss.Time.Cycle)
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
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.Time.NewPhase()
	if viewUpdt == axon.Phase {
		ss.UpdateView(train)
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		if !train {
			ss.Log(elog.Test, elog.Cycle)
			if ss.GUI.CycleUpdateRate > 0 && (ss.Time.Cycle%ss.GUI.CycleUpdateRate) == 0 {
				ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Test, elog.Cycle))
			}

		}
		if !ss.CmdArgs.NoGui {
			ss.RecSpikes(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.TrialStatsFunc(ss, train)

	if train {
		ss.Net.DWt()
	}

	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.UpdateView(train)
	}
	// TODO check why this is being called here instead of in plus or minus phase
	if ss.GUI.CycleUpdateRate > 0 && (ss.Time.Cycle%ss.GUI.CycleUpdateRate) == 0 {
		ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Test, elog.Cycle))
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en Environment) {
	// ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := (en).State(ly.Nm)
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
		ss.Log(elog.Train, elog.Epoch)
		ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Train, elog.Epoch))
		ss.LrateSched(epc)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc == 0 || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
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
	ss.ThetaCyc(true)
	ss.Log(elog.Train, elog.Trial)
	ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Train, elog.Trial))
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(elog.Train, elog.Run)
	ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Train, elog.Run))
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
	ss.InitStats()
	ss.Logs.Table(elog.Train, elog.Epoch).SetNumRows(0)
	ss.Logs.Table(elog.Test, elog.Epoch).SetNumRows(0)
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

// Train runs the full training from this point onward
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
	ss.GUI.IsRunning = false
	if ss.GUI.Win != nil {
		vp := ss.GUI.Win.WinViewport2D()
		if ss.GUI.ToolBar != nil {
			ss.GUI.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
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
			ss.UpdateView(false)
		}
		ss.Log(elog.Test, elog.Epoch)
		ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Test, elog.Epoch))
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.Log(elog.Test, elog.Trial)
	ss.GUI.UpdatePlot(elog.GenScopeKey(elog.Test, elog.Trial))
	if ss.CmdArgs.NetData != nil { // offline record net data from testing, just final state
		ss.CmdArgs.NetData.Record(ss.Counters(false))
	}
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	TestEnv := ss.TestEnv
	cur := TestEnv.Trial().Cur
	TestEnv.Trial().Cur = idx
	ss.ApplyInputs(ss.TestEnv)
	ss.ThetaCyc(false) // !train
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
