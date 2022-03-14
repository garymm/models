package sim

import (
	"bytes"
	"fmt"
	"log"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/env"
	_ "github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
)

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// ThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
// TODO Using this it doesn't learn
func (ss *Sim) ThetaCyc() {
	if ss.Trainer.ThetaCycleOverride != nil {
		ss.Trainer.ThetaCycleOverride(ss)
		return
	}

	train := ss.Trainer.EvalMode == elog.Train

	ss.Trainer.OnThetaCycleStart()

	// TODO Parameterize these.
	minusCyc := 150 // 150
	plusCyc := 50   // 50

	ss.Net.NewState()
	ss.Time.NewState(train)
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

		ss.Trainer.OnMillisecondEnd()
	}
	ss.Time.NewPhase()
	ss.StatCounters(train)
	ss.Trainer.OnPlusPhaseStart()
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

		ss.Trainer.OnMillisecondEnd()
	}
	ss.TrialStatsFunc(ss, train)
	ss.StatCounters(train)

	if !train {
		ss.GUI.UpdatePlot(elog.Test, elog.Cycle) // make sure always updated at end
	}
	ss.Trainer.OnThetaCycleEnd()
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

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(elog.Train, elog.Run)

	ss.Trainer.OnRunEnd()
}

// NewRun intializes a new run of the model, using the ss.Run counter
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

func (ss *Sim) TrainTrial() {
	ss.Trainer.EvalMode = elog.Train
	if ss.TrainEnv.Trial().Cur == -1 {
		// This is a hack, and it should be initialized at 0
		ss.TrainEnv.Trial().Cur = 0
	}
	ss.StatCounters(true)
	ss.ApplyInputs(ss.TrainEnv)

	ss.ThetaCyc()

	ss.Log(elog.Train, elog.Trial)
	ss.Trainer.OnTrialEnd()
}

// TrainEpoch runs until the end of the Epoch, then updates logs.
func (ss *Sim) TrainEpoch() {
	ss.Trainer.EvalMode = elog.Train
	ss.Trainer.OnEpochStart()
	for ; ss.TrainEnv.Trial().Cur < ss.TrainEnv.Trial().Max; ss.TrainEnv.Trial().Cur += 1 {
		ss.TrainTrial()
		if ss.GUI.StopNow == true {
			ss.Stopped()
			return
		}
	}
	ss.TrainEnv.Trial().Cur = 0

	ss.Trainer.OnEpochEnd()
	// Log after OnEpochEnd.
	ss.Log(elog.Train, elog.Epoch)
}

func (ss *Sim) TrainRun() {
	ss.Trainer.EvalMode = elog.Train
	ss.NewRun()
	if ss.TrainEnv.Trial().Cur == -1 {
		// This is a hack, and it should be initialized at 0
		ss.TrainEnv.Trial().Cur = 0
	}
	for ; ss.TrainEnv.Epoch().Cur < ss.TrainEnv.Epoch().Max; ss.TrainEnv.Epoch().Cur += 1 {
		ss.TrainEpoch()
		if ss.GUI.StopNow == true {
			return
		}
		if ss.Trainer.EarlyStopping() {
			// End this run early
			break
		}
	}
	ss.TrainEnv.Epoch().Cur = 0
	ss.RunEnd()
}

// Train trains until the end of runs, unless stopped early by the GUI.
func (ss *Sim) Train() {
	ss.Trainer.EvalMode = elog.Train
	// Note that Run, Epoch, and Trial are not initialized at zero to allow Train to restart where it left off.
	for ; ss.Run.Cur < ss.Run.Max; ss.Run.Cur += 1 {
		ss.TrainRun()
		if ss.GUI.StopNow == true {
			ss.GUI.StopNow = false
			return
		}
	}
	ss.GUI.StopNow = true
	ss.GUI.Stopped()
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

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
// TODO Rewrite this the same as TrainTrial, or merge them
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
	ss.ThetaCyc()
	ss.Log(elog.Test, elog.Trial)
	if ss.CmdArgs.NetData != nil { // offline record net data from testing, just final state
		ss.CmdArgs.NetData.Record(ss.GUI.NetViewText)
	}
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	ss.Trainer.EvalMode = elog.Test
	TestEnv := ss.TestEnv
	cur := TestEnv.Trial().Cur
	TestEnv.Trial().Cur = idx
	ss.ApplyInputs(ss.TestEnv)
	ss.ThetaCyc()
	TestEnv.Trial().Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Trainer.EvalMode = elog.Test
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
