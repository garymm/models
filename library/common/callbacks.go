package common

import (
	"fmt"
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/goki/gi/gi"
)

func AddDefaultTrainCallbacks(ss *sim.Sim) {
	// Save Weights
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnRunEnd: func() {
			if ss.CmdArgs.SaveWts {
				fnm := ss.WeightsFileName()
				fmt.Printf("Saving Weights to: %s\n", fnm)
				ss.Net.SaveWtsJSON(gi.FileName(fnm))
			}
		},
	})

	// Weight storage and update delta.
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		Name: "DWt",
		OnThetaStart: func() {
			// update prior weight changes at start, so any DWt values remain visible at end
			// you might want to do this less frequently to achieve a mini-batch update
			// in which case, move it out to the TrainTrial method where the relevant
			// counters are being dealt with.

			// update prior weight changes at start, so any DWt values remain visible at end
			if ss.Trainer.EvalMode == elog.Train {
				// Apply delta weight.
				ss.Net.WtFmDWt(&ss.Time)
			}
		},
		OnThetaEnd: func() {
			if ss.Trainer.EvalMode == elog.Train {
				// Compute delta weight.
				ss.Net.DWt(&ss.Time)
			}
		},
	})

	// Learning Rate Schedule
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == elog.Train {
				LrateSched(ss, ss.TrainEnv.Epoch().Cur)
			}
		},
	})

	// Testing
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == elog.Train {
				if (ss.TestInterval > 0) && ((ss.TrainEnv.Epoch().Cur+1)%ss.TestInterval == 0) {
					ss.TestAll()
				}
			}
		},
	})

	// PCA Stats
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == elog.Train {
				// Should run on first epoch, needs to run before Log.
				if (ss.PCAInterval > 0) && (ss.TrainEnv.Epoch().Cur%ss.PCAInterval == 0) {
					ss.PCAStats()
				}
			}
		},
		OnTrialEnd: func() {
			if ss.Trainer.EvalMode == elog.Train {
				if (ss.PCAInterval > 0) && (ss.TrainEnv.Epoch().Cur%ss.PCAInterval == 0) {
					ss.Log(elog.Analyze, elog.Trial)
				}
			}
		},
	})

	// First Zero Early Stopping
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		RunStopEarly: func() bool {
			return ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop
		},
	})

	AddPlusAndMinusPhases(ss)

	AddDefaultGUICallbacks(ss)
}

func AddDefaultGUICallbacks(ss *sim.Sim) {
	var viewUpdt axon.TimeScales // Reset at the top of theta cycle.
	viewUpdtCallbacks := sim.TrainingCallbacks{
		OnThetaStart: func() {
			// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine

			viewUpdt = ss.TrainUpdt
			if ss.Trainer.EvalMode == elog.Test {
				viewUpdt = ss.TestUpdt
			}
			if viewUpdt == axon.Phase {
				ss.GUI.UpdateNetView()
			}
		},
		OnMillisecondEnd: func() {
			if ss.ViewOn {
				ss.UpdateViewTime(viewUpdt)
			}
		},
	}
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, viewUpdtCallbacks)
}

func AddPlusAndMinusPhases(ss *sim.Sim) {
	ss.Trainer.Phases = []sim.ThetaPhase{sim.ThetaPhase{
		Name:     "Minus",
		Duration: 150,
		OnMillisecondEnd: func() {
			switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
			case 75:
				ss.Net.ActSt1(&ss.Time)
			case 100:
				ss.Net.ActSt2(&ss.Time)
			}
		},
		PhaseStart: func() {
			ss.Time.PlusPhase = false
		},
		PhaseEnd: func() {
			ss.Net.MinusPhase(&ss.Time)
		},
	}, sim.ThetaPhase{
		Name:     "Plus",
		Duration: 50,
		PhaseStart: func() {
			ss.Time.PlusPhase = true
			ss.StatCounters(ss.Trainer.EvalMode == elog.Train)
		},
		PhaseEnd: func() {
			ss.Net.PlusPhase(&ss.Time)
		},
	}}
}

// LrateSched implements the learning rate schedule
func LrateSched(ss *sim.Sim, epc int) {
	switch epc {
	case 40:
		ss.Net.LrateMod(0.5)
		fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
	}
}

func AddHipCallbacks(ss *sim.Sim) {
	// TODO Make sure these are gotten at the correct time.
	ca1 := ss.Net.LayerByName("CA1").(axon.AxonLayer).AsAxon()
	ca3 := ss.Net.LayerByName("CA3").(axon.AxonLayer).AsAxon()
	// ecin := ss.Net.LayerByName("ECin").(axon.AxonLayer).AsAxon()
	ecout := ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(axon.AxonPrjn).AsAxon()
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(axon.AxonPrjn).AsAxon()
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(axon.AxonPrjn).AsAxon()
	absGain := float32(2)

	// Notes on durations: / 100, 25, 25, 50 best so far, vs 75,50 at start, 50,50 instead of 25..
	//	// cycPerQtr := []int{100, 1, 1, 50} // 150, 1, 1, 50 works for EcCa1Prjn, but 100, 1, 1, 50 does not

	var dgwtscale float32

	// Override Default Phases
	ss.Trainer.Phases = []sim.ThetaPhase{sim.ThetaPhase{
		Name:     "Q1",
		Duration: 50,
		PhaseEnd: func() {
			// Second, Third Quarters: CA1 is driven by CA3 recall
			ss.Net.ActSt1(&ss.Time)
			ca1FmECin.PrjnScale.Abs = 0
			ca1FmCa3.PrjnScale.Abs = absGain
			if ss.Trainer.EvalMode == elog.Train {
				ca3FmDg.PrjnScale.Rel = dgwtscale // restore after 1st quarter
			} else {
				ca3FmDg.PrjnScale.Rel = dgwtscale - 0 //TODO 3 Should be replaced with HipSim.MossyDel, and that brings up doubts about our overall approach to HipSim
				//ca3FmDg.PrjnScale.Rel = dgwtscale - ss.Hip.MossyDelTest // testing
			}
			ss.Net.InitGScale() // update computed scaling factors
		},
	}, sim.ThetaPhase{
		Name:     "Q2",
		Duration: 50,
		PhaseEnd: func() {
			ss.Net.ActSt2(&ss.Time)
		},
	}, sim.ThetaPhase{
		Name:     "Q3",
		Duration: 50,
		PhaseEnd: func() { // Fourth Quarter: CA1 back to ECin drive only
			train := ss.Trainer.EvalMode == elog.Train
			if train { // clamp ECout from ECin
				ca1FmECin.PrjnScale.Abs = absGain
				ca1FmCa3.PrjnScale.Abs = 0
				ss.Net.InitGScale() // update computed scaling factors
				// ecin.UnitVals(&ss.TmpVals, "Act")
				// ecout.ApplyExt1D32(ss.TmpVals)
			}
			ss.Net.MinusPhase(&ss.Time)

			ss.MemStats(train) // must come after QuarterFinal
		},
	}, sim.ThetaPhase{
		Name:     "Q4",
		Duration: 50,
		PhaseEnd: func() {
			ss.Net.PlusPhase(&ss.Time)
		},
	}}

	// Hip Theta Cycle
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnThetaStart: func() {
			// First Quarter: CA1 is driven by ECin, not by CA3 recall
			// (which is not really active yet anyway)
			ca1FmECin.PrjnScale.Abs = absGain
			ca1FmCa3.PrjnScale.Abs = 0

			dgwtscale = ca3FmDg.PrjnScale.Rel

			//ca3FmDg.PrjnScale.Rel = dgwtscale - ss.Hip.MossyDel
			ca3FmDg.PrjnScale.Rel = dgwtscale - 3 // turn off DG input to CA3 in first quarter // TODO 3 Should be replaced with HipSim.MossyDel, and that brings up doubts about our overall approach to HipSim

			if ss.Trainer.EvalMode == elog.Train {
				ecout.SetType(emer.Target) // clamp a plus phase during testing todo: ask randy why this is the case
			} else {
				ecout.SetType(emer.Compare) // don't clamp
			}
			ecout.UpdateExtFlags() // call this after updating type

			ss.Net.InitGScale() // update computed scaling factors

		},
		OnThetaEnd: func() {
			ca3FmDg.PrjnScale.Rel = dgwtscale // restore
			ca1FmCa3.PrjnScale.Abs = absGain
		},
		OnMillisecondEnd: func() {
			if ss.Trainer.EvalMode != elog.Train {
				ss.Log(elog.Test, elog.Cycle)
			}
		},
		OnEveryPhaseEnd: func() {
			if ss.GetViewUpdate() == axon.Phase {
				ss.GUI.UpdateNetView()
			}
		},
	})

}
