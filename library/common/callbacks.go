package common

import (
	"fmt"
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
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
		OnThetaEnd: func() {
			if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
				ss.GUI.UpdateNetView()
			}
		},
		OnPlusPhaseStart: func() {
			if viewUpdt == axon.Phase {
				ss.GUI.UpdateNetView()
			}
		},
		OnEpochEnd: func() {
			if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
				ss.GUI.UpdateNetView()
			}
		},
	}
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, viewUpdtCallbacks)
}

// LrateSched implements the learning rate schedule
func LrateSched(ss *sim.Sim, epc int) {
	switch epc {
	case 40:
		ss.Net.LrateMod(0.5)
		fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
	}
}
