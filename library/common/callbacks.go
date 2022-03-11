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

	// Weight Visibility
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnThetaCycleStart: func() {
			// update prior weight changes at start, so any DWt values remain visible at end
			if ss.Trainer.EvalMode == elog.Train {
				ss.Net.WtFmDWt(&ss.Time)
			}
		},
		OnThetaCycleEnd: func() {
			if ss.Trainer.EvalMode == elog.Train {
				ss.Net.WtFmDWt(&ss.Time)
			}
		},
	})

	// Learning Rate Schedule
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == elog.Train {
				ss.LrateSched(ss.TrainEnv.Epoch().Cur)
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

	AddDefaultGUICallbacks(ss)
}

func AddDefaultGUICallbacks(ss *sim.Sim) {
	var viewUpdt axon.TimeScales // Reset at the top of theta cycle.
	viewUpdtCallbacks := sim.TrainingCallbacks{
		OnThetaCycleStart: func() {
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
		OnThetaCycleEnd: func() {
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
