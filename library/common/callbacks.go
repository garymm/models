package common

import (
	"fmt"
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/goki/gi/gi"
)

func AddDefaultTrainCallbacks(ss *sim.Sim) {
	saveWeights := sim.TrainingCallbacks{
		OnRunEnd: func() {
			if ss.CmdArgs.SaveWts {
				fnm := ss.WeightsFileName()
				fmt.Printf("Saving Weights to: %s\n", fnm)
				ss.Net.SaveWtsJSON(gi.FileName(fnm))
			}
		},
	}

	weightVisiblity := sim.TrainingCallbacks{
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
	}

	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, saveWeights, weightVisiblity)

	AddDefaultGUICallbacks(ss)
}

func AddDefaultGUICallbacks(ss *sim.Sim) {
	guiview := InitGUIViewHandler(ss)
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, guiview.TrainingCallbacks)
}

type GUIViewHandler struct {
	sim.TrainingCallbacks
	viewUpdt axon.TimeScales
	ss       *sim.Sim
}

func InitGUIViewHandler(ss *sim.Sim) *GUIViewHandler {
	gui := &GUIViewHandler{ss: ss}
	gui.TrainingCallbacks.OnThetaCycleStart = gui.OnThetaCycleStart
	gui.TrainingCallbacks.OnMillisecondEnd = gui.OnMillisecondEnd
	gui.TrainingCallbacks.OnThetaCycleEnd = gui.OnThetaCycleEnd
	return gui
}

func (guiview *GUIViewHandler) OnThetaCycleStart() {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := guiview.ss.TrainUpdt
	if !(guiview.ss.Trainer.EvalMode == elog.Train) {
		viewUpdt = guiview.ss.TestUpdt
	}
	if viewUpdt == axon.Phase {
		guiview.ss.GUI.UpdateNetView()
	}
	guiview.viewUpdt = viewUpdt
}

func (guiview *GUIViewHandler) OnMillisecondEnd() {
	if guiview.ss.ViewOn {
		guiview.ss.UpdateViewTime(guiview.viewUpdt)
	}
}

func (guiview *GUIViewHandler) OnThetaCycleEnd() {
	if guiview.viewUpdt == axon.Phase || guiview.viewUpdt == axon.AlphaCycle || guiview.viewUpdt == axon.ThetaCycle {
		guiview.ss.GUI.UpdateNetView()
	}
}
