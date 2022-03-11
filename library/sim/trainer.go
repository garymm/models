package sim

import "github.com/emer/emergent/elog"

type TrainingCallbacks struct {
	OnRunStart   func()
	OnRunEnd     func()
	OnEpochStart func()
	OnEpochEnd   func()
	OnTrialStart func()
	OnTrialEnd   func()
	OnCycleStart func()
	OnCycleEnd   func()
}

type Trainer struct {
	EvalMode elog.EvalModes `desc:"The current training mode."`
	//Callbacks []TrainingCallbacks // TODO Move from Sim
	TrainRunOverride   func()
	TrainEpochOverride func()
	TrainTrialOverride func()
	ThetaCycleOverride func(sim *Sim)
}
