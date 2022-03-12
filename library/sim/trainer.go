package sim

import "github.com/emer/emergent/elog"

type TrainingCallbacks struct {
	OnRunStart   func()
	OnRunEnd     func()
	OnEpochStart func()
	OnEpochEnd   func()

	// TODO Trial only does ThetaCyc, maybe don't need?
	OnTrialStart func()
	OnTrialEnd   func()

	OnThetaCycleStart func()
	OnThetaCycleEnd   func()
	OnMillisecondEnd  func()
	OnPlusPhaseStart  func()
	OnMinusPhaseStart func()
	EarlyStopping     func() bool
}

type Trainer struct {
	EvalMode           elog.EvalModes `desc:"The current training mode."`
	Callbacks          []TrainingCallbacks
	TrainRunOverride   func()
	TrainEpochOverride func()
	TrainTrialOverride func()
	ThetaCycleOverride func(sim *Sim)
}

// Boiler-plate functions to prevent copy-pasting a for-loop.

func (trainer *Trainer) OnRunStart() {
	for _, callback := range trainer.Callbacks {
		if callback.OnRunStart != nil {
			callback.OnRunStart()
		}
	}
}

func (trainer *Trainer) OnRunEnd() {
	for _, callback := range trainer.Callbacks {
		if callback.OnRunEnd != nil {
			callback.OnRunEnd()
		}
	}
}

func (trainer *Trainer) OnEpochStart() {
	for _, callback := range trainer.Callbacks {
		if callback.OnEpochStart != nil {
			callback.OnEpochStart()
		}
	}
}

func (trainer *Trainer) OnEpochEnd() {
	for _, callback := range trainer.Callbacks {
		if callback.OnEpochEnd != nil {
			callback.OnEpochEnd()
		}
	}
}

func (trainer *Trainer) OnTrialStart() {
	for _, callback := range trainer.Callbacks {
		if callback.OnTrialStart != nil {
			callback.OnTrialStart()
		}
	}
}

func (trainer *Trainer) OnTrialEnd() {
	for _, callback := range trainer.Callbacks {
		if callback.OnTrialEnd != nil {
			callback.OnTrialEnd()
		}
	}
}

func (trainer *Trainer) OnThetaCycleStart() {
	for _, callback := range trainer.Callbacks {
		if callback.OnThetaCycleStart != nil {
			callback.OnThetaCycleStart()
		}
	}
}

func (trainer *Trainer) OnThetaCycleEnd() {
	for _, callback := range trainer.Callbacks {
		if callback.OnThetaCycleEnd != nil {
			callback.OnThetaCycleEnd()
		}
	}
}

func (trainer *Trainer) OnMillisecondEnd() {
	for _, callback := range trainer.Callbacks {
		if callback.OnMillisecondEnd != nil {
			callback.OnMillisecondEnd()
		}
	}
}

func (trainer *Trainer) OnPlusPhaseStart() {
	for _, callback := range trainer.Callbacks {
		if callback.OnPlusPhaseStart != nil {
			callback.OnPlusPhaseStart()
		}
	}
}

func (trainer *Trainer) OnMinusPhaseStart() {
	for _, callback := range trainer.Callbacks {
		if callback.OnMinusPhaseStart != nil {
			callback.OnMinusPhaseStart()
		}
	}
}

func (trainer *Trainer) EarlyStopping() bool {
	for _, callback := range trainer.Callbacks {
		if callback.EarlyStopping != nil {
			if callback.EarlyStopping() {
				return true
			}
		}
	}
	return false
}
