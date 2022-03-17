package sim

import "github.com/emer/emergent/elog"

// TODO Use these maybe
type ThetaPhase struct {
	Duration   int
	PhaseStart func()
	PhaseEnd   func()
}

type TrainingCallbacks struct {
	Name string

	OnRunStart        func()
	OnRunEnd          func()
	OnEpochStart      func()
	OnEpochEnd        func()
	OnTrialStart      func()
	OnTrialEnd        func()
	OnThetaStart      func()
	OnThetaEnd        func()
	OnMillisecondEnd  func()
	OnPlusPhaseStart  func()
	OnMinusPhaseStart func()
	RunStopEarly      func() bool
	EpochStopEarly    func() bool
	TrialStopEarly    func() bool
	ThetaStopEarly    func() bool

	Phases []ThetaPhase

	// TODO Add theta phases, each of which is an object with duration, name, callbacks
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

func (trainer *Trainer) OnThetaStart() {
	for _, callback := range trainer.Callbacks {
		if callback.OnThetaStart != nil {
			callback.OnThetaStart()
		}
	}
}

func (trainer *Trainer) OnThetaEnd() {
	for _, callback := range trainer.Callbacks {
		if callback.OnThetaEnd != nil {
			callback.OnThetaEnd()
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

func (trainer *Trainer) RunStopEarly() bool {
	for _, callback := range trainer.Callbacks {
		if callback.RunStopEarly != nil {
			if callback.RunStopEarly() {
				return true
			}
		}
	}
	return false
}
