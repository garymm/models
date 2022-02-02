package elog

import "github.com/goki/ki/kit"

// EvalModes the mode enum
type EvalModes int32

//go:generate stringer -type=EvalModes

var KiT_EvalModes = kit.Enums.AddEnum(EvalModesN, kit.BitFlag, nil)

func (ev EvalModes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *EvalModes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The evaluation modes
const (
	UnknownEvalMode EvalModes = iota

	// AllEvalModes indicates that the log should occur over all modes present in other items.
	AllEvalModes

	// Train is this a training mode for the env
	Train

	// Test is this a test mode for the env
	Test

	// Validate is this a validation mode for the env
	Validate

	EvalModesN
)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
