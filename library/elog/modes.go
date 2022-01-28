package elog

import "github.com/goki/ki/kit"

// Modes the mode enum
type Modes int32

//go:generate stringer -type=Modes

var KiT_Modes = kit.Enums.AddEnum(ModesN, kit.NotBitFlag, nil)

func (ev Modes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Modes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The time scales
const (

	// AllModes represents the kind of situation where your data is being used
	AllModes Modes = iota

	// Train is this a training mode for the env
	Train

	// Test is this a test mode for the env
	Test

	// Validate is this a validation mode for the env
	Validate

	ModesN
)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
