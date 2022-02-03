package egui

import "github.com/goki/ki/kit"

// ToolGhosting the mode enum
type ToolGhosting int32

//go:generate stringer -type=ToolGhosting

var KiT_ToolGhosting = kit.Enums.AddEnum(ToolGhostingN, kit.BitFlag, nil)

func (ev ToolGhosting) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *ToolGhosting) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The evaluation modes
const (
	ActiveStopped ToolGhosting = iota

	ActiveRunning

	ActiveAlways

	ToolGhostingN
)
