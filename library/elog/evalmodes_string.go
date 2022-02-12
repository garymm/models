// Code generated by "stringer -type=EvalModes"; DO NOT EDIT.

package elog

import (
	"errors"
	"strconv"
)

var _ = errors.New("dummy error")

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[NoEvalMode-0]
	_ = x[AllModes-1]
	_ = x[Train-2]
	_ = x[Test-3]
	_ = x[Validate-4]
	_ = x[EvalModesN-5]
}

const _EvalModes_name = "NoEvalModeAllModesTrainTestValidateEvalModesN"

var _EvalModes_index = [...]uint8{0, 10, 18, 23, 27, 35, 45}

func (i EvalModes) String() string {
	if i < 0 || i >= EvalModes(len(_EvalModes_index)-1) {
		return "EvalModes(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _EvalModes_name[_EvalModes_index[i]:_EvalModes_index[i+1]]
}

func (i *EvalModes) FromString(s string) error {
	for j := 0; j < len(_EvalModes_index)-1; j++ {
		if s == _EvalModes_name[_EvalModes_index[j]:_EvalModes_index[j+1]] {
			*i = EvalModes(j)
			return nil
		}
	}
	return errors.New("String: " + s + " is not a valid option for type: EvalModes")
}
