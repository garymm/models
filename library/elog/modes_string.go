// Code generated by "stringer -type=TrainOrTest"; DO NOT EDIT.

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
	_ = x[AllModes-0]
	_ = x[Train-1]
	_ = x[Test-2]
	_ = x[Validate-3]
	_ = x[ModesN-4]
}

const _Modes_name = "AllModesTrainTestValidateModesN"

var _Modes_index = [...]uint8{0, 8, 13, 17, 25, 31}

func (i TrainOrTest) String() string {
	if i < 0 || i >= TrainOrTest(len(_Modes_index)-1) {
		return "TrainOrTest(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _Modes_name[_Modes_index[i]:_Modes_index[i+1]]
}

func (i *TrainOrTest) FromString(s string) error {
	for j := 0; j < len(_Modes_index)-1; j++ {
		if s == _Modes_name[_Modes_index[j]:_Modes_index[j+1]] {
			*i = TrainOrTest(j)
			return nil
		}
	}
	return errors.New("String: " + s + " is not a valid option for type: TrainOrTest")
}
