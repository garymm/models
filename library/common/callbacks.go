package common

import (
	"fmt"
	"github.com/Astera-org/models/library/sim"
	"github.com/goki/gi/gi"
)

func AddDefaultTrainCallbacks(ss *sim.Sim) {
	cc := sim.TrainingCallbacks{
		OnRunEnd: func() {
			if ss.CmdArgs.SaveWts {
				fnm := ss.WeightsFileName()
				fmt.Printf("Saving Weights to: %s\n", fnm)
				ss.Net.SaveWtsJSON(gi.FileName(fnm))
			}
		},
	}
	ss.Callbacks = append(ss.Callbacks, cc)
}
