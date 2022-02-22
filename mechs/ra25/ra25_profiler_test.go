package main

import (
	"github.com/Astera-org/models/library/sim"

	"os"
	"testing"
)

// In GoLand, right-click the play triangle by this method and select "Profile TestModelTraining with 'CPU Profiler'"s
func TestModelTraining(t *testing.T) {
	var TheSim sim.Sim
	TheSim.New()
	os.Args = append(os.Args, "-nogui=true")
	os.Args = append(os.Args, "-runs=1")
	os.Args = append(os.Args, "-epochs=2")
	Config(&TheSim)
	TheSim.RunFromArgs()
}
