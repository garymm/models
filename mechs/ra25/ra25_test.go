package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/etable/etable"
	"os"
	"testing"
)

// Commented out because the args parsing within Config can only be called once in a tests file.
//func TestConfigNet(t *testing.T) {
//	simo := sim.Sim{}
//	simo.New()
//	simo.CmdArgs.NoGui = true
//	Config(&simo)
//	neto := axon.Network{}
//	ConfigNet(&simo, &neto)
//	if len(neto.Layers) != 4 {
//		t.Errorf("Expected network to be configured differently")
//	}
//}

func TestConfigPats(t *testing.T) {
	ss := sim.Sim{}
	ss.Pats = &etable.Table{}
	ConfigPats(&ss)
	if ss.Pats.Rows < 10 {
		t.Errorf("Expected more patterns than that!")
	}
}

// In GoLand, right-click the play triangle by this method and select "Profile TestModelTraining with 'CPU Profiler'"s
// TODO Modify this test to ensure that LastZero is achieved in under 100 epochs
func TestModelTraining(t *testing.T) {
	var TheSim sim.Sim
	TheSim.New()
	// TODO Add params
	os.Args = append(os.Args, "-nogui=true")
	os.Args = append(os.Args, "-runs=1")
	os.Args = append(os.Args, "-epochs=2")
	Config(&TheSim)
	TheSim.RunFromArgs()
}
