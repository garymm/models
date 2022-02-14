package main

import (
	"fmt"
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/etable"
	"testing"
)

func TestConfigNet(t *testing.T) {
	simo := sim.Sim{}
	neto := axon.Network{}
	ConfigNet(&simo, &neto)
	if len(neto.Layers) != 4 {
		t.Errorf("Expected network to be configured differently")
	}
}

func TestConfigPats(t *testing.T) {
	ss := One2Sim{}
	ss.Pats = &etable.Table{}
	ss.NInputs = 5
	ss.NOutputs = 2
	ConfigPats(&ss)
	if ss.Pats.Rows < 10 {
		fmt.Println(*ss.Pats)
		t.Errorf("Expected more patterns than that!")
	}
}