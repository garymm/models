// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip_bench runs a hippocampus model for testing parameters and new learning ideas
package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/elog"
)

type HipSim struct {
	sim.Sim
}

func (hip *HipSim) New() {
	hip.Stats.Init()
}

func (hip *HipSim) NewRun() {
	//hip.InitRndSeed()
	//hip.Time.Reset()
	hip.InitStats()
	//hip.StatCounters(true)
}

func (hip *HipSim) InitStats() {
	hip.Sim.InitStats()
	hip.Stats.SetInt("TrgOnWasOffAll", 0)
	hip.Stats.SetInt("TrgOnWasOffAll", 0)
	hip.Stats.SetInt("TrgOffWasOn", 0)
	hip.Stats.SetInt("Mem", 0)
	hip.Stats.SetInt("CntErr", 0)
}

func (hip *HipSim) ConfigLogs() {
	hip.ConfigLogItems()
	hip.Logs.CreateTables()
	hip.Logs.SetContext(&hip.Stats, hip.Net)
	// don't plot certain combinations we don't use
	hip.Logs.NoPlot(elog.Train, elog.Cycle)
	hip.Logs.NoPlot(elog.Test, elog.Run)
	// note: Analyze not plotted by default
	hip.Logs.SetMeta(elog.Train, elog.Run, "LegendCol", "Params")
	hip.Stats.ConfigRasters(hip.Net, hip.Net.LayersByClass())
}
