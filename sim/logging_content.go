package sim

import (
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"strings"
)

func GetEpochWindowLast5(ss *Sim) *etable.IdxView {
	epochlog := ss.Logs.GetTable(elog.Train, elog.Epoch)
	epochwindow := etable.NewIdxView(epochlog)

	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epochwindow.Len()-1 {
		nlast = epochwindow.Len() - 1
	}
	epochwindow.Idxs = epochwindow.Idxs[epochwindow.Len()-nlast:]

	return epochwindow
}

type logsComputeHelper struct {
	// Use either Modes+Times or Mode+Time
	Modes   []elog.TrainOrTest `desc:"a variable list of modes that this item can exist in"`
	Times   []elog.Times       `desc:"a variable list of times that this item can exist in"`
	Mode    elog.TrainOrTest   `desc:"a single mode that this item can exist in"`
	Time    elog.Times         `desc:"a single time that this item can exist in"`
	Compute elog.ComputeFunc   `desc:"For this timescale and mode, how is this value computed?"`
}

func addLogsItem(ss *Sim, item elog.Item, computes []logsComputeHelper) {
	item.Compute = map[elog.ScopeKey]elog.ComputeFunc{}
	if item.Plot == elog.DUnknown {
		item.Plot = elog.DTrue
	}
	if item.FixMin == elog.DUnknown {
		item.FixMin = elog.DTrue
	}
	if item.FixMax == elog.DUnknown {
		item.FixMax = elog.DFalse
	}
	for _, compute := range computes {
		item.AssignComputeFunc(compute.Mode, compute.Time, compute.Compute)
		item.AssignComputeFuncOver(compute.Modes, compute.Times, compute.Compute)
	}
	ss.Logs.AddItem(&item)
}

func (ss *Sim) ConfigLogSpec() {
	// Train epoch
	addLogsItem(ss, elog.Item{
		Name: "Run",
		Type: etensor.INT64,
		Plot: elog.DFalse,
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run.Cur))
	}}, {Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run.Cur))
	}}, {Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run.Cur))
	}}, {Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run.Cur))
	}}})
	addLogsItem(ss, elog.Item{
		Name: "Params",
		Type: etensor.STRING,
		Plot: elog.DFalse,
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellString(item.Name, row, ss.RunName())
	}}})
	addLogsItem(ss, elog.Item{
		Name:  "FirstZero",
		Type:  etensor.FLOAT64,
		Plot:  elog.DFalse,
		Range: minmax.F64{Min: -1},
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.FirstZero))
	}}})
	addLogsItem(ss, elog.Item{
		Name: "Epoch",
		Type: etensor.INT64,
		Plot: elog.DFalse,
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch.Prv))
	}}, {Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch.Prv))
	}}, {Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch.Prv))
	}}})
	addLogsItem(ss, elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcUnitErr)
	}}, {Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		epochWin := GetEpochWindowLast5(ss)
		dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcPctErr)
	}}, {Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		epochWin := GetEpochWindowLast5(ss)
		dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcPctCor)
	}}, {Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		epochWin := GetEpochWindowLast5(ss)
		dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcCosDiff)
	}}, {Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		epochWin := GetEpochWindowLast5(ss)
		dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "Correl",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcCorrel)
	}}, {Mode: elog.Train, Time: elog.Run, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		epochWin := GetEpochWindowLast5(ss)
		dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name: "PerTrlMSec",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
	}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcPerTrlMSec)
	}}})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		curlname := lnm
		ly := ss.Net.LayerByName(curlname).(axon.AxonLayer).AsAxon()
		addLogsItem(ss, elog.Item{
			Name:   curlname + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.ActMAvg))
		}}})
		addLogsItem(ss, elog.Item{
			Name:  curlname + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.AvgMaxGeM))
		}}})
		addLogsItem(ss, elog.Item{
			Name:  curlname + "_AvgGe",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Avg))
		}}})
		addLogsItem(ss, elog.Item{
			Name:  curlname + "_MaxGe",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Max))
		}}})
		addLogsItem(ss, elog.Item{
			Name:  curlname + "_Gi",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Gi))
		}}})

		///
		addLogsItem(ss, elog.Item{
			Name:  curlname + "_AvgDifAvg",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].AvgDif.Avg))
		}}})
		addLogsItem(ss, elog.Item{
			Name:  curlname + "_AvgDifMax",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Train, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].AvgDif.Max))
		}}})
	}

	// Test trial and epoch and cycle
	//addLogsItem(ss, elog.Item{
	//	Name: "Run",
	//	Type: etensor.INT64,
	//}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run.Cur))
	//}}, {Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run.Cur))
	//}}})
	//addLogsItem(ss, elog.Item{
	//	Name: "Epoch",
	//	Type: etensor.INT64,
	//}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch.Prv))
	//}}, {Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch.Prv))
	//}}})
	addLogsItem(ss, elog.Item{
		Name: "Trial",
		Type: etensor.INT64,
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TestEnv.Trial.Cur))
	}}})
	addLogsItem(ss, elog.Item{
		Name: "TrialName",
		Type: etensor.STRING,
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellString(item.Name, row, strings.Join(ss.TestEnv.CurWords, " "))
	}}})
	addLogsItem(ss, elog.Item{
		Name: "Err",
		Type: etensor.FLOAT64,
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrlErr))
	}}})
	addLogsItem(ss, elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrlUnitErr))
	}}, {Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		trl := ss.Logs.GetTable(elog.Test, elog.Trial)
		tix := etable.NewIdxView(trl)
		dt.SetCellFloat(item.Name, row, agg.Sum(tix, "UnitErr")[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		trl := ss.Logs.GetTable(elog.Test, elog.Trial)
		tix := etable.NewIdxView(trl)
		dt.SetCellFloat(item.Name, row, agg.Mean(tix, "Err")[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		trl := ss.Logs.GetTable(elog.Test, elog.Trial)
		tix := etable.NewIdxView(trl)
		dt.SetCellFloat(item.Name, row, 1-agg.Mean(tix, "Err")[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrlCosDiff))
	}}, {Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		trl := ss.Logs.GetTable(elog.Test, elog.Trial)
		tix := etable.NewIdxView(trl)
		dt.SetCellFloat(item.Name, row, agg.Sum(tix, "CosDiff")[0])
	}}})
	addLogsItem(ss, elog.Item{
		Name:   "Correl",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrlCorrel))
	}}, {Mode: elog.Test, Time: elog.Epoch, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		trl := ss.Logs.GetTable(elog.Test, elog.Trial)
		tix := etable.NewIdxView(trl)
		dt.SetCellFloat(item.Name, row, agg.Sum(tix, "Correl")[0])
	}}})
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	addLogsItem(ss, elog.Item{
		Name:      "InAct",
		Type:      etensor.FLOAT64,
		CellShape: inp.Shp.Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		ivt := ss.ValsTsr("Input")
		inp.UnitValsTensor(ivt, "Act")
		dt.SetCellTensor(item.Name, row, ivt)
	}}})
	addLogsItem(ss, elog.Item{
		Name:      "OutActM",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		ovt := ss.ValsTsr("Output")
		out.UnitValsTensor(ovt, "ActM")
		dt.SetCellTensor(item.Name, row, ovt)
	}}})
	addLogsItem(ss, elog.Item{
		Name:      "OutActP",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		ovt := ss.ValsTsr("Output")
		out.UnitValsTensor(ovt, "ActP")
		dt.SetCellTensor(item.Name, row, ovt)
	}}})
	// Cycle
	addLogsItem(ss, elog.Item{
		Name: "Cycle",
		Type: etensor.INT64,
	}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Cycle, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat("Cycle", row, float64(row))
	}}})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		curlname := lnm
		ly := ss.Net.LayerByName(curlname).(axon.AxonLayer).AsAxon()
		addLogsItem(ss, elog.Item{
			Name:   curlname + " ActM.Avg",
			Type:   etensor.FLOAT64,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Trial, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.ActMAvg))
		}}})
		addLogsItem(ss, elog.Item{
			Name:  curlname + " Ge.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Cycle, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Avg))
		}}})
		addLogsItem(ss, elog.Item{
			Name:  curlname + " Act.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
		}, []logsComputeHelper{{Mode: elog.Test, Time: elog.Cycle, Compute: func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Act.Avg))
		}}})
	}
}
