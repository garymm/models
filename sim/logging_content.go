package sim

import (
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// Helper function for one of the compute functions below.
func getEpochWindowLast5(ss *Sim) *etable.IdxView {
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

func updateItemForAll(ss *Sim, item *elog.Item) {
	// This could be refactored with a set object. It's not efficient to call this for every item, but it also doesn't matter.
	// DO NOT SUBMIT Store usedmodes and usedtimes on Logs
	usedModes := []elog.EvalModes{}
	usedTimes := []elog.Times{}
	for _, item := range ss.Logs.Items {
		for sk, _ := range item.Compute {
			modes, times := sk.GetModesAndTimes()
			for _, m := range modes {
				foundMode := false
				for _, um := range usedModes {
					if m == um {
						foundMode = true
						break
					}
				}
				if !foundMode && m != elog.UnknownEvalMode {
					usedModes = append(usedModes, m)
				}
			}
			for _, t := range times {
				foundTime := false
				for _, ut := range usedTimes {
					if t == ut {
						foundTime = true
						break
					}
				}
				if !foundTime && t != elog.UnknownTimescale {
					usedTimes = append(usedTimes, t)
				}
			}
		}
	}
	newMap := elog.ComputeMap{}
	for sk, c := range item.Compute {
		newsk := sk
		useAllModes := false
		useAllTimes := false
		modes, times := sk.GetModesAndTimes()
		for _, m := range modes {
			if m == elog.AllEvalModes {
				useAllModes = true
			}
		}
		for _, t := range times {
			if t == elog.AllTimes {
				useAllTimes = true
			}
		}
		if useAllModes && useAllTimes {
			newsk = elog.GenScopesKey(usedModes, usedTimes)
		} else if useAllModes {
			newsk = elog.GenScopesKey(usedModes, times)
		} else if useAllTimes {
			newsk = elog.GenScopesKey(modes, usedTimes)
		}
		newMap[newsk] = c
	}
	item.Compute = newMap
}

// DO NOT SUBMIT Move this to Logs
func ProcessLogItems(ss *Sim) {
	for _, item := range ss.Logs.Items {
		if item.Plot == elog.DUnknown {
			item.Plot = elog.DTrue
		}
		if item.FixMin == elog.DUnknown {
			item.FixMin = elog.DTrue
		}
		if item.FixMax == elog.DUnknown {
			item.FixMax = elog.DFalse
		}
		for scope, _ := range item.Compute {
			//updateItemForAll(ss, item)
			item.UpdateModesAndTimesFromScope(scope)
		}
	}
}

func (ss *Sim) ConfigLogSpec() {
	// Train epoch
	//TODO(Andrew) implement the ability to specify that something covers all times and ESPECIALLY all modes
	ss.Logs.AddItem(&elog.Item{
		Name: "Run",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run().Cur))
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run().Cur))
			}, elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run().Cur))
			}, elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run().Cur))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Params",
		Type: etensor.STRING,
		Plot: elog.DFalse, //TODO(Andrew) refactor this to actually assign maps this way instead via logscomputerHelper
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellString(item.Name, row, ss.RunName())
			},
		},
	})

	ss.Logs.AddItem(&elog.Item{
		Name:  "FirstZero",
		Type:  etensor.FLOAT64,
		Plot:  elog.DFalse,
		Range: minmax.F64{Min: -1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.FirstZero))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Epoch",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenScopesKey([]elog.EvalModes{elog.Train, elog.Test}, []elog.Times{elog.Epoch, elog.Trial}): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch().Prv))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.EpcUnitErr)
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				epochWin := getEpochWindowLast5(ss)
				dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.EpcPctErr)
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				epochWin := getEpochWindowLast5(ss)
				dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.EpcPctCor)
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				epochWin := getEpochWindowLast5(ss)
				dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.EpcCosDiff)
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				epochWin := getEpochWindowLast5(ss)
				dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "Correl",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.EpcCorrel)
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				epochWin := getEpochWindowLast5(ss)
				dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PerTrlMSec",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.EpcPerTrlMSec)
			}}})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		curlname := lnm
		ly := ss.Net.LayerByName(curlname).(axon.AxonLayer).AsAxon()
		ss.Logs.AddItem(&elog.Item{
			Name:   curlname + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.ActMAvg))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.AvgMaxGeM))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_AvgGe",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Avg))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_MaxGe",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Max))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_Gi",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Gi))
				}}})

		///
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_AvgDifAvg",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].AvgDif.Avg))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_AvgDifMax",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].AvgDif.Max))
				}}})
	}

	// Test trial and epoch and cycle
	//ss.Logs.AddItem(&elog.Item{
	//	Name: "Run",
	//	Type: etensor.INT64,
	//Compute: elog.ComputeMap{
	//elog.GenScopeKey(elog.Test, elog.Trial):  func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run().Cur))
	//}, elog.GenScopeKey(elog.Test, elog.Epoch):  func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run().Cur))
	//}}})
	//ss.Logs.AddItem(&elog.Item{
	//	Name: "Epoch",
	//	Type: etensor.INT64,
	//Compute: elog.ComputeMap{
	//elog.GenScopeKey(elog.Test, elog.Trial):  func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch().Prv))
	//}, elog.GenScopeKey(elog.Test, elog.Epoch):  func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch().Prv))
	//}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Trial",
		Type: etensor.INT64,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TestEnv.Trial().Cur))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TrialName",
		Type: etensor.STRING,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellString(item.Name, row, ss.TestEnv.GetCurrentTrialName())
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Err",
		Type: etensor.FLOAT64,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrlErr))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrlUnitErr))
			}, elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				trl := ss.Logs.GetTable(elog.Test, elog.Trial)
				tix := etable.NewIdxView(trl)
				dt.SetCellFloat(item.Name, row, agg.Sum(tix, "UnitErr")[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				trl := ss.Logs.GetTable(elog.Test, elog.Trial)
				tix := etable.NewIdxView(trl)
				dt.SetCellFloat(item.Name, row, agg.Mean(tix, "Err")[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				trl := ss.Logs.GetTable(elog.Test, elog.Trial)
				tix := etable.NewIdxView(trl)
				dt.SetCellFloat(item.Name, row, 1-agg.Mean(tix, "Err")[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrlCosDiff))
			}, elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				trl := ss.Logs.GetTable(elog.Test, elog.Trial)
				tix := etable.NewIdxView(trl)
				dt.SetCellFloat(item.Name, row, agg.Sum(tix, "CosDiff")[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "Correl",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrlCorrel))
			}, elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				trl := ss.Logs.GetTable(elog.Test, elog.Trial)
				tix := etable.NewIdxView(trl)
				dt.SetCellFloat(item.Name, row, agg.Sum(tix, "Correl")[0])
			}}})

	//TODO move inp and out to compute function in some helper that goes inside the function
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Logs.AddItem(&elog.Item{
		Name:      "InAct",
		Type:      etensor.FLOAT64,
		CellShape: inp.Shp.Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				ivt := ss.ValsTsr("Input")
				inp.UnitValsTensor(ivt, "Act")
				dt.SetCellTensor(item.Name, row, ivt)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:      "OutActM",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				ovt := ss.ValsTsr("Output")
				out.UnitValsTensor(ovt, "ActM")
				dt.SetCellTensor(item.Name, row, ovt)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:      "OutActP",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				ovt := ss.ValsTsr("Output")
				out.UnitValsTensor(ovt, "ActP")
				dt.SetCellTensor(item.Name, row, ovt)
			}}})
	// Cycle
	ss.Logs.AddItem(&elog.Item{
		Name: "Cycle",
		Type: etensor.INT64,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Cycle): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat("Cycle", row, float64(row))
			}}})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		curlname := lnm
		ly := ss.Net.LayerByName(curlname).(axon.AxonLayer).AsAxon()
		ss.Logs.AddItem(&elog.Item{
			Name:   curlname + " ActM.Avg",
			Type:   etensor.FLOAT64,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.ActMAvg))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + " Ge.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Test, elog.Cycle): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Avg))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + " Act.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenScopeKey(elog.Test, elog.Cycle): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
					dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Act.Avg))
				}}})
	}

	// Important! For each item that's been added, set defaults and extract Modes and Times from the compute function.
	ProcessLogItems(ss)
}
