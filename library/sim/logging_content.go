package sim

import (
	"time"

	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// Helper function for one of the compute functions below.
func getEpochWindowLast5(ss *Sim) *etable.IdxView {
	epochlog := ss.Logs.Table(elog.Train, elog.Epoch)
	epochwindow := etable.NewIdxView(epochlog)

	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epochwindow.Len()-1 {
		nlast = epochwindow.Len() - 1
	}
	epochwindow.Idxs = epochwindow.Idxs[epochwindow.Len()-nlast:]

	return epochwindow
}

func (ss *Sim) ConfigLogSpec() {
	// Train epoch
	ss.Logs.AddItem(&elog.Item{
		Name: "Run",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.AllEvalModes, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.Run.Cur))
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.Run.Cur))
			}, elog.GenScopeKey(elog.Test, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.Run.Cur))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Params",
		Type: etensor.STRING,
		Plot: elog.DFalse,
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
			elog.GenScopesKey([]elog.EvalModes{elog.AllEvalModes}, []elog.Times{elog.Epoch, elog.Trial}): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch().Prv))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.TrlUnitErr)
			}, elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), item.Name)[0])
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
			elog.GenScopeKey(elog.Train, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.TrlErr)
			}, elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), item.Name)[0])
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
				dt.SetCellFloat(item.Name, row, 1-agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), "PctErr")[0])
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
			elog.GenScopeKey(elog.Train, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.TrlCosDiff)
			}, elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), item.Name)[0])
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
			elog.GenScopeKey(elog.Train, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, ss.TrlCorrel)
			}, elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), item.Name)[0])
			}, elog.GenScopeKey(elog.Train, elog.Run): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				epochWin := getEpochWindowLast5(ss)
				dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PerEpcMSec",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Train, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				epcPerTrlMSec := 0.0
				if ss.LastEpcTime.IsZero() {
					epcPerTrlMSec = 0
				} else {
					iv := time.Now().Sub(ss.LastEpcTime)
					// TODO This should be normalized by number of trials rather than 1.0 and renamed back to PerTrlMSec
					epcPerTrlMSec = float64(iv) / (1.0 * float64(time.Millisecond))
				}
				ss.LastEpcTime = time.Now()
				dt.SetCellFloat(item.Name, row, epcPerTrlMSec)
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
	//	dt.SetCellFloat(item.Name, row, float64(ss.Run.Cur))
	//}, elog.GenScopeKey(elog.Test, elog.Epoch):  func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
	//	dt.SetCellFloat(item.Name, row, float64(ss.Run.Cur))
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
			elog.GenScopeKey(elog.AllEvalModes, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, float64(ss.TestEnv.Trial().Cur))
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TrialName",
		Type: etensor.STRING,
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.AllEvalModes, elog.Trial): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
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
				dt.SetCellFloat(item.Name, row, agg.Sum(ss.Logs.IdxView(elog.Test, elog.Trial), "UnitErr")[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, agg.Mean(ss.Logs.IdxView(elog.Test, elog.Trial), "Err")[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenScopeKey(elog.Test, elog.Epoch): func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
				dt.SetCellFloat(item.Name, row, 1-agg.Mean(ss.Logs.IdxView(elog.Test, elog.Trial), "Err")[0])
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
				dt.SetCellFloat(item.Name, row, agg.Sum(ss.Logs.IdxView(elog.Test, elog.Trial), "CosDiff")[0])
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
				dt.SetCellFloat(item.Name, row, agg.Sum(ss.Logs.IdxView(elog.Test, elog.Trial), "Correl")[0])
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
}
