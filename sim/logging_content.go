package sim

import (
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"strings"
)

func (ss *Sim) GetEpochWindow() *etable.IdxView {
	epochlog := ss.TrnEpcLog
	epochwindow := etable.NewIdxView(epochlog)

	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epochwindow.Len()-1 {
		nlast = epochwindow.Len() - 1
	}
	epochwindow.Idxs = epochwindow.Idxs[epochwindow.Len()-nlast:]

	return epochwindow
}

func (ss *Sim) InitPerLayerDefault(name string) elog.Item {
	item := elog.Item{Name: name,
		Type:   etensor.FLOAT64,
		Plot:   eplot.Off,
		FixMin: eplot.FixMin,
		FixMax: eplot.FloatMax,
		Modes:  []elog.Modes{elog.Train},
		Times:  []elog.Times{elog.Epoch},
		Range:  minmax.F64{Max: 1}}
	return item
}

func (ss *Sim) InitLogItemDefault(name string, tensorType etensor.Type) elog.Item {
	item := elog.Item{Name: name,
		Type:   tensorType,
		Plot:   eplot.Off,
		FixMin: eplot.FixMin,
		FixMax: eplot.FloatMax,
		Modes:  []elog.Modes{elog.Train},
		Times:  []elog.Times{elog.Run}}
	return item
}

func (ss *Sim) InitLogItemDefaultTest(name string, tensorType etensor.Type) elog.Item {
	item := elog.Item{Name: name,
		Type:   tensorType,
		Plot:   eplot.On,
		FixMin: eplot.FixMin,
		FixMax: eplot.FloatMax,
		Modes:  []elog.Modes{elog.Test},
		Times:  []elog.Times{elog.Trial}}
	return item
}

func (ss *Sim) ConfigLogger() {
	//A function to calculate mean of epochs
	meanCalculationFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		epochWin := ss.GetEpochWindow()
		dt.SetCellFloat(item.Name, row, agg.Mean(epochWin, item.Name)[0])
	}

	runItem := ss.InitLogItemDefault("Run", etensor.INT64)
	runItem.Times = append(runItem.Times, elog.Epoch)
	runFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Run.Cur))
	}
	runItem.AssignComputeFunc(runFunc)
	ss.Logs.AddItem(&runItem)
	/////
	paramsItem := ss.InitLogItemDefault("Params", etensor.STRING)
	paramsFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellString(item.Name, row, ss.RunName())
	}
	paramsItem.AssignComputeFunc(paramsFunc)
	ss.Logs.AddItem(&paramsItem)
	////
	firstZeroItem := ss.InitLogItemDefault("FirstZero", etensor.FLOAT64)
	firstZeroItem.Range = minmax.F64{Min: -1}
	firstZeroFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.FirstZero))
	}
	firstZeroItem.AssignComputeFunc(firstZeroFunc)
	ss.Logs.AddItem(&firstZeroItem)
	/////
	epochItem := ss.InitLogItemDefault("Epoch", etensor.INT64)
	epochFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, float64(ss.TrainEnv.Epoch.Prv))
	}
	epochItem.AssignComputeFunc(epochFunc)
	ss.Logs.AddItem(&epochItem)
	/////
	unitErrItem := ss.InitLogItemDefault("UnitErr", etensor.FLOAT64)
	unitErrItem.Times = append(unitErrItem.Times, elog.Epoch) //add epoch modes
	unitErrEpochFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcUnitErr)
	}
	unitErrItem.Compute[unitErrItem.GetScopeKey(elog.Train, elog.Run)] = meanCalculationFunc
	unitErrItem.Compute[unitErrItem.GetScopeKey(elog.Train, elog.Epoch)] = unitErrEpochFunc
	ss.Logs.AddItem(&unitErrItem)
	/////
	PctErrItem := ss.InitLogItemDefault("PctErr", etensor.FLOAT64)
	PctErrItem.Range = minmax.F64{Max: 1}
	PctErrItem.FixMax = eplot.FixMax
	PctErrItem.Plot = eplot.On
	PctErrItem.Times = append(PctErrItem.Times, elog.Epoch) //add epoch modes
	PctErrEpochFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcPctErr)
	}
	PctErrItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Run)] = meanCalculationFunc
	PctErrItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Epoch)] = PctErrEpochFunc
	ss.Logs.AddItem(&PctErrItem)
	/////
	PctCorItem := ss.InitLogItemDefault("PctCor", etensor.FLOAT64)
	PctCorItem.Range = minmax.F64{Max: 1}
	PctCorItem.FixMax = eplot.FixMax
	PctCorItem.Plot = eplot.On
	PctCorItem.Times = append(PctCorItem.Times, elog.Epoch) //add epoch modes
	PctCorEpochFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcPctCor)
	}
	PctCorItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Run)] = meanCalculationFunc
	PctCorItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Epoch)] = PctCorEpochFunc
	ss.Logs.AddItem(&PctCorItem)
	/////
	CosDiffItem := ss.InitLogItemDefault("CosDiff", etensor.FLOAT64)
	CosDiffItem.Range = minmax.F64{Max: 1}
	CosDiffItem.FixMax = eplot.FixMax
	CosDiffItem.Plot = eplot.On
	CosDiffItem.Times = append(CosDiffItem.Times, elog.Epoch) //add epoch modes
	CosDiffEpochFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcCosDiff)
	}
	CosDiffItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Run)] = meanCalculationFunc
	CosDiffItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Epoch)] = CosDiffEpochFunc
	ss.Logs.AddItem(&CosDiffItem)
	//////
	CorrelItem := ss.InitLogItemDefault("Correl", etensor.FLOAT64)
	CorrelItem.Range = minmax.F64{Max: 1}
	CorrelItem.FixMax = eplot.FixMax
	CorrelItem.Plot = eplot.On
	CorrelItem.Times = append(CorrelItem.Times, elog.Epoch) //add epoch modes
	CorrelEpochFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcCorrel)
	}
	CorrelItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Run)] = meanCalculationFunc
	CorrelItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Epoch)] = CorrelEpochFunc
	ss.Logs.AddItem(&CorrelItem)
	///////
	PerTrlMSecItem := ss.InitLogItemDefault("PerTrlMSec", etensor.FLOAT64)
	PerTrlMSecItem.Times = append(PerTrlMSecItem.Times, elog.Epoch) //add epoch modes
	PerTrlMSecEpochFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
		dt.SetCellFloat(item.Name, row, ss.EpcPerTrlMSec)
	}
	PerTrlMSecItem.Compute[PctErrItem.GetScopeKey(elog.Train, elog.Epoch)] = PerTrlMSecEpochFunc
	ss.Logs.AddItem(&PerTrlMSecItem)

	for _, lnm := range ss.LayStatNms {
		currName := lnm
		///////
		actAvgFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			ly := ss.Net.LayerByName(currName).(axon.AxonLayer).AsAxon()
			dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.ActMAvg))
		}
		actAvgItem := ss.InitPerLayerDefault(lnm + "_ActAvg")
		actAvgItem.FixMax = eplot.FixMax //Only one that uses fixmax
		actAvgItem.AssignComputeFunc(actAvgFunc)
		ss.Logs.AddItem(&actAvgItem)
		/////
		maxGenFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			ly := ss.Net.LayerByName(currName).(axon.AxonLayer).AsAxon()
			dt.SetCellFloat(item.Name, row, float64(ly.ActAvg.AvgMaxGeM))
		}
		maxGenItem := ss.InitPerLayerDefault(lnm + "_MaxGeM")
		maxGenItem.AssignComputeFunc(maxGenFunc)
		ss.Logs.AddItem(&maxGenItem)
		/////
		avgGeFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			ly := ss.Net.LayerByName(currName).(axon.AxonLayer).AsAxon()
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Avg))
		}
		avgGeItem := ss.InitPerLayerDefault(lnm + "_AvgGeM")
		actAvgItem.AssignComputeFunc(avgGeFunc)
		ss.Logs.AddItem(&avgGeItem)
		/////
		maxGeFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			ly := ss.Net.LayerByName(currName).(axon.AxonLayer).AsAxon()
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Ge.Max))
		}
		maxGeItem := ss.InitPerLayerDefault(lnm + "_MaxGe")
		maxGeItem.AssignComputeFunc(maxGeFunc)
		ss.Logs.AddItem(&maxGeItem)
		//////
		giFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			ly := ss.Net.LayerByName(currName).(axon.AxonLayer).AsAxon()
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].Inhib.Gi))
		}
		giItem := ss.InitPerLayerDefault(lnm + "_Gi")
		giItem.AssignComputeFunc(giFunc)
		ss.Logs.AddItem(&giItem)
		/////
		avgDiffFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			ly := ss.Net.LayerByName(currName).(axon.AxonLayer).AsAxon()
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].AvgDif.Avg))
		}
		avgDiffItem := ss.InitPerLayerDefault(lnm + "_AvgDifAvg")
		avgDiffItem.Compute[actAvgItem.GetScopeKey(elog.Train, elog.Epoch)] = avgDiffFunc
		ss.Logs.AddItem(&avgDiffItem)
		////
		avgDiffMaxFunc := func(item *elog.Item, scope elog.ScopeKey, dt *etable.Table, row int) {
			ly := ss.Net.LayerByName(currName).(axon.AxonLayer).AsAxon()
			dt.SetCellFloat(item.Name, row, float64(ly.Pools[0].AvgDif.Max))
		}
		avgDiffMaxItem := ss.InitPerLayerDefault(lnm + "_AvgDifMax")
		avgDiffMaxItem.AssignComputeFunc(avgDiffMaxFunc)
		ss.Logs.AddItem(&avgDiffMaxItem)
		////
	}

	runItemTest := ss.InitLogItemDefaultTest("Run", etensor.INT64)
	runItemTest.Times = append(runItemTest.Times, elog.Epoch)
	runItemTest.AssignComputeFunc(runFunc)
	ss.Logs.AddItem(&runItemTest)

}

func (ss *Sim) ConfigLogSpec() {
	// Train epoch
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Run",
		Type: etensor.INT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Run.Cur))
		}, axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Run.Cur))
		}},
		Plot:     eplot.Off,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Params",
		Type: etensor.STRING},
		Compute: map[axon.TimeScales]LogFunc{
			axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
				dt.SetCellString(name, row, ss.RunName())
			}},
		Plot:     eplot.Off,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "FirstZero",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{
			axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
				dt.SetCellFloat(name, row, float64(ss.FirstZero))
			}},
		Plot:     eplot.Off,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		Range:    minmax.F64{Min: -1},
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Epoch",
		Type: etensor.INT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Epoch.Prv))
		}},
		Plot:     eplot.Off,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "UnitErr",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcUnitErr)
		}, axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
			epochWin := ss.GetEpochWindow()
			dt.SetCellFloat(name, row, agg.Mean(epochWin, name)[0])
		}},
		Plot:     eplot.Off,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctErr",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcPctErr)
		}, axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
			epochWin := ss.GetEpochWindow()
			dt.SetCellFloat(name, row, agg.Mean(epochWin, name)[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctCor",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcPctCor)
		}, axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
			epochWin := ss.GetEpochWindow()
			dt.SetCellFloat(name, row, agg.Mean(epochWin, name)[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "CosDiff",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcCosDiff)
		}, axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
			epochWin := ss.GetEpochWindow()
			dt.SetCellFloat(name, row, agg.Mean(epochWin, name)[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Correl",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcCorrel)
		}, axon.Run: func(ss *Sim, dt *etable.Table, row int, name string) {
			epochWin := ss.GetEpochWindow()
			dt.SetCellFloat(name, row, agg.Mean(epochWin, name)[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PerTrlMSec",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcPerTrlMSec)
		}},
		Plot:     eplot.Off,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Train})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		//curlname := lnm
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_ActAvg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.ActAvg.ActMAvg))
			}},
			Plot:      eplot.Off,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FixMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_MaxGeM",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.ActAvg.AvgMaxGeM))
			}},
			Plot:      eplot.Off,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_AvgGe",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].Inhib.Ge.Avg))
			}},
			Plot:      eplot.Off,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_MaxGe",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].Inhib.Ge.Max))
			}},
			Plot:      eplot.Off,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_Gi",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].Inhib.Gi))
			}},
			Plot:      eplot.Off,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Train,
			LayerName: lnm})

		///
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_AvgDifAvg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].AvgDif.Avg))
			}},
			Plot:      eplot.Off,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_AvgDifMax",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].AvgDif.Max))
			}},
			Plot:      eplot.Off,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Train,
			LayerName: lnm})
	}

	// Test trial and epoch and cycle
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Run",
		Type: etensor.INT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Run.Cur))
		}, axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Run.Cur))
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Epoch",
		Type: etensor.INT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Epoch.Prv))
		}, axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Epoch.Prv))
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Trial",
		Type: etensor.INT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TestEnv.Trial.Cur))
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "TrialName",
		Type: etensor.STRING},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellString(name, row, strings.Join(ss.TestEnv.CurWords, " "))
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Err",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlErr))
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "UnitErr",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlUnitErr))
		}, axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Sum(tix, "UnitErr")[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctErr",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Mean(tix, "Err")[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctCor",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, 1-agg.Mean(tix, "Err")[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "CosDiff",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlCosDiff))
		}, axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Sum(tix, "CosDiff")[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Correl",
		Type: etensor.FLOAT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlCorrel))
		}, axon.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Sum(tix, "Correl")[0])
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Test})
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name:      "InAct",
		Type:      etensor.FLOAT64,
		CellShape: inp.Shp.Shp},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			ivt := ss.ValsTsr("Input")
			inp.UnitValsTensor(ivt, "Act")
			dt.SetCellTensor(name, row, ivt)
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name:      "OutActM",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			ovt := ss.ValsTsr("Output")
			out.UnitValsTensor(ovt, "ActM")
			dt.SetCellTensor(name, row, ovt)
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name:      "OutActP",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp},
		Compute: map[axon.TimeScales]LogFunc{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			ovt := ss.ValsTsr("Output")
			out.UnitValsTensor(ovt, "ActP")
			dt.SetCellTensor(name, row, ovt)
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FixMax,
		Range:    minmax.F64{Max: 1},
		EvalType: Test})
	// Cycle
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Cycle",
		Type: etensor.INT64},
		Compute: map[axon.TimeScales]LogFunc{axon.Cycle: func(ss *Sim, dt *etable.Table, cyc int, name string) {
			dt.SetCellFloat("Cycle", cyc, float64(cyc))
		}},
		Plot:     eplot.On,
		FixMin:   eplot.FixMin,
		FixMax:   eplot.FloatMax,
		EvalType: Test})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + " ActM.Avg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Trial: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.ActAvg.ActMAvg))
			}},
			Plot:      eplot.On,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FixMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Test,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + " Ge.Avg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Cycle: func(ss *Sim, dt *etable.Table, cyc int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
			}},
			Plot:      eplot.On,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Test,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + " Act.Avg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[axon.TimeScales]LogFuncLayer{axon.Cycle: func(ss *Sim, dt *etable.Table, cyc int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, cyc, float64(ly.Pools[0].Inhib.Act.Avg))
			}},
			Plot:      eplot.On,
			FixMin:    eplot.FixMin,
			FixMax:    eplot.FloatMax,
			Range:     minmax.F64{Max: 1},
			EvalType:  Test,
			LayerName: lnm})
	}
}
