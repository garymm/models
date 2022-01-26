package main

import (
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
