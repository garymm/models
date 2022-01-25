package main

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"strings"
)

func (ss *Sim) ConfigLogSpec() {
	// Train epoch
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Run",
		Type: etensor.INT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Run.Cur))
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Epoch",
		Type: etensor.INT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Epoch.Prv))
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "UnitErr",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcUnitErr)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctErr",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcPctErr)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctCor",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcPctCor)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "CosDiff",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcCosDiff)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Correl",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcCorrel)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PerTrlMSec",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, ss.EpcPerTrlMSec)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Train})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_ActAvg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.ActAvg.ActMAvg))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_MaxGeM",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.ActAvg.AvgMaxGeM))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_AvgGe",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].Inhib.Ge.Avg))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_MaxGe",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].Inhib.Ge.Max))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_Gi",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].Inhib.Gi))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_AvgDifAvg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].AvgDif.Avg))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Train,
			LayerName: lnm})
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "_AvgDifMax",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.Pools[0].AvgDif.Max))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Train,
			LayerName: lnm})
	}

	// Test trial and epoch
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Run",
		Type: etensor.INT64},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Run.Cur))
		}, env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Run.Cur))
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Epoch",
		Type: etensor.INT64},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Epoch.Prv))
		}, env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrainEnv.Epoch.Prv))
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Trial",
		Type: etensor.INT64},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TestEnv.Trial.Cur))
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "TrialName",
		Type: etensor.STRING},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellString(name, row, strings.Join(ss.TestEnv.CurWords, " "))
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Err",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlErr))
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "UnitErr",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlUnitErr))
		}, env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Sum(tix, "UnitErr")[0])
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctErr",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Mean(tix, "Err")[0])
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "PctCor",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, 1-agg.Mean(tix, "Err")[0])
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "CosDiff",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlCosDiff))
		}, env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Sum(tix, "CosDiff")[0])
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name: "Correl",
		Type: etensor.FLOAT64},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			dt.SetCellFloat(name, row, float64(ss.TrlCorrel))
		}, env.Epoch: func(ss *Sim, dt *etable.Table, row int, name string) {
			trl := ss.TstTrlLog
			tix := etable.NewIdxView(trl)
			dt.SetCellFloat(name, row, agg.Sum(tix, "Correl")[0])
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name:      "InAct",
		Type:      etensor.FLOAT64,
		CellShape: inp.Shp.Shp},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			ivt := ss.ValsTsr("Input")
			inp.UnitValsTensor(ivt, "Act")
			dt.SetCellTensor(name, row, ivt)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name:      "OutActM",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			ovt := ss.ValsTsr("Output")
			out.UnitValsTensor(ovt, "ActM")
			dt.SetCellTensor(name, row, ovt)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
		Name:      "OutActP",
		Type:      etensor.FLOAT64,
		CellShape: out.Shp.Shp},
		Compute: map[env.TimeScales]LogFunc{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string) {
			ovt := ss.ValsTsr("Output")
			out.UnitValsTensor(ovt, "ActP")
			dt.SetCellTensor(name, row, ovt)
		}},
		Plot:     true,
		FixMin:   true,
		FixMax:   false,
		EvalType: Test})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		ss.LogSpec.AddItem(&LogItem{Column: etable.Column{
			Name: lnm + "ActM.Avg",
			Type: etensor.FLOAT64},
			ComputeLayer: map[env.TimeScales]LogFuncLayer{env.Trial: func(ss *Sim, dt *etable.Table, row int, name string, ly axon.Layer) {
				dt.SetCellFloat(name, row, float64(ly.ActAvg.ActMAvg))
			}},
			Plot:      true,
			FixMin:    true,
			FixMax:    false,
			EvalType:  Test,
			LayerName: lnm})
	}
}
