package sim

import (
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

func (ss *Sim) ConfigLogSpec() {
	ss.Logs.AddItem(&elog.Item{
		Name: "Run",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.AllTimes): func(ctx *elog.Context) {
				ctx.SetStatInt("Run")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Params",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.AllTimes): func(ctx *elog.Context) {
				ctx.SetString(ss.RunName())
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Epoch",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scopes([]elog.EvalModes{elog.AllModes}, []elog.Times{elog.Epoch, elog.Trial}): func(ctx *elog.Context) {
				ctx.SetStatInt("Epoch")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Trial",
		Type: etensor.INT64,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatInt("Trial")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TrialName",
		Type: etensor.STRING,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatString("TrialName")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Cycle",
		Type: etensor.INT64,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Cycle): func(ctx *elog.Context) {
				ctx.SetStatInt("Cycle")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:  "FirstZero",
		Type:  etensor.FLOAT64,
		Plot:  elog.DFalse,
		Range: minmax.F64{Min: -1},
		Write: elog.WriteMap{
			elog.Scope(elog.Train, elog.Run): func(ctx *elog.Context) {
				ctx.SetStatInt("FirstZero")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlUnitErr")
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Err",
		Type: etensor.FLOAT64,
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAggItem(ctx.Mode, elog.Trial, "Err", agg.AggMean)
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5) // cached
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(1 - ctx.ItemFloatScope(ctx.Scope, "PctErr"))
			}, elog.Scope(elog.AllModes, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, elog.Epoch, 5) // cached
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetFloat64(ss.Stats.Float("TrlCosDiff"))
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.Train, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(elog.Train, elog.Epoch, 5) // cached
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "Correl",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetFloat64(ss.Stats.Float("TrlCorrel"))
			}, elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean)
			}, elog.Scope(elog.Train, elog.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(elog.Train, elog.Epoch, 5) // cached
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PerTrlMSec",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				nm := ctx.Item.Name
				tmr := ctx.Stats.StopTimer(nm)
				trls := ctx.Logs.Table(ctx.Mode, elog.Trial)
				tmr.N = trls.Rows
				pertrl := tmr.AvgMSecs()
				ctx.Stats.SetFloat(nm, pertrl)
				ctx.SetFloat64(pertrl)
				tmr.ResetStart()
			}}})

	// Standard stats for Ge and AvgAct tuning -- for all hidden, output layers
	layers := []string{"Hidden1", "Hidden2", "Output"}
	for _, lnm := range layers {
		clnm := lnm
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.AvgMaxGeM)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifAvg",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Avg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifMax",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Max)
				}}})
		// Test Cycle activity plot
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + " Ge.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Test, elog.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Ge.Avg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + " Act.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Test, elog.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Avg)
				}}})
	}

	// input layer average activity -- important for tuning
	layers = []string{"Input"}
	for _, lnm := range layers {
		clnm := lnm
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
	}

	// input / output layer activity patterns during testing
	layers = []string{"Input", "Output"}
	for _, lnm := range layers {
		clnm := lnm
		cly := ss.Net.LayerByName(clnm)
		ss.Logs.AddItem(&elog.Item{
			Name:      clnm + "_Act",
			Type:      etensor.FLOAT64,
			CellShape: cly.Shape().Shp,
			FixMax:    elog.DTrue,
			Range:     minmax.F64{Max: 1},
			Write: elog.WriteMap{
				elog.Scope(elog.Test, elog.Trial): func(ctx *elog.Context) {
					ctx.SetLayerTensor(clnm, "Act")
				}}})
		if cly.Type() == emer.Target {
			ss.Logs.AddItem(&elog.Item{
				Name:      clnm + "_ActM",
				Type:      etensor.FLOAT64,
				CellShape: cly.Shape().Shp,
				FixMax:    elog.DTrue,
				Range:     minmax.F64{Max: 1},
				Write: elog.WriteMap{
					elog.Scope(elog.Test, elog.Trial): func(ctx *elog.Context) {
						ctx.SetLayerTensor(clnm, "ActM")
					}}})
		}
	}
}
