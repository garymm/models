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
			elog.GenKey(elog.AllModes, elog.AllTimes): func(ctx *elog.Context) {
				ctx.SetInt(ss.Run.Cur)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Params",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.AllModes, elog.AllTimes): func(ctx *elog.Context) {
				ctx.SetString(ss.RunName())
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:  "FirstZero",
		Type:  etensor.FLOAT64,
		Plot:  elog.DFalse,
		Range: minmax.F64{Min: -1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Train, elog.Run): func(ctx *elog.Context) {
				ctx.SetStatInt("FirstZero")
			},
		}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Epoch",
		Type: etensor.INT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenKeys([]elog.EvalModes{elog.AllModes}, []elog.Times{elog.Epoch, elog.Trial}): func(ctx *elog.Context) {
				ctx.SetInt(ss.TrainEnv.Epoch().Prv)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Train, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlUnitErr")
			}, elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(elog.Train, elog.Trial, agg.AggMean)
			}, elog.GenKey(elog.Train, elog.Run): func(ctx *elog.Context) {
				epochWin := getEpochWindowLast5(ss)
				ctx.SetFloat64(agg.Mean(epochWin, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Train, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr")
			}, elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), ctx.Item.Name)[0])
			}, elog.GenKey(elog.Train, elog.Run): func(ctx *elog.Context) {
				epochWin := getEpochWindowLast5(ss)
				ctx.SetFloat64(agg.Mean(epochWin, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(1 - agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), "PctErr")[0])
			}, elog.GenKey(elog.Train, elog.Run): func(ctx *elog.Context) {
				epochWin := getEpochWindowLast5(ss)
				ctx.SetFloat64(agg.Mean(epochWin, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Train, elog.Trial): func(ctx *elog.Context) {
				ctx.SetFloat64(ss.Stats.Float("TrlCosDiff"))
			}, elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), ctx.Item.Name)[0])
			}, elog.GenKey(elog.Train, elog.Run): func(ctx *elog.Context) {
				epochWin := getEpochWindowLast5(ss)
				ctx.SetFloat64(agg.Mean(epochWin, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "Correl",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Train, elog.Trial): func(ctx *elog.Context) {
				ctx.SetFloat64(ss.Stats.Float("TrlCorrel"))
			}, elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(agg.Mean(ss.Logs.IdxView(elog.Train, elog.Trial), ctx.Item.Name)[0])
			}, elog.GenKey(elog.Train, elog.Run): func(ctx *elog.Context) {
				epochWin := getEpochWindowLast5(ss)
				ctx.SetFloat64(agg.Mean(epochWin, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PerEpcMSec",
		Type: etensor.FLOAT64,
		Plot: elog.DFalse,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
				epcPerTrlMSec := 0.0
				if ss.LastEpcTime.IsZero() {
					epcPerTrlMSec = 0
				} else {
					iv := time.Now().Sub(ss.LastEpcTime)
					// TODO This should be normalized by number of trials rather than 1.0 and renamed back to PerTrlMSec
					epcPerTrlMSec = float64(iv) / (1.0 * float64(time.Millisecond))
				}
				ss.LastEpcTime = time.Now()
				ctx.SetFloat64(epcPerTrlMSec)
			}}})
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		curlname := lnm
		ss.Logs.AddItem(&elog.Item{
			Name:   curlname + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
		// ss.Logs.AddItem(&elog.Item{
		// 	Name:  curlname + "_MaxGeM",
		// 	Type:  etensor.FLOAT64,
		// 	Plot:  elog.DFalse,
		// 	Range: minmax.F64{Max: 1},
		// 	Compute: elog.ComputeMap{
		// 		elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
		//				ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
		// 			ctx.SetFloat32(ly.ActAvg.AvgMaxGeM))
		// 		}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_AvgGe",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Ge.Avg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_MaxGe",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Ge.Max)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_Gi",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Gi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_AvgDifAvg",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Avg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + "_AvgDifMax",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Train, elog.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Max)
				}}})
	}

	// Test trial and epoch and cycle
	//ss.Logs.AddItem(&elog.Item{
	//	Name: "Run",
	//	Type: etensor.INT64,
	//Compute: elog.ComputeMap{
	//elog.GenKey(elog.Test, elog.Trial):  func(ctx *elog.Context) {
	//	ctx.SetInt(ss.Run.Cur))
	//}, elog.GenKey(elog.Test, elog.Epoch):  func(ctx *elog.Context) {
	//	ctx.SetInt(ss.Run.Cur))
	//}}})
	//ss.Logs.AddItem(&elog.Item{
	//	Name: "Epoch",
	//	Type: etensor.INT64,
	//Compute: elog.ComputeMap{
	//elog.GenKey(elog.Test, elog.Trial):  func(ctx *elog.Context) {
	//	ctx.SetInt(ss.TrainEnv.Epoch().Prv))
	//}, elog.GenKey(elog.Test, elog.Epoch):  func(ctx *elog.Context) {
	//	ctx.SetInt(ss.TrainEnv.Epoch().Prv))
	//}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Trial",
		Type: etensor.INT64,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetInt(ss.TestEnv.Trial().Cur)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TrialName",
		Type: etensor.STRING,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetString(ss.TestEnv.GetCurrentTrialName())
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "Err",
		Type: etensor.FLOAT64,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "UnitErr",
		Type: etensor.FLOAT64,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlUnitErr")
			}, elog.GenKey(elog.Test, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(elog.Test, elog.Trial, agg.AggSum)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctErr",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(elog.Test, elog.Trial, agg.AggMean)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "PctCor",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetFloat64(1 - agg.Mean(ctx.Logs.IdxView(elog.Test, elog.Trial), "Err")[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "CosDiff",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlCosDiff")
			}, elog.GenKey(elog.Test, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(elog.Test, elog.Trial, agg.AggSum)
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "Correl",
		Type:   etensor.FLOAT64,
		FixMax: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlCorrel")
			}, elog.GenKey(elog.Test, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(elog.Test, elog.Trial, agg.AggSum)
			}}})

	//TODO move inp and out to compute function in some helper that goes inside the function
	inp := ss.Net.LayerByName("Input")
	out := ss.Net.LayerByName("Output")

	ss.Logs.AddItem(&elog.Item{
		Name:      "InAct",
		Type:      etensor.FLOAT64,
		CellShape: inp.Shape().Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
				ctx.SetLayerTensor("Input", "Act")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:      "OutActM",
		Type:      etensor.FLOAT64,
		CellShape: out.Shape().Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
				ctx.SetLayerTensor("Output", "ActM")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name:      "OutActP",
		Type:      etensor.FLOAT64,
		CellShape: out.Shape().Shp,
		FixMax:    elog.DTrue,
		Range:     minmax.F64{Max: 1},
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
				ctx.SetLayerTensor("Output", "ActP")
			}}})
	// Cycle
	ss.Logs.AddItem(&elog.Item{
		Name: "Cycle",
		Type: etensor.INT64,
		Compute: elog.ComputeMap{
			elog.GenKey(elog.Test, elog.Cycle): func(ctx *elog.Context) {
				ctx.SetInt(ctx.Row)
			}}})

	// TODO: add iterator for this?
	// Add for each layer
	for _, lnm := range ss.LayStatNms {
		curlname := lnm
		ss.Logs.AddItem(&elog.Item{
			Name:   curlname + " ActM.Avg",
			Type:   etensor.FLOAT64,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Test, elog.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + " Ge.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Test, elog.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Ge.Avg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  curlname + " Act.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Compute: elog.ComputeMap{
				elog.GenKey(elog.Test, elog.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(curlname).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Avg)
				}}})
	}
}
