package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/elog"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
)

// InitStats initializes all the statistics.
// called at start of new run
func InitHipStats(ss *sim.Sim) {
	// TODO HIP Copied
	// accumulators
	ss.Stats.SetFloat("SumUnitErr", 0)
	ss.Stats.SetFloat("SumCosDiff", 0)
	ss.Stats.SetInt("CntErr", 0)
	ss.Stats.SetInt("HipFirstZero", -1)
	ss.Stats.SetInt("HipNZero", 0)
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("Mem", 0)
	ss.Stats.SetFloat("TrgOnWasOff", 0)
	ss.Stats.SetFloat("TrgOnWasOffCmp", 0)
	ss.Stats.SetFloat("TrgOffWasOn", 0)
	ss.Stats.SetFloat("TrlUnitErr", 0)
	ss.Stats.SetFloat("EpcUnitErr", 0)
	ss.Stats.SetFloat("EpcPctErr", 0)
	ss.Stats.SetFloat("EpcCosDiff", 0)

	// TODO These need to be initialized. Maybe all stats should initialize to 0?
	ss.Stats.SetFloat("MemThr", 0.34)
	ss.Stats.SetInt("NZeroStop", 1)
}

func ConfigHipItems(ss *sim.Sim) {
	ss.Logs.AddItem(&elog.Item{
		Name:   "Mem",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		FixMin: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("Mem")
			},
			elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean) // TODO how is this referencing Mem name
			},
		}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "TrgOnWasOff",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		FixMin: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrgOnWasOff")
			},
			elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean) // TODO how is this referencing Mem name
			},
		}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "TrgOffWasOn",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		FixMin: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			elog.Scope(elog.AllModes, elog.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrgOffWasOn")
			},
			elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean) // TODO how is this referencing Mem name
			},
		}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TestNm",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			elog.Scope(elog.Test, elog.Trial): func(ctx *elog.Context) {
				// TODO Actually get this
				ctx.SetString("AB")
			},
		}})

	// TODO Add AB Mem and stuff
	tstNms := []string{"AB", "AC"} // TODO Add in "Lure"
	tstStatNms := []string{"Mem", "TrgOnWasOff", "TrgOffWasOn"}

	for _, tn := range tstNms {
		for _, ts := range tstStatNms {
			plot := elog.DFalse
			if ts == "Mem" {
				plot = elog.DTrue
			}
			ss.Logs.AddItem(&elog.Item{
				Name:   tn + " " + ts,
				Type:   etensor.FLOAT64,
				Plot:   plot,
				FixMax: elog.DTrue,
				FixMin: elog.DTrue,
				Range:  minmax.F64{Max: 1},
				Write: elog.WriteMap{
					// TODO These are not right
					elog.Scope(elog.Test, elog.Epoch): func(ctx *elog.Context) {
						trl := ss.Logs.Log(elog.Test, elog.Trial)
						trix := etable.NewIdxView(trl)
						spl := split.GroupBy(trix, []string{"TestNm"})
						for _, ts := range tstStatNms {
							split.Agg(spl, ts, agg.AggMean)
						}
						tstStats := spl.AggsToTable(etable.ColNameOnly)

						for ri := 0; ri < tstStats.Rows; ri++ {
							tst := tstStats.CellString("TestNm", ri)
							for _, tso := range tstStatNms {
								if tst+" "+tso == tn+" "+ts {
									ctx.SetFloat64(tstStats.CellFloat(ts, ri))
								}
								//epcLog.SetCellFloat(tst+" "+tso, row, ss.TstStats.CellFloat(ts, ri))
							}
						}

						ctx.SetStatFloat("TrgOffWasOn")
					},
					//elog.Scope(elog.AllModes, elog.Epoch): func(ctx *elog.Context) {
					//	ctx.SetAgg(ctx.Mode, elog.Trial, agg.AggMean) // TODO how is this referencing Mem name
					//},
				}})
		}
	}

	//ss.Logs.Log(elog.Test,elog.Epoch).CellFloat("AB Mem")
}
