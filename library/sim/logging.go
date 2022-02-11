package sim

import (
	"fmt"

	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/norm"
	"github.com/emer/etable/split"
)

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Logs.CreateTables()
}

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.Logs.ValsTsrs == nil {
		ss.Logs.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.Logs.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.Logs.ValsTsrs[name] = tsr
	}
	return tsr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	rn := ""
	if ss.Tag != "" {
		rn += ss.Tag + "_"
	}
	rn += ss.Params.Name()
	if ss.CmdArgs.StartRun > 0 {
		rn += fmt.Sprintf("_%03d", ss.CmdArgs.StartRun)
	}
	return rn
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.Run.Cur, (ss.TrainEnv).Epoch().Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// TODO Unify these functions
// TODO move these calculations to the logger add items compute function
// Create a general Log(mode, time) function

func (ss *Sim) Log(mode elog.EvalModes, time elog.Times) {
	dt := ss.Logs.Table(mode, time)

	row := dt.Rows
	if time == elog.Cycle {
		row = ss.Time.Cycle
	}
	if time == elog.Trial {
		// TODO Why is this not stored on ss.Time?
		if mode == elog.Test {
			row = (ss.TestEnv).Trial().Cur
		} else {
			row = (ss.TrainEnv).Trial().Cur
		}
	}

	// TODO These should be callback functions
	if mode == elog.Train && time == elog.Epoch {
		ss.UpdateTrnEpc()
	}

	ss.Logs.LogRow(mode, time, row)

	// TODO These should be callback functions
	if mode == elog.Test && time == elog.Epoch {
		ss.UpdateTstEpcErrors()
	}
	if mode == elog.Train && time == elog.Run {
		ss.UpdateRun(dt)
	}
}

// Callback functions that update miscellaneous logs
// TODO Move these to logging_content.go

func (ss *Sim) UpdateTrnEpc() {
	epc := (ss.TrainEnv).Epoch().Prv // this is triggered by increment so use previous value

	sumErr := ss.Stats.FloatMetric("SumErr")
	epcSumErr := float64(sumErr)
	ss.Stats.SetFloatMetric("SumErr", 0)

	if ss.Stats.IntMetric("FirstZero") < 0 && epcSumErr == 0 {
		ss.Stats.SetIntMetric("FirstZero", epc)
	}
	if epcSumErr == 0 {
		nzero := ss.Stats.IntMetric("NZero")
		ss.Stats.SetIntMetric("NZero", nzero+1)
	} else {
		ss.Stats.SetIntMetric("NZero", 0)
	}
}

func (ss *Sim) UpdateTstEpcErrors() {
	// Record those test trials which had errors
	trl := ss.Logs.Table(elog.Test, elog.Trial)
	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("UnitErr", row) > 0 // include error trials
	})

	ss.Logs.MiscTables["TestErrorLog"] = trlix.NewTable()

	allsp := split.All(trlix)
	split.Agg(allsp, "UnitErr", agg.AggSum)
	split.Agg(allsp, "InAct", agg.AggMean)
	split.Agg(allsp, "OutActM", agg.AggMean)
	split.Agg(allsp, "OutActP", agg.AggMean)

	ss.Logs.MiscTables["TestErrorStats"] = allsp.AggsToTable(etable.AddAggName)
}

//////////////////////////////////////////////
//  RunLog

// UpdateRun adds data from current run to the RunLog table.
func (ss *Sim) UpdateRun(dt *etable.Table) {
	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")

	ss.Logs.MiscTables["RunStats"] = spl.AggsToTable(etable.AddAggName)
}

//////////////////////////////////////////////
//  SpikeRasters

// SpikeRastTsr gets spike raster tensor of given name, creating if not yet made
func (ss *Sim) SpikeRastTsr(name string) *etensor.Float32 {
	if ss.Logs.SpikeRasters == nil {
		ss.Logs.SpikeRasters = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.Logs.SpikeRasters[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.Logs.SpikeRasters[name] = tsr
	}
	return tsr
}

// SpikeRastGrid gets spike raster grid of given name, creating if not yet made
func (ss *Sim) SpikeRastGrid(name string) *etview.TensorGrid {
	if ss.Logs.SpikeRastGrids == nil {
		ss.Logs.SpikeRastGrids = make(map[string]*etview.TensorGrid)
	}
	tsr, ok := ss.Logs.SpikeRastGrids[name]
	if !ok {
		tsr = &etview.TensorGrid{}
		ss.Logs.SpikeRastGrids[name] = tsr
	}
	return tsr
}

// SetSpikeRastCol sets column of given spike raster from data
func (ss *Sim) SetSpikeRastCol(sr, vl *etensor.Float32, col int) {
	for ni, v := range vl.Values {
		sr.Set([]int{ni, col}, v)
	}
}

// ConfigSpikeGrid configures the spike grid
func (ss *Sim) ConfigSpikeGrid(tg *etview.TensorGrid, sr *etensor.Float32) {
	tg.SetStretchMax()
	sr.SetMetaData("grid-fill", "1")
	tg.SetTensor(sr)
}

// ConfigSpikeRasts configures spike rasters
func (ss *Sim) ConfigSpikeRasts() {
	ncy := 200 // max cycles
	// spike rast
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		sr := ss.SpikeRastTsr(lnm)
		sr.SetShape([]int{ly.Shp.Len(), ncy}, nil, []string{"Nrn", "Cyc"})
	}
}

// RecSpikes records spikes
func (ss *Sim) RecSpikes(cyc int) {
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		tv := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tv, "Spike")
		sr := ss.SpikeRastTsr(lnm)
		ss.SetSpikeRastCol(sr, tv, cyc)
	}
}

// AvgLayVal returns average of given layer variable value
func (ss *Sim) AvgLayVal(ly *axon.Layer, vnm string) float32 {
	tv := ss.ValsTsr(ly.Name())
	ly.UnitValsTensor(tv, vnm)
	return norm.Mean32(tv.Values)
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {

	// accumulators
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloatMetric("Sumerr", 0.0)
	ss.Stats.SetFloatMetric("TrlErr", 0.0)
	ss.Stats.SetStringMetric("TrlClosest", "")
	ss.Stats.SetFloatMetric("TrlCorrel", 0.0)
	ss.Stats.SetFloatMetric("TrlUnitErr", 0.0)
	ss.Stats.SetFloatMetric("TrlCosDiff", 0.0)

	ss.Stats.SetIntMetric("FirstZero", -1)
	ss.Stats.SetIntMetric("NZero", 0)
	//stats.SetFloatMetric("TrlCosDiff", 0, 0)
	// internal state - view:"-"
	//SumErr float64 `view:"-" inactive:"+" desc:"Sum of errors throughout epoch. This way we can know when an epoch is error free, for early stopping."`

	// statistics: note use float64 as that is best for etable.Table
	// TODO Maybe put this on a Stats object - moved to map
	//TrlErr     float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on UnitErr = 0 (subject to .5 unit-wise tolerance)"`
	//TrlClosest string  `inactive:"+" desc:"Name of the pattern with the closest output"`
	//TrlCorrel  float64 `inactive:"+" desc:"Correlation with closest output"`
	//TrlUnitErr float64 `inactive:"+" desc:"current trial's unit-level pct error"`
	//TrlCosDiff float64 `inactive:"+" desc:"current trial's cosine difference"`

	// TODO Move these to a newly created func EpochStats
	// State about how long there's been zero error.
	//FirstZero   int       `inactive:"+" desc:"epoch at when all TrlErr first went to zero"`
	//NZero       int       `inactive:"+" desc:"number of epochs in a row with no TrlErr"`

}
