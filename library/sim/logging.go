package sim

import (
	"fmt"

	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/split"
)

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(elog.Train, elog.Cycle)
	ss.Logs.NoPlot(elog.Test, elog.Run)
	ss.Logs.NoPlot(elog.Analyze, elog.Trial)
	ss.Logs.SetMeta(elog.Train, elog.Run, "LegendCol", "Params")
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

	sumErr := ss.Stats.Float("SumErr")
	epcSumErr := float64(sumErr)
	ss.Stats.SetFloat("SumErr", 0)

	if ss.Stats.Int("FirstZero") < 0 && epcSumErr == 0 {
		ss.Stats.SetInt("FirstZero", epc)
	}
	if epcSumErr == 0 {
		nzero := ss.Stats.Int("NZero")
		ss.Stats.SetInt("NZero", nzero+1)
	} else {
		ss.Stats.SetInt("NZero", 0)
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
		sr := ss.Stats.F32Tensor("Raster_" + lnm)
		sr.SetShape([]int{ly.Shp.Len(), ncy}, nil, []string{"Nrn", "Cyc"})
	}
}

// RecSpikes records spikes
func (ss *Sim) RecSpikes(cyc int) {
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		tv := ss.Stats.F32Tensor(lnm)
		ly.UnitValsTensor(tv, "Spike")
		sr := ss.Stats.F32Tensor("Raster_" + lnm)
		ss.SetSpikeRastCol(sr, tv, cyc)
	}
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("TrlErr", 0.0)
	ss.Stats.SetString("TrlClosest", "")
	ss.Stats.SetFloat("TrlCorrel", 0.0)
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCosDiff", 0.0)
	ss.Stats.SetInt("FirstZero", -1)
	ss.Stats.SetInt("NZero", 0)
}
