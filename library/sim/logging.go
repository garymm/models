package sim

import (
	"fmt"
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/norm"
	"github.com/emer/etable/split"
	"time"
)

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Logs.CreateTables()
}

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
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
	rn += ss.ParamsName()
	if ss.StartRun > 0 {
		rn += fmt.Sprintf("_%03d", ss.StartRun)
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
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName((ss.TrainEnv).Run().Cur, (ss.TrainEnv).Epoch().Cur) + ".wts"
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
	dt := ss.Logs.GetTable(mode, time)

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
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	// TODO These should be callback functions
	if mode == elog.Train && time == elog.Epoch {
		ss.UpdateTrnEpc()
	}

	// TODO(optimize): A map would be more efficient than a filter loop.
	for _, item := range ss.Logs.Items {
		callback, ok := item.GetComputeFunc(mode, time)
		if ok {
			callback(item, item.GetScopeKey(mode, time), dt, row)
		}
	}

	// TODO These should be callback functions
	if mode == elog.Test && time == elog.Epoch {
		ss.UpdateTstEpcErrors()
	}
	if mode == elog.Train && time == elog.Run {
		ss.UpdateRun(dt)
	}

	// TODO Put these in a map or on LogTable
	var plt *eplot.Plot2D
	if mode == elog.Train && time == elog.Epoch {
		plt = ss.TrnEpcPlot
	}
	if mode == elog.Test && time == elog.Trial {
		plt = ss.TstTrlPlot
	}
	if mode == elog.Test && time == elog.Epoch {
		plt = ss.TstEpcPlot
	}
	// Note this special case
	if mode == elog.Test && time == elog.Cycle && row == 10 {
		plt = ss.TstCycPlot
	}
	if mode == elog.Train && time == elog.Run {
		plt = ss.RunPlot
	}
	if plt != nil {
		plt.GoUpdate()
	}

	// TODO Check LogTable for details on saving the plot
	if mode == elog.Train && time == elog.Epoch {
		if ss.TrnEpcFile != nil {
			if (ss.TrainEnv).Run().Cur == ss.StartRun && row == 0 {
				// note: can't just use row=0 b/c reset table each run
				dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
			}
			dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
		}
	}
	if mode == elog.Train && time == elog.Run {
		if ss.RunFile != nil {
			if row == 0 {
				dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
			}
			dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
		}
	}
}

func (ss *Sim) UpdateTrnEpc() {
	epc := (ss.TrainEnv).Epoch().Prv // this is triggered by increment so use previous value
	//nt := float64(len((*ss.TrainEnv).Order)) // number of trials in view
	nt := float64((ss.TrainEnv).Trial().Max) //TODO: figure out the appropriate normalization term for the loss
	ss.EpcUnitErr = ss.SumUnitErr / nt
	ss.SumUnitErr = 0
	ss.EpcPctErr = float64(ss.SumErr) / nt
	ss.SumErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.EpcCorrel = ss.SumCorrel / nt
	ss.SumCosDiff = 0
	ss.SumCorrel = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}
	if ss.EpcPctErr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) UpdateTstEpcErrors() {
	// Record those test trials which had errors
	trl := ss.Logs.GetTable(elog.Test, elog.Trial)
	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("UnitErr", row) > 0 // include error trials
	})
	ss.TstErrLog = trlix.NewTable()
	allsp := split.All(trlix)
	split.Agg(allsp, "UnitErr", agg.AggSum)
	split.Agg(allsp, "InAct", agg.AggMean)
	split.Agg(allsp, "OutActM", agg.AggMean)
	split.Agg(allsp, "OutActP", agg.AggMean)
	ss.TstErrStats = allsp.AggsToTable(etable.AddAggName)
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) UpdateRun(dt *etable.Table) {
	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)
}

//////////////////////////////////////////////
//  SpikeRasters

// SpikeRastTsr gets spike raster tensor of given name, creating if not yet made
func (ss *Sim) SpikeRastTsr(name string) *etensor.Float32 {
	if ss.SpikeRasters == nil {
		ss.SpikeRasters = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.SpikeRasters[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.SpikeRasters[name] = tsr
	}
	return tsr
}

// SpikeRastGrid gets spike raster grid of given name, creating if not yet made
func (ss *Sim) SpikeRastGrid(name string) *etview.TensorGrid {
	if ss.SpikeRastGrids == nil {
		ss.SpikeRastGrids = make(map[string]*etview.TensorGrid)
	}
	tsr, ok := ss.SpikeRastGrids[name]
	if !ok {
		tsr = &etview.TensorGrid{}
		ss.SpikeRastGrids[name] = tsr
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
	ss.SumErr = 0
	ss.SumUnitErr = 0
	ss.SumCosDiff = 0
	ss.SumCorrel = 0
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlErr = 0
	ss.TrlUnitErr = 0
	ss.EpcUnitErr = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}
