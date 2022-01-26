package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/norm"
	"github.com/emer/etable/split"
	"strconv"
	"strings"
	"time"
)

// LogPrec is precision for saving float values in logs
const LogPrec = 4

type EvaluationType int64

type LogFunc func(ss *Sim, dt *etable.Table, row int, name string)
type LogFuncLayer func(ss *Sim, dt *etable.Table, row int, name string, layer axon.Layer)

const (
	Train EvaluationType = 0
	Test                 = 1
)

type LogItem struct {
	etable.Column                                  // Inherits elements Name, Type, CellShape, DimNames
	Range         minmax.F32                       `desc:"The minimum and maximum"`
	Compute       map[axon.TimeScales]LogFunc      `desc:"For each timescale, how is this value computed?"`
	ComputeLayer  map[axon.TimeScales]LogFuncLayer `desc:"For each timescale, how is this value computed? This is for layer specific callbacks."`
	Plot          bool                             `desc:"Whether or not to plot it"`
	FixMin        bool                             `desc:"Whether to fix the minimum in the display"`
	FixMax        bool                             `desc:"Whether to fix the maximum in the display"`
	EvalType      EvaluationType                   `desc:"Describes what the evaluation of the type"`
	LayerName     string                           `desc:"The name of the layer that this should apply to. This will only not be empty for items that are logged per layer"`
}

type LogSpec struct {
	Items []*LogItem `desc:""`
	//PerLayerDetails []*LogItem `desc:""` // DO NOT SUBMIT delete this
}

func (logSpec *LogSpec) AddItem(item *LogItem) {
	logSpec.Items = append(logSpec.Items, item)
}

//func (logSpec *LogSpec) AddLayerItem(item *LogItem) {
//	logSpec.PerLayerDetails = append(logSpec.PerLayerDetails, item)
//}

//func (logSpec *LogSpec) DuplicateForTest() {
//
//	var length = len(logSpec.Items)
//	for i := 0; i < length; i++ {
//		copiedLog := *logSpec.Items[i]
//		copiedLog.EvalType = Test
//		logSpec.AddItem(&copiedLog)
//	}
//
//	//var lengthLayer = len(logSpec.PerLayerDetails)
//	//for i := 0; i < lengthLayer; i++ {
//	//	copiedLog := *logSpec.PerLayerDetails[i]
//	//	copiedLog.EvalType = Test
//	//	logSpec.AddLayerItem(&copiedLog)
//	//}
//}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

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
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	//nt := float64(len(ss.TrainEnv.Order)) // number of trials in view
	nt := float64(ss.TrainEnv.Trial.Max) //TODO: figure out the appropriate normalization term for the loss
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

	for _, item := range ss.LogSpec.Items {
		if item.EvalType == Train {
			callback, ok := item.Compute[axon.Epoch]
			if ok {
				callback(ss, dt, row, item.Name)
			}
		}
	}
	//// TODO(Logging) Need to get callbacks working.
	//dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	//dt.SetCellFloat("Epoch", row, float64(epc))
	//dt.SetCellFloat("UnitErr", row, ss.EpcUnitErr)
	//dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	//dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	//dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	//dt.SetCellFloat("Correl", row, ss.EpcCorrel)
	//dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		for _, item := range ss.LogSpec.Items {
			if item.EvalType == Train {
				callback, ok := item.ComputeLayer[axon.Epoch]
				if ok && item.LayerName == lnm {
					// TODO(optimize) is this copying ly?
					callback(ss, dt, row, item.Name, *ly)
				}
			}
		}

		//// DO NOT SUBMIT Delete this stuff
		//// ffpj := ly.RecvPrjn(0).(*axon.Prjn)
		//// dt.SetCellFloat(ly.Nm+"_FF_AvgMaxG", row, float64(ffpj.GScale.AvgMax))
		//// dt.SetCellFloat(ly.Nm+"_FF_Scale", row, float64(ffpj.GScale.Scale))
		//// if ly.NRecvPrjns() > 1 {
		//// 	fbpj := ly.RecvPrjn(1).(*axon.Prjn)
		//// 	dt.SetCellFloat(ly.Nm+"_FB_AvgMaxG", row, float64(fbpj.GScale.AvgMax))
		//// 	dt.SetCellFloat(ly.Nm+"_FB_Scale", row, float64(fbpj.GScale.Scale))
		//// }
		//dt.SetCellFloat(ly.Nm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
		//dt.SetCellFloat(ly.Nm+"_MaxGeM", row, float64(ly.ActAvg.AvgMaxGeM))
		//dt.SetCellFloat(ly.Nm+"_AvgGe", row, float64(ly.Pools[0].Inhib.Ge.Avg))
		//dt.SetCellFloat(ly.Nm+"_MaxGe", row, float64(ly.Pools[0].Inhib.Ge.Max))
		//dt.SetCellFloat(ly.Nm+"_Gi", row, float64(ly.Pools[0].Inhib.Gi))
		//// dt.SetCellFloat(ly.Nm+"_GiMult", row, float64(ly.ActAvg.GiMult))
		//dt.SetCellFloat(ly.Nm+"_AvgDifAvg", row, float64(ly.Pools[0].AvgDif.Avg))
		//dt.SetCellFloat(ly.Nm+"_AvgDifMax", row, float64(ly.Pools[0].AvgDif.Max))
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TrnEpcPlot != nil {
		ss.TrnEpcPlot.GoUpdate()
	}
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == ss.StartRun && row == 0 {
			// note: can't just use row=0 b/c reset table each run
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	sch := etable.Schema{}
	for _, val := range ss.LogSpec.Items {
		// Compute records which timescales are logged. It also records how, but we don't need that here.
		_, ok := val.Compute[axon.Epoch]
		if ok && val.EvalType == Train {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
		_, ok = val.ComputeLayer[axon.Epoch]
		if ok && val.EvalType == Train {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
	}

	//
	//sch := etable.Schema{
	//	{"Run", etensor.INT64, nil, nil},
	//	{"Epoch", etensor.INT64, nil, nil},
	//	{"UnitErr", etensor.FLOAT64, nil, nil},
	//	{"PctErr", etensor.FLOAT64, nil, nil},
	//	{"PctCor", etensor.FLOAT64, nil, nil},
	//	{"CosDiff", etensor.FLOAT64, nil, nil},
	//	{"Correl", etensor.FLOAT64, nil, nil},
	//	{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	//}
	//for _, lnm := range ss.LayStatNms {
	//	for _, val := range ss.LogSpec.Items {
	//		_, ok := val.ComputeLayer[axon.Epoch]
	//		if ok && val.EvalType == Train {
	//			sch = append(sch, etable.Column{val.Name, val.Type, nil, nil})
	//		}
	//	}
	//
	//	//sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
	//	//sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
	//	//sch = append(sch, etable.Column{lnm + "_AvgGe", etensor.FLOAT64, nil, nil})
	//	//sch = append(sch, etable.Column{lnm + "_MaxGe", etensor.FLOAT64, nil, nil})
	//	//sch = append(sch, etable.Column{lnm + "_Gi", etensor.FLOAT64, nil, nil})
	//	//sch = append(sch, etable.Column{lnm + "_AvgDifAvg", etensor.FLOAT64, nil, nil})
	//	//sch = append(sch, etable.Column{lnm + "_AvgDifMax", etensor.FLOAT64, nil, nil})
	//}
	dt.SetFromSchema(sch, 0)
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	//epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TestEnv.Trial.Cur
	row := trl // TODO(clean) Is this making a copy? Is it necessary?
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	for _, item := range ss.LogSpec.Items {
		if item.EvalType == Test {
			callback, ok := item.Compute[axon.Trial]
			if ok {
				callback(ss, dt, row, item.Name)
			}
		}
	}

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		for _, item := range ss.LogSpec.Items {
			if item.EvalType == Test {
				callback, ok := item.ComputeLayer[axon.Trial]
				if ok && item.LayerName == lnm {
					// TODO(optimize) is this copying ly?
					callback(ss, dt, row, item.Name, *ly)
				}
			}
		}
	}
	//dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	//dt.SetCellFloat("Epoch", row, float64(epc))
	//dt.SetCellFloat("Trial", row, float64(trl))
	//dt.SetCellString("TrialName", row, strings.Join(ss.TestEnv.CurWords, " "))
	//dt.SetCellFloat("Err", row, ss.TrlErr)
	//dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	//dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	//dt.SetCellFloat("Correl", row, ss.TrlCosDiff)

	//for _, lnm := range ss.LayStatNms {
	//	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	//	dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	//}

	//inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	//out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	//ivt := ss.ValsTsr("Input")
	//ovt := ss.ValsTsr("Output")
	//inp.UnitValsTensor(ivt, "Act")
	//dt.SetCellTensor("InAct", row, ivt)
	//out.UnitValsTensor(ovt, "ActM")
	//dt.SetCellTensor("OutActM", row, ovt)
	//out.UnitValsTensor(ovt, "ActP")
	//dt.SetCellTensor("OutActP", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstTrlPlot != nil {
		ss.TstTrlPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	//inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	//out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := len(ss.TestEnv.NGrams) // 1 //ss.TestEnv.Table.Len() // number in view
	//sch := etable.Schema{
	//	{"Run", etensor.INT64, nil, nil},
	//	{"Epoch", etensor.INT64, nil, nil},
	//	{"Trial", etensor.INT64, nil, nil},
	//	{"TrialName", etensor.STRING, nil, nil},
	//	{"Err", etensor.FLOAT64, nil, nil},
	//	{"UnitErr", etensor.FLOAT64, nil, nil},
	//	{"CosDiff", etensor.FLOAT64, nil, nil},
	//	{"Correl", etensor.FLOAT64, nil, nil},
	//}
	sch := etable.Schema{}
	for _, val := range ss.LogSpec.Items {
		// Compute records which timescales are logged. It also records how, but we don't need that here.
		_, ok := val.Compute[axon.Trial]
		if ok && val.EvalType == Test {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
		_, ok = val.ComputeLayer[axon.Trial]
		if ok && val.EvalType == Test {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
	}
	//for _, lnm := range ss.LayStatNms {
	//	sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	//}
	//sch = append(sch, etable.Schema{
	//	{"InAct", etensor.FLOAT64, inp.Shp.Shp, nil},
	//	{"OutActM", etensor.FLOAT64, out.Shp.Shp, nil},
	//	{"OutActP", etensor.FLOAT64, out.Shp.Shp, nil},
	//}...)
	dt.SetFromSchema(sch, nt)
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	//trl := ss.TstTrlLog
	//tix := etable.NewIdxView(trl)
	//epc := ss.TrainEnv.Epoch.Prv // ?

	//// note: this shows how to use agg methods to compute summary data from another
	//// data table, instead of incrementing on the Sim
	//// TODO(Logging) Need to get callbacks working.
	//dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	////dt.SetCellFloat("Epoch", row, float64(epc))
	//dt.SetCellFloat("UnitErr", row, agg.Sum(tix, "UnitErr")[0])
	//dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
	//dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])
	//dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])
	//dt.SetCellFloat("Correl", row, agg.Mean(tix, "Correl")[0])

	for _, item := range ss.LogSpec.Items {
		if item.EvalType == Test {
			callback, ok := item.Compute[axon.Epoch]
			if ok {
				callback(ss, dt, row, item.Name)
			}
		}
	}

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		for _, item := range ss.LogSpec.Items {
			if item.EvalType == Test {
				callback, ok := item.ComputeLayer[axon.Epoch]
				if ok && item.LayerName == lnm {
					// TODO(optimize) is this copying ly?
					callback(ss, dt, row, item.Name, *ly)
				}
			}
		}
	}

	// Record those test trials which had errors
	trl := ss.TstTrlLog
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

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstEpcPlot != nil {
		ss.TstEpcPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	//sch := etable.Schema{
	//	{"Run", etensor.INT64, nil, nil},
	//	{"Epoch", etensor.INT64, nil, nil},
	//	{"UnitErr", etensor.FLOAT64, nil, nil},
	//	{"PctErr", etensor.FLOAT64, nil, nil},
	//	{"PctCor", etensor.FLOAT64, nil, nil},
	//	{"CosDiff", etensor.FLOAT64, nil, nil},
	//	{"Correl", etensor.FLOAT64, nil, nil},
	//}

	sch := etable.Schema{}
	for _, val := range ss.LogSpec.Items {
		// Compute records which timescales are logged. It also records how, but we don't need that here.
		_, ok := val.Compute[axon.Epoch]
		if ok && val.EvalType == Test {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
		_, ok = val.ComputeLayer[axon.Epoch]
		if ok && val.EvalType == Test {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
	}
	dt.SetFromSchema(sch, 0)
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

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	//dt.SetCellFloat("Cycle", cyc, float64(cyc))
	//for _, lnm := range ss.LayStatNms {
	//	ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	//	dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
	//	dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	//}

	for _, item := range ss.LogSpec.Items {
		if item.EvalType == Test {
			callback, ok := item.Compute[axon.Cycle]
			if ok {
				callback(ss, dt, cyc, item.Name)
			}
		}
	}

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		for _, item := range ss.LogSpec.Items {
			if item.EvalType == Test {
				callback, ok := item.ComputeLayer[axon.Cycle]
				if ok && item.LayerName == lnm {
					// TODO(optimize) is this copying ly?
					callback(ss, dt, cyc, item.Name, *ly)
				}
			}
		}
	}

	if ss.TstCycPlot != nil && cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{}
	for _, val := range ss.LogSpec.Items {
		// Compute records which timescales are logged. It also records how, but we don't need that here.
		_, ok := val.Compute[axon.Cycle]
		if ok && val.EvalType == Test {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
		_, ok = val.ComputeLayer[axon.Cycle]
		if ok && val.EvalType == Test {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
	}
	//sch := etable.Schema{
	//	{"Cycle", etensor.INT64, nil, nil},
	//}
	//for _, lnm := range ss.LayStatNms {
	//	sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
	//	sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	//}
	dt.SetFromSchema(sch, np)
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	epclog := ss.TrnEpcLog
	epcix := etable.NewIdxView(epclog)
	if epcix.Len() == 0 {
		return
	}

	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("UnitErr", row, agg.Mean(epcix, "UnitErr")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])
	dt.SetCellFloat("Correl", row, agg.Mean(epcix, "Correl")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	if ss.RunPlot != nil {
		ss.RunPlot.GoUpdate()
	}
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Correl", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
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

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ss.TrlCosDiff = float64(out.CosDiff.Cos)

	_, cor, closestWord := ss.ClosestStat(ss.Net, "Output", "ActM", ss.Pats, "Pattern", "Word")
	ss.TrlClosest = closestWord
	ss.TrlCorrel = float64(cor)
	contextWords := strings.Join(ss.TrainEnv.CurWords, " ")

	//Check if the closest word that is found is one of the potential following words
	_, ok := ss.TrainEnv.NGrams[contextWords][closestWord]
	if ok {
		ss.TrlErr = 0
	} else {
		ss.TrlErr = 1
	}

	if accum {
		ss.SumErr += ss.TrlErr
		ss.SumUnitErr += ss.TrlUnitErr
		ss.SumCosDiff += ss.TrlCosDiff
		ss.SumCorrel += ss.TrlCorrel
	}
}
