package sim

import (
	"fmt"

	"github.com/emer/axon/axon"

	"github.com/emer/emergent/elog"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/split"
)

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them to the GUI, if the GUI is active
func (ss *Sim) StatCounters(train bool) {
	ev := ss.TrainEnv
	if !train {
		ev = ss.TestEnv
	}
	ss.Stats.SetInt("Run", ss.Run.Cur)
	ss.Stats.SetInt("Epoch", ss.TrainEnv.Epoch().Cur)
	ss.Stats.SetInt("Trial", ev.Trial().Cur)
	ss.Stats.SetString("TrialName", ev.CurTrialName())
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	displayText := fmt.Sprintf("%s\tRun:\t%d/%d\tEpoch:%d/%d\tTrial:\t%d/%d\tCycle:\t%d\t", ss.Trainer.EvalMode, ss.Run.Cur, ss.Run.Max, ev.Epoch().Cur, ev.Epoch().Max, ev.Trial().Cur, ev.Trial().Max, ss.Time.Cycle) + "\t" + ss.Stats.Print([]string{"TrlErr", "TrlCosDiff"})
	//println(displayText)
	ss.GUI.NetViewText = displayText
}

func (ss *Sim) ConfigLogsFromArgs() {
	elog.LogDir = "logs"
	if ss.CmdArgs.saveEpcLog {
		fnm := ss.LogFileName("epc")
		ss.Logs.SetLogFile(elog.Train, elog.Epoch, fnm)

		//Save test as well as train epoch logs
		testfnm := ss.LogFileName("testepc")
		ss.Logs.SetLogFile(elog.Test, elog.Epoch, testfnm)
	}
	if ss.CmdArgs.saveRunLog {
		fnm := ss.LogFileName("run")
		ss.Logs.SetLogFile(elog.Train, elog.Run, fnm)
	}
}

func (ss *Sim) ConfigLogs() {
	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(elog.Train, elog.Cycle)
	ss.Logs.NoPlot(elog.Test, elog.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(elog.Train, elog.Run, "LegendCol", "Params")
	ss.Stats.ConfigRasters(ss.Net, 200, ss.Net.LayersByClass())
	ss.ConfigLogsFromArgs() // This must occur after logs are configged.
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

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode elog.EvalModes, time elog.Times) {
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows
	switch {
	case mode == elog.Test && time == elog.Epoch:
		ss.LogTestErrors()
	case time == elog.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == elog.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == elog.Cycle {
		ss.GUI.UpdateCyclePlot(elog.Test, ss.Time.Cycle)
	} else {
		ss.GUI.UpdatePlot(mode, time)
	}

	switch {
	case mode == elog.Train && time == elog.Run:
		ss.LogRunStats()
	}
}

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func (ss *Sim) LogTestErrors() {
	sk := elog.Scope(elog.Test, elog.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("TestErrors")
	ix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Err", row) > 0 // include error trials
	})
	ss.Logs.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.Agg(allsp, "UnitErr", agg.AggSum)
	// note: can add other stats to compute
	ss.Logs.MiscTables["TestErrorStats"] = allsp.AggsToTable(etable.AddAggName)
}

// LogRunStats records stats across all runs, at Train Run scope
func (ss *Sim) LogRunStats() {
	sk := elog.Scope(elog.Train, elog.Run)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("RunStats")

	spl := split.GroupBy(ix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.Logs.MiscTables["RunStats"] = spl.AggsToTable(etable.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func (ss *Sim) PCAStats() {
	ss.Stats.PCAStats(ss.Logs.IdxView(elog.Analyze, elog.Trial), "ActM", ss.Net.LayersByClass("Hidden"))
	ss.Logs.ResetLog(elog.Analyze, elog.Trial)
}

// RasterRec updates spike raster record for given cycle
func (ss *Sim) RasterRec(cyc int) {
	ss.Stats.RasterRec(ss.Net, cyc, "Spike")
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Targ values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations
// TODO Hippocampus specific
func (ss *Sim) MemStats(train bool) {
	ecout := ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon() // note: must be input b/c ECin can be active
	nn := ecout.Shape().Len()
	actThr := float32(0.2)
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	actMi, _ := ecout.UnitVarIdx("ActM")
	targi, _ := ecout.UnitVarIdx("Targ")
	actQ1i, _ := ecout.UnitVarIdx("ActSt1")
	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitVal1D(actMi, ni)
		trg := ecout.UnitVal1D(targi, ni) // full pattern target
		inact := inp.UnitVal1D(actQ1i, ni)
		if trg < actThr { // trgOff
			trgOffN += 1
			if actm > actThr {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < actThr { // missing in ECin -- completion target
				cmpN += 1
				if actm < actThr {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < actThr {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	trgOnWasOffAll /= trgOnN
	trgOffWasOn /= trgOffN
	if train { // no cmp
		if trgOnWasOffAll < ss.Stats.Float("MemThr") && trgOffWasOn < ss.Stats.Float("MemThr") {
			ss.Stats.SetFloat("Mem", 1)
		} else {
			ss.Stats.SetFloat("Mem", 0)
		}
	} else { // test
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < ss.Stats.Float("MemThr") && trgOffWasOn < ss.Stats.Float("MemThr") {
				ss.Stats.SetFloat("Mem", 1)
			} else {
				ss.Stats.SetFloat("Mem", 0)
			}
		}
	}
	ss.Stats.SetFloat("TrgOnWasOff", trgOnWasOffAll)
	ss.Stats.SetFloat("TrgOnWasOffCmp", trgOnWasOffCmp)
	ss.Stats.SetFloat("TrgOffWasOn", trgOffWasOn)
}
