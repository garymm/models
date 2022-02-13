package sim

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/metric"
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
	ss.GUI.NetViewText = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "TrlErr", "TrlCosDiff"})
}

//TODO: should be placed in a library or package pertaining to calculating stats related to one to many

// ClosestStat finds the closest pattern in given column of given table to
// given layer activation pattern using given variable.  Returns the row number,
// correlation value, and value of a column named namecol for that row if non-empty.
// Column must be etensor.Float32
func (ss *Sim) ClosestStat(net emer.Network, lnm, varnm string, dt *etable.Table, colnm, namecol string) (int, float32, string) {
	vt := ss.Stats.F32Tensor(lnm)
	ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	ly.UnitValsTensor(vt, varnm)
	col := dt.ColByName(colnm)
	// note: requires Increasing metric so using Inv
	row, cor := metric.ClosestRow32(vt, col.(*etensor.Float32), metric.InvCorrelation32)
	cor = 1 - cor // convert back to correl
	nm := ""
	if namecol != "" {
		nm = dt.CellString(namecol, row)
	}
	return row, cor, nm
}
