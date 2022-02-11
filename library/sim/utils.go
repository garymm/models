package sim

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/metric"
)

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion.
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\t\t\t", ss.Run.Cur, ss.TrainEnv.Epoch().Cur, ss.TrainEnv.Trial().Cur, ss.Time.Cycle)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\t\t\t", ss.Run.Cur, ss.TestEnv.Epoch().Cur, ss.TestEnv.Trial().Cur, ss.Time.Cycle)
	}
}

//TODO: should be placed in a library or package pertaining to calculating stats related to one to many

// ClosestStat finds the closest pattern in given column of given table to
// given layer activation pattern using given variable.  Returns the row number,
// correlation value, and value of a column named namecol for that row if non-empty.
// Column must be etensor.Float32
func (ss *Sim) ClosestStat(net emer.Network, lnm, varnm string, dt *etable.Table, colnm, namecol string) (int, float32, string) {
	vt := ss.ValsTsr(lnm)
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
