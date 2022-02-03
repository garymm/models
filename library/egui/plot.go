package egui

import (
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/etable/eplot"
)

type Plots struct {
	PlotMap map[elog.ScopeKey]*eplot.Plot2D
}
type PlotItem struct {
	Title     string
	XAxisCol  string
	LegendCol string
	evalMode  elog.EvalModes
	timeScale elog.Times
}
