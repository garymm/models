package elog

import (
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// ComputeFunc function that computes value at the
type ComputeFunc func(item *Item, scope ScopeKey, dt *etable.Table, row int)

// Item describes the logging functionality
type Item struct {
	Name      string                   `desc:"name of column -- must be unique for a table"`
	Type      etensor.Type             `desc:"data type, using etensor types which are isomorphic with arrow.Type"`
	CellShape []int                    `desc:"shape of a single cell in the column (i.e., without the row dimension) -- for scalars this is nil -- tensor column will add the outer row dimension to this shape"`
	DimNames  []string                 `desc:"names of the dimensions within the CellShape -- 'Row' will be added to outer dimension"`
	Modes     []Modes                  `desc:"a variable list of modes that this item can exist in"`
	Times     []Times                  `desc:"a variable list of times that this item can exist in"`
	ScopeKey  ScopeKey                 `desc:"a string representation of the combined enum name"`
	Compute   map[ScopeKey]ComputeFunc `desc:"For each timescale and mode, how is this value computed?"`
	Plot      bool                     `desc:"Whether or not to plot it"`
	FixMin    bool                     `desc:"Whether to fix the minimum in the display"`
	FixMax    bool                     `desc:"Whether to fix the maximum in the display"`
	Range     minmax.F64               `desc:"The minimum and maximum"`
}

func (item *Item) GetScopeKey(mode Modes, time Times) ScopeKey {
	ss := ScopeKey("")
	ss.FromScope(mode, time)
	return ss
}

func (item *Item) GetComputeFunc(mode Modes, time Times) ComputeFunc {
	item.ScopeKey.FromScope(mode, time)
	return item.Compute[item.ScopeKey]
}

func (item *Item) AssignComputeFunc(theFunc ComputeFunc) {
	for _, mode := range item.Modes {
		for _, time := range item.Times {
			item.Compute[item.GetScopeKey(mode, time)] = theFunc
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ScopeKey the associated string representation of a scope or scopes
type ScopeKey string

// FromScopes create an associated scope merging the modes and times that are specified
func (sk *ScopeKey) FromScopes(modes []Modes, times []Times) {
	var mstr string
	var tstr string
	for _, mode := range modes {
		str := mode.String()
		if mstr == "" {
			mstr = str
		} else {
			mstr += "|" + str
		}
	}
	for _, time := range times {
		str := time.String()
		if tstr == "" {
			tstr = str
		} else {
			tstr += "|" + str
		}
	}
	*sk = ScopeKey(mstr + "&" + tstr)
}

// FromScope create an associated scope merging the modes and times that are specified
func (sk *ScopeKey) FromScope(mode Modes, time Times) {
	sk.FromScopes([]Modes{mode}, []Times{time})
}
