// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elog

import (
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// ComputeFunc function that computes value at the
type ComputeFunc func(item *Item, scope ScopeKey, dt *etable.Table, row int)

type DefaultBool int64

const (
	DUnknown DefaultBool = iota
	DTrue
	DFalse
)

func (db *DefaultBool) ToBool() bool {
	return *db == DTrue
}

// ComputeMap holds compute function for a defined scope key
type ComputeMap map[ScopeKey]ComputeFunc

// Item describes one item to be logged -- has all the info
// for this item, across all scopes where it is relevant.
type Item struct {
	Name      string       `desc:"name of column -- must be unique for a table"`
	Type      etensor.Type `desc:"data type, using etensor types which are isomorphic with arrow.Type"`
	CellShape []int        `desc:"shape of a single cell in the column (i.e., without the row dimension) -- for scalars this is nil -- tensor column will add the outer row dimension to this shape"`
	DimNames  []string     `desc:"names of the dimensions within the CellShape -- 'Row' will be added to outer dimension"`
	Compute   ComputeMap   `desc:"For each timescale and mode, how is this value computed? The key should be a single mode and timescale, from GenScopeKey(mode, time) -- use All* when used across all instances of a given scope -- map will be updated with specific cases during final processing of items."`
	Plot      DefaultBool  `desc:"Whether or not to plot it"`
	FixMin    DefaultBool  `desc:"Whether to fix the minimum in the display"`
	FixMax    DefaultBool  `desc:"Whether to fix the maximum in the display"`
	Range     minmax.F64   `desc:"The minimum and maximum"`

	// following are updated in final Process step
	Modes map[EvalModes]bool `desc:"map of eval modes that this item has a compute function for"`
	Times map[Times]bool     `desc:"map of times that this item has a compute function for"`
}

func (item *Item) ComputeFunc(mode EvalModes, time Times) (ComputeFunc, bool) {
	val, ok := item.Compute[GenScopeKey(mode, time)]
	return val, ok
}

// SetComputeFuncAll sets the compute function for all existing Modes and Times
// Can be used to replace a compute func after the fact.
func (item *Item) SetComputeFuncAll(theFunc ComputeFunc) {
	for mode := range item.Modes {
		for time := range item.Times {
			item.Compute[GenScopeKey(mode, time)] = theFunc
		}
	}
}

// SetComputeFuncOver sets the compute function over range of modes and times
func (item *Item) SetComputeFuncOver(modes []EvalModes, times []Times, theFunc ComputeFunc) {
	for _, mode := range modes {
		for _, time := range times {
			item.Compute[GenScopeKey(mode, time)] = theFunc
		}
	}
}

// SetComputeFunc sets compute function for one mode, time
func (item *Item) SetComputeFunc(mode EvalModes, time Times, theFunc ComputeFunc) {
	item.SetComputeFuncOver([]EvalModes{mode}, []Times{time}, theFunc)
}

// SetEachScopeKey updates the Compute map so that it only contains entries
// for a unique Mode,Time pair, where multiple modes and times may have
// originally been specified.
func (item *Item) SetEachScopeKey() {
	newCompute := ComputeMap{}
	doReplace := false
	for sk, c := range item.Compute {
		modes, times := sk.ModesAndTimes()
		if len(modes) > 1 || len(times) > 1 {
			doReplace = true
			for _, m := range modes {
				for _, t := range times {
					newCompute[GenScopeKey(m, t)] = c
				}
			}
		} else {
			newCompute[sk] = c
		}
	}
	if doReplace {
		item.Compute = newCompute
	}
}

// CompileModesAndTimes compiles maps of modes and times where this item appears.
// Based on the final updated Compute map
func (item *Item) CompileModesAndTimes() {
	item.Modes = make(map[EvalModes]bool)
	item.Times = make(map[Times]bool)
	for scope, _ := range item.Compute {
		modes, times := scope.ModesAndTimes()
		for _, mode := range modes {
			item.Modes[mode] = true
		}
		for _, time := range times {
			item.Times[time] = true
		}
	}
}

func (item *Item) HasMode(mode EvalModes) bool {
	_, has := item.Modes[mode]
	return has
}

func (item *Item) HasTime(time Times) bool {
	_, has := item.Times[time]
	return has
}
