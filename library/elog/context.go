// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elog

import (
	"github.com/Astera-org/models/library/estats"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// ComputeFunc function that computes and sets log values
// The Context provides information typically needed for logging
type ComputeFunc func(ctxt *Context)

// Context provides the context for logging compute functions.
// SetContext must be called on Logs to set the Stats and Net values
// Provides various convenience functions for setting log values
// and other commonly-used operations.
type Context struct {
	Logs  *Logs         `desc:"pointer to the Logs object with all log data"`
	Stats *estats.Stats `desc:"pointer to stats"`
	Net   emer.Network  `desc:"network"`
	Item  *Item         `desc:"the current log Item"`
	Scope ScopeKey      `desc:"the current eval mode and time scale"`
	Table *etable.Table `desc:"current table to record value to"`
	Row   int           `desc:"current row in table"`
}

// SetFloat64 sets a float64 to current table, item, row
func (ctx *Context) SetFloat64(val float64) {
	ctx.Table.SetCellFloat(ctx.Item.Name, ctx.Row, val)
}

// SetFloat32 sets a float32 to current table, item, row
func (ctx *Context) SetFloat32(val float32) {
	ctx.Table.SetCellFloat(ctx.Item.Name, ctx.Row, float64(val))
}

// SetInt sets an int to current table, item, row
func (ctx *Context) SetInt(val int) {
	ctx.Table.SetCellFloat(ctx.Item.Name, ctx.Row, float64(val))
}

// SetString sets a string to current table, item, row
func (ctx *Context) SetString(val string) {
	ctx.Table.SetCellString(ctx.Item.Name, ctx.Row, val)
}

// SetStatFloat sets a Stats Float of given name to current table, item, row
func (ctx *Context) SetStatFloat(name string) {
	ctx.Table.SetCellFloat(ctx.Item.Name, ctx.Row, ctx.Stats.Float(name))
}

// SetStatInt sets a Stats int of given name to current table, item, row
func (ctx *Context) SetStatInt(name string) {
	ctx.Table.SetCellFloat(ctx.Item.Name, ctx.Row, float64(ctx.Stats.Int(name)))
}

// SetStatString sets a Stats string of given name to current table, item, row
func (ctx *Context) SetStatString(name string) {
	ctx.Table.SetCellString(ctx.Item.Name, ctx.Row, ctx.Stats.String(name))
}

// SetTensor sets a Tensor to current table, item, row
func (ctx *Context) SetTensor(val etensor.Tensor) {
	ctx.Table.SetCellTensor(ctx.Item.Name, ctx.Row, val)
}

///////////////////////////////////////////////////
//  Aggregation

// SetAgg sets an aggregated scalar value computed from given eval mode
// and time scale with same Item name, to current item, row
func (ctx *Context) SetAgg(mode EvalModes, time Times, ag agg.Aggs) {
	ctx.SetAggScope(GenKey(mode, time), ag)
}

// SetAggScope sets an aggregated scalar value computed from
// another scope (SkopeKey) with same Item name, to current item, row
func (ctx *Context) SetAggScope(scope ScopeKey, ag agg.Aggs) {
	ix := ctx.Logs.IdxViewScope(scope)
	val := agg.Agg(ix, ctx.Item.Name, ag)[0]
	ctx.SetFloat64(val)
}

///////////////////////////////////////////////////
//  Network

// Layer returns layer by name as the emer.Layer interface --
// you may then need to convert to a concrete type depending.
func (ctx *Context) Layer(layNm string) emer.Layer {
	return ctx.Net.LayerByName(layNm)
}

// SetLayerTensor sets tensor of Unit values on a layer for given variable
func (ctx *Context) SetLayerTensor(layNm, unitVar string) {
	ly := ctx.Layer(layNm)
	tsr := ctx.Stats.F32Tensor(layNm)
	ly.UnitValsTensor(tsr, unitVar)
	ctx.SetTensor(tsr)
}
