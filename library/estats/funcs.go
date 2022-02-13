// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package estats

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/metric"
)

// funcs contains misc stats functions

// SetLayerTensor sets tensor of Unit values on a layer for given variable
// to a F32Tensor with name = layNm
func (st *Stats) SetLayerTensor(net emer.Network, layNm, unitVar string) *etensor.Float32 {
	ly := net.LayerByName(layNm)
	tsr := st.F32Tensor(layNm)
	ly.UnitValsTensor(tsr, unitVar)
	return tsr
}

// ClosestStat finds the closest pattern in given column of given pats table to
// given layer activation pattern using given variable.  Returns the row number,
// correlation value, and value of a column named namecol for that row if non-empty.
// Column must be etensor.Float32
func (st *Stats) ClosestPat(net emer.Network, layNm, unitVar string, pats *etable.Table, colnm, namecol string) (int, float32, string) {
	tsr := st.SetLayerTensor(net, layNm, unitVar)
	col := pats.ColByName(colnm)
	// note: requires Increasing metric so using Inv
	row, cor := metric.ClosestRow32(tsr, col.(*etensor.Float32), metric.InvCorrelation32)
	cor = 1 - cor // convert back to correl
	nm := ""
	if namecol != "" {
		nm = pats.CellString(namecol, row)
	}
	return row, cor, nm
}
