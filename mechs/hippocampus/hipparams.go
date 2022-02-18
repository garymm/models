package main

import "github.com/emer/emergent/evec"

// HipParams have the hippocampus size and connectivity parameters
type HipParams struct {
	ECSize       evec.Vec2i `desc:"size of EC in terms of overall pools (outer dimension)"`
	ECPool       evec.Vec2i `desc:"size of one EC pool"`
	CA1Pool      evec.Vec2i `desc:"size of one CA1 pool"`
	CA3Size      evec.Vec2i `desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	DGSize       evec.Vec2i `inactive:"+" desc:"size of DG"`
	DGPCon       float32    `desc:"percent connectivity into DG"`
	CA3PCon      float32    `desc:"percent connectivity into CA3"`
	MossyPCon    float32    `desc:"percent connectivity into CA3 from DG"`
	ECPctAct     float32    `desc:"percent activation in EC pool"`
	MossyDel     float32    `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32    `desc:"delta in mossy strength for testing (relative to base param)"`
}

// TODO add to hipParams.go
func (hp *HipParams) Defaults() {
	// size
	hp.ECSize.Set(2, 3)
	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(15, 15) // using MedHip now
	hp.CA3Size.Set(30, 30) // using MedHip now
	hp.DGRatio = 2.236     // c.f. Ketz et al., 2013

	// ratio
	hp.DGPCon = 0.25 // .35 is sig worse, .2 learns faster but AB recall is worse
	hp.CA3PCon = 0.25
	hp.MossyPCon = 0.02 // .02 > .05 > .01 (for small net)
	hp.ECPctAct = 0.2

	hp.MossyDel = 3     // 4 > 2 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 0 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount..
}

func (hp *HipParams) Update() {
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)
}
