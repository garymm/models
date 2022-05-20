package main

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"log"
)

func DefineNetworkCharacteristics(ss *Sim) { //todo this does not to be stored, will moved to network structure, and not make global, need to validate it will work first though

	ss.Prjn4x4Skp2 = prjn.NewPoolTile()
	ss.Prjn4x4Skp2.Size.Set(4, 4)
	ss.Prjn4x4Skp2.Skip.Set(2, 2)
	ss.Prjn4x4Skp2.Start.Set(-1, -1)
	ss.Prjn4x4Skp2.TopoRange.Min = 0.5

	ss.Prjn4x4Skp2Recip = prjn.NewPoolTileRecip(ss.Prjn4x4Skp2)

	ss.Prjn4x3Skp2 = prjn.NewPoolTile()
	ss.Prjn4x3Skp2.Size.Set(3, 4)
	ss.Prjn4x3Skp2.Skip.Set(0, 2)
	ss.Prjn4x3Skp2.Start.Set(0, -1)
	ss.Prjn4x3Skp2.TopoRange.Min = 0.5

	ss.Prjn4x3Skp2Recip = prjn.NewPoolTileRecip(ss.Prjn4x3Skp2)
}

func DefineNetworkStructure(ss *Sim, net *deep.Network) {

	net.InitName(net, "Emery")

	full := prjn.NewFull()
	sameu := prjn.NewPoolSameUnit()
	sameu.SelfCon = false
	p1to1 := prjn.NewPoolOneToOne()

	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1
	_ = rndcut

	var parprjn prjn.Pattern
	parprjn = full

	fsz := 1 + 2*ss.FoveaSize
	// popsize = 12

	// input / output layers:
	v2wd, v2wdp := net.AddInputTRC4D("V2Wd", ss.DepthPools, ss.NFOVRays, ss.DepthSize/ss.DepthPools, 1)
	v2fd, v2fdp := net.AddInputTRC4D("V2Fd", ss.DepthPools, fsz, ss.DepthSize/ss.DepthPools, 1) // FovDepth
	v2wd.SetClass("Depth")
	v2wdp.SetClass("Depth")
	v2fd.SetClass("Depth")
	v2fdp.SetClass("Depth")

	v1f, v1fp := net.AddInputTRC4D("V1F", 1, fsz, ss.PatSize.Y, ss.PatSize.X) // Fovea
	v1f.SetClass("Fovea")
	v1fp.SetClass("Fovea")

	s1s, s1sp := net.AddInputTRC4D("S1S", 1, 4, 2, 1) // ProxSoma
	s1s.SetClass("S1S")
	s1sp.SetClass("S1S")

	s1v, s1vp := net.AddInputTRC4D("S1V", 1, 2, ss.PopSize, 1) // Vestibular
	s1v.SetClass("S1V")
	s1vp.SetClass("S1V")

	ins := net.AddLayer4D("Ins", 1, len(ss.Inters), ss.PopSize, 1, emer.Input) // Inters = Insula
	ins.SetClass("Ins")

	m1 := net.AddLayer2D("M1", 10, 10, emer.Hidden)
	vl := net.AddLayer2D("VL", ss.PatSize.Y, ss.PatSize.X, emer.Target)  // Action
	act := net.AddLayer2D("Act", ss.PatSize.Y, ss.PatSize.X, emer.Input) // Action

	m1p := net.AddTRCLayer2D("M1P", 10, 10)
	m1p.Driver = "M1"

	mstd, mstdct, mstdp := net.AddSuperCTTRC4D("MSTd", ss.DepthPools/2, ss.NFOVRays/2, 8, 8)
	mstdct.RecvPrjns().SendName(mstd.Name()).SetPattern(p1to1)                                // todo: try ss.Prjn3x3Skp1 orig: p1to1
	net.ConnectLayers(mstdct, v2wdp, ss.Prjn4x4Skp2Recip, emer.Forward).SetClass("CTToPulv2") // 3 is too high
	net.ConnectLayers(v2wdp, mstd, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v2wdp, mstdct, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")

	cipl, ciplct, ciplp := deep.AddSuperCTTRC4D(net.AsAxon(), "cIPL", 3, 3, 8, 8)
	ciplct.RecvPrjns().SendName(cipl.Name()).SetPattern(full)

	pcc, pccct := deep.AddSuperCT4D(net.AsAxon(), "PCC", 2, 2, 7, 7)
	pccct.RecvPrjns().SendName(pcc.Name()).SetPattern(parprjn)

	sma, smact := net.AddSuperCT2D("SMA", 10, 10)
	smact.RecvPrjns().SendName(sma.Name()).SetPattern(full)

	net.ConnectLayers(smact, m1p, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(m1p, sma, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(m1p, smact, full, emer.Back).SetClass("FmPulv")

	it, itct := net.AddSuperCT2D("IT", 10, 10)
	itct.RecvPrjns().SendName(it.Name()).SetPattern(parprjn)
	net.ConnectLayers(itct, v1fp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(v1fp, itct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v1fp, it, full, emer.Back).SetClass("FmPulv")

	lip, lipct := net.AddSuperCT4D("LIP", ss.DepthPools/2, 1, 8, 8)
	lipct.RecvPrjns().SendName(lip.Name()).SetPattern(full)
	net.ConnectLayers(lipct, v2fdp, ss.Prjn4x3Skp2Recip, emer.Forward).SetClass("CTToPulv3")
	net.ConnectLayers(v2fdp, lipct, ss.Prjn4x3Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(v2fdp, lip, ss.Prjn4x3Skp2, emer.Back).SetClass("FmPulv")

	// todo: LIP fovea is not topo organized for left, middle right positions -- groups are depth organized
	// not enough resolution to really map that out here.
	net.ConnectLayers(lipct, v1fp, full, emer.Back).SetClass("ToPulv1") // attention

	m1.SetClass("M1")
	vl.SetClass("M1")
	act.SetClass("M1")
	m1p.SetClass("M1")

	mstd.SetClass("MSTd")
	mstdct.SetClass("MSTd")
	mstdp.SetClass("MSTd")

	cipl.SetClass("cIPL")
	ciplct.SetClass("cIPL")
	ciplp.SetClass("cIPL")

	pcc.SetClass("PCC")
	pccct.SetClass("PCC")

	sma.SetClass("SMA")
	smact.SetClass("SMA")

	it.SetClass("IT")
	itct.SetClass("IT")

	lip.SetClass("LIP")
	lipct.SetClass("LIP")

	////////////////////
	// basic super cons

	net.ConnectLayers(v2wd, mstd, ss.Prjn4x4Skp2, emer.Forward).SetClass("SuperFwd")

	// MStd <-> CIPl
	net.ConnectLayers(mstd, cipl, ss.Prjn4x4Skp2, emer.Forward).SetClass("SuperFwd")
	net.ConnectLayers(cipl, mstd, ss.Prjn4x4Skp2Recip, emer.Back)
	net.ConnectLayers(ciplct, mstdct, ss.Prjn4x4Skp2Recip, emer.Back).SetClass("CTBack")

	net.ConnectLayers(mstdp, ciplct, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(mstdp, cipl, ss.Prjn4x4Skp2, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(ciplct, mstdp, ss.Prjn4x4Skp2Recip, emer.Forward).SetClass("CTToPulv3")

	net.ConnectLayers(smact, mstdct, full, emer.Back).SetClass("CTBack")
	net.ConnectLayers(sma, mstd, full, emer.Back)

	// CIPl <-> PCC
	net.ConnectLayers(cipl, pcc, parprjn, emer.Forward).SetClass("SuperFwd")
	net.ConnectLayers(pcc, cipl, parprjn, emer.Back)
	net.ConnectLayers(pccct, ciplct, parprjn, emer.Back).SetClass("CTBack")

	net.ConnectLayers(ciplp, pccct, parprjn, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(ciplp, pcc, parprjn, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(pccct, ciplp, parprjn, emer.Forward).SetClass("CTToPulv2")

	// PCC <-> SMA
	net.ConnectLayers(pcc, sma, parprjn, emer.Forward).SetClass("SuperFwd")
	net.ConnectLayers(sma, pcc, parprjn, emer.Back)
	net.ConnectLayers(smact, pccct, parprjn, emer.Back).SetClass("CTBack")

	// SMA <-> M1
	net.ConnectLayers(sma, m1, parprjn, emer.Forward).SetClass("SuperFwd")

	net.BidirConnectLayers(m1, vl, full)

	net.ConnectLayers(v1f, it, full, emer.Forward)
	net.ConnectLayers(v2fd, lip, ss.Prjn4x3Skp2, emer.Forward)

	net.ConnectCtxtToCT(m1p, mstdct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, ciplct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, smact, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, pccct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, itct, full).SetClass("FmPulv")
	net.ConnectCtxtToCT(m1p, lipct, full).SetClass("FmPulv")

	////////////////////
	// to cIPL

	// net.ConnectLayers(sma, cipl, parprjn, emer.Back)
	net.ConnectLayers(s1s, cipl, full, emer.Back)
	net.ConnectLayers(s1v, cipl, full, emer.Back)
	// net.ConnectLayers(vl, cipl, full, emer.Back) // todo: m1?

	net.ConnectLayers(pccct, ciplct, parprjn, emer.Back).SetClass("CTBack")
	net.ConnectLayers(smact, ciplct, parprjn, emer.Back).SetClass("CTBack")

	// S1 vestibular
	net.ConnectLayers(ciplct, s1vp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1vp, ciplct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1vp, cipl, full, emer.Back).SetClass("FmPulv")

	////////////////////
	// to PCC

	net.ConnectLayers(s1s, pcc, full, emer.Forward)
	net.ConnectLayers(s1v, pcc, full, emer.Forward)

	net.ConnectLayers(smact, pccct, parprjn, emer.Back).SetClass("CTBack")

	// S1 vestibular
	net.ConnectLayers(pccct, s1vp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1vp, pccct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1vp, pcc, full, emer.Back).SetClass("FmPulv")

	// S1 soma
	net.ConnectLayers(pccct, s1sp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1sp, pccct, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1sp, pcc, full, emer.Back).SetClass("FmPulv")

	////////////////////
	// to SMA

	net.ConnectLayers(it, sma, full, emer.Forward)
	net.ConnectLayers(lip, sma, full, emer.Forward)
	net.ConnectLayers(s1s, sma, full, emer.Forward)

	// S1 vestibular
	net.ConnectLayers(smact, s1vp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1vp, smact, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1vp, sma, full, emer.Back).SetClass("FmPulv")

	// S1 soma
	net.ConnectLayers(smact, s1sp, full, emer.Forward).SetClass("CTToPulv")
	net.ConnectLayers(s1sp, smact, full, emer.Back).SetClass("FmPulv")
	net.ConnectLayers(s1sp, sma, full, emer.Back).SetClass("FmPulv")

	////////////////////
	// to M1

	net.ConnectLayers(smact, vl, full, emer.Forward)
	// net.ConnectLayers(sma, vl, full, emer.Forward) // no, right?

	////////////////////
	// to IT

	net.ConnectLayers(sma, it, full, emer.Back)

	net.ConnectLayers(smact, itct, parprjn, emer.Back).SetClass("CTBack") // needs to know how moving.

	////////////////////
	// to LIP

	net.ConnectLayers(sma, lip, full, emer.Back)
	// net.ConnectLayers(pcc, lip, full, emer.Back) // not useful

	net.ConnectLayers(smact, lipct, full, emer.Back).SetClass("CTBack") // always need sma to predict action outcome
	// net.ConnectLayers(pccct, lipct, full, emer.Back).SetClass("CTBack")

	////////////////////
	// lateral inhibition

	net.LateralConnectLayerPrjn(mstd, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(mstdct, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(cipl, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(ciplct, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(pcc, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(pccct, p1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(sma, full, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(smact, full, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(m1, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	////////////////////
	// Shortcuts

	net.ConnectLayers(v2wd, cipl, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, ciplct, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, pcc, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, pccct, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, sma, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v2wd, smact, rndcut, emer.Forward).SetClass("V1SC")

	//////////////////////////////////////
	// collect // TODO(refactor): this is used for like logging and stuff?

	ss.PulvLays = make([]string, 0, 10)
	ss.HidLays = make([]string, 0, 10)
	ss.SuperLays = make([]string, 0, 10)
	ss.InputLays = make([]string, 0, 10)
	for _, ly := range net.Layers {
		if ly.IsOff() {
			continue
		}
		switch ly.Type() {
		case emer.Input:
			ss.InputLays = append(ss.InputLays, ly.Name())
		case deep.TRC:
			ss.PulvLays = append(ss.PulvLays, ly.Name())
		case emer.Hidden:
			ss.SuperLays = append(ss.SuperLays, ly.Name())
			fallthrough
		case deep.CT:
			ss.HidLays = append(ss.HidLays, ly.Name())
		}
	}
	ss.PulvLays = append(ss.PulvLays, "VL")

	net.Defaults()
	SetParams("Network", ss.LogSetParams, net, &ss.Params, ss.ParamSet, ss) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()

}
