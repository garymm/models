package sim

// TODO Move this to an egui package

import (
	"github.com/Astera-org/models/library/egui"
	"github.com/emer/axon/axon"
	"github.com/goki/gi/gi"
	"github.com/goki/mat32"
)

func GuiRun(TheSim *Sim) {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {

	title := "Axon Random Associator"
	ss.GUI.MakeWindow(ss, "one2many", title, `This demonstrates a basic Axon model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	ss.GUI.AddPlots(title,ss.Logs)

	stb := ss.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.SpikeRecLays {
		sr := ss.SpikeRastTsr(lnm)
		tg := ss.SpikeRastGrid(lnm)
		tg.SetName(lnm + "Spikes")
		gi.AddNewLabel(stb, lnm, lnm+":")
		stb.AddChild(tg)
		gi.AddNewSpace(stb, lnm+"_spc")
		ss.ConfigSpikeGrid(tg, sr)
	}
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.ViewPort.SetNeedsFullRender()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Train",
		Icon: "run",
		Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval."
		Active: egui.ActiveStopped,
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Stop",
		Icon: "stop",
		Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.",
		Active: egui.ActiveRunning,
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Trial",
		Icon: "step-fwd",
		Tooltip: "Advances one training trial at a time.",
		Active: egui.ActiveStopped,
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "Step Epoch",
		Icon: "fast-fwd",
		Tooltip: "Advances one epoch (complete set of training patterns) at a time.",
		Active: egui.ActiveStopped,
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "Step Run",
		Icon: "fast-fwd",
		Tooltip: "Advances one full training Run at a time.",
		Active: egui.ActiveStopped,
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("test")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "Test Trial",
		Icon: "fast-fwd",
		Tooltip: "Runs the next testing trial.",
		Active: egui.ActiveStopped,
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "Test Item",
		Icon: "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active: egui.ActiveStopped,
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "Test All",
		Icon: "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active: egui.ActiveStopped,
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "Reset RunLog",
		Icon: "reset",
		Tooltip:  "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active: egui.ActiveAlways,
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "New Seed",
		Icon:  "new",
		Tooltip:  "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active: egui.ActiveAlways,
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label:  "README",
		Icon:  "file-markdown",
		Tooltip:  "Opens your browser on the README file that contains instructions for how to run this model.",
		Active: egui.ActiveAlways,
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}


func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.GUI.UpdateView(ss,train)
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.GUI.UpdateView(ss,train)
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.GUI.UpdateView(ss,train)
		}
	case axon.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.GUI.UpdateView(ss,train)
		}
	}
}
