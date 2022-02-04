package sim

import (
	"fmt"
	"github.com/Astera-org/models/library/egui"
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func GuiRun(TheSim *Sim, appname, title, about string) {
	TheSim.Init()
	win := TheSim.ConfigGui(appname, title, about)
	win.StartEventLoop()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui(appname, title, about string) *gi.Window {
	ss.GUI.MakeWindow(ss, appname, title, about)
	ss.GUI.CycleUpdateRate = 10
	ss.GUI.NetView.SetNet(ss.Net) // TODO ask Randy what this is doing

	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	ss.GUI.AddPlots(title, ss.Logs)

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
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Train",
		Icon:    "run",
		Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.Train()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Stop",
		Icon:    "stop",
		Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.",
		Active:  egui.ActiveRunning,
		Func: func() {
			ss.Stop()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Trial",
		Icon:    "step-fwd",
		Tooltip: "Advances one training trial at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.TrainTrial()
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Epoch",
		Icon:    "fast-fwd",
		Tooltip: "Advances one epoch (complete set of training patterns) at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.TrainEpoch()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Run",
		Icon:    "fast-fwd",
		Tooltip: "Advances one full training Run at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.TrainRun()
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("test")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Trial",
		Icon:    "fast-fwd",
		Tooltip: "Runs the next testing trial.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.TestTrial(false) // don't return on change -- wrap
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Item",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {

			gi.StringPromptDialog(ss.GUI.ViewPort, "", "Test Item",
				gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
				ss.GUI.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					dlg := send.(*gi.Dialog)
					if sig == int64(gi.DialogAccepted) {
						val := gi.StringPromptDialogValue(dlg)
						idxs := []int{0} //TODO: //ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
						if len(idxs) == 0 {
							gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
						} else {
							if !ss.GUI.IsRunning {
								ss.GUI.IsRunning = true
								fmt.Printf("testing index: %d\n", idxs[0])
								ss.TestItem(idxs[0])
								ss.GUI.IsRunning = false
								ss.GUI.ViewPort.SetNeedsFullRender()
							}
						}
					}
				})

		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.GetTable(elog.Train, elog.Run).SetNumRows(0)
			runPlot := ss.GUI.PlotMap[elog.GenScopeKey(elog.Train, elog.Run)]
			runPlot.Update()
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "New Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.NewRndSeed()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/one2many/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

// UpdateView updates the gui visualization of the network
func (ss *Sim) UpdateView(train bool) {
	if ss.GUI.NetView != nil && ss.GUI.NetView.IsVisible() {
		ss.GUI.NetView.Record(ss.Counters(train))

		// note: essential to use Go version of update when called from another goroutine
		ss.GUI.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

// UpdateViewTime based on differetn time scales change the values accoridngly
func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.UpdateView(train)
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.UpdateView(train)
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.UpdateView(train)
		}
	case axon.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.UpdateView(train)
		}
	}
}
