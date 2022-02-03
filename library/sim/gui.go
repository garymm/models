package sim

// TODO Move this to an egui package

import (
	"fmt"
	"github.com/Astera-org/models/library/egui"
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func GuiRun(TheSim *Sim) {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {

	ss.GUI.MakeWindow(ss, "one2many", "Axon Random Associator", `This demonstrates a basic Axon model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	ss.GUI.AddPlots(ss.Logs)

	// TODO(andrew) Replace these with a loop
	//plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	//ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.Logs.GetTable(elog.Train, elog.Epoch))
	//
	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	//ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.Logs.GetTable(elog.Test, elog.Trial))
	//
	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	//ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.Logs.GetTable(elog.Test, elog.Cycle))
	//
	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	//ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.Logs.GetTable(elog.Test, elog.Epoch))
	//
	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	//ss.RunPlot = ss.ConfigRunPlot(plt, ss.Logs.GetTable(elog.Train, elog.Run))

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
		Active:  egui.ActiveRunning,
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


	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on change -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := []int{0} //TODO: //ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %d\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Logs.GetTable(elog.Train, elog.Run).SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/one2many/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	/* uncomment below to prompt before closing
	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})
	*/

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// TODO Replace all these functions
func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	for _, item := range ss.Logs.Items {
		_, ok := item.GetComputeFunc(elog.Train, elog.Epoch)
		if ok {
			// order of params: on, fixMin, min, fixMax, max
			plt.SetColParams(item.Name, item.Plot.ToBool(), item.FixMin.ToBool(), item.Range.Min, item.FixMax.ToBool(), item.Range.Max)
		}
	}
	return plt
}

// SimProps  register Save methods so they can be used
/**
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}
**/

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	for _, item := range ss.Logs.Items {
		_, ok := item.GetComputeFunc(elog.Test, elog.Trial)
		if ok {
			// order of params: on, fixMin, min, fixMax, max
			plt.SetColParams(item.Name, item.Plot.ToBool(), item.FixMin.ToBool(), item.Range.Min, item.FixMax.ToBool(), item.Range.Max)
		}
	}
	return plt
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	for _, item := range ss.Logs.Items {
		_, ok := item.GetComputeFunc(elog.Test, elog.Epoch)
		if ok {
			// order of params: on, fixMin, min, fixMax, max
			plt.SetColParams(item.Name, item.Plot.ToBool(), item.FixMin.ToBool(), item.Range.Min, item.FixMax.ToBool(), item.Range.Max)
		}
	}
	return plt
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	for _, item := range ss.Logs.Items {
		_, ok := item.GetComputeFunc(elog.Test, elog.Cycle)
		if ok {
			// order of params: on, fixMin, min, fixMax, max
			plt.SetColParams(item.Name, item.Plot.ToBool(), item.FixMin.ToBool(), item.Range.Min, item.FixMax.ToBool(), item.Range.Max)
		}
	}
	return plt
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.Params.LegendCol = "Params"
	plt.SetTable(dt)
	for _, item := range ss.Logs.Items {
		_, ok := item.GetComputeFunc(elog.Train, elog.Run)
		if ok {
			// order of params: on, fixMin, min, fixMax, max
			plt.SetColParams(item.Name, item.Plot.ToBool(), item.FixMin.ToBool(), item.Range.Min, item.FixMax.ToBool(), item.Range.Max)
		}
	}
	return plt
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

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
