package egui

import (
	"github.com/Astera-org/models/library/elog"
	"github.com/emer/emergent/netview"
	"github.com/emer/etable/eplot"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

type GUI struct {
	IsRunning bool `view:"-" desc:"true if sim is running"`
	StopNow   bool `view:"-" desc:"flag to stop running"`

	Win      *gi.Window       `view:"-" desc:"main GUI gui.Window"`
	NetView  *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar  *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	ViewPort *gi.Viewport2D

	StructView *giv.StructView
	TabView    *gi.TabView

	PlotMap map[elog.ScopeKey]*eplot.Plot2D
}

func (gui *GUI) UpdateWindow() {
	gui.ViewPort.SetNeedsFullRender()
}

func (gui *GUI) MakeWindow(sim interface{}, appname, title, about string) {
	width := 1600
	height := 1200

	gi.SetAppName(appname)
	gi.SetAppAbout(about)

	gui.Win = gi.NewMainWindow(appname, title, width, height)

	gui.ViewPort = gui.Win.WinViewport2D()
	gui.ViewPort.UpdateStart()

	mfr := gui.Win.SetMainFrame()

	gui.ToolBar = gi.AddNewToolBar(mfr, "tbar")
	gui.ToolBar.SetStretchMaxWidth()

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	gui.StructView = giv.AddNewStructView(split, "sv")
	gui.StructView.SetStruct(sim)

	gui.TabView = gi.AddNewTabView(split, "tv")

	gui.NetView = gui.TabView.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	gui.NetView.Var = "Act"

	split.SetSplits(.2, .8)

}

//func (gui *GUI) UpdateView(ss *sim.Sim, train bool) {
//	if gui.NetView != nil && gui.NetView.IsVisible() {
//		gui.NetView.Record(ss.Counters(train))
//		// note: essential to use Go version of update when called from another goroutine
//		gui.NetView.GoUpdate() // note: using counters is significantly slower..
//	}
//}

func (gui *GUI) AddToolbarItem(item ToolbarItem) {
	switch item.Active {
	case ActiveStopped:
		gui.ToolBar.AddAction(gi.ActOpts{Label: item.Label, Icon: item.Icon, Tooltip: item.Tooltip, UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!gui.IsRunning)
		}}, gui.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			item.Func()
		})
	case ActiveRunning:
		gui.ToolBar.AddAction(gi.ActOpts{Label: item.Label, Icon: item.Icon, Tooltip: item.Tooltip, UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(gui.IsRunning)
		}}, gui.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
			item.Func()
		})
	case ActiveAlways:
		gui.ToolBar.AddAction(gi.ActOpts{Label: item.Label, Icon: item.Icon, Tooltip: item.Tooltip}, gui.Win.This(),
			func(recv, send ki.Ki, sig int64, data interface{}) {
				item.Func()
			})
	}
}

func (gui *GUI) AddPlots(title string, Log elog.Logs) {
	gui.PlotMap = make(map[elog.ScopeKey]*eplot.Plot2D)
	for key, table := range Log.Tables {

		plt := gui.TabView.AddNewTab(eplot.KiT_Plot2D, string(key)+"Plot").(*eplot.Plot2D)
		gui.PlotMap[key] = plt
		plt.SetTable(table)
		//This is so inefficient even if it's run once, this is ugly
		for _, item := range Log.Items {
			_, ok := item.Compute[key]
			if ok {
				plt.SetColParams(item.Name, item.Plot.ToBool(), item.FixMin.ToBool(), item.Range.Min, item.FixMax.ToBool(), item.Range.Max)
				modes, times := key.GetModesAndTimes()
				timeName := modes[0].String()
				plt.Params.Title = title + " " + timeName + " Plot"
				plt.Params.XAxisCol = timeName
				if times[0] == elog.Run { //The one exception
					plt.Params.LegendCol = "Params"
				}
			}
		}
	}

}

func (gui *GUI) FinalizeGUI(closePrompt bool) {
	vp := gui.Win.WinViewport2D()
	vp.UpdateEndNoSig(true)

	// main menu
	appnm := gi.AppName()
	mmen := gui.Win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := gui.Win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(gui.Win)

	emen := gui.Win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(gui.Win)

	if closePrompt {

		inQuitPrompt := false
		gi.SetQuitReqFunc(func() {
			if inQuitPrompt {
				return
			}
			inQuitPrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
				Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
				gui.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inQuitPrompt = false
					}
				})
		})

		inClosePrompt := false
		gui.Win.SetCloseReqFunc(func(w *gi.Window) {
			if inClosePrompt {
				return
			}
			inClosePrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close gui.Window?",
				Prompt: "Are you <i>sure</i> you want to close the gui.Window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
				gui.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inClosePrompt = false
					}
				})
		})
	}

	gui.Win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main gui.Window is closed, quit
	})

	gui.Win.MainMenuUpdated()
}
