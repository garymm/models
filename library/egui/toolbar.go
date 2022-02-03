package egui

type ToolbarItem struct {
	Label   string
	Icon    string
	Tooltip string
	Active  ToolGhosting
	Func    func()
}
