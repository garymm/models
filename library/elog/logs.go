package elog

import (
	"github.com/emer/etable/etable"
)

type Logs struct {
	Items      []*Item
	ItemIdxMap map[string]int
	Tables     map[ScopeKey]*etable.Table

	// TODO Remove these
	TrnEpcLog *etable.Table `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog *etable.Table `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog *etable.Table `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog *etable.Table `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstCycLog *etable.Table `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog    *etable.Table `view:"no-inline" desc:"summary log of each run"`
}

// DELETE THIS DO NOT SUBMIT
// DO NOT SUBMIT cats
//ss.TrnEpcLog = &etable.Table{}
//ss.TstEpcLog = &etable.Table{}
//ss.TstTrlLog = &etable.Table{}
//ss.TstCycLog = &etable.Table{}
//ss.RunLog = &etable.Table{}

// AddItem adds an item to the list
func (lg *Logs) AddItem(item *Item) {
	lg.Items = append(lg.Items, item)
	if lg.ItemIdxMap == nil {
		lg.ItemIdxMap = make(map[string]int)
	}
	// TODO Name is not unique
	lg.ItemIdxMap[item.Name] = len(lg.Items) - 1
}

func (lg *Logs) AddItemScoped(item *Item, modes []TrainOrTest, times []Times) {
	item.ScopeKey.FromScopes(modes, times)
	lg.AddItem(item)
}

func (lg *Logs) CreateTables() {
	uniqueTables := make(map[ScopeKey]*etable.Table)
	for _, item := range lg.Items {
		for _, mode := range item.Modes {
			for _, time := range item.Times {
				tempScopeKey := ScopeKey("")
				tempScopeKey.FromScopes([]TrainOrTest{mode}, []Times{time})
				_, ok := uniqueTables[tempScopeKey]
				if ok == false {
					uniqueTables[tempScopeKey] = &etable.Table{}
				}
			}
		}
	}
	lg.Tables = uniqueTables
}

func (lg *Logs) GetTables(mode TrainOrTest, time Times) *etable.Table {
	tempScopeKey := ScopeKey("")
	tempScopeKey.FromScopes([]TrainOrTest{mode}, []Times{time})
	return lg.Tables[tempScopeKey]
}
