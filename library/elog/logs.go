package elog

import (
	"github.com/emer/etable/etable"
)

type Logs struct {
	Items      []*Item
	ItemIdxMap map[string]int
	Tables     map[ScopeKey]*etable.Table
}

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
