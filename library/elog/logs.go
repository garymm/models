package elog

import (
	"github.com/emer/etable/etable"
	"strconv"
)

// LogPrec is precision for saving float values in logs
const LogPrec = 4

type Logs struct {
	Items      []*Item
	ItemIdxMap map[string]int
	Tables     map[ScopeKey]*etable.Table
	TableOrder []ScopeKey
	TableFuncs ComputeMap
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

func (lg *Logs) AddItemScoped(item *Item, modes []EvalModes, times []Times) {
	item.ScopeKey.FromScopes(modes, times)
	lg.AddItem(item)
}

func (lg *Logs) configLogTable(dt *etable.Table, mode EvalModes, time Times) {
	dt.SetMetaData("name", mode.String()+time.String()+"Log")
	dt.SetMetaData("desc", "Record of performance over "+time.String()+" of "+mode.String())
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	sch := etable.Schema{}
	if len(lg.TableFuncs) == 0 {
		lg.TableFuncs = make(ComputeMap)
	}
	for _, val := range lg.Items {
		// Compute records which timescales are logged. It also records how, but we don't need that here.
		theFunction, ok := val.GetComputeFunc(mode, time)
		if ok {
			lg.TableFuncs[GenScopeKey(mode, time)] = theFunction
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (lg *Logs) CreateTables() {

	uniqueTables := make(map[ScopeKey]*etable.Table)
	tableOrder := make([]ScopeKey, 0) //initial size
	for _, item := range lg.Items {
		for _, mode := range item.Modes {
			for _, time := range item.Times {
				tempScopeKey := GenScopeKey(mode, time)
				_, ok := uniqueTables[tempScopeKey]
				if ok == false {
					uniqueTables[tempScopeKey] = &etable.Table{}
					tableOrder = append(tableOrder, tempScopeKey)
					lg.configLogTable(uniqueTables[tempScopeKey], mode, time)
				}
			}
		}
	}
	lg.Tables = uniqueTables
	lg.TableOrder = tableOrder
}

func (lg *Logs) GetTable(mode EvalModes, time Times) *etable.Table {
	tempScopeKey := ScopeKey("")
	tempScopeKey.FromScopes([]EvalModes{mode}, []Times{time})
	return lg.Tables[tempScopeKey]
}
