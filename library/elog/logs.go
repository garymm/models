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

	// TODO Remove these
	TrnEpcLog *etable.Table `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog *etable.Table `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog *etable.Table `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog *etable.Table `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstCycLog *etable.Table `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog    *etable.Table `view:"no-inline" desc:"summary log of each run"`
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

func (lg *Logs) configLogTable(dt *etable.Table, mode TrainOrTest, time Times) {
	dt.SetMetaData("name", mode.String()+time.String()+"Log")
	dt.SetMetaData("desc", "Record of performance over "+time.String()+" of "+mode.String())
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	sch := etable.Schema{}
	for _, val := range lg.Items {
		// Compute records which timescales are logged. It also records how, but we don't need that here.
		_, ok := val.GetComputeFunc(mode, time)
		if ok {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
	}
	dt.SetFromSchema(sch, 0)
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
					lg.configLogTable(uniqueTables[tempScopeKey], mode, time)
				}
			}
		}
	}
	lg.Tables = uniqueTables
}

func (lg *Logs) GetTable(mode TrainOrTest, time Times) *etable.Table {
	tempScopeKey := ScopeKey("")
	tempScopeKey.FromScopes([]TrainOrTest{mode}, []Times{time})
	return lg.Tables[tempScopeKey]
}
