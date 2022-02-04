package elog

import (
	"fmt"
	"github.com/emer/etable/etable"
	"os"
	"strconv"
)

// LogPrec is precision for saving float values in logs
const LogPrec = 4

type LogTable struct {
	Table *etable.Table `desc:"Actual data stored."`
	// TODO Use this to cache the IdxView if speed becomes an issue.
	TableView     etable.IdxView `desc:"View of the table."`
	File          *os.File       `desc:"File to store the log."`
	HeaderWritten bool           `desc:"If true, header has been written already."`
	// DO NOT SUBMIT Add callback functions here
}

type Logs struct {
	Items      []*Item `desc:"A list of the items that should be logged. Each item should describe one column that you want to log, and how."`
	ItemIdxMap map[string]int

	// TODO Replace this with a struct that stores etable.Table, File, IdxView, HeaderWrittenBool
	Tables     map[ScopeKey]LogTable `desc:"Tables of logs."`
	EvalModes  []EvalModes           `desc:"All the eval modes that appear in any of the items of this log."`
	Timescales []Times               `desc:"All the timescales that appear in any of the items of this log."`

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

func (lg *Logs) CompileAllModesAndTimes() {
	// It's not efficient to call this for every item, but it also doesn't matter.
	for _, item := range lg.Items {
		for sk, _ := range item.Compute {
			modes, times := sk.GetModesAndTimes()
			for _, m := range modes {
				foundMode := false
				for _, um := range lg.EvalModes {
					if m == um {
						foundMode = true
						break
					}
				}
				if !foundMode && m != UnknownEvalMode && m != AllEvalModes {
					lg.EvalModes = append(lg.EvalModes, m)
				}
			}
			for _, t := range times {
				foundTime := false
				for _, ut := range lg.Timescales {
					if t == ut {
						foundTime = true
						break
					}
				}
				if !foundTime && t != UnknownTimescale && t != AllTimes {
					lg.Timescales = append(lg.Timescales, t)
				}
			}
		}
	}
}

func (lg *Logs) UpdateItemForAll(item *Item) {
	// This could be refactored with a set object.
	newMap := ComputeMap{}
	for sk, c := range item.Compute {
		newsk := sk
		useAllModes := false
		useAllTimes := false
		modes, times := sk.GetModesAndTimes()
		for _, m := range modes {
			if m == AllEvalModes {
				useAllModes = true
			}
		}
		for _, t := range times {
			if t == AllTimes {
				useAllTimes = true
			}
		}
		if useAllModes && useAllTimes {
			newsk = GenScopesKey(lg.EvalModes, lg.Timescales)
		} else if useAllModes {
			newsk = GenScopesKey(lg.EvalModes, times)
		} else if useAllTimes {
			newsk = GenScopesKey(modes, lg.Timescales)
		}
		newMap[newsk] = c
	}
	item.Compute = newMap
}

func (lg *Logs) ProcessLogItems() {
	for _, item := range lg.Items {
		if item.Plot == DUnknown {
			item.Plot = DTrue
		}
		if item.FixMin == DUnknown {
			item.FixMin = DTrue
		}
		if item.FixMax == DUnknown {
			item.FixMax = DFalse
		}
		for scope, _ := range item.Compute {
			item.UpdateModesAndTimesFromScope(scope)
		}
	}
	lg.CompileAllModesAndTimes()
	for _, item := range lg.Items {
		// This needs to go after CompileAllModesAndTimes
		lg.UpdateItemForAll(item)
		// This needs to happen after UpdateItemForAll
		item.ExpandModesAndTimesIfNecessary()
	}
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
	uniqueTables := make(map[ScopeKey]LogTable)
	tableOrder := make([]ScopeKey, 0) //initial size
	for _, item := range lg.Items {
		for scope, _ := range item.Compute {
			_, ok := uniqueTables[scope]
			modes, times := scope.GetModesAndTimes()
			if len(modes) != 1 || len(times) != 1 {
				fmt.Errorf("Unexpected too long modes or times in " + string(scope))
			}
			if ok == false {
				uniqueTables[scope] = LogTable{Table: &etable.Table{}}
				tableOrder = append(tableOrder, scope)
				lg.configLogTable(uniqueTables[scope].Table, modes[0], times[0])
			}
		}
	}
	lg.Tables = uniqueTables
	lg.TableOrder = tableOrder
}

func (lg *Logs) GetTable(mode EvalModes, time Times) *etable.Table {
	tempScopeKey := ScopeKey("")
	tempScopeKey.FromScopes([]EvalModes{mode}, []Times{time})
	return lg.Tables[tempScopeKey].Table
}

func (lg *Logs) GetTableView(mode EvalModes, time Times) *etable.IdxView {
	tempScopeKey := ScopeKey("")
	tempScopeKey.FromScopes([]EvalModes{mode}, []Times{time})
	// TODO(optimize) Cache this in the TableView
	return etable.NewIdxView(lg.Tables[tempScopeKey].Table)
}

func (lg *Logs) GetTableDetails(mode EvalModes, time Times) LogTable {
	tempScopeKey := ScopeKey("")
	tempScopeKey.FromScopes([]EvalModes{mode}, []Times{time})
	return lg.Tables[tempScopeKey]
}
