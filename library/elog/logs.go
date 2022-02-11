// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elog

import (
	"fmt"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"log"
	"os"
	"strconv"

	"github.com/emer/etable/etable"
)

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// LogTable contains all the data for one log table
type LogTable struct {
	Table        *etable.Table   `desc:"Actual data stored."`
	IdxView      *etable.IdxView `desc:"Index View of the table -- automatically updated when a new row of data is logged to the table."`
	File         *os.File        `desc:"File to store the log into."`
	WroteHeaders bool            `desc:"true if headers for File have already been written"`
}

// Logs contains all logging state and API for doing logging.
// do AddItem to add any number of items, at different eval mode, time scopes.
// Each Item has its own Compute functions, at each scope as neeeded.
// Then call CreateTables to generate log tables from those items.
// Call Log with a mode and time to add a new row of data to the log
// and ResetLog to reset the log to empty.
type Logs struct {
	Items      []*Item        `desc:"A list of the items that should be logged. Each item should describe one column that you want to log, and how.  Order in list determines order in logs."`
	ItemIdxMap map[string]int `view:"-" desc:"map of item indexes by name, for rapid access"`

	Tables map[ScopeKey]*LogTable `desc:"Tables storing log data, auto-generated from Items."`
	Modes  map[string]bool        `view:"-" desc:"All the eval modes that appear in any of the items of this log."`
	Times  map[string]bool        `view:"-" desc:"All the timescales that appear in any of the items of this log."`

	TableOrder []ScopeKey `view:"-" desc:"sorted order of table scopes"`

	// TODO Move this to Logs
	ValsTsrs map[string]*etensor.Float32 `view:"-" desc:"Value Tensors. A buffer for holding layer values. This helps avoid reallocating memory every time"`

	SpikeRasters   map[string]*etensor.Float32   `desc:"spike raster data for different layers"`
	SpikeRastGrids map[string]*etview.TensorGrid `desc:"spike raster plots for different layers"`

	MiscTables map[string]*etable.Table `desc:"gets additional tables that are not typical"`
}

// MiscTable gets a miscellaneous table that is not specified or typically expected
func (lg *Logs) MiscTable(name string) *etable.Table {
	return lg.MiscTables[name]
}

// AddItem adds an item to the list
func (lg *Logs) AddItem(item *Item) {
	lg.Items = append(lg.Items, item)
	if lg.ItemIdxMap == nil {
		lg.ItemIdxMap = make(map[string]int)
	}
	// note: we're not really in a position to track errors in a big list of
	// AddItem statements, so don't both with error return
	if _, has := lg.ItemIdxMap[item.Name]; has {
		// This error is not currently a problem as this map is not used.
		// TODO: If we want to use this map, we need to clean up logging_content.go.
		// log.Printf("elog.AddItem Warning: item name repeated: %s\n", item.Name)
	}
	lg.ItemIdxMap[item.Name] = len(lg.Items) - 1
}

// Table returns the table for given mode, time
func (lg *Logs) Table(mode EvalModes, time Times) *etable.Table {
	sk := GenScopeKey(mode, time)
	return lg.Tables[sk].Table
}

// TODO we could add facility for named index views that are also cached
// just make the IdxView a named map

// IdxView returns the Index View of a log table for a given mode, time
// This is used for data aggregation, filtering etc.  This view
// should not be altered and always shows the whole table
// Create new ones as needed.
func (lg *Logs) IdxView(mode EvalModes, time Times) *etable.IdxView {
	sk := GenScopeKey(mode, time)
	ld := lg.Tables[sk]
	if ld.IdxView == nil {
		ld.IdxView = etable.NewIdxView(ld.Table)
	}
	return ld.IdxView
}

// TableDetails returns the LogTable record of associated info for given table
func (lg *Logs) TableDetails(mode EvalModes, time Times) *LogTable {
	return lg.Tables[GenScopeKey(mode, time)]
}

// CreateTables creates the log tables based on all the specified log items
// It first calls ProcessItems to instantiate specific scopes.
func (lg *Logs) CreateTables() error {
	lg.ProcessItems()
	uniqueTables := make(map[ScopeKey]*LogTable)
	tableOrder := make([]ScopeKey, 0) //initial size
	var err error
	for _, item := range lg.Items {
		for scope, _ := range item.Compute {
			_, has := uniqueTables[scope]
			modes, times := scope.ModesAndTimes()
			if len(modes) != 1 || len(times) != 1 {
				err = fmt.Errorf("Unexpected too long modes or times in: " + string(scope))
				log.Println(err) // actually print the err
			}
			if !has {
				uniqueTables[scope] = &LogTable{Table: &etable.Table{}}
				tableOrder = append(tableOrder, scope)
				lg.ConfigTable(uniqueTables[scope].Table, modes[0], times[0])
			}
		}
	}
	lg.Tables = uniqueTables
	lg.TableOrder = SortScopes(tableOrder)
	lg.MiscTables = make(map[string]*etable.Table)

	return err
}

// Log performs logging for given mode, time.
// Adds a new row and computes all the items.
// and saves data to file if open.
func (lg *Logs) Log(mode EvalModes, time Times) *etable.Table {
	sk := GenScopeKey(mode, time)
	ld := lg.Tables[sk]
	return lg.LogRow(mode, time, ld.Table.Rows)
}

// LogRow performs logging for given mode, time, at given row.
// Saves data to file if open.
func (lg *Logs) LogRow(mode EvalModes, time Times, row int) *etable.Table {
	sk := GenScopeKey(mode, time)
	ld := lg.Tables[sk]
	dt := ld.Table
	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}
	lg.ComputeScope(sk, dt, row)
	ld.IdxView = nil // dirty that so it is regenerated later when needed
	lg.WriteLastLogRow(ld)
	return dt
}

// ResetLog resets the log for given mode, time, at given row.
// by setting number of rows = 0
func (lg *Logs) ResetLog(mode EvalModes, time Times) {
	sk := GenScopeKey(mode, time)
	ld := lg.Tables[sk]
	dt := ld.Table
	dt.SetNumRows(0)
	ld.IdxView = nil // dirty that so it is regenerated later when needed
}

// SetLogFile sets the log filename for given scope
func (lg *Logs) SetLogFile(mode EvalModes, time Times, fnm string) {
	lt := lg.TableDetails(mode, time)
	var err error
	lt.File, err = os.Create("logs/" + fnm)
	if err != nil {
		log.Println(err)
		lt.File = nil
	} else {
		fmt.Printf("Saving log to: %s\n", fnm)
	}
}

// CloseLogFiles closes all open log files
func (lg *Logs) CloseLogFiles() {
	for _, ld := range lg.Tables {
		if ld.File != nil {
			ld.File.Close()
			ld.File = nil
		}
	}
}

///////////////////////////////////////////////////////////////////////////
//   Internal infrastructure below, main user API above

// ComputeScope calls all item compute functions within given scope
func (lg *Logs) ComputeScope(sk ScopeKey, dt *etable.Table, row int) {
	for _, item := range lg.Items {
		callback, ok := item.Compute[sk]
		if ok {
			callback(item, sk, dt, row)
		}
	}
}

// WriteLastLogRow writes the last row of table to file, if File != nil
func (lg *Logs) WriteLastLogRow(ld *LogTable) {
	if ld.File == nil {
		return
	}
	dt := ld.Table
	if !ld.WroteHeaders {
		dt.WriteCSVHeaders(ld.File, etable.Tab)
		ld.WroteHeaders = true
	}
	dt.WriteCSVRow(ld.File, dt.Rows-1, etable.Tab)
}

// ProcessItems is called in CreateTables, after all items have been added.
// It instantiates All scopes, and compiles multi-list scopes into
// single mode, item pairs
func (lg *Logs) ProcessItems() {
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
	}
	lg.CompileAllModesAndTimes()
	for _, item := range lg.Items {
		lg.ItemBindAllScopes(item)
		item.SetEachScopeKey()
		item.CompileModesAndTimes()
	}
}

// CompileAllModesAndTimes gathers all the modes and times used across all items
func (lg *Logs) CompileAllModesAndTimes() {
	lg.Modes = make(map[string]bool)
	lg.Times = make(map[string]bool)
	for _, item := range lg.Items {
		for sk, _ := range item.Compute {
			modes, times := sk.ModesAndTimes()
			for _, m := range modes {
				if m == "AllEvalModes" || m == "NoEvalMode" {
					continue
				}
				lg.Modes[m] = true
			}
			for _, t := range times {
				if t == "AllTimes" || t == "NoTime" {
					continue
				}
				lg.Times[t] = true
			}
		}
	}
}

// ItemBindAllScopes translates the AllEvalModes or AllTimes scopes into
// a concrete list of actual Modes and Times used across all items
func (lg *Logs) ItemBindAllScopes(item *Item) {
	newMap := ComputeMap{}
	for sk, c := range item.Compute {
		newsk := sk
		useAllModes := false
		useAllTimes := false
		modes, times := sk.ModesAndTimesMap()
		for m := range modes {
			if m == "AllEvalModes" {
				useAllModes = true
			}
		}
		for t := range times {
			if t == "AllTimes" {
				useAllTimes = true
			}
		}
		if useAllModes && useAllTimes {
			newsk = GenScopesKeyMap(lg.Modes, lg.Times)
		} else if useAllModes {
			newsk = GenScopesKeyMap(lg.Modes, times)
		} else if useAllTimes {
			newsk = GenScopesKeyMap(modes, lg.Times)
		}
		newMap[newsk] = c
	}
	item.Compute = newMap
}

// ConfigTable configures given table for given unique mode, time scope
func (lg *Logs) ConfigTable(dt *etable.Table, mode, time string) {
	dt.SetMetaData("name", mode+time+"Log")
	dt.SetMetaData("desc", "Record of performance over "+time+" of "+mode)
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	sch := etable.Schema{}
	for _, val := range lg.Items {
		// Compute is the definive record for which timescales are logged.
		// It also records how, but we don't need that here.
		if _, ok := val.ComputeFunc(mode, time); ok {
			sch = append(sch, etable.Column{val.Name, val.Type, val.CellShape, val.DimNames})
		}
	}
	dt.SetFromSchema(sch, 0)
}
