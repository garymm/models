package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

type HipTableTypes string
type TableMaps map[HipTableTypes]*etable.Table

const (
	TrainAB      HipTableTypes = "TrainAB"
	TrainAC      HipTableTypes = "TrainAC"
	TestAB       HipTableTypes = "TestAB"
	TestAC       HipTableTypes = "TestAC"
	PretrainLure HipTableTypes = "PreTrainLure"
	TestLure     HipTableTypes = "TestLure"
	TrainAll     HipTableTypes = "TrainAll" //needs special logic
)

// PatParams have the pattern parameters
type PatParams struct {
	ListSize    int     `desc:"number of A-B, A-C patterns each"`
	MinDiffPct  float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`
	DriftCtxt   bool    `desc:"use drifting context representations -- otherwise does bit flips from prototype"`
	CtxtFlipPct float32 `desc:"proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting"`
	DriftPct    float32 `desc:"percentage of active bits that drift, per step, for drifting context"`
}

type EnvHip struct {
	sim.Environment
	env.FixedTable
	EvalTables TableMaps `desc:"a map of tables used for handling stuff"`
	Pat        PatParams
	IsTest     bool `desc:"Whether this is the Test environment or the Train"`
}

func (pp *PatParams) Defaults() {
	pp.ListSize = 10 // 20 def
	pp.MinDiffPct = 0.5
	pp.CtxtFlipPct = .25
}

func (envhip *EnvHip) InitTables(tableNames ...HipTableTypes) {
	envhip.EvalTables = make(TableMaps)
	for _, nm := range tableNames {
		envhip.EvalTables[nm] = &etable.Table{}
	}
}

func (envhip *EnvHip) AssignTable(name string) {
	envhip.Table = etable.NewIdxView(envhip.EvalTables[HipTableTypes(name)])
}

func (envhip *EnvHip) SetName(name string) {
	envhip.FixedTable.Nm = name
}
func (envhip *EnvHip) Name() string {
	return envhip.FixedTable.Nm
}
func (envhip *EnvHip) SetDesc(desc string) {
	envhip.FixedTable.Dsc = desc
}
func (envhip *EnvHip) Desc() string {
	return envhip.FixedTable.Dsc
}

func (envhip *EnvHip) Order() []int {
	return envhip.FixedTable.Order
}
func (envhip *EnvHip) Sequential() bool {
	return envhip.FixedTable.Sequential
}
func (envhip *EnvHip) SetSequential(s bool) {
	envhip.FixedTable.Sequential = s
}

func (envhip *EnvHip) Run() *env.Ctr {
	return &envhip.FixedTable.Run
}
func (envhip *EnvHip) Epoch() *env.Ctr {
	return &envhip.FixedTable.Epoch
}
func (envhip *EnvHip) Trial() *env.Ctr {
	return &envhip.FixedTable.Trial
}
func (envhip *EnvHip) TrialName() *env.CurPrvString {
	return &envhip.FixedTable.TrialName
}
func (envhip *EnvHip) GroupName() *env.CurPrvString {
	return &envhip.FixedTable.GroupName
}
func (envhip *EnvHip) NameCol() string {
	return envhip.FixedTable.NameCol
}
func (envhip *EnvHip) SetNameCol(s string) {
	envhip.FixedTable.NameCol = s
}
func (envhip *EnvHip) GroupCol() string {
	return envhip.FixedTable.GroupCol
}
func (envhip *EnvHip) SetGroupCol(s string) {
	envhip.FixedTable.GroupCol = s
}

func (envhip *EnvHip) Validate() error {

	return envhip.FixedTable.Validate()
}
func (envhip *EnvHip) Init(run int) {
	if !envhip.IsTest {
		envhip.AssignTable("TrainAB")
	}
	envhip.FixedTable.Init(run)
	envhip.Trial().Cur = 0
	envhip.Pat.Defaults()
}

func (envhip *EnvHip) CurTrialName() string {
	return envhip.TrialName().Cur
}
func (envhip *EnvHip) Step() {
	envhip.FixedTable.Step()
}

func (envhip *EnvHip) State(s string) etensor.Tensor {
	return envhip.FixedTable.State(s)
}

func (envhip *EnvHip) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	return envhip.FixedTable.Counter(scale)
}

func (envhip *EnvHip) InputAndOutputLayers() []string {
	return []string{"Input", "ECout"}
}

func (envhip *EnvHip) AddTaskSwitching(ss *sim.Sim) *sim.TrainingCallbacks {

	taskSwitching := sim.TrainingCallbacks{}
	//This occurs after testall on a different onepochend
	taskSwitching.OnEpochEnd = func() {
		if ss.Trainer.EvalMode == elog.Train {
			//For clarity, this is calculating Nzero and First zero in regards to testing AC and testing AB,
			//the reason is that at end of training epoch we at least one epoch of testing
			updateNZeroAndFirstZero(ss)
		}

		numberZero := ss.Stats.Int("NZero")
		nzeroStop := ss.Stats.Int("NZeroStop")
		learned := (numberZero > 0 && nzeroStop >= numberZero)
		max := TrainEnv.Epoch().Max
		cur := TrainEnv.Epoch().Cur

		if TrainEnv.EvalTables[HipTableTypes(TrainAB)] == TrainEnv.Table.Table {
			if learned || cur == max/2 {
				TrainEnv.AssignTable(string(TrainAC))
				ss.Stats.SetInt("NZero", 0)
			}
		}
	}

	taskSwitching.RunStopEarly = func() bool {
		numberZero := ss.Stats.Int("NZero")
		nzeroStop := ss.Stats.Int("NZeroStop")
		learned := (numberZero > 0 && nzeroStop >= numberZero)

		if TrainEnv.EvalTables[HipTableTypes(TrainAC)] == TrainEnv.Table.Table {
			if learned {
				return true
			}
		}
		return false

	}

	return &taskSwitching
}

func calcMem(ss *sim.Sim) float64 {
	// base zero on testing performance! -
	//this should be in trainenv, or used when adding log items,
	isAB := TrainEnv.Table.Table == TrainEnv.EvalTables[TrainAB]
	var mem float64
	if isAB {
		mem = ss.Logs.Context.ItemFloat(elog.Test, elog.Epoch, "AB Mem")
	} else {
		mem = ss.Logs.Context.ItemFloat(elog.Test, elog.Epoch, "AC Mem")
	}
	return mem
}

//Move this to log items
func updateNZeroAndFirstZero(ss *sim.Sim) {
	mem := calcMem(ss)
	if ss.Stats.Int("FirstZero") < 0 && mem == 1 {
		ss.Stats.SetInt("FirstZero", ss.TrainEnv.Epoch().Cur)
	}
	if mem == 1 {
		ss.Stats.SetInt("NZero", ss.Stats.Int("NZero")+1)
	} else {
		ss.Stats.SetInt("NZero", 0)
	}
}
