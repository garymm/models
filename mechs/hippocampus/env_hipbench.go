package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

type HipTableTypes string
type TableMaps map[HipTableTypes]*etable.Table

const (
	TrainAB      HipTableTypes = "TrainAB"
	TrainBC      HipTableTypes = "TrainBC"
	TestAB       HipTableTypes = "TestAB"
	TestAC       HipTableTypes = "TestAC"
	TrainAC      HipTableTypes = "TestAC"
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

type EnvHipBench struct {
	sim.Environment
	env.FixedTable
	EvalTables TableMaps //a map of tables used for handling stuff
	Pat        PatParams
}

func (pp *PatParams) Defaults() {
	pp.ListSize = 10 // 20 def
	pp.MinDiffPct = 0.5
	pp.CtxtFlipPct = .25
}

func (envhip *EnvHipBench) InitTables(tableNames ...HipTableTypes) {
	envhip.EvalTables = make(TableMaps)
	for _, nm := range tableNames {
		envhip.EvalTables[nm] = &etable.Table{}
	}
}

func (envhip *EnvHipBench) SetName(name string) {
	envhip.FixedTable.Nm = name
}
func (envhip *EnvHipBench) Name() string {
	return envhip.FixedTable.Nm
}
func (envhip *EnvHipBench) SetDesc(desc string) {
	envhip.FixedTable.Dsc = desc
}
func (envhip *EnvHipBench) Desc() string {
	return envhip.FixedTable.Dsc
}

func (envhip *EnvHipBench) Order() []int {
	return envhip.FixedTable.Order
}
func (envhip *EnvHipBench) Sequential() bool {
	return envhip.FixedTable.Sequential
}
func (envhip *EnvHipBench) SetSequential(s bool) {
	envhip.FixedTable.Sequential = s
}

func (envhip *EnvHipBench) Run() *env.Ctr {
	return &envhip.FixedTable.Run
}
func (envhip *EnvHipBench) Epoch() *env.Ctr {
	return &envhip.FixedTable.Epoch
}
func (envhip *EnvHipBench) Trial() *env.Ctr {
	return &envhip.FixedTable.Trial
}
func (envhip *EnvHipBench) TrialName() *env.CurPrvString {
	return &envhip.FixedTable.TrialName
}
func (envhip *EnvHipBench) GroupName() *env.CurPrvString {
	return &envhip.FixedTable.GroupName
}
func (envhip *EnvHipBench) NameCol() string {
	return envhip.FixedTable.NameCol
}
func (envhip *EnvHipBench) SetNameCol(s string) {
	envhip.FixedTable.NameCol = s
}
func (envhip *EnvHipBench) GroupCol() string {
	return envhip.FixedTable.GroupCol
}
func (envhip *EnvHipBench) SetGroupCol(s string) {
	envhip.FixedTable.GroupCol = s
}

func (envhip *EnvHipBench) Validate() error {

	return envhip.FixedTable.Validate()
}
func (envhip *EnvHipBench) Init(run int) {
	envhip.FixedTable.Init(run)
	envhip.Pat.Defaults()
}

func (envhip *EnvHipBench) CurTrialName() string {
	return envhip.TrialName().Cur
}
func (envhip *EnvHipBench) Step() {
	envhip.FixedTable.Step()
}

func (envhip *EnvHipBench) State(s string) etensor.Tensor {
	return envhip.FixedTable.State(s)
}

func (envhip *EnvHipBench) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	return envhip.FixedTable.Counter(scale)
}
