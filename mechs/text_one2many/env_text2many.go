package main

import (
	"github.com/Astera-org/models/sim"
	"github.com/emer/emergent/env"
)

type EnvText2Many struct {
	sim.CorpusEnv
	sim.Environment
}

func (env *EnvText2Many) SetName(name string) {
	env.CorpusEnv.Nm = name
}
func (env *EnvText2Many) Name() string {
	return env.CorpusEnv.Nm
}
func (env *EnvText2Many) SetDesc(desc string) {
	env.CorpusEnv.Dsc = desc
}
func (env *EnvText2Many) Desc() string {
	return env.CorpusEnv.Dsc
}

func (env *EnvText2Many) Order() []int {
	return []int{}
}
func (env *EnvText2Many) Sequential() bool {
	return false
}
func (env *EnvText2Many) SetSequential(s bool) {
}
func (env *EnvText2Many) Run() *env.Ctr {
	return &env.CorpusEnv.Run
}
func (env *EnvText2Many) Epoch() *env.Ctr {
	return &env.CorpusEnv.Epoch
}
func (env *EnvText2Many) Trial() *env.Ctr {
	return &env.CorpusEnv.Trial
}
func (env *EnvText2Many) TrialName() *env.CurPrvString {
	return nil
}
func (env *EnvText2Many) GroupName() *env.CurPrvString {
	return nil
}
func (env *EnvText2Many) NameCol() string {
	return ""
}
func (env *EnvText2Many) SetNameCol(s string) {
}

func (env *EnvText2Many) GroupCol() string {
	return ""
}
func (env *EnvText2Many) SetGroupCol(s string) {
}

func (env *EnvText2Many) Validate() error {
	return env.CorpusEnv.Validate()
}
func (env *EnvText2Many) Init(run int) {
	env.CorpusEnv.Init(run)
}
