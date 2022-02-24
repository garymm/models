package sim

import (
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
)

// Environment defines expected getters and setters for our environment variables
type Environment interface {
	SetName(name string)
	Name() string
	SetDesc(desc string)
	Desc() string

	Order() []int
	Sequential() bool
	SetSequential(s bool)

	Epoch() *env.Ctr
	Trial() *env.Ctr
	Run() *env.Ctr

	TrialName() *env.CurPrvString

	CurTrialName() string
	GroupName() *env.CurPrvString

	NameCol() string
	SetNameCol(s string)
	GroupCol() string
	SetGroupCol(s string)

	AssignTable(s string)

	Validate() error

	Step()
	State(s string) etensor.Tensor
	Counter(scale env.TimeScales) (cur, prv int, chg bool)

	Init(run int)
}
