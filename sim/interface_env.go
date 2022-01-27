package sim

import (
	"github.com/emer/emergent/env"
)

//Environment defines expected getters and setters for our environment variables
type Environment interface {
	SetName(name string)
	Name() string
	SetDesc(desc string)
	Desc() string

	Order() []int
	Sequential() bool
	SetSequential(s bool)

	Run() *env.Ctr
	Epoch() *env.Ctr
	Trial() *env.Ctr

	TrialName() *env.CurPrvString

	GroupName() *env.CurPrvString

	NameCol() string
	SetNameCol(s string)
	GroupCol() string
	SetGroupCol(s string)

	Validate() error

	Init(run int)
}
