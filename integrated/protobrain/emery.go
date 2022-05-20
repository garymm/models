// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/Astera-org/worlds/network_agent"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/etensor"
)

// Protobrain demonstrates a network model that has elements of cortical visual perception and a rudimentary action system. It is not reward motivated, and instead it learns to approximate a behavior heuristic. It is intended to be used with the world found in github.com/Astera-org/worlds/integrated/example_worlds.

func main() {
	var sim Sim
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()
	world, serverFunc := network_agent.GetWorldAndServerFunc(sim.Loops)
	sim.WorldEnv = world

	userInterface := egui.UserInterface{
		StructForView:             &sim,
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
		AppName:                   "Attempted refactor of Fworld",
		AppTitle:                  "Fworld with refactored code",
		AppAbout:                  `Learn to mimic patterns coming from a teacher signal in a flat grid world.`,
		AddNetworkLoggingCallback: axon.AddCommonLogItemsForOutputLayers,
		DoLogging:                 true,
		HaveGui:                   true,
		StartAsServer:             true,
		ServerFunc:                serverFunc,
	}
	userInterface.Start() // Start blocks, so don't put any code after this.
}

// Sim encapsulates working data for the simulation model, keeping all relevant state information organized and available without having to pass everything around.
type Sim struct {
	Net      *deep.Network        `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	NetDeets NetworkDeets         `desc:"Contains details about the network."`
	Loops    *looper.Manager      `view:"no-inline" desc:"contains looper control loops for running sim"`
	WorldEnv agent.WorldInterface `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time     axon.Time            `desc:"axon timing parameters and state"`
	LoopTime string               `desc:"Printout of the current time."`
}

func (ss *Sim) ConfigNet() *deep.Network {
	net := &deep.Network{}
	DefineNetworkStructure(&ss.NetDeets, net)
	return net
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigLoops() *looper.Manager {
	manager := looper.Manager{}.Init()
	manager.Stacks[etime.Train] = &looper.Stack{}
	manager.Stacks[etime.Train].Init().AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 1).AddTime(etime.Cycle, 200)
	axon.AddPlusAndMinusPhases(manager, &ss.Time, ss.Net.AsAxon())

	plusPhase := &manager.GetLoop(etime.Train, etime.Cycle).Events[1]
	plusPhase.OnEvent.Add("Sim:PlusPhase:SendActionsThenStep", func() {
		axon.SendActionAndStep(ss.Net.AsAxon(), ss.WorldEnv)
	})

	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Sim:ResetState", func() {
		ss.Net.NewState()
		ss.Time.NewState(mode.String())
	})

	stack.Loops[etime.Trial].OnStart.Add("Sim:Trial:Observe", func() {
		// If not using an AgentProxyWithWorldCache, the full list of observations is returned from StepWorld().
		for name, _ := range ss.WorldEnv.(*agent.AgentProxyWithWorldCache).CachedObservations {
			axon.ApplyInputs(ss.Net.AsAxon(), ss.WorldEnv, name, func(spec agent.SpaceSpec) etensor.Tensor {
				return ss.WorldEnv.Observe(name)
			})
		}

	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)
	//manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewPatterns", func() { ss.WorldEnv.InitWorld(nil) })
	axon.AddDefaultLoopSimLogic(manager, &ss.Time, ss.Net.AsAxon())

	// Initialize and print loop structure, then add to Sim
	manager.Init()
	fmt.Println(manager.DocString())

	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Sim:Trial:QuickScore", func() {
		loss := ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon().PctUnitErr()
		s := fmt.Sprintf("%f", loss)
		fmt.Println("the pctuniterror is " + s)
		//simple error calculation
	})

	return manager
}

// NewRun intializes a new run of the model, using the WorldMailbox.GetCounter(etime.Run) counter for the new run value
func (ss *Sim) NewRun() {
	ss.NetDeets.PctCortex = 0
	ss.WorldEnv.InitWorld(nil)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.NetDeets.InitStats()
}
