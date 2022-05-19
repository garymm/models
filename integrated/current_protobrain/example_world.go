package main

import (
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

type ExampleWorld struct {
	agent.WorldInterface
	netAttributes    NetCharacteristics
	observationShape map[string][]int
	observations     map[string]*etensor.Float32
}

// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
func (world *ExampleWorld) Observe(name string) etensor.Tensor {
	if name == "VL" { //if type target
		return world.observations[name]
	} else { //if an input
		if world.observations[name] == nil {
			spaceSpec := world.netAttributes.ObservationMapping[name]
			world.observations[name] = etensor.NewFloat32(spaceSpec.ContinuousShape, spaceSpec.Stride, nil)
			patgen.PermutedBinaryRows(world.observations[name], 1, 1, 0)
		}

	}
	return world.observations[name]
}

// StepWorld steps the index of the current pattern.
func (world *ExampleWorld) StepWorld(actions map[string]agent.Action, agentDone bool) (worldDone bool, debug string) {
	return false, ""
}

// Init Initializes or reinitialize the world, todo, change from being hardcoded for emery
func (world *ExampleWorld) InitWorld(details map[string]string) (actionSpace map[string]agent.SpaceSpec, observationSpace map[string]agent.SpaceSpec) {

	world.netAttributes = (&NetCharacteristics{}).Init()
	world.observations = make(map[string]*etensor.Float32)
	world.observations["VL"] = etensor.NewFloat32(world.netAttributes.ActionMapping["VL"].ContinuousShape, world.netAttributes.ActionMapping["VL"].Stride, nil)

	patgen.PermutedBinaryRows(world.observations["VL"], 1, 1, 0)

	return map[string]agent.SpaceSpec{"VL": world.netAttributes.ObservationMapping["VL"]}, world.netAttributes.ObservationMapping
}
