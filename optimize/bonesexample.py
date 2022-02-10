import math
from collections import OrderedDict
import numpy as np
import json
import copy
import csv
import os
import optimize


#NOTE bones must be pulled and installed locally from
#https://gitlab.com/generally-intelligent/bones/
from bones import BONES
from bones import BONESParams
from bones import LogSpace
from bones import LogitSpace
from bones import ObservationInParam
from bones import ParamDictType
from bones import LinearSpace


def get_bones_suggestion(current_suggestions:dict,parametername, guidelines):
    return current_suggestions[parametername]

def create_bones_suggested_params(params, suggestions):
    cparams = copy.deepcopy(params)
    parameters_to_modify = optimize.generate_list_iterate(cparams)
    print("PARAMETERS TO MODIFY in Bones")
    print(parameters_to_modify)
    for info in parameters_to_modify:
        value_to_assign = get_bones_suggestion(suggestions, info["uniquename"], info["values"]["Hypers"][info["paramname"]])
        info["values"]["Params"][info["paramname"]] = value_to_assign
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = (optimize.create_hyperonly(cparams))
    # print(updated_parameters)
    return updated_parameters


def optimize_bones(params, suggestions:dict):
    print("BEGIN OPTIMIZE")
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = create_bones_suggested_params(params, suggestions)
    # Save the hyperparameters so that they can be read by the model
    with open("hyperparams.json", "w") as outfile:
        json.dump(updated_parameters, outfile)
    # Run go program with -params arg
    optimize.run_model("-paramsFile=hyperparams.json -nogui=true -epclog=true -params=Searching -runs=5 -epochs=1")
    # Get valuation from logs
    # TODO Make sure this name is unique for parallelization.
    with open('One2Many_Searching_testepc.tsv', newline='') as csvfile: # TODO this should have a name that corresponds to project, leaving for now as it will cause a problem in optimize
        f = csv.reader(csvfile, delimiter='\t', quotechar='|')
        rows = []
        for row in f:
            rows.append(row)
        # Get the last UnitErr
        # TODO Parse this tsv file more carefully
        score = rows[-1][2]
        print("GOT SCORE: " + str(score))
    return float(score)


def prepare_hyperparams_bones(the_params):
    #todo check if ordinal cases, is int, and determine approapriate distribution, as well as naive min max
    hyperparams = optimize.generate_list_iterate(the_params)

    initial_params = {}
    params_space_by_name = OrderedDict()
    for vals in hyperparams:
        uniquename = vals["uniquename"]
        relevantvalues = vals["values"]["Hypers"][vals["paramname"]]
        assert "Val" in relevantvalues, "a default value needs to exist in relevant values in hyperparmas"

        value = float(relevantvalues["Val"])
        min, max, sigma = -1.0, -1.0, -1.0
        if ("Min" in relevantvalues) == False: min = value * .5
        else: min = float(relevantvalues["Min"])
        if ("Max" in relevantvalues) == False: max = value * 1.5
        else: max = float(relevantvalues["Max"])
        if ("Sigma" in relevantvalues) == False: sigma = .5
        else: sigma = float(relevantvalues["Sigma"])
        min = 0
        initial_params.update({uniquename:value})
        distribution_type =  LinearSpace(scale=sigma,min=min,max = max) #this is naive, and assume int false
        params_space_by_name.update([(uniquename,distribution_type)])

    return {"initial_params":initial_params, "paramspace_conditions":params_space_by_name}


def run_bones(bones_obj,trialnumber, params, optimize_fn):
    for i in range(trialnumber):
        suggestions = bones_obj.suggest().suggestion
        observed_value = optimize_fn(params, suggestions)
        bones_obj.observe(ObservationInParam(input=suggestions, output=observed_value))




if __name__ == '__main__':
    os.chdir('../')  # Move into the models/ directory
    hyperFile = "hyperparamsExample.json"
    # Run go with -hyperFile cmd arg to save them to file
    print("GETTING HYPERPARAMETERS")
    #optimize.run_model("-hyperFile=" + hyperFile)
    # Load hypers from file
    f = open(hyperFile)
    params = json.load(f)
    f.close()
    print("GOT PARAMS")

    prep_params_dict = prepare_hyperparams_bones(params)
    initial_params = prep_params_dict["initial_params"]
    params_space_by_name =  prep_params_dict["paramspace_conditions"]
    bone_params = BONESParams(
        better_direction_sign=-1, is_wandb_logging_enabled=False, initial_search_radius=0.5, resample_frequency=-1
    )
    bones = BONES(bone_params, params_space_by_name)
    bones.set_search_center(initial_params)
    run_bones(bones,2,params,optimize_fn=optimize_bones)