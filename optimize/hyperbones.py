import sys
from collections import OrderedDict
import json
import copy
import os
import optimization
import pandas as pd

# NOTE bones must be pulled and installed locally from
# https://gitlab.com/generally-intelligent/bones/
from bones import BONES
from bones import BONESParams
from bones import ObservationInParam
from bones import LinearSpace


OBSERVATIONS_FILE = "bones_obs.txt"


def get_bones_suggestion(current_suggestions: dict, parametername, guidelines):
    return current_suggestions[parametername]


# TODO Pull this into optimization.py
def create_bones_suggested_params(params, suggestions, trial_name: str):
    cparams = copy.deepcopy(params)
    parameters_to_modify = optimization.enumerate_parameters_to_modify(cparams)
    print("PARAMETERS TO MODIFY IN BONES")
    print(parameters_to_modify)
    for info in parameters_to_modify:
        value_to_assign = get_bones_suggestion(suggestions, info["uniquename"],
                                               info["values"]["Hypers"][info["paramname"]])
        info["values"]["Params"][info["paramname"]] = value_to_assign
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = (optimization.create_hyperonly(cparams, trial_name))
    # print(updated_parameters)
    return updated_parameters


# Bones needs the space to be defined at start.
def prepare_hyperparams_bones(the_params):
    hyperparams = optimization.enumerate_parameters_to_modify(the_params)
    initial_params = {}
    params_space_by_name = OrderedDict()
    for vals in hyperparams:
        uniquename = vals["uniquename"]
        relevantvalues = vals["values"]["Hypers"][vals["paramname"]]
        assert "Val" in relevantvalues, "a default value needs to exist in relevant values in hyperparmas"
        value = float(relevantvalues["Val"])
        # Assume that we always supply a standard deviation.
        stddev = value / 4.0  # Default
        if "StdDev" in relevantvalues:
            stddev = float(relevantvalues["StdDev"])
        if "Sigma" in relevantvalues:
            stddev = float(relevantvalues["Sigma"])
        # TODO Have a parameter for Linear/LogLinear/Etc.
        # TODO Allow integer and categorical spaces.
        distribution_type = LinearSpace(scale=stddev)
        if "Min" in relevantvalues and "Max" in relevantvalues:
            distribution_type = LinearSpace(scale=stddev, min=float(relevantvalues["Min"]), max=float(relevantvalues["Max"]))
        elif "Min" in relevantvalues:
            distribution_type = LinearSpace(scale=stddev, min=float(relevantvalues["Min"]))
        elif "Max" in relevantvalues:
            distribution_type = LinearSpace(scale=stddev, min=float(relevantvalues["Max"]))
        initial_params.update({uniquename: value})
        params_space_by_name.update([(uniquename, distribution_type)])

    return {"initial_params": initial_params, "paramspace_conditions": params_space_by_name}


def optimize_bones(params, suggestions: dict, trial_name: str):
    print("BEGIN OPTIMIZE")
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = create_bones_suggested_params(params, suggestions, trial_name)

    # Save the hyperparameters so that they can be read by the model
    with open("hyperparams.json", "w") as outfile:
        json.dump(updated_parameters, outfile)

    # Run go program with -params arg
    optimization.run_model(
        "-paramsFile=hyperparams.json -nogui=true -epclog=true -params={0} -runs={1} -epochs={2}".format(
            trial_name, str(optimization.NUM_RUNS), str(optimization.NUM_EPOCHS)))

    # Get valuation from logs
    return optimization.get_score_from_logs(trial_name)


def run_bones(bones_obj, trialnumber, params):
    best_suggest = {}
    best_score = sys.float_info.max
    for i in range(trialnumber):
        trial_name = "Searching_" + str(i)
        suggestions = bones_obj.suggest().suggestion
        print("TRYING THESE SUGGESTIONS")
        print(suggestions)
        observed_value = optimize_bones(params, suggestions, trial_name)
        print(observed_value)
        bones_obj.observe(ObservationInParam(input=suggestions, output=observed_value))
        if observed_value < best_score:
            best_score = observed_value
            best_suggest = suggestions
        with open(OBSERVATIONS_FILE, "a") as file_object:
            file_object.write("\n{}\t{}\t{}\t{}".
                              format(str(suggestions), str(observed_value), str(best_suggest), str(best_score)))
    return best_suggest, best_score


def main():
    os.chdir('../')  # Move into the models/ directory
    params = optimization.get_hypers()

    prep_params_dict = prepare_hyperparams_bones(params)
    initial_params = prep_params_dict["initial_params"]
    params_space_by_name = prep_params_dict["paramspace_conditions"]
    bone_params = BONESParams(
        better_direction_sign=-1, is_wandb_logging_enabled=False, initial_search_radius=0.5, resample_frequency=-1
    )
    bones = BONES(bone_params, params_space_by_name)
    bones.set_search_center(initial_params)
    best, best_score = run_bones(bones, optimization.NUM_TRIALS, params)
    print("Best parameters at: " + str(best) + " with score: " + str(best_score))


if __name__ == '__main__':
    print("Starting optimization main func")
    main()
