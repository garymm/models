import collections
import sys
import time
from collections import OrderedDict
import json
import copy
import os

import wandb

import optimization
import concurrent.futures
import threading

# NOTE bones must be pulled and installed locally from
# https://gitlab.com/generally-intelligent/bones/
from bones import BONES
from bones import BONESParams
from bones import ObservationInParam
from bones import LinearSpace
from bones import LogSpace


OBSERVATIONS_FILE = "bones_obs.txt" #todo doesn't seem to actually write to file


def get_bones_suggestion(current_suggestions: dict, parametername, guidelines):
    return current_suggestions[parametername]


# TODO Pull this into optimization.py
def create_bones_suggested_params(params, suggestions, trial_name: str):
    cparams = copy.deepcopy(params)
    parameters_to_modify = optimization.enumerate_parameters_to_modify(cparams)
    # print("PARAMETERS TO MODIFY IN BONES")
    # print(parameters_to_modify)
    for info in parameters_to_modify:
        value_to_assign = get_bones_suggestion(suggestions, info["uniquename"],
                                               info["values"]["Hypers"][info["paramname"]])
        info["values"]["Params"][info["paramname"]] = str(value_to_assign)
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
        bones_args = {
            "is_integer": ("Type" in relevantvalues and relevantvalues["Type"] == "Int"),
            "scale": stddev
        }
        if "Min" in relevantvalues:
            bones_args["min"] = float(relevantvalues["Min"])
        if "Max" in relevantvalues:
            bones_args["max"] = float(relevantvalues["Max"])
        if value <= 0 or ("Space" in relevantvalues and relevantvalues["Space"] == "Linear"):
            distribution_type = LinearSpace(**bones_args)
        else:
            distribution_type = LogSpace(**bones_args)
        initial_params.update({uniquename: value})
        params_space_by_name.update([(uniquename, distribution_type)])

    return initial_params, params_space_by_name


def optimize_bones(params, suggestions: dict, trial_name: str):
    print("BEGIN OPTIMIZE")
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = create_bones_suggested_params(params, suggestions, trial_name)

    # Save the hyperparameters so that they can be read by the model
    hyperfile = "hyperparams{}.json".format(trial_name)
    with open(hyperfile, "w") as outfile:
        json.dump(updated_parameters, outfile)

    # Run go program with -params arg
    optimization.run_model(
        "-paramsFile={} -nogui=true -epclog=true -params={} -runs={} -epochs={}".format(
            hyperfile, trial_name, str(optimization.NUM_RUNS), str(optimization.NUM_EPOCHS)))

    # Get valuation from logs
    return optimization.get_score_from_logs(trial_name)


def run_bones(bones_obj, trialnumber, params):
    best_suggest = {}
    best_score = sys.float_info.max
    for i in range(trialnumber):
        trial_name = "Searching_" + str(i)
        current_suggestion = None
        suggestions = bones_obj.suggest().suggestion
        # print("TRYING THESE SUGGESTIONS")
        # print(suggestions)
        observed_value = optimize_bones(params, suggestions, trial_name)
        # print(observed_value)
        bones_obj.observe(ObservationInParam(input=suggestions, output=observed_value))
        if observed_value < best_score:
            best_score = observed_value
            best_suggest = suggestions
        with open(OBSERVATIONS_FILE, "a") as file_object:
            file_object.write("\n{}\t{}\t{}\t{}".
                              format(str(suggestions), str(observed_value), str(best_suggest), str(best_score)))
    return best_suggest, best_score


all_observations = collections.deque()
start_time = -1


def single_bones_trial(bones_obj, params, lock, i):
    trial_name = "Searching_" + str(i)
    print("Starting trial: " + trial_name)
    with lock:
        suggestions = bones_obj.suggest().suggestion
    print("TRYING THESE SUGGESTIONS")
    print(suggestions)
    observed_value = optimize_bones(params, suggestions, trial_name)
    print("GOT OBSERVED VALUE")
    print(observed_value)
    with lock:
        bones_obj.observe(ObservationInParam(input=suggestions, output=observed_value))
        all_observations.append((observed_value, suggestions, trial_name))
        print("Finished this many observations: " + str(len(all_observations)))
        # for so in all_observations:
        #     print(so[2] + " Score: " + str(so[0]) + " From Sugg: " + str(so[1]))
        best = sorted(all_observations, key=lambda a: a[0])[0]
        print("BEST RESULT: " + str(best))
    print("Average elapsed time: " + str((time.time() - start_time) / len(all_observations)))


def run_bones_parallel(bones_obj, trialnumber, params):
    locky = threading.Lock()
    global start_time
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=optimization.NUM_PARALLEL) as executor:
        for i in range(trialnumber):
            print("Starting to execute: " + str(i))
            executor.submit(single_bones_trial, bones_obj, params, locky, i)

    best = sorted(all_observations, key=lambda a: a[0])[0]
    return best[1], best[0]


def main():
    os.chdir('../')  # Move into the models/ directory
    params = optimization.get_hypers()

    initial_params, params_space_by_name = prepare_hyperparams_bones(params)
    wandb_key = ""
    if wandb_key != "":
        wandb.login(key = wandb_key)
    bone_params = BONESParams(
        better_direction_sign=-1, is_wandb_logging_enabled=wandb_key!="", initial_search_radius=0.5, resample_frequency=-1
    )

    bones = BONES(bone_params, params_space_by_name)
    bones.set_search_center(initial_params)
    best, best_score = run_bones_parallel(bones, optimization.NUM_TRIALS, params)
    print("Best parameters at: " + str(best) + " with score: " + str(best_score))


if __name__ == '__main__':
    print("Starting optimization main func")
    main()
