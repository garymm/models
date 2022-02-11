import subprocess

import optuna
import json
import copy
import csv
import os
from optuna import Trial
import pandas as pd
#todo integrate so that can easily swap between bones and optuna
#todo specify the number iterations per epoch and epochs
#todo oneday wrap this in a clear object with comments
MECHNAME = "RA25" #"One2Many", "RA25", these are app names defined at the top of each mech file
EXECUTABLE_PATH = "ra25" #the directory the file comes from
VARIABLE_TO_OPTIMIZE = "#PctErr"

def generate_list_iterate(params: list):
    params_relations = []
    assert len(params) == 1
    index = 0
    for name in params[0]["Sheets"]:
        for idx in range(len(params[0]["Sheets"][name])):
            element = params[0]["Sheets"][name][idx]
            if ("Hypers" in element) & (type(element["Hypers"]) != type(None)):
                for paramname in element["Hypers"]:
                    uniquename = "{}_{}".format(index, paramname)
                    index += 1
                    params_relations.append(
                        {"uniquename": uniquename, "paramname": paramname, "sheetidx": idx, "values": element})

    return params_relations


def create_hyperonly(params):
    duplicate = {}
    duplicate["Name"] = "Searching"
    duplicate["Desc"] = "Parameters suggested by optimizer"
    duplicate["Sheets"] = {}
    assert len(params) == 1
    for name in params[0]["Sheets"]:
        duplicate["Sheets"][name] = []
        for idx in range(len(params[0]["Sheets"][name])):
            element = params[0]["Sheets"][name][idx]
            if ("Hypers" in element) & (type(element["Hypers"]) != type(None)):
                dup_element = copy.deepcopy(element)
                for pname in element["Params"]:
                    if (pname in element["Hypers"]) == False:
                        del dup_element["Params"][pname]
                del dup_element["Hypers"]
                duplicate["Sheets"][name].append(dup_element)

    return [duplicate]


def run_model(args):
    gocommand = "go"
    # If you encounter an error here, add another line to get your local go file.
    if str(os.popen("test -f /usr/local/go/bin/go && echo linux").read()).strip() == "linux":
        gocommand = "/usr/local/go/bin/go"
    elif str(os.popen("test -f /usr/local/opt/go/libexec/bin/go && echo mac").read()).strip() == "mac":
        gocommand = "/usr/local/opt/go/libexec/bin/go"
    os.system(gocommand + " run mechs/{}/*.go ".format(EXECUTABLE_PATH) + args)


def get_opt_value(trial: Trial, parametername, guidelines):
    val = float(guidelines["Val"])
    if guidelines.get("Min") is not None and guidelines.get("Max") is not None:
        print("GUIDELINES")
        print(guidelines)
        mino = float(guidelines["Min"])
        maxo = float(guidelines["Max"])
    else:
        # TODO Make this better
        mino = val * .5
        maxo = val * 1.5
    return trial.suggest_float(parametername, mino, maxo)


def create_suggested_params(params, trial):
    cparams = copy.deepcopy(params)
    parameters_to_modify = generate_list_iterate(cparams)
    print("PARAMETERS TO MODIFY")
    print(parameters_to_modify)
    for info in parameters_to_modify:
        value_to_assign = get_opt_value(trial, info["uniquename"], info["values"]["Hypers"][info["paramname"]])
        info["values"]["Params"][info["paramname"]] = value_to_assign
    # This creates a version of Params that has stripped out everything that didn't have Hypers
    updated_parameters = (create_hyperonly(cparams))
    print("UPDATED PARAMS")
    print(updated_parameters)
    return updated_parameters



def main():
    os.chdir('../')  # Move into the models/ directory

    hyperFile = "hyperparamsExample.json"
    # Run go with -hyperFile cmd arg to save them to file
    print("GETTING HYPERPARAMETERS")
    run_model("-hyperFile=" + hyperFile)
    # Load hypers from file
    f = open(hyperFile)
    params = json.load(f)
    f.close()
    print("GOT PARAMS")
    print(params)

    # Study definition
    study = optuna.create_study(direction='minimize', study_name="Find_Some_Hypers")

    def optimize(trial: Trial):
        print("BEGIN OPTIMIZE")
        # This creates a version of Params that has stripped out everything that didn't have Hypers
        updated_parameters = create_suggested_params(params, trial)

        # Save the hyperparameters so that they can be read by the model
        with open("hyperparams.json", "w") as outfile:
            json.dump(updated_parameters, outfile)

        # Run go program with -params arg
        run_model("-paramsFile=hyperparams.json -nogui=true -epclog=true -params=Searching -runs=5 -epochs=1")

        # Get valuation from logs
        # TODO Make sure this name is unique for parallelization.
        score = pd.read_csv('logs/{}_Searching_testepc.tsv'.format(MECHNAME), sep="\t")[VARIABLE_TO_OPTIMIZE].values[-1]
        return float(score)

    # Starts optimization
    study.optimize(optimize, n_trials=4)
    print("BEST PARAMS")
    print(study.best_params)

    # TODO Create a full parameters set that uses the best params


if __name__ == '__main__':
    print("Starting optimization main func")
    main()
