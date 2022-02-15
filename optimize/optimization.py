import copy
import os
import pandas as pd
import json

# todo integrate so that can easily swap between bones and optuna
# todo specify the number iterations per epoch and epochs
# todo oneday wrap this in a clear object with comments
MECHNAME = "RA25"  # "One2Many", "RA25", these are app names defined at the top of each mech file
EXECUTABLE_PATH = "ra25"  # the directory the file comes from
VARIABLE_TO_OPTIMIZE = "#PctErr"
NUM_EPOCHS = 100
NUM_RUNS = 10
NUM_TRIALS = 20


def get_hypers():
    hyper_file = "hyperparamsExample.json" #this
    # Run go with -hyperFile cmd arg to save them to file
    print("GETTING HYPERPARAMETERS")
    run_model("-hyperFile=" + hyper_file)
    # Load hypers from file
    f = open(hyper_file)
    params = json.load(f)
    f.close()
    print("GOT PARAMS")
    print(params)
    return params


def get_score_from_logs():
    # TODO Make sure this name is unique for parallelization.
    score = pd.read_csv('logs/{}_Searching_run.tsv'.format(MECHNAME), sep="\t")[VARIABLE_TO_OPTIMIZE].values[-1]
    return float(score)


def enumerate_parameters_to_modify(params: list):
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
                    if paramname in element["Params"]:
                        element["Hypers"][paramname]["Val"] = element["Params"][paramname]
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
    os.system(gocommand + " run mechs/{}/*[^test].go ".format(EXECUTABLE_PATH) + args)
