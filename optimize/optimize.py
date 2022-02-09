import optuna
import json
import copy
import csv
import os

from optuna import Trial


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
    #
    return [duplicate]


def run_model(args):
    os.system("GOROOT=/usr/local/opt/go/libexec") #gosetup
    os.system("GOPATH=/Users/garbar/go") #gosetup
    # TODO Make this more general.
    os.system("/usr/local/opt/go/libexec/bin/go build -o /private/var/folders/wq/b74k_01n1v14krlqryxn2_s80000gn/T/GoLand/___textone2many_nogui github.com/Astera-org/models/mechs/text_one2many")
    os.system("/private/var/folders/wq/b74k_01n1v14krlqryxn2_s80000gn/T/GoLand/___textone2many_nogui -nogui=true " + args)


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
    # print(updated_parameters)
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
        run_model("-paramsFile=hyperparams.json -nogui=true -epclog=true -params=Searching -runs=1 -epochs=3")

        # Get valuation from logs
        # TODO Make sure this name is unique for parallelization.
        with open('One2Many_Searching_epc.tsv', newline='') as csvfile:
            f = csv.reader(csvfile, delimiter='\t', quotechar='|')
            rows = []
            for row in f:
                rows.append(row)
            # Get the last UnitErr
            # TODO Parse this tsv file more carefully
            score = rows[-1][2]
            print("GOT SCORE: " + str(score))
        return float(score)

    # Starts optimization
    study.optimize(optimize, n_trials=4)
    print("BEST PARAMS")
    print(study.best_params)

    # TODO Create a full parameters set that uses the best params


if __name__ == '__main__':
    print("Starting optimization main func")
    main()
