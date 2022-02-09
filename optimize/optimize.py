import optuna
import json
import copy
import csv


# TODO These are the steps for hyperparameter optimization
#  ✅ Find an optimizer package like optuna and import it
#  Take our Hypers object and translate it into the hyperparameters
#  Get the test performance from logs as a val function
#  Connect hyperspace (hyperparameter space) and val function to package
#  Call one2many on the command line

# Python program starts
# ✅ Python calls go model with a --hypers cmd arg, which tells it to do nothing except print out its hyperparameter specificiations
# Python reads that file to get hypers, then converts into a format that optuna can use
# Python writes params to file
# Python calls the go model with a cmdline arg telling it to run without gui, and telling it where to find params
# Go model computes a single eval metric and writes it to file
# When go model finishes, python reads output from file, getting objective metric
# Python iterates
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

    return [duplicate]


# def _generate_list(fullset: list, params: dict):
#     for i in params:
#         if type(params) == type([]):
#             child = i
#         else:
#             child = params[i]
#
#             if (i == "Hypers") & (type(child) != type(None)):
#                 for key in child:
#                     fullset.append({key: child[key]})
#                 return
#         if type(child) == type({}):
#             _generate_list(fullset, child)
#         elif type(child) == type([]):
#             _generate_list(fullset, child)


# def generate_hyperlist(params: dict) -> list:
#     fullset = []
#     newset = []
#     _generate_list(fullset, params)
#
#     for i in range(len(fullset)):
#         for key in fullset[i]:
#             name = "{}_{}".format(i, key)
#             newset.append({name: fullset[i][key]})
#     return newset


# def optimizer(parametername, guidelines: {}):
#     print("DOING OPTIMIZATION: " + str(parametername) + " " + str(guidelines))
#     return str(float(guidelines["Val"]) + 6666)


def generate_parameters(params: dict, hyperparams: dict, optimizer_func):
    pass


def run_model(args):
    os.system("GOROOT=/usr/local/go #gosetup")
    os.system("GOPATH=/home/keenan/go #gosetup")
    os.system("/usr/local/go/bin/go build -o /tmp/GoLand/___text_one2many_load_params_from_file github.com/Astera-org/models/mechs/text_one2many #gosetup")
    os.system("/tmp/GoLand/___text_one2many_load_params_from_file " + args)


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



def main():
    # TODO(andrew and michael) Clean up this file

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
    study = optuna.create_study(direction='minimize')

    def optimize(trial: Trial):
        print("BEGIN OPTIMIZE")
        # This block modifies params in place to include the new optimized values
        # TODO Don't modify in place
        parameters_to_modify = generate_list_iterate(params)
        print("PARAMETERS TO MODIFY")
        print(parameters_to_modify)
        for info in parameters_to_modify:
            # TODO Invoke Optuna or Bones here
            # value_to_assign = optimizer(info["uniquename"], info["values"]["Hypers"][info["paramname"]])
            value_to_assign = get_opt_value(trial, info["uniquename"], info["values"]["Hypers"][info["paramname"]])
            info["values"]["Params"][info["paramname"]] = value_to_assign
        # print("OPTIMIZED PARAMETERS")
        # print(parameters_to_modify)
        # print("MODIFIED PARAMS")
        # print(params)

        # This creates a version of Params that has stripped out everything that didn't have Hypers
        updated_parameters = (create_hyperonly(params))
        # print("UPDATED PARAMETERS")
        print(updated_parameters)

        # Save the hyperparameters so that they can be read by the model
        # print("SAVING HYPERS")
        with open("hyperparams.json", "w") as outfile:
            json.dump(updated_parameters, outfile)

        # Run go program with -params arg
        run_model("-paramsFile=hyperparams.json -nogui=true -params=Searching -runs=1 -epochs=3")

        # Get valuation from logs
        score = 0
        with open('One2Many_Searching_epc.tsv', newline='') as csvfile:
            f = csv.reader(csvfile, delimiter='\t', quotechar='|')
            rows = []
            for row in f:
                rows.append(row)
                # print(', '.join(row))
            # Get the last UnitErr
            # TODO Parse this tsv file more carefully
            score = rows[-1][2]
            print("GOT SCORE: " + str(score))
        return float(score)

    # Starts optimization
    study.optimize(optimize, n_trials=4)

    # Iterate through many runs
    # TODO Parallelization
    # maxtries = 1
    # i = 0
    # while i < maxtries:
    #     i += 1
    #     print("EVALUATING PARAMS TRY " + str(i))

        # hyperparameterlist = generate_hyperlist(params)
        # # TODO(michael) Why isn't this parameter used? Can this code be deleted?
        # print("GOT HYPERPARAMS")
        # print(hyperparameterlist)

        # TODO Communicate with optimizer

    ##faux optuna
    # for source, duplicate in zip(listofhypers, hyperparameterlist):
    #    pass
    # updatedval = optimizer(j, hyperparameterlist[j])

    # print("DONE")
    # print(listofhypers)
    # Construct hypers object for optuna

    # Run one2many
    # Read logs


if __name__ == '__main__':
    print("Starting optimization main func")
    main()
