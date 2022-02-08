# import optuna
import json
import copy


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


def _generate_list(fullset: list, params: dict):
    for i in params:
        if type(params) == type([]):
            child = i
        else:
            child = params[i]

            if (i == "Hypers") & (type(child) != type(None)):
                for key in child:
                    fullset.append({key: child[key]})
                return
        if type(child) == type({}):
            _generate_list(fullset, child)
        elif type(child) == type([]):
            _generate_list(fullset, child)


def generate_hyperlist(params: dict) -> list:
    fullset = []
    newset = []
    _generate_list(fullset, params)

    for i in range(len(fullset)):
        for key in fullset[i]:
            name = "{}_{}".format(i, key)
            newset.append({name: fullset[i][key]})
    return fullset, newset


def optimizer(parametername, guidelines: {}):
    return str(float(guidelines["Val"]) + 6666)


def generate_parameters(params: dict, hyperparams: dict, optimizer_func):
    pass


# def faux_optimize
def main():
    # Load hypers from file
    hyperFile = "../hyperparamsExample.json"
    # TODO Run go with -hyperFile cmd arg
    f = open(hyperFile)
    params = json.load(f)
    f.close()
    print(params)

    listofhypers, hyperparameterlist = generate_hyperlist(params)

    # This is where optuna would be running
    parameters_to_modify = (generate_list_iterate(params))
    for info in parameters_to_modify:
        value_to_assign = optimizer(info["uniquename"], info["values"]["Hypers"][info["paramname"]])
        info["values"]["Params"][info["paramname"]] = value_to_assign


    updated_parameters = (create_hyperonly(params))
    print(updated_parameters)

    with open("../hyperparams.json", "w") as outfile:
        json_object = json.dump(updated_parameters,outfile)
    ##faux optuna
    # for source, duplicate in zip(listofhypers, hyperparameterlist):
    #    pass
    # updatedval = optimizer(j, hyperparameterlist[j])

    # print("DONE")
    # print(listofhypers)
    # Construct hypers object for optuna

    # Run one2many
    # Read logs


# print(listofhypers)
# with open("hyperparams.json", "w") as outfile:
#    json_object = json.dumps(listofhypers)


if __name__ == '__main__':
    print("HEA")
    main()
