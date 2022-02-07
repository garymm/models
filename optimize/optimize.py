import optuna
import json

# TODO These are the steps for hyperparameter optimization
#  Find an optimizer package like optuna and import it
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

def main():
    # Load hypers from file
    hyperFile = "../test.json"
    # TODO Run go with -hyperFile cmd arg
    f = open(hyperFile)
    params = json.load(f)
    f.close()
    print(params)

    # Construct hypers object for optuna

    # Run one2many
    # Read logs

if __name__ == '__main__':
    main()
