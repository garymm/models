package sim

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/Astera-org/models/library/elog"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/goki/gi/gi"
)

var saveEpcLog bool
var saveRunLog bool
var saveNetData bool
var note string
var hyperFile string
var paramsFile string

// ParseArgs updates the Sim object with command line arguments.
func (ss *Sim) ParseArgs() {
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&saveNetData, "netdata", false, "if true, save network activation etc data from testing trials, for later viewing in netview")
	flag.BoolVar(&ss.NoGui, "nogui", len(os.Args) > 1, "if not passing any other args and want to run nogui, use nogui")
	flag.StringVar(&hyperFile, "hyperFile", "", "Name of the file to output hyperparameter data. If not empty string, program should write and then exit")
	flag.StringVar(&paramsFile, "paramsFile", "", "Name of the file to input parameters from.")
	flag.Parse()

	if hyperFile != "" {
		file, _ := json.MarshalIndent(ss.Params, "", "  ")
		_ = ioutil.WriteFile(hyperFile, file, 0644)
		// TODO This no longer prevents the run
		return
	}
	if paramsFile != "" {
		jsonFile, err := os.Open(paramsFile)
		if err != nil {
			fmt.Println(err)
			return
		}
		defer jsonFile.Close()
		byteValue, _ := ioutil.ReadAll(jsonFile)
		loadedParams := params.Sets{}
		json.Unmarshal(byteValue, &loadedParams)
		if len(loadedParams) == 0 {
			fmt.Println("Unable to load parameters from file: " + paramsFile)
			return
		}
		ss.Params = append(ss.Params, loadedParams[0])
	}
}

// RunFromArgs uses command line arguments to run the model.
func (ss *Sim) RunFromArgs() {
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		fnm := ss.LogFileName("epc")
		ss.Logs.SetLogFile(elog.Train, elog.Epoch, fnm)
	}
	if saveRunLog {
		fnm := ss.LogFileName("run")
		ss.Logs.SetLogFile(elog.Train, elog.Run, fnm)
	}
	if saveNetData {
		ss.NetData = &netview.NetData{}
		ss.NetData.Init(ss.Net, 200) // 200 = amount to save
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs starting at %d\n", ss.MaxRuns, ss.StartRun)
	(ss.TrainEnv).Run().Set(ss.StartRun)
	(ss.TrainEnv).Run().Max = ss.StartRun + ss.MaxRuns
	ss.NewRun()
	ss.Train()

	ss.Logs.CloseLogFiles()

	if saveNetData {
		ndfn := ss.Net.Nm + "_" + ss.RunName() + ".netdata.gz"
		ss.NetData.SaveJSON(gi.FileName(ndfn))
	}
}
