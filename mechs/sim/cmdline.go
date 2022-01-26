package sim

import (
	"flag"
	"fmt"
	"github.com/emer/emergent/netview"
	"github.com/goki/gi/gi"
	"log"
	"os"
)

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var saveNetData bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&saveNetData, "netdata", false, "if true, save network activation etc data from testing trials, for later viewing in netview")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %s\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %s\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if saveNetData {
		ss.NetData = &netview.NetData{}
		ss.NetData.Init(ss.Net, 200) // 200 = amount to save
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs starting at %d\n", ss.MaxRuns, ss.StartRun)
	ss.TrainEnv.Run.Set(ss.StartRun)
	ss.TrainEnv.Run.Max = ss.StartRun + ss.MaxRuns
	ss.NewRun()
	ss.Train()

	if saveNetData {
		ndfn := ss.Net.Nm + "_" + ss.RunName() + ".netdata.gz"
		ss.NetData.SaveJSON(gi.FileName(ndfn))
	}
}
