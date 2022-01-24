package main

import (
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"log"
)

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

func (ss *Sim) ConfigPatsFromEnv() {

	dt := ss.Pats
	dt.SetMetaData("name", "SuccessorPatterns")
	dt.SetMetaData("desc", "SuccessorPatterns")
	sch := etable.Schema{
		{"Word", etensor.STRING, nil, nil},
		{"Pattern", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, ss.NInputs*ss.NOutputs)

	i := 0
	for _, word := range ss.TrainEnv.Words {
		idx := ss.TrainEnv.WordMap[word]
		mytensor := ss.TrainEnv.WordReps.SubSpace([]int{idx})
		dt.SetCellString("Word", i, word)
		dt.SetCellTensor("Pattern", i, mytensor)
		i++

	}

	dt.SaveCSV("random_5x5_25_gen.tsv", etable.Tab, etable.Headers)
}
