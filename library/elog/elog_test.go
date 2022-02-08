package elog

import (
	"testing"

	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

func TestScopeKeyStringing(t *testing.T) {
	sk := GenScopeKey(Train, Epoch)
	if sk != "Train&Epoch" {
		t.Errorf("Got unexpected scopekey " + string(sk))
	}
	sk2 := GenScopesKey([]EvalModes{Train, Test}, []Times{Epoch, Cycle})
	if sk2 != "Train|Test&Epoch|Cycle" {
		t.Errorf("Got unexpected scopekey " + string(sk2))
	}
	modes, times := sk2.ModesAndTimes()
	if len(modes) != 2 || len(times) != 2 {
		t.Errorf("Error parsing scopekey")
	}
}

func TestItem(t *testing.T) {
	item := Item{
		Name: "Testo",
		Type: etensor.STRING,
		Compute: ComputeMap{"Train|Test&Epoch|Cycle": func(item *Item, scope ScopeKey, dt *etable.Table, row int) {
			// DO NOTHING
		}},
	}
	item.SetEachScopeKey()
	_, ok := item.ComputeFunc("Train", "Epoch")
	if !ok {
		t.Errorf("Error getting compute function")
	}
	if item.HasMode(Validate) || item.HasTime(Run) {
		t.Errorf("Item has mode or time it shouldn't")
	}
}
