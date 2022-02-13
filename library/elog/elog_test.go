package elog

import (
	"testing"

	"github.com/emer/etable/etensor"
)

func TestScopeKeyStringing(t *testing.T) {
	sk := Scope(Train, Epoch)
	if sk != "Train&Epoch" {
		t.Errorf("Got unexpected scopekey " + string(sk))
	}
	sk2 := Scopes([]EvalModes{Train, Test}, []Times{Epoch, Cycle})
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
		Write: WriteMap{"Train|Test&Epoch|Cycle": func(ctx *Context) {
			// DO NOTHING
		}},
	}
	item.SetEachScopeKey()
	_, ok := item.WriteFunc("Train", "Epoch")
	if !ok {
		t.Errorf("Error getting compute function")
	}
	if item.HasMode(Validate) || item.HasTime(Run) {
		t.Errorf("Item has mode or time it shouldn't")
	}
}
