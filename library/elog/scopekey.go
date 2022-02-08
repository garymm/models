// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elog

import "strings"

// ScopeKey the associated string representation of a scope or scopes.
// They include one or more EvalModes and one or more Times.
type ScopeKey string

// Like "Train&Test|Epoch&Trial"
var ScopeKeyBetweenModeAndTime = "&"
var ScopeKeyComma = "|"

// FromScopes create an associated scope merging the modes and times that are specified
// If you modify this, also modify ModesAndTimes, below.
func (sk *ScopeKey) FromScopes(modes []EvalModes, times []Times) {
	var mstr string
	var tstr string
	for _, mode := range modes {
		str := mode.String()
		if mstr == "" {
			mstr = str
		} else {
			mstr += ScopeKeyComma + str
		}
	}
	for _, time := range times {
		str := time.String()
		if tstr == "" {
			tstr = str
		} else {
			tstr += ScopeKeyComma + str
		}
	}
	*sk = ScopeKey(mstr + ScopeKeyBetweenModeAndTime + tstr)
}

// FromScope create an associated scope merging the modes and times that are specified
func (sk *ScopeKey) FromScope(mode EvalModes, time Times) {
	sk.FromScopes([]EvalModes{mode}, []Times{time})
}

// ModesAndTimes needs to be the inverse mirror of FromScopes
func (sk *ScopeKey) ModesAndTimes() (modes []EvalModes, times []Times) {
	skstr := strings.Split(string(*sk), ScopeKeyBetweenModeAndTime)
	modestr := skstr[0]
	timestr := skstr[1]
	modestrs := strings.Split(modestr, ScopeKeyComma)
	timestrs := strings.Split(timestr, ScopeKeyComma)
	for _, m := range modestrs {
		mo := UnknownEvalMode
		mo.FromString(m)
		modes = append(modes, mo)
	}
	for _, t := range timestrs {
		tim := UnknownTime
		tim.FromString(t)
		times = append(times, tim)
	}
	return modes, times
}

// FromScopesMap create an associated scope merging the modes and times that are specified
// If you modify this, also modify ModesAndTimesMap, below.
func (sk *ScopeKey) FromScopesMap(modes map[EvalModes]bool, times map[Times]bool) {
	ml := make([]EvalModes, len(modes))
	tl := make([]Times, len(times))
	idx := 0
	for m := range modes {
		ml[idx] = m
		idx++
	}
	idx = 0
	for t := range times {
		tl[idx] = t
		idx++
	}
	sk.FromScopes(ml, tl)
}

// ModesAndTimesMap returns maps of modes and times
func (sk *ScopeKey) ModesAndTimesMap() (modes map[EvalModes]bool, times map[Times]bool) {
	ml, tl := sk.ModesAndTimes()
	modes = make(map[EvalModes]bool)
	times = make(map[Times]bool)
	for _, m := range ml {
		modes[m] = true
	}
	for _, t := range tl {
		times[t] = true
	}
	return
}

func GenScopeKey(mode EvalModes, time Times) ScopeKey {
	ss := ScopeKey("")
	ss.FromScope(mode, time)
	return ss
}

func GenScopesKey(modes []EvalModes, times []Times) ScopeKey {
	ss := ScopeKey("")
	ss.FromScopes(modes, times)
	return ss
}

func GenScopesKeyMap(modes map[EvalModes]bool, times map[Times]bool) ScopeKey {
	ss := ScopeKey("")
	ss.FromScopesMap(modes, times)
	return ss
}

func GetScopeName(mode EvalModes, time Times) string {
	return mode.String() + time.String()
}
