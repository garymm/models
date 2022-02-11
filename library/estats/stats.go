package estats

import "fmt"

type Stats struct {
	FloatMetrics  map[string]float64
	StringMetrics map[string]string
	IntMetrics    map[string]int
}

func InitStats() Stats {
	stats := Stats{}
	stats.FloatMetrics = make(map[string]float64)
	stats.StringMetrics = make(map[string]string)
	stats.IntMetrics = make(map[string]int)
	return stats
}

func (stats *Stats) SetFloatMetric(name string, value float64) {
	stats.FloatMetrics[name] = value
}
func (stats *Stats) SetStringMetric(name string, value string) {
	stats.StringMetrics[name] = value
}
func (stats *Stats) SetIntMetric(name string, value int) {
	stats.IntMetrics[name] = value
}

func (stats *Stats) FloatMetric(name string) float64 {
	val, has := stats.FloatMetrics[name]
	if has {
		return val
	}
	fmt.Println("Value not found in map!")
	return 0
}
func (stats *Stats) StringMetric(name string) string {
	val, has := stats.StringMetrics[name]
	if has {
		return val
	}
	fmt.Println("Value not found in map!")
	return ""
}
func (stats *Stats) IntMetric(name string) int {
	val, has := stats.IntMetrics[name]
	if has {
		return val
	}
	fmt.Println("Value not found in map!")
	return 0
}
