package estats

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
	return stats.FloatMetrics[name]
}
func (stats *Stats) StringMetric(name string) string {
	return stats.StringMetrics[name]
}
func (stats *Stats) IntMetric(name string) int {
	return stats.IntMetrics[name]
}
