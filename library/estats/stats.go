package estats

type Stats struct {
	FloatMetrics  map[string]float64
	StringMetrics map[string]string
	IntMetrics    map[string]int
}

func (stats *Stats) Init() {
	floatMetrics := make(map[string]float64)
	stringMetrics := make(map[string]string)
	intMetrics := make(map[string]int)
	stats.FloatMetrics = floatMetrics
	stats.StringMetrics = stringMetrics
	stats.IntMetrics = intMetrics
}

func (stats *Stats) SetFloatMetric(name string)  {}
func (stats *Stats) SetStringMetric(name string) {}
func (stats *Stats) SetIntMetric(name string)    {}

func (stats *Stats) FloatMetric(name string) float64 {
	return stats.FloatMetrics[name]
}
func (stats *Stats) StringMetric(name string) string {
	return stats.StringMetrics[name]
}
func (stats *Stats) IntMetric(name string) int {
	return stats.IntMetrics[name]
}
