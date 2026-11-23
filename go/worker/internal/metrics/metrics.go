// Package metrics declares the worker's Prometheus metrics with exactly the
// names the Python worker exports, so dashboards and alerts work with both.
package metrics

import (
	"math"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// LatencyBucketsMs must equal LAT_BUCKETS_MS in echostream/serving/worker.py
// (a golden test checks it). The last bucket (+Inf) catches everything else.
var LatencyBucketsMs = []float64{
	0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750,
	1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 30000, 45000, 60000, 90000,
	120000, 180000, 300000, math.Inf(1),
}

// BucketIndex returns the first bucket whose upper bound is >= ms. NaN (a
// corrupt produced_ms) lands in the last bucket instead of crashing.
func BucketIndex(ms float64) int {
	for i, b := range LatencyBucketsMs {
		if ms <= b {
			return i
		}
	}
	return len(LatencyBucketsMs) - 1
}

func secondsBuckets() []float64 {
	out := make([]float64, 0, len(LatencyBucketsMs)-1)
	for _, b := range LatencyBucketsMs[:len(LatencyBucketsMs)-1] {
		out = append(out, b/1000)
	}
	return out
}

var (
	Applied = promauto.NewCounter(prometheus.CounterOpts{
		Name: "echostream_worker_events_applied_total", Help: "events applied to the feature store"})
	Duplicates = promauto.NewCounter(prometheus.CounterOpts{
		Name: "echostream_worker_events_duplicate_total", Help: "redeliveries skipped by dedupe"})
	Dead = promauto.NewCounter(prometheus.CounterOpts{
		Name: "echostream_worker_events_dlq_total", Help: "events sent to the dead-letter stream"})
	Retries = promauto.NewCounter(prometheus.CounterOpts{
		Name: "echostream_worker_retries_total", Help: "batches left pending because a feature node failed"})
	EventToFeature = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "echostream_event_to_feature_seconds", Help: "event produced -> feature updated",
		Buckets: secondsBuckets()})
)
