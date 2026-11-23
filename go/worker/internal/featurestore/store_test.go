package featurestore

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/xungirl/echostream/go/worker/internal/events"
	"github.com/xungirl/echostream/go/worker/internal/golden"
	"github.com/xungirl/echostream/go/worker/internal/metrics"
)

type workerGolden struct {
	ApplyEventLua string `json:"apply_event_lua"`
	LatBucketsMs  []any  `json:"lat_buckets_ms"`
}

func TestLuaAndBucketsMatchPython(t *testing.T) {
	var g workerGolden
	golden.Load(t, "worker_golden.json", &g)
	if ApplyEventLua != g.ApplyEventLua {
		t.Fatal("apply_event.lua differs from APPLY_EVENT_LUA in feature_store.py")
	}
	if len(g.LatBucketsMs) != len(metrics.LatencyBucketsMs) {
		t.Fatalf("%d buckets, Python has %d", len(metrics.LatencyBucketsMs), len(g.LatBucketsMs))
	}
	for i, b := range g.LatBucketsMs {
		want := math.Inf(1)
		if f, ok := b.(float64); ok {
			want = f
		}
		if metrics.LatencyBucketsMs[i] != want {
			t.Fatalf("bucket %d: %v, Python %v", i, metrics.LatencyBucketsMs[i], want)
		}
	}
}

func TestApplyArgsFormatsLikeRedisPy(t *testing.T) {
	ev := events.Event{EventID: "kr7", UserID: 42, ItemID: 9, Action: "complete", WatchRatio: 2.0,
		Ts: 1593878903.438, Cats: "3,5"}
	c := ApplyArgs(ev, 3600, 100, "2.0")
	if got := fmt.Sprint(c.Keys); got != "[ev:kr7 rt:42:hist rt:42:cnt rec:42]" {
		t.Fatal(got)
	}
	if got := fmt.Sprint(c.Args); got != "[3600 9 2.0 1593878903.438 complete 3,5 100 2.0]" {
		t.Fatal(got)
	}
}

type fakeRedisErr string

func (e fakeRedisErr) Error() string { return string(e) }
func (fakeRedisErr) RedisError()     {}

func TestPyErrorMessageStripsKnownCodes(t *testing.T) {
	cases := map[string]string{
		"ERR value is not a valid float script: abc, on @user_script:10.": "value is not a valid float script: abc, on @user_script:10.",
		"BUSY Redis is busy running a script":                             "BUSY Redis is busy running a script",
		"OOM command not allowed":                                         "command not allowed",
	}
	for in, want := range cases {
		if got := PyErrorMessage(fakeRedisErr(in)); got != want {
			t.Errorf("%q -> %q, want %q", in, got, want)
		}
	}
	if !isServerError(fakeRedisErr("ERR x")) || isServerError(errors.New("dial tcp: connection refused")) {
		t.Fatal("isServerError")
	}
}
