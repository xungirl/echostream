package worker

// These tests run real workers (in-process goroutines) against real
// redis-server processes on random ports, like the Python integration tests.
// They are skipped if redis-server is not installed.

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/xungirl/echostream/go/worker/internal/config"
	"github.com/xungirl/echostream/go/worker/internal/hashing"
)

type testCluster struct {
	cfg    config.Config
	events *redis.Client
	feats  map[string]*redis.Client
	procs  map[string]*exec.Cmd // "events", "feat0", ...
}

func redisServerPath(t *testing.T) string {
	if p, err := exec.LookPath("redis-server"); err == nil {
		return p
	}
	if _, err := os.Stat("/opt/homebrew/bin/redis-server"); err == nil {
		return "/opt/homebrew/bin/redis-server"
	}
	t.Skip("redis-server not installed")
	return ""
}

func portsFree(base, n int) bool {
	for i := 0; i < n; i++ {
		l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", base+20+i))
		if err != nil {
			return false
		}
		l.Close()
	}
	return true
}

// startCluster starts an event-bus Redis and nFeat feature-node Redis servers.
func startCluster(t *testing.T, nFeat, nPartitions int) *testCluster {
	t.Helper()
	bin := redisServerPath(t)
	var base int
	for {
		base = 30000 + 100*rand.Intn(300)
		if portsFree(base, nFeat+1) {
			break
		}
	}
	c := &testCluster{feats: map[string]*redis.Client{}, procs: map[string]*exec.Cmd{}}
	c.cfg = config.Config{
		PortBase: base, NPartitions: nPartitions, NFeatureNodes: nFeat, StreamPrefix: "events", Group: "workers",
		DLQStream: "events:dlq", DedupeTTLSec: 3600, HistMax: 100, LeaseMs: 600, ClaimMinIdleMs: 200, Batch: 50,
		PositiveThresh: "2.0",
	}
	for i := 0; i < nFeat; i++ {
		c.cfg.FeatureNodeName = append(c.cfg.FeatureNodeName, fmt.Sprintf("feat%d", i))
	}
	start := func(name string, port int) *redis.Client {
		cmd := exec.Command(bin, "--port", strconv.Itoa(port), "--bind", "127.0.0.1", "--save", "",
			"--appendonly", "no", "--protected-mode", "no")
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		c.procs[name] = cmd
		t.Cleanup(func() {
			_ = cmd.Process.Signal(syscall.SIGCONT)
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		})
		cl := redis.NewClient(&redis.Options{Addr: fmt.Sprintf("127.0.0.1:%d", port), Protocol: 2})
		t.Cleanup(func() { cl.Close() })
		waitFor(t, 5*time.Second, func() bool { return cl.Ping(context.Background()).Err() == nil })
		return cl
	}
	c.events = start("events", base+20)
	for i, name := range c.cfg.FeatureNodeName {
		c.feats[name] = start(name, base+21+i)
	}
	return c
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("condition not met after %v", timeout)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// runWorker starts a worker in the background; the returned stop func shuts
// it down (gracefully when release is true, like SIGTERM; otherwise like a
// crash that leaves its leases and pending entries behind).
func runWorker(t *testing.T, c *testCluster, id string) (w *Worker, stop func(release bool)) {
	w = New(id, c.cfg)
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	stopped := false
	stop = func(release bool) {
		if stopped {
			return
		}
		stopped = true
		w.Shutdown()
		if err := <-done; err != nil {
			t.Errorf("worker %s: %v", id, err)
		}
		if release {
			w.ReleaseLeases(context.Background())
		}
		w.Close()
	}
	t.Cleanup(func() { stop(true) })
	return w, stop
}

type ev struct {
	id     string
	user   int64
	item   int
	wr     float64
	action string
	ts     float64
}

func (c *testCluster) produce(t *testing.T, evs []ev) {
	ctx := context.Background()
	pipe := c.events.Pipeline()
	for _, e := range evs {
		stream := c.cfg.StreamName(hashing.PartitionFor(e.user, c.cfg.NPartitions))
		pipe.XAdd(ctx, &redis.XAddArgs{Stream: stream, Values: []any{
			"event_id", e.id, "user_id", strconv.FormatInt(e.user, 10), "item_id", strconv.Itoa(e.item),
			"action", e.action, "watch_ratio", strconv.FormatFloat(e.wr, 'f', -1, 64),
			"ts", strconv.FormatFloat(e.ts, 'f', 1, 64), "cats", "1,2",
			"produced_ms", strconv.FormatFloat(float64(time.Now().UnixNano())/1e6, 'f', 3, 64)}})
	}
	if _, err := pipe.Exec(ctx); err != nil {
		t.Fatal(err)
	}
}

// makeEvents: per user a sequence 0..perUser-1 (ts = seq, so order is checkable),
// interleaved across users; every dupEvery-th event is sent twice (a client retry).
func makeEvents(users []int64, perUser int, prefix string, dupEvery int) []ev {
	var out []ev
	for seq := 0; seq < perUser; seq++ {
		for _, u := range users {
			e := ev{fmt.Sprintf("%s-%d-%d", prefix, u, seq), u, seq % 50, 0.1, "skip", float64(seq)}
			if seq%3 == 0 {
				e.wr, e.action = 2.5, "complete"
			}
			out = append(out, e)
			if dupEvery > 0 && seq%dupEvery == 0 {
				out = append(out, e)
			}
		}
	}
	return out
}

func (c *testCluster) drained() bool {
	ctx := context.Background()
	for p := 0; p < c.cfg.NPartitions; p++ {
		n, err := c.events.XLen(ctx, c.cfg.StreamName(p)).Result()
		if err != nil || n > 0 {
			return false
		}
		pend, err := c.events.XPending(ctx, c.cfg.StreamName(p), c.cfg.Group).Result()
		if err != nil || pend.Count > 0 {
			return false
		}
	}
	return true
}

// checkUser: no loss, no double count, per-user order.
func (c *testCluster) checkUser(t *testing.T, u int64, perUser int) {
	t.Helper()
	ctx := context.Background()
	node := hashing.NewRing(c.cfg.FeatureNodeName, 100).NodeForUser(u)
	cnt, err := c.feats[node].HGetAll(ctx, fmt.Sprintf("rt:%d:cnt", u)).Result()
	if err != nil {
		t.Fatal(err)
	}
	if cnt["n"] != strconv.Itoa(perUser) {
		t.Fatalf("user %d: n=%s, want %d (%v)", u, cnt["n"], perUser, cnt)
	}
	hist, _ := c.feats[node].LRange(ctx, fmt.Sprintf("rt:%d:hist", u), 0, -1).Result()
	for i := range hist { // newest first
		seq := len(hist) - 1 - i
		if parts := strings.Split(hist[i], ":"); parts[2] != fmt.Sprintf("%d.0", seq) {
			t.Fatalf("user %d: history out of order at %d: %v", u, i, hist)
		}
	}
}

func statSum(t *testing.T, c *testCluster, field string, ids ...string) float64 {
	total := 0.0
	for _, id := range ids {
		v, err := c.events.HGet(context.Background(), "stats:worker:"+id, field).Float64()
		if err == nil {
			total += v
		}
	}
	return total
}

func users(from, n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(from + i)
	}
	return out
}

func TestNoLossNoDoubleCountOrderWithDuplicatesAndPoison(t *testing.T) {
	c := startCluster(t, 3, 4)
	us := users(1000, 40)
	c.produce(t, makeEvents(us, 30, "dup", 4))
	ctx := context.Background()
	s := c.cfg.StreamName(hashing.PartitionFor(1000, 4))
	c.events.XAdd(ctx, &redis.XAddArgs{Stream: s, Values: []any{"event_id", "bad-1", "user_id", "not-a-number"}})
	c.events.XAdd(ctx, &redis.XAddArgs{Stream: s, Values: []any{"garbage", "x"}})
	c.events.XAdd(ctx, &redis.XAddArgs{Stream: s, Values: []any{"event_id", "\xff\xfe", "user_id", "1"}})

	_, stop := runWorker(t, c, "w0")
	waitFor(t, 20*time.Second, c.drained)
	time.Sleep(200 * time.Millisecond) // one more heartbeat publishes the stats
	for _, u := range us {
		c.checkUser(t, u, 30)
	}
	waitFor(t, 5*time.Second, func() bool { return statSum(t, c, "applied", "w0") == 40*30 })
	if d := statSum(t, c, "duplicates", "w0"); d != 40*8 { // seq 0,4,...,28 -> 8 re-sends per user
		t.Fatalf("duplicates=%v", d)
	}
	dlq, _ := c.events.XRange(ctx, "events:dlq", "-", "+").Result()
	if len(dlq) != 3 {
		t.Fatalf("dlq has %d entries", len(dlq))
	}
	reasons := []string{dlq[0].Values["_reason"].(string), dlq[1].Values["_reason"].(string), dlq[2].Values["_reason"].(string)}
	want := []string{"missing fields ['item_id', 'action', 'watch_ratio', 'ts', 'produced_ms']",
		"missing fields ['event_id', 'user_id', 'item_id', 'action', 'watch_ratio', 'ts', 'produced_ms']",
		"fields are not valid UTF-8"}
	for i := range want {
		if reasons[i] != want[i] {
			t.Errorf("dlq reason %d = %q, want %q", i, reasons[i], want[i])
		}
	}
	if dlq[2].Values["event_id"] != "\xff\xfe" {
		t.Errorf("raw bytes not kept in the DLQ copy: %q", dlq[2].Values["event_id"])
	}
	stop(true)
	// graceful shutdown released every lease and left the alive set
	for p := 0; p < 4; p++ {
		if n, _ := c.events.Exists(ctx, leaseKey(p)).Result(); n != 0 {
			t.Errorf("lease:%d still held after graceful shutdown", p)
		}
	}
	if _, err := c.events.ZScore(ctx, "workers:alive", "w0").Result(); err != redis.Nil {
		t.Errorf("w0 still in workers:alive")
	}
}

func TestPendingEntriesOfDeadConsumerAreClaimed(t *testing.T) {
	c := startCluster(t, 2, 3)
	ctx := context.Background()
	for p := 0; p < 3; p++ {
		c.events.XGroupCreateMkStream(ctx, c.cfg.StreamName(p), c.cfg.Group, "0")
	}
	us := users(5000, 20)
	c.produce(t, makeEvents(us, 15, "ghost", 0))
	read := 0
	for p := 0; p < 3; p++ { // a consumer reads everything and dies before XACK
		res, err := c.events.XReadGroup(ctx, &redis.XReadGroupArgs{Group: c.cfg.Group, Consumer: "ghost",
			Streams: []string{c.cfg.StreamName(p), ">"}, Count: 10000}).Result()
		if err != nil {
			t.Fatal(err)
		}
		read += len(res[0].Messages)
	}
	if read != 20*15 {
		t.Fatalf("ghost read %d", read)
	}
	runWorker(t, c, "w0")
	waitFor(t, 20*time.Second, c.drained)
	for _, u := range us {
		c.checkUser(t, u, 15)
	}
	waitFor(t, 5*time.Second, func() bool { return statSum(t, c, "claimed", "w0") == float64(read) })
}

func TestLeasesRebalanceAndTakeOverAfterCrash(t *testing.T) {
	c := startCluster(t, 2, 6)
	w0, _ := runWorker(t, c, "w0")
	w1, stop1 := runWorker(t, c, "w1")
	waitFor(t, 10*time.Second, func() bool { return w0.ownedCount() == 3 && w1.ownedCount() == 3 })

	us := users(3000, 30)
	evs := makeEvents(us, 40, "crash", 0)
	c.produce(t, evs[:len(evs)/2])
	time.Sleep(50 * time.Millisecond)
	stop1(false) // crash: leases and pending entries stay behind
	c.produce(t, evs[len(evs)/2:])
	waitFor(t, 10*time.Second, func() bool { return w0.ownedCount() == 6 }) // after the lease TTL
	waitFor(t, 20*time.Second, c.drained)
	for _, u := range us {
		c.checkUser(t, u, 40)
	}
}

func TestFeatureNodeFreezeKeepsOrderAndRecovers(t *testing.T) {
	c := startCluster(t, 2, 2)
	w, _ := runWorker(t, c, "w0")
	waitFor(t, 5*time.Second, func() bool { return w.ownedCount() == 2 })
	frozen := c.procs["feat1"].Process
	if err := frozen.Signal(syscall.SIGSTOP); err != nil {
		t.Fatal(err)
	}
	us := users(4000, 20)
	c.produce(t, makeEvents(us, 20, "stall", 0))
	time.Sleep(1500 * time.Millisecond) // events for feat1 users keep failing and stay pending
	if err := frozen.Signal(syscall.SIGCONT); err != nil {
		t.Fatal(err)
	}
	waitFor(t, 30*time.Second, c.drained)
	for _, u := range us {
		c.checkUser(t, u, 20)
	}
	waitFor(t, 5*time.Second, func() bool { return statSum(t, c, "retries", "w0") > 0 })
}
