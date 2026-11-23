// Package worker is the Go port of echostream/serving/worker.py: a Redis
// Streams consumer with partition leases. The algorithm is deliberately the
// same, step by step, so both workers can run side by side in one cluster.
//
// Guarantees and how they are achieved
//
//   - at-least-once: an entry is XACKed only after its effect is stored; a
//     crashed worker's un-acked entries stay in the group's pending list (PEL)
//     and are taken over with XAUTOCLAIM by whoever gets the partition next.
//   - no double counting: the effect is one Lua script that first sets a
//     dedupe key ev:{event_id} (SET NX), so a redelivery is a no-op.
//   - per-user order: a user always maps to the same partition, and each
//     partition has exactly one owner at a time (a lease key with a TTL,
//     renewed by a heartbeat). The owner handles its pending entries first,
//     then new ones, in stream order; if a feature node is down, that node's
//     events stay pending and nothing newer for that node is applied first.
//   - poison events: malformed entries or deterministic script errors go to
//     the dead-letter stream (events:dlq) and are acked, so they cannot block
//     a partition.
//
// Concurrency: two goroutines (lease loop, process loop) share the set of
// owned partitions and the stats, guarded by mu. Inside one batch the
// per-feature-node pipelines run in parallel goroutines.
package worker

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/xungirl/echostream/go/worker/internal/archive"
	"github.com/xungirl/echostream/go/worker/internal/config"
	"github.com/xungirl/echostream/go/worker/internal/events"
	"github.com/xungirl/echostream/go/worker/internal/featurestore"
	"github.com/xungirl/echostream/go/worker/internal/metrics"
	"github.com/xungirl/echostream/go/worker/internal/pycompat"
)

// Lease scripts: only the current owner may renew or release a lease.
const (
	renewLua   = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('PEXPIRE', KEYS[1], ARGV[2]) else return 0 end"
	releaseLua = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"
)

// Worker consumes the partitions it holds a lease for.
type Worker struct {
	id      string
	cfg     config.Config
	r       *redis.Client // event bus
	fs      *featurestore.Store
	archive *archive.Archiver // nil unless ECHO_ARCHIVE=1

	mu         sync.Mutex // guards everything below
	owned      map[int]bool
	needsClaim map[int]bool
	stats      map[string]float64 // like Python's defaultdict(float): a key exists once touched
	latHist    []int64

	stop chan struct{} // closed by Shutdown
	once sync.Once
}

// New creates a worker; connections are opened lazily.
func New(id string, cfg config.Config) *Worker {
	w := &Worker{
		id:  id,
		cfg: cfg,
		r: redis.NewClient(&redis.Options{
			Addr: cfg.EventsRedisAddr(), Protocol: 2,
			DialTimeout: 2 * time.Second, ReadTimeout: 2 * time.Second, WriteTimeout: 2 * time.Second,
		}),
		fs:         featurestore.New(cfg.FeatureNodeAddrs(), cfg.FeatureNodeName),
		owned:      map[int]bool{},
		needsClaim: map[int]bool{},
		stats:      map[string]float64{},
		latHist:    make([]int64, len(metrics.LatencyBucketsMs)),
		stop:       make(chan struct{}),
	}
	if cfg.Archive {
		w.archive = archive.New(cfg.PostgresDSN)
	}
	return w
}

// ------------------------------------------------------------------ setup

func (w *Worker) ensureGroups(ctx context.Context) error {
	for p := 0; p < w.cfg.NPartitions; p++ {
		err := w.r.XGroupCreateMkStream(ctx, w.cfg.StreamName(p), w.cfg.Group, "0").Err()
		if err != nil && !strings.Contains(err.Error(), "BUSYGROUP") {
			return err
		}
	}
	return nil
}

// Run blocks until Shutdown is called (returns nil) or a non-transient error
// happens (returns it, so main exits and the supervisor restarts the process).
func (w *Worker) Run(ctx context.Context) error {
	// Redis may still be loading its AOF or not listening yet: retry for up to ~2 minutes
	for attempt := 0; attempt < 600; attempt++ {
		if err := w.ensureGroups(ctx); err == nil {
			break
		}
		if w.sleep(200 * time.Millisecond) {
			return nil
		}
	}
	errs := make(chan error, 2)
	go func() { errs <- w.leaseLoop(ctx) }()
	go func() { errs <- w.processLoop(ctx) }()
	first := <-errs // a loop only returns early on a fatal error or on stop
	w.Shutdown()    // make the other loop stop too
	second := <-errs
	if first != nil {
		return first
	}
	return second
}

// Shutdown asks both loops to stop after their current step.
func (w *Worker) Shutdown() { w.once.Do(func() { close(w.stop) }) }

// ReleaseLeases gives every partition back right away (graceful shutdown), so
// other workers can take over without waiting for the lease TTL.
func (w *Worker) ReleaseLeases(ctx context.Context) {
	for _, p := range w.ownedSorted() {
		w.r.Eval(ctx, releaseLua, []string{leaseKey(p)}, w.id)
	}
	w.r.ZRem(ctx, "workers:alive", w.id)
}

// Close closes all connections.
func (w *Worker) Close() {
	w.fs.Close()
	_ = w.r.Close()
	if w.archive != nil {
		w.archive.Close()
	}
}

// sleep waits d or until stop; it reports whether the worker is stopping.
func (w *Worker) sleep(d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-w.stop:
		return true
	case <-t.C:
		return false
	}
}

func (w *Worker) stopping() bool {
	select {
	case <-w.stop:
		return true
	default:
		return false
	}
}

// ----------------------------------------------------------------- leases

func leaseKey(p int) string { return fmt.Sprintf("lease:%d", p) }

func (w *Worker) leaseLoop(ctx context.Context) error {
	for !w.stopping() {
		if err := w.balanceLeases(ctx); err != nil && !isTransient(err) {
			return err
		}
		if w.sleep(time.Duration(w.cfg.LeaseMs) * time.Millisecond / 6) {
			return nil
		}
	}
	return nil
}

// balanceLeases is one heartbeat: announce we are alive, renew our leases,
// give partitions back if we hold more than our fair share, take free ones if
// we hold fewer, and publish our stats.
func (w *Worker) balanceLeases(ctx context.Context) error {
	now := nowMs()
	if err := w.r.ZAdd(ctx, "workers:alive", redis.Z{Score: now, Member: w.id}).Err(); err != nil {
		return err
	}
	if err := w.r.ZRemRangeByScore(ctx, "workers:alive", "0", pycompat.FormatFloat(now-float64(w.cfg.LeaseMs))).Err(); err != nil {
		return err
	}
	alive, err := w.r.ZCard(ctx, "workers:alive").Result()
	if err != nil {
		return err
	}
	fair := int(math.Ceil(float64(w.cfg.NPartitions) / float64(max(alive, 1))))

	for _, p := range w.ownedSorted() { // renew; drop leases we lost
		ok, err := w.r.Eval(ctx, renewLua, []string{leaseKey(p)}, w.id, w.cfg.LeaseMs).Int64()
		if err != nil {
			return err
		}
		if ok == 0 {
			w.setOwned(p, false)
		}
	}
	for { // a new worker joined: give partitions back, highest first
		owned := w.ownedSorted()
		if len(owned) <= fair {
			break
		}
		p := owned[len(owned)-1]
		w.setOwned(p, false)
		if err := w.r.Eval(ctx, releaseLua, []string{leaseKey(p)}, w.id).Err(); err != nil {
			return err
		}
	}
	if w.ownedCount() < fair {
		// start at a worker-specific offset so workers don't all race for partition 0
		start := 0
		for _, r := range w.id {
			start += int(r)
		}
		start %= w.cfg.NPartitions
		for k := 0; k < w.cfg.NPartitions && w.ownedCount() < fair; k++ {
			p := (start + k) % w.cfg.NPartitions
			if w.isOwned(p) {
				continue
			}
			got, err := w.r.SetNX(ctx, leaseKey(p), w.id, time.Duration(w.cfg.LeaseMs)*time.Millisecond).Result()
			if err != nil {
				return err
			}
			if got {
				w.mu.Lock()
				w.owned[p] = true
				w.needsClaim[p] = true
				w.mu.Unlock()
			}
		}
	}
	return w.r.HSet(ctx, "stats:worker:"+w.id, w.statsSnapshot(now)).Err()
}

// statsSnapshot renders the stats hash exactly like the Python worker
// (floats via repr(), "partitions" and "lat_hist" as comma-separated lists).
func (w *Worker) statsSnapshot(now float64) []any {
	w.mu.Lock()
	defer w.mu.Unlock()
	keys := make([]string, 0, len(w.stats))
	for k := range w.stats {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]any, 0, 2*len(keys)+8)
	for _, k := range keys {
		out = append(out, k, pycompat.FormatFloat(w.stats[k]))
	}
	parts := make([]string, 0, len(w.owned))
	for _, p := range sortedKeys(w.owned) {
		parts = append(parts, strconv.Itoa(p))
	}
	hist := make([]string, len(w.latHist))
	for i, n := range w.latHist {
		hist[i] = strconv.FormatInt(n, 10)
	}
	return append(out,
		"partitions", strings.Join(parts, ","),
		"lat_hist", strings.Join(hist, ","),
		"updated_ms", pycompat.FormatFloat(now),
		"pid", strconv.Itoa(os.Getpid()))
}

// ------------------------------------------------------------- processing

// claim takes over pending entries left by a previous owner. It returns true
// once no other consumer holds pending entries for this partition; until then
// we must not read new entries (they would overtake the old ones).
func (w *Worker) claim(ctx context.Context, p int) (bool, error) {
	stream := w.cfg.StreamName(p)
	cursor := "0-0"
	for {
		msgs, next, err := w.r.XAutoClaim(ctx, &redis.XAutoClaimArgs{
			Stream: stream, Group: w.cfg.Group, Consumer: w.id,
			MinIdle: time.Duration(w.cfg.ClaimMinIdleMs) * time.Millisecond, Start: cursor, Count: 1000,
		}).Result()
		if err != nil {
			return false, err
		}
		w.addStat("claimed", float64(len(msgs)))
		cursor = next
		if cursor == "0-0" {
			break
		}
	}
	summary, err := w.r.XPending(ctx, stream, w.cfg.Group).Result()
	if err != nil {
		return false, err
	}
	for name, n := range summary.Consumers {
		if name != w.id && n > 0 {
			return false, nil // previous owner's entries not idle long enough yet
		}
	}
	w.mu.Lock()
	delete(w.needsClaim, p)
	w.mu.Unlock()
	return true, nil
}

func (w *Worker) processLoop(ctx context.Context) error {
	for !w.stopping() {
		did, err := w.processOnce(ctx)
		if err != nil {
			if !isTransient(err) {
				return err
			}
			did = false
			if w.sleep(200 * time.Millisecond) {
				return nil
			}
		}
		if !did && w.sleep(5*time.Millisecond) {
			return nil
		}
	}
	return nil
}

// blockedNodes records, per partition, the feature nodes that failed in this
// round: newer events for those nodes must wait so they don't overtake.
type blockedNodes map[int]map[string]bool

func (b blockedNodes) add(p int, node string) {
	if b[p] == nil {
		b[p] = map[string]bool{}
	}
	b[p][node] = true
}

func (w *Worker) processOnce(ctx context.Context) (bool, error) {
	owned := w.ownedSorted()
	if len(owned) == 0 {
		w.sleep(100 * time.Millisecond)
		return false, nil
	}
	ready := make([]int, 0, len(owned))
	for _, p := range owned {
		if w.isNeedsClaim(p) {
			ok, err := w.claim(ctx, p)
			if err != nil {
				return false, err
			}
			if !ok {
				continue
			}
		}
		ready = append(ready, p)
	}
	if len(ready) == 0 {
		w.sleep(50 * time.Millisecond)
		return false, nil
	}
	streams := make([]string, len(ready))
	for i, p := range ready {
		streams[i] = w.cfg.StreamName(p)
	}
	blocked := blockedNodes{}
	did := false
	// 1) our own pending entries first (retries, claimed entries), in stream order
	pending, err := readGroup(ctx, w.r, w.cfg.Group, w.id, streams, "0", w.cfg.Batch, -1)
	if err != nil {
		return false, err
	}
	for _, sb := range pending {
		if len(sb.Entries) > 0 {
			did = true
			if err := w.handle(ctx, sb, blocked); err != nil {
				return did, err
			}
		}
	}
	// 2) new entries
	fresh, err := readGroup(ctx, w.r, w.cfg.Group, w.id, streams, ">", w.cfg.Batch, 50*time.Millisecond)
	if err != nil {
		return did, err
	}
	for _, sb := range fresh {
		if len(sb.Entries) > 0 {
			did = true
			if err := w.handle(ctx, sb, blocked); err != nil {
				return did, err
			}
		}
	}
	return did, nil
}

type item struct {
	entry events.Entry
	ev    events.Event
}

// handle applies one stream's batch: validate, group by feature node, run one
// pipeline per node in parallel, then dead-letter / count / archive / ack.
func (w *Worker) handle(ctx context.Context, sb streamBatch, blocked blockedNodes) error {
	p, err := strconv.Atoi(sb.Stream[strings.LastIndex(sb.Stream, ":")+1:])
	if err != nil {
		return err
	}
	if !w.isOwned(p) { // lease lost meanwhile: leave entries for the new owner
		return nil
	}
	var toAck []string
	var nodeOrder []string // first-seen order, like a Python dict
	byNode := map[string][]item{}
	for _, e := range sb.Entries {
		if e.Fields == nil { // entry was deleted (already processed + XDEL'd)
			toAck = append(toAck, e.ID)
			continue
		}
		ev, err := events.Parse(e.Fields)
		if err != nil {
			if derr := w.deadLetter(ctx, sb.Stream, e, err.Error()); derr != nil {
				return derr
			}
			toAck = append(toAck, e.ID)
			continue
		}
		node := w.fs.NodeFor(ev.UserID)
		if blocked[p][node] {
			continue // keep pending -> preserves order behind the failed event
		}
		if _, seen := byNode[node]; !seen {
			nodeOrder = append(nodeOrder, node)
		}
		byNode[node] = append(byNode[node], item{e, ev})
	}

	// one pipeline per feature node, all nodes in parallel (asyncio.gather in Python)
	type nodeResult struct {
		results []featurestore.Result
		err     error
	}
	out := make([]nodeResult, len(nodeOrder))
	var wg sync.WaitGroup
	for i, node := range nodeOrder {
		wg.Add(1)
		go func() {
			defer wg.Done()
			calls := make([]featurestore.Call, len(byNode[node]))
			for j, it := range byNode[node] {
				calls[j] = featurestore.ApplyArgs(it.ev, w.cfg.DedupeTTLSec, w.cfg.HistMax, w.cfg.PositiveThresh)
			}
			res, err := w.fs.ApplyBatch(ctx, node, calls)
			out[i] = nodeResult{res, err}
		}()
	}
	wg.Wait()

	now := nowMs()
	var rows []archive.Row
	archived := map[string]bool{}
	for i, node := range nodeOrder {
		if out[i].err != nil {
			blocked.add(p, node) // transient: retry later, keep order
			metrics.Retries.Inc()
			w.addStat("retries", 1)
			continue
		}
		for j, it := range byNode[node] {
			res := out[i].results[j]
			switch res.Outcome {
			case featurestore.ScriptError:
				if err := w.deadLetter(ctx, sb.Stream, it.entry, "script error: "+res.ErrMsg); err != nil {
					return err
				}
			case featurestore.Applied:
				metrics.Applied.Inc()
				lat := now - it.ev.ProducedMs
				metrics.EventToFeature.Observe(lat / 1000)
				w.mu.Lock()
				w.stats["applied"]++
				w.latHist[metrics.BucketIndex(lat)]++
				w.mu.Unlock()
			default:
				metrics.Duplicates.Inc()
				w.addStat("duplicates", 1)
			}
			if w.archive != nil && res.Outcome != featurestore.ScriptError {
				ev := it.ev
				rows = append(rows, archive.Row{EventID: ev.EventID, UserID: ev.UserID, ItemID: ev.ItemID,
					Action: ev.Action, WatchRatio: ev.WatchRatio, Ts: ev.Ts, Cats: ev.Cats, Partition: p,
					StreamID: it.entry.ID})
				archived[it.entry.ID] = true
			}
			toAck = append(toAck, it.entry.ID)
		}
	}
	if len(rows) > 0 && !w.writeArchive(ctx, rows) {
		// Postgres unavailable: features are already applied (idempotent), but we keep the
		// entries pending so the archive write is retried -> the archive never misses an event
		kept := toAck[:0]
		for _, id := range toAck {
			if !archived[id] {
				kept = append(kept, id)
			}
		}
		toAck = kept
		for _, node := range w.fs.Ring.Nodes() { // keep order: nothing newer in this partition this round
			blocked.add(p, node)
		}
	}
	if len(toAck) > 0 {
		pipe := w.r.Pipeline()
		pipe.XAck(ctx, sb.Stream, w.cfg.Group, toAck...)
		pipe.XDel(ctx, sb.Stream, toAck...) // keep the stream bounded: processed entries are gone
		if _, err := pipe.Exec(ctx); err != nil {
			return err
		}
	}
	return nil
}

func (w *Worker) writeArchive(ctx context.Context, rows []archive.Row) bool {
	if err := w.archive.Write(ctx, rows); err != nil {
		w.addStat("archive_failures", 1)
		return false
	}
	w.addStat("archived", float64(len(rows)))
	return true
}

// deadLetter copies the entry to events:dlq with the reason and acks nothing
// itself (the caller acks). Fields keep their order; the _source/_entry/
// _reason/_worker fields overwrite same-named fields like Python's {**a, ...}.
func (w *Worker) deadLetter(ctx context.Context, stream string, e events.Entry, reason string) error {
	fields := append([]events.Field(nil), e.Fields...)
	fields = setField(fields, "_source", stream)
	fields = setField(fields, "_entry", e.ID)
	fields = setField(fields, "_reason", events.Truncate(reason, 200))
	fields = setField(fields, "_worker", w.id)
	values := make([]any, 0, 2*len(fields))
	for _, f := range fields {
		values = append(values, f.Key, f.Value)
	}
	if err := w.r.XAdd(ctx, &redis.XAddArgs{Stream: w.cfg.DLQStream, Values: values}).Err(); err != nil {
		return err
	}
	metrics.Dead.Inc()
	w.addStat("dlq", 1)
	return nil
}

// ------------------------------------------------------------ small helpers

func (w *Worker) addStat(k string, v float64) {
	w.mu.Lock()
	w.stats[k] += v
	w.mu.Unlock()
}

func (w *Worker) ownedSorted() []int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return sortedKeys(w.owned)
}

func (w *Worker) ownedCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.owned)
}

func (w *Worker) isOwned(p int) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.owned[p]
}

func (w *Worker) isNeedsClaim(p int) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.needsClaim[p]
}

func (w *Worker) setOwned(p int, v bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if v {
		w.owned[p] = true
	} else {
		delete(w.owned, p)
	}
}

func sortedKeys(m map[int]bool) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

func nowMs() float64 { return float64(time.Now().UnixNano()) / 1e6 }

// isTransient: connection problems, timeouts and "Redis is still loading" are
// retried; any other error reply is a bug and stops the worker (the Python
// worker crashes on those too, and the supervisor restarts it).
func isTransient(err error) bool {
	var rerr redis.Error
	if !errors.As(err, &rerr) {
		return true
	}
	return redis.HasErrorPrefix(err, "LOADING")
}
