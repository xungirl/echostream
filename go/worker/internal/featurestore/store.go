// Package featurestore is the Go twin of echostream/serving/feature_store.py:
// the real-time feature store sharded over several Redis nodes.
//
// All keys of one user live on ONE node (consistent hashing on the user id),
// so applying an event is one atomic Lua script on that node:
//
//	ev:{event_id}   dedupe marker with TTL -> a redelivered event is a no-op
//	rt:{uid}:hist   list "item:watch_ratio:ts", newest first, capped
//	rt:{uid}:cnt    hash of counters (n, play, complete, skip, pos, sum_wr, ...)
//	rec:{uid}       cached recommendations, deleted because they are stale now
//
// The Lua source is byte-identical to Python's APPLY_EVENT_LUA (a golden test checks it).
package featurestore

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/xungirl/echostream/go/worker/internal/events"
	"github.com/xungirl/echostream/go/worker/internal/hashing"
	"github.com/xungirl/echostream/go/worker/internal/pycompat"
)

//go:embed apply_event.lua
var ApplyEventLua string

// Outcome of applying one event.
type Outcome int

const (
	Duplicate   Outcome = iota // script returned 0: the dedupe key already existed
	Applied                    // script returned 1
	ScriptError                // deterministic Redis error (e.g. NaN watch ratio): dead-letter it
)

// Result is the outcome for one event of a batch.
type Result struct {
	Outcome Outcome
	ErrMsg  string // for ScriptError: the message as Python's redis client prints it
}

// Call is the KEYS and ARGV of one script invocation.
type Call struct {
	Keys []string
	Args []any
}

// Store holds one client per feature node and the ring that picks the node.
type Store struct {
	Ring    *hashing.Ring
	clients map[string]*redis.Client

	mu  sync.Mutex
	sha map[string]string // node -> SHA1 of the loaded script
}

// New connects (lazily) to every node. 0.5 s timeouts like the Python store.
func New(nodes map[string]string, names []string) *Store {
	s := &Store{Ring: hashing.NewRing(names, 100), clients: map[string]*redis.Client{}, sha: map[string]string{}}
	for name, addr := range nodes {
		s.clients[name] = redis.NewClient(&redis.Options{
			Addr:         addr,
			Protocol:     2, // RESP2, like redis-py
			DialTimeout:  500 * time.Millisecond,
			ReadTimeout:  500 * time.Millisecond,
			WriteTimeout: 500 * time.Millisecond,
		})
	}
	return s
}

// NodeFor returns the feature node that owns a user.
func (s *Store) NodeFor(userID int64) string { return s.Ring.NodeForUser(userID) }

// ApplyArgs == FeatureStore.apply_args(): numbers are formatted exactly as
// redis-py would send them (int -> "42", float -> repr()).
func ApplyArgs(ev events.Event, ttlSec, histMax int, posThreshold string) Call {
	uid := strconv.FormatInt(ev.UserID, 10)
	return Call{
		Keys: []string{"ev:" + ev.EventID, "rt:" + uid + ":hist", "rt:" + uid + ":cnt", "rec:" + uid},
		Args: []any{
			strconv.Itoa(ttlSec), strconv.FormatInt(ev.ItemID, 10),
			pycompat.FormatFloat(ev.WatchRatio), pycompat.FormatFloat(ev.Ts),
			ev.Action, ev.Cats, strconv.Itoa(histMax), posThreshold,
		},
	}
}

// ApplyBatch runs the script for several events on one node, in order, in
// one round trip (a pipeline). It returns one Result per call. A non-nil
// error means the node is unreachable (transient): the caller keeps the
// events pending and retries later.
func (s *Store) ApplyBatch(ctx context.Context, node string, calls []Call) ([]Result, error) {
	c := s.clients[node]
	sha, err := s.scriptSHA(ctx, node)
	if err != nil {
		return nil, err
	}
	pipe := c.Pipeline()
	cmds := make([]*redis.Cmd, len(calls))
	for i, call := range calls {
		cmds[i] = pipe.EvalSha(ctx, sha, call.Keys, call.Args...)
	}
	_, _ = pipe.Exec(ctx) // per-command errors are inspected below

	results := make([]Result, len(calls))
	for i, cmd := range cmds {
		n, err := cmd.Int64()
		switch {
		case err == nil && n == 1:
			results[i] = Result{Outcome: Applied}
		case err == nil:
			results[i] = Result{Outcome: Duplicate}
		case redis.HasErrorPrefix(err, "NOSCRIPT"):
			// the node restarted and lost its script cache: reload and re-run the whole
			// batch -- safe because the script is idempotent (dedupe key)
			s.mu.Lock()
			delete(s.sha, node)
			s.mu.Unlock()
			return s.ApplyBatch(ctx, node, calls)
		case isServerError(err):
			results[i] = Result{Outcome: ScriptError, ErrMsg: PyErrorMessage(err)}
		default:
			return nil, err // network/timeout: transient
		}
	}
	return results, nil
}

func (s *Store) scriptSHA(ctx context.Context, node string) (string, error) {
	s.mu.Lock()
	sha, ok := s.sha[node]
	s.mu.Unlock()
	if ok {
		return sha, nil
	}
	sha, err := s.clients[node].ScriptLoad(ctx, ApplyEventLua).Result()
	if err != nil {
		return "", err
	}
	s.mu.Lock()
	s.sha[node] = sha
	s.mu.Unlock()
	return sha, nil
}

// Close closes all node clients.
func (s *Store) Close() {
	for _, c := range s.clients {
		_ = c.Close()
	}
}

// isServerError reports whether err is an error reply from Redis (as opposed
// to a broken connection or a timeout).
func isServerError(err error) bool {
	var rerr redis.Error
	return errors.As(err, &rerr) && !errors.Is(err, redis.Nil)
}

// redis-py strips these error codes from the message (redis/_parsers/base.py).
var pyStrippedCodes = map[string]bool{
	"ERR": true, "OOM": true, "WRONGPASS": true, "EXECABORT": true, "LOADING": true, "NOSCRIPT": true,
	"READONLY": true, "NOAUTH": true, "NOPERM": true, "ASK": true, "TRYAGAIN": true, "MOVED": true,
	"CLUSTERDOWN": true, "CROSSSLOT": true, "MASTERDOWN": true,
}

// PyErrorMessage formats a Redis error reply the way str(ResponseError) does in redis-py.
func PyErrorMessage(err error) string {
	msg := err.Error()
	code, rest, _ := strings.Cut(msg, " ")
	if pyStrippedCodes[code] {
		return rest
	}
	return msg
}

// String is for debugging.
func (r Result) String() string {
	return fmt.Sprintf("%d %s", r.Outcome, r.ErrMsg)
}
