// Package events is the Go twin of echostream/serving/events.py: the event
// schema shared by the API (producer) and the workers (consumers), and the
// validation that decides which stream entries are poison.
package events

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/xungirl/echostream/go/worker/internal/pycompat"
)

// Field is one field of a stream entry. Entries keep their fields in order
// (a Go map would lose it) because the DLQ copy and some error messages print
// them in the order the producer wrote them, exactly like Python's dict.
type Field struct {
	Key, Value string
}

// Entry is one stream entry. Fields == nil means the entry was deleted
// (XDEL) after it had been delivered; Redis then returns only its id.
type Entry struct {
	ID     string
	Fields []Field
}

// Event is a validated entry.
type Event struct {
	EventID    string
	UserID     int64
	ItemID     int64
	Action     string
	WatchRatio float64
	Ts         float64
	Cats       string // comma-separated category ids, "" if absent
	ProducedMs float64
}

// PoisonError marks an entry that can never be applied; it goes to the DLQ.
type PoisonError struct{ Msg string }

func (e *PoisonError) Error() string { return e.Msg }

var (
	actions  = []string{"play", "complete", "skip"}
	required = []string{"event_id", "user_id", "item_id", "action", "watch_ratio", "ts", "produced_ms"}
)

// NotUTF8 is the DLQ reason for entries whose bytes are not valid UTF-8.
const NotUTF8 = "fields are not valid UTF-8"

// Parse validates a raw entry, mirroring parse_fields() check by check so a
// bad entry gets the same DLQ reason text in both languages.
func Parse(fields []Field) (Event, error) {
	for _, f := range fields {
		if !utf8.ValidString(f.Key) || !utf8.ValidString(f.Value) {
			return Event{}, &PoisonError{NotUTF8}
		}
	}
	get := func(k string) (string, bool) {
		for _, f := range fields {
			if f.Key == k {
				return f.Value, true
			}
		}
		return "", false
	}
	var missing []string
	for _, k := range required {
		if _, ok := get(k); !ok {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		return Event{}, &PoisonError{"missing fields " + reprList(missing)}
	}

	// conversions in the same order as the Python dict literal: the first failure wins
	var ev Event
	var err error
	ev.EventID, _ = get("event_id")
	userID, _ := get("user_id")
	if ev.UserID, err = pycompat.ParseInt(userID); err != nil {
		return Event{}, &PoisonError{err.Error()}
	}
	itemID, _ := get("item_id")
	if ev.ItemID, err = pycompat.ParseInt(itemID); err != nil {
		return Event{}, &PoisonError{err.Error()}
	}
	ev.Action, _ = get("action")
	wr, _ := get("watch_ratio")
	if ev.WatchRatio, err = pycompat.ParseFloat(wr); err != nil {
		return Event{}, &PoisonError{err.Error()}
	}
	ts, _ := get("ts")
	if ev.Ts, err = pycompat.ParseFloat(ts); err != nil {
		return Event{}, &PoisonError{err.Error()}
	}
	ev.Cats, _ = get("cats")
	produced, _ := get("produced_ms")
	if ev.ProducedMs, err = pycompat.ParseFloat(produced); err != nil {
		return Event{}, &PoisonError{err.Error()}
	}

	// NaN watch_ratio passes `wr < 0` in Python too; Redis rejects it later (script error -> DLQ)
	if !isAction(ev.Action) || ev.WatchRatio < 0 || ev.UserID < 0 || ev.ItemID < 0 ||
		ev.EventID == "" || strings.Contains(ev.EventID, ":") {
		return Event{}, &PoisonError{"invalid values " + reprDict(fields)}
	}
	return ev, nil
}

func isAction(a string) bool {
	for _, x := range actions {
		if a == x {
			return true
		}
	}
	return false
}

// reprList renders []string like Python's repr(list): ['a', 'b'].
func reprList(xs []string) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = pycompat.QuoteString(x)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// reprDict renders fields like Python's repr(dict): {'k': 'v', ...}.
func reprDict(fields []Field) string {
	parts := make([]string, len(fields))
	for i, f := range fields {
		parts[i] = pycompat.QuoteString(f.Key) + ": " + pycompat.QuoteString(f.Value)
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// Truncate keeps the first n characters (runes), like Python's s[:n].
func Truncate(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	i := 0
	for pos := range s {
		if i == n {
			return s[:pos]
		}
		i++
	}
	return s
}

// String renders an event for debugging.
func (e Event) String() string {
	return fmt.Sprintf("%s(user=%d item=%d %s wr=%s)", e.EventID, e.UserID, e.ItemID, e.Action,
		pycompat.FormatFloat(e.WatchRatio))
}
