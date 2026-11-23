package events

import (
	"errors"
	"testing"

	"github.com/xungirl/echostream/go/worker/internal/golden"
	"github.com/xungirl/echostream/go/worker/internal/pycompat"
)

type eventsGolden struct {
	Cases []struct {
		Fields [][2]string `json:"fields"`
		Poison string      `json:"poison"`
		Event  *struct {
			EventID    string `json:"event_id"`
			UserID     int64  `json:"user_id"`
			ItemID     int64  `json:"item_id"`
			Action     string `json:"action"`
			WatchRatio string `json:"watch_ratio"`
			Ts         string `json:"ts"`
			Cats       string `json:"cats"`
			ProducedMs string `json:"produced_ms"`
		} `json:"event"`
	} `json:"cases"`
}

// Every case was run through Python's parse_fields(); Go must accept the same
// entries with the same values and reject the rest with the same DLQ reason.
func TestParseMatchesPython(t *testing.T) {
	var g eventsGolden
	golden.Load(t, "events_golden.json", &g)
	for _, c := range g.Cases {
		fields := make([]Field, len(c.Fields))
		for i, kv := range c.Fields {
			fields[i] = Field{kv[0], kv[1]}
		}
		ev, err := Parse(fields)
		if c.Poison != "" {
			var pe *PoisonError
			if !errors.As(err, &pe) || pe.Msg != c.Poison {
				t.Errorf("%v:\n got  %v\n want poison %q", c.Fields, err, c.Poison)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: unexpected poison %v", c.Fields, err)
			continue
		}
		want := c.Event
		got := [8]string{ev.EventID, pycompat.FormatFloat(float64(ev.UserID)), pycompat.FormatFloat(float64(ev.ItemID)),
			ev.Action, pycompat.FormatFloat(ev.WatchRatio), pycompat.FormatFloat(ev.Ts), ev.Cats,
			pycompat.FormatFloat(ev.ProducedMs)}
		exp := [8]string{want.EventID, pycompat.FormatFloat(float64(want.UserID)),
			pycompat.FormatFloat(float64(want.ItemID)), want.Action, want.WatchRatio, want.Ts, want.Cats, want.ProducedMs}
		if got != exp {
			t.Errorf("%v:\n got  %v\n want %v", c.Fields, got, exp)
		}
	}
}

func TestInvalidUTF8IsPoison(t *testing.T) {
	_, err := Parse([]Field{{"event_id", "\xff\xfe"}, {"user_id", "1"}})
	var pe *PoisonError
	if !errors.As(err, &pe) || pe.Msg != NotUTF8 {
		t.Fatalf("got %v", err)
	}
}

func TestTruncateCountsCharactersNotBytes(t *testing.T) {
	if got := Truncate("日本語abc", 4); got != "日本語a" {
		t.Fatal(got)
	}
	if got := Truncate("short", 200); got != "short" {
		t.Fatal(got)
	}
}
