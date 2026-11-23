package worker

import (
	"fmt"
	"testing"
)

func TestParseReadReplyKeepsFieldOrderAndDeletedEntries(t *testing.T) {
	reply := []any{
		[]any{"events:3", []any{
			[]any{"1-1", []any{"z", "1", "a", "2", "z", "3"}}, // repeated key: first position, last value
			[]any{"1-2", nil}, // deleted after delivery
		}},
		[]any{"events:5", []any{}},
	}
	got, err := parseReadReply(reply)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Stream != "events:3" || len(got[1].Entries) != 0 {
		t.Fatalf("%+v", got)
	}
	if s := fmt.Sprint(got[0].Entries[0].Fields); s != "[{z 3} {a 2}]" {
		t.Fatalf("fields %s", s)
	}
	if got[0].Entries[1].ID != "1-2" || got[0].Entries[1].Fields != nil {
		t.Fatalf("deleted entry %+v", got[0].Entries[1])
	}
}

func TestParseReadReplyRejectsGarbage(t *testing.T) {
	if _, err := parseReadReply("nope"); err == nil {
		t.Fatal("want error")
	}
	if _, err := parseReadReply([]any{[]any{"s"}}); err == nil {
		t.Fatal("want error")
	}
}
