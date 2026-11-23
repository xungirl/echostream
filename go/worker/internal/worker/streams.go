package worker

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/xungirl/echostream/go/worker/internal/events"
)

// streamBatch is what XREADGROUP returns for one stream.
type streamBatch struct {
	Stream  string
	Entries []events.Entry
}

// readGroup runs XREADGROUP and parses the raw reply ourselves instead of
// using go-redis' XReadGroup, because that one returns each entry's fields as
// a map and loses their order (the DLQ copy keeps the producer's order).
//
// id "0" reads this consumer's own pending entries; ">" reads new ones.
// block < 0 means do not block.
func readGroup(ctx context.Context, r *redis.Client, group, consumer string, streams []string, id string,
	count int, block time.Duration) ([]streamBatch, error) {
	args := []any{"XREADGROUP", "GROUP", group, consumer, "COUNT", count}
	if block >= 0 {
		args = append(args, "BLOCK", block.Milliseconds())
	}
	args = append(args, "STREAMS")
	for _, s := range streams {
		args = append(args, s)
	}
	for range streams {
		args = append(args, id)
	}
	reply, err := r.Do(ctx, args...).Result()
	if errors.Is(err, redis.Nil) { // BLOCK timed out with nothing new
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return parseReadReply(reply)
}

// parseReadReply turns  [[stream, [[id, [k, v, ...]], [id, nil], ...]], ...]  into Go types.
func parseReadReply(reply any) ([]streamBatch, error) {
	list, ok := reply.([]any)
	if !ok {
		return nil, fmt.Errorf("XREADGROUP: unexpected reply %T", reply)
	}
	out := make([]streamBatch, 0, len(list))
	for _, item := range list {
		pair, ok := item.([]any)
		if !ok || len(pair) != 2 {
			return nil, fmt.Errorf("XREADGROUP: bad stream item %v", item)
		}
		name, _ := pair[0].(string)
		rawEntries, _ := pair[1].([]any)
		sb := streamBatch{Stream: name, Entries: make([]events.Entry, 0, len(rawEntries))}
		for _, re := range rawEntries {
			e, ok := re.([]any)
			if !ok || len(e) != 2 {
				return nil, fmt.Errorf("XREADGROUP: bad entry %v", re)
			}
			id, _ := e[0].(string)
			entry := events.Entry{ID: id}
			if kv, ok := e[1].([]any); ok && len(kv) > 0 {
				entry.Fields = fieldsFromPairs(kv)
			} // nil or empty: the entry was deleted after delivery
			sb.Entries = append(sb.Entries, entry)
		}
		out = append(out, sb)
	}
	return out, nil
}

// fieldsFromPairs builds ordered fields with Python dict semantics: a repeated
// key keeps its first position and takes the last value.
func fieldsFromPairs(kv []any) []events.Field {
	fields := make([]events.Field, 0, len(kv)/2)
	for i := 0; i+1 < len(kv); i += 2 {
		k, _ := kv[i].(string)
		v, _ := kv[i+1].(string)
		fields = setField(fields, k, v)
	}
	return fields
}

// setField overwrites key in place if present, else appends it.
func setField(fields []events.Field, key, value string) []events.Field {
	for i := range fields {
		if fields[i].Key == key {
			fields[i].Value = value
			return fields
		}
	}
	return append(fields, events.Field{Key: key, Value: value})
}
