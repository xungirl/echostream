package hashing

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/xungirl/echostream/go/worker/internal/golden"
)

type hashingGolden struct {
	UserIDs    []int64 `json:"user_ids"`
	StableHash []struct {
		Key  string `json:"key"`
		Hash string `json:"hash"`
	} `json:"stable_hash"`
	Rings []struct {
		Nodes     []string `json:"nodes"`
		VNodes    int      `json:"vnodes"`
		NodeIndex []int    `json:"node_index"`
	} `json:"rings"`
	Partitions []struct {
		N         int   `json:"n"`
		Partition []int `json:"partition"`
	} `json:"partitions"`
}

// Python computed these; Go must agree on every single one.
func TestGoldenFromPython(t *testing.T) {
	var g hashingGolden
	golden.Load(t, "hashing_golden.json", &g)

	for _, c := range g.StableHash {
		want, _ := strconv.ParseUint(c.Hash, 10, 64)
		if got := StableHash(c.Key); got != want {
			t.Errorf("StableHash(%q) = %d, Python says %d", c.Key, got, want)
		}
	}
	checked := 0
	for _, rc := range g.Rings {
		ring := NewRing(rc.Nodes, rc.VNodes)
		mismatches := 0
		for i, uid := range g.UserIDs {
			if got, want := ring.NodeForUser(uid), rc.Nodes[rc.NodeIndex[i]]; got != want {
				mismatches++
				if mismatches < 5 {
					t.Errorf("%d nodes: user %d -> %s, Python says %s", len(rc.Nodes), uid, got, want)
				}
			}
			checked++
		}
		if mismatches > 0 {
			t.Errorf("%d nodes: %d/%d users on a different node than Python", len(rc.Nodes), mismatches, len(g.UserIDs))
		}
	}
	for _, pc := range g.Partitions {
		for i, uid := range g.UserIDs {
			if got := PartitionFor(uid, pc.N); got != pc.Partition[i] {
				t.Fatalf("PartitionFor(%d, %d) = %d, Python says %d", uid, pc.N, got, pc.Partition[i])
			}
			checked++
		}
	}
	t.Logf("%d (user, ring/partition) assignments identical to Python", checked)
}

func TestRingMovesOnlyAFractionOfKeys(t *testing.T) {
	before := NewRing([]string{"n0", "n1", "n2"}, 100)
	after := NewRing([]string{"n0", "n1", "n2", "n3"}, 100)
	moved := 0
	for u := 0; u < 9000; u++ {
		k := fmt.Sprintf("user:%d", u)
		if a, b := before.GetNode(k), after.GetNode(k); a != b {
			if b != "n3" {
				t.Fatalf("%s moved %s -> %s; only moves to the new node are allowed", k, a, b)
			}
			moved++
		}
	}
	if f := float64(moved) / 9000; f < 0.15 || f > 0.35 {
		t.Fatalf("moved fraction %.2f, want about 1/4", f)
	}
}

func TestNodes(t *testing.T) {
	r := NewRing([]string{"feat2", "feat0", "feat1"}, 100)
	if got := fmt.Sprint(r.Nodes()); got != "[feat0 feat1 feat2]" {
		t.Fatal(got)
	}
}

func BenchmarkNodeForUser(b *testing.B) {
	r := NewRing([]string{"feat0", "feat1", "feat2"}, 100)
	for i := 0; i < b.N; i++ {
		r.NodeForUser(int64(i % 7176))
	}
}
