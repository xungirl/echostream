// Package hashing is the Go twin of echostream/core/hashing.py.
//
// Both languages must send a user to the same stream partition and the same
// feature node, otherwise a Go worker would write a user's features to a node
// where the Python API never looks. So the hash is a fixed, portable function
// (64-bit BLAKE2b, big-endian), not Go's or Python's built-in map hashing.
// Golden tests (testdata/hashing_golden.json, written by Python) pin it.
package hashing

import (
	"encoding/binary"
	"fmt"
	"sort"

	"golang.org/x/crypto/blake2b"
)

// StableHash == stable_hash() in Python: blake2b(key, digest_size=8) as a big-endian uint64.
func StableHash(key string) uint64 {
	h, err := blake2b.New(8, nil) // an 8-byte digest is its own BLAKE2b variant, not a truncated 64-byte one
	if err != nil {
		panic(err) // only possible for an invalid size/key
	}
	h.Write([]byte(key))
	return binary.BigEndian.Uint64(h.Sum(nil))
}

// PartitionFor == partition_for(): the stream a user's events go to.
// Same user -> same partition, which is what keeps per-user order.
func PartitionFor(userID int64, nPartitions int) int {
	return int(StableHash(fmt.Sprintf("user:%d", userID)) % uint64(nPartitions))
}

type point struct {
	hash uint64
	node string
}

// Ring is a consistent-hash ring with virtual nodes. Each node is placed on
// the ring `vnodes` times at hash("<node>#<v>"); a key belongs to the first
// point clockwise from hash(key). Adding a node moves only ~1/N of the keys.
type Ring struct {
	points []point // sorted by (hash, node), like Python's bisect.insort on tuples
}

// NewRing builds the ring; nodes and vnodes must match the Python side (vnodes=100).
func NewRing(nodes []string, vnodes int) *Ring {
	r := &Ring{}
	for _, n := range nodes {
		for v := 0; v < vnodes; v++ {
			r.points = append(r.points, point{StableHash(fmt.Sprintf("%s#%d", n, v)), n})
		}
	}
	sort.Slice(r.points, func(i, j int) bool {
		if r.points[i].hash != r.points[j].hash {
			return r.points[i].hash < r.points[j].hash
		}
		return r.points[i].node < r.points[j].node
	})
	return r
}

// GetNode returns the node owning key: the first point with hash > hash(key),
// wrapping around (Python: bisect_right(ring, (h, "￿")) % len(ring)).
func (r *Ring) GetNode(key string) string {
	if len(r.points) == 0 {
		panic("empty ring")
	}
	h := StableHash(key)
	i := sort.Search(len(r.points), func(i int) bool { return r.points[i].hash > h })
	return r.points[i%len(r.points)].node
}

// NodeForUser is FeatureStore.node_for(): all of a user's keys live on one node.
func (r *Ring) NodeForUser(userID int64) string {
	return r.GetNode(fmt.Sprintf("user:%d", userID))
}

// Nodes returns the distinct node names, sorted (Python: ring.nodes).
func (r *Ring) Nodes() []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range r.points {
		if !seen[p.node] {
			seen[p.node] = true
			out = append(out, p.node)
		}
	}
	sort.Strings(out)
	return out
}
