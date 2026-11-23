// Package config mirrors echostream/serving/config.py: the same ECHO_* env
// vars with the same defaults, so the supervisor can start either worker with
// one environment.
package config

import (
	"fmt"
	"os"
	"strconv"
)

// Config holds what the worker needs from ClusterConfig + EventConfig.
type Config struct {
	PortBase        int
	NPartitions     int
	NFeatureNodes   int
	StreamPrefix    string // "events" -> streams events:0 .. events:{P-1}
	Group           string
	DLQStream       string
	DedupeTTLSec    int
	HistMax         int
	LeaseMs         int
	ClaimMinIdleMs  int
	Batch           int
	PositiveThresh  string // passed to Lua as text; Python sends repr(2.0) == "2.0"
	Archive         bool   // ECHO_ARCHIVE=1: write applied events to Postgres before acking
	PostgresDSN     string
	FeatureNodeName []string // feat0, feat1, ... in order
}

// FromEnv reads the same variables, with the same defaults, as config.py.
func FromEnv() (Config, error) {
	var c Config
	var err error
	get := func(name string, def int) int {
		if err != nil {
			return 0
		}
		v, ok := os.LookupEnv(name)
		if !ok {
			return def
		}
		n, e := strconv.Atoi(v)
		if e != nil {
			err = fmt.Errorf("%s=%q: %w", name, v, e)
		}
		return n
	}
	c.PortBase = get("ECHO_PORT_BASE", 17000)
	c.NPartitions = get("ECHO_PARTITIONS", 8)
	c.NFeatureNodes = get("ECHO_FEATURE_NODES", 3)
	c.DedupeTTLSec = get("ECHO_DEDUPE_TTL", 86400)
	c.LeaseMs = get("ECHO_LEASE_MS", 3000)
	c.ClaimMinIdleMs = get("ECHO_CLAIM_IDLE_MS", 1000)
	c.Batch = get("ECHO_WORKER_BATCH", 500)
	if err != nil {
		return c, err
	}
	c.StreamPrefix, c.Group, c.DLQStream = "events", "workers", "events:dlq"
	c.HistMax = 100
	c.PositiveThresh = "2.0"
	c.Archive = os.Getenv("ECHO_ARCHIVE") == "1"
	c.PostgresDSN = os.Getenv("ECHO_PG_DSN")
	if c.PostgresDSN == "" { // same fallback as echostream/store/db.py:dsn()
		c.PostgresDSN = fmt.Sprintf("postgresql://echostream@127.0.0.1:%d/echostream", c.PortBase+40)
	}
	for i := 0; i < c.NFeatureNodes; i++ {
		c.FeatureNodeName = append(c.FeatureNodeName, fmt.Sprintf("feat%d", i))
	}
	return c, nil
}

// EventsRedisAddr is the event bus (port_base + 20).
func (c Config) EventsRedisAddr() string { return fmt.Sprintf("127.0.0.1:%d", c.PortBase+20) }

// FeatureNodeAddrs maps feat{i} -> 127.0.0.1:(port_base + 21 + i).
func (c Config) FeatureNodeAddrs() map[string]string {
	m := map[string]string{}
	for i, name := range c.FeatureNodeName {
		m[name] = fmt.Sprintf("127.0.0.1:%d", c.PortBase+21+i)
	}
	return m
}

// StreamName == stream_name(p) in events.py.
func (c Config) StreamName(p int) string { return fmt.Sprintf("%s:%d", c.StreamPrefix, p) }
