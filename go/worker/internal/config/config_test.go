package config

import (
	"os"
	"testing"
)

// unset removes an env var for the duration of the test (t.Setenv restores the old value).
func unset(t *testing.T, k string) {
	t.Setenv(k, "")
	if err := os.Unsetenv(k); err != nil {
		t.Fatal(err)
	}
}

func TestDefaultsMatchConfigPy(t *testing.T) {
	for _, k := range []string{"ECHO_PORT_BASE", "ECHO_PARTITIONS", "ECHO_FEATURE_NODES", "ECHO_DEDUPE_TTL",
		"ECHO_LEASE_MS", "ECHO_CLAIM_IDLE_MS", "ECHO_WORKER_BATCH", "ECHO_ARCHIVE", "ECHO_PG_DSN"} {
		unset(t, k)
	}
	c, err := FromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if c.PortBase != 17000 || c.NPartitions != 8 || c.NFeatureNodes != 3 || c.DedupeTTLSec != 86400 ||
		c.LeaseMs != 3000 || c.ClaimMinIdleMs != 1000 || c.Batch != 500 || c.Archive {
		t.Fatalf("%+v", c)
	}
	if c.EventsRedisAddr() != "127.0.0.1:17020" || c.FeatureNodeAddrs()["feat2"] != "127.0.0.1:17023" {
		t.Fatalf("addresses %s %v", c.EventsRedisAddr(), c.FeatureNodeAddrs())
	}
	if c.PostgresDSN != "postgresql://echostream@127.0.0.1:17040/echostream" || c.StreamName(3) != "events:3" {
		t.Fatal(c.PostgresDSN, c.StreamName(3))
	}
}

func TestOverridesAndBadValues(t *testing.T) {
	t.Setenv("ECHO_PORT_BASE", "27000")
	t.Setenv("ECHO_PARTITIONS", "6")
	t.Setenv("ECHO_ARCHIVE", "1")
	c, err := FromEnv()
	if err != nil || c.PortBase != 27000 || c.NPartitions != 6 || !c.Archive {
		t.Fatalf("%+v %v", c, err)
	}
	t.Setenv("ECHO_LEASE_MS", "soon")
	if _, err := FromEnv(); err == nil {
		t.Fatal("want error for a non-integer ECHO_LEASE_MS")
	}
}
