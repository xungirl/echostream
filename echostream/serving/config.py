"""Cluster topology and timeouts. Everything runs as separate processes on ONE
laptop ("multi-node on localhost"); ports stand in for hosts.

Values can be overridden with env vars (ECHO_*) so tests can run a smaller cluster.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(os.environ.get("ECHO_RUN_DIR", ROOT / ".run"))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class ClusterConfig:
    port_base: int = _int("ECHO_PORT_BASE", 17000)
    n_shards: int = _int("ECHO_SHARDS", 4)
    n_replicas: int = _int("ECHO_REPLICAS", 2)
    n_partitions: int = _int("ECHO_PARTITIONS", 8)
    n_workers: int = _int("ECHO_WORKERS", 4)
    n_feature_nodes: int = _int("ECHO_FEATURE_NODES", 3)
    n_api: int = _int("ECHO_API_REPLICAS", 4)
    bundle_dir: str = os.environ.get("ECHO_BUNDLE", str(ROOT / "artifacts" / "serving" / "current"))

    # ---- ports (all derived from port_base) ----
    @property
    def lb_port(self) -> int:
        return self.port_base

    @property
    def supervisor_port(self) -> int:
        return self.port_base + 9

    def api_port(self, i: int) -> int:
        return self.port_base + 1 + i

    @property
    def events_redis_port(self) -> int:
        return self.port_base + 20

    def feature_redis_port(self, i: int) -> int:
        return self.port_base + 21 + i

    def shard_port(self, shard: int, replica: int) -> int:
        return self.port_base + 100 + 10 * shard + replica

    def worker_metrics_port(self, i: int) -> int:
        return self.port_base + 60 + i

    @property
    def postgres_port(self) -> int:
        return self.port_base + 40

    # ---- urls ----
    @property
    def events_redis_url(self) -> str:
        return f"redis://127.0.0.1:{self.events_redis_port}/0"

    @property
    def feature_nodes(self) -> dict[str, str]:
        return {f"feat{i}": f"redis://127.0.0.1:{self.feature_redis_port(i)}/0" for i in range(self.n_feature_nodes)}

    def shard_urls(self) -> list[list[str]]:
        return [[f"http://127.0.0.1:{self.shard_port(s, r)}" for r in range(self.n_replicas)] for s in range(self.n_shards)]


@dataclass(frozen=True)
class Timeouts:
    """Latency budget for /recommend (milliseconds). The overall deadline is
    propagated: every stage gets min(its own cap, time left)."""
    total_ms: float = _float("ECHO_BUDGET_MS", 250.0)
    features_ms: float = _float("ECHO_FEATURES_MS", 20.0)
    shard_ms: float = _float("ECHO_SHARD_MS", 40.0)        # per attempt, per shard
    ranker_ms: float = _float("ECHO_RANKER_MS", 60.0)
    eject_after_failures: int = 3                           # outlier ejection of a bad replica
    eject_ms: float = 2000.0


@dataclass(frozen=True)
class EventConfig:
    stream_prefix: str = "events"
    group: str = "workers"
    dlq_stream: str = "events:dlq"
    dedupe_ttl_s: int = _int("ECHO_DEDUPE_TTL", 86400)
    hist_max: int = 100
    lease_ms: int = _int("ECHO_LEASE_MS", 3000)
    claim_min_idle_ms: int = _int("ECHO_CLAIM_IDLE_MS", 1000)
    batch: int = _int("ECHO_WORKER_BATCH", 500)
    max_deliveries: int = 5
    positive_threshold: float = 2.0


CLUSTER = ClusterConfig()
TIMEOUTS = Timeouts()
EVENTS = EventConfig()
