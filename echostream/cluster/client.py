"""Helpers to start/stop the supervisor and wait until the cluster is ready (tests, scripts, make)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import redis

from echostream.serving.config import ClusterConfig


def fresh_run_dir(name: str) -> Path:
    """An empty .run/<name> directory (Redis AOF, Postgres data, logs) for a clean benchmark run."""
    import shutil
    d = Path(__file__).resolve().parents[2] / ".run" / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def wait_ready(cfg: ClusterConfig, profile: str, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    checks = []
    if profile in ("full", "events", "serving"):
        checks.append(lambda: redis.Redis(port=cfg.events_redis_port, socket_timeout=0.5).ping())
        for i in range(cfg.n_feature_nodes):
            checks.append(lambda i=i: redis.Redis(port=cfg.feature_redis_port(i), socket_timeout=0.5).ping())
    if profile in ("full", "serving", "shards"):
        for s in range(cfg.n_shards):
            for r in range(cfg.n_replicas):
                checks.append(lambda s=s, r=r: httpx.get(f"http://127.0.0.1:{cfg.shard_port(s, r)}/health",
                                                         timeout=0.5).raise_for_status())
    if profile in ("full", "serving"):
        for i in range(cfg.n_api):
            checks.append(lambda i=i: httpx.get(f"http://127.0.0.1:{cfg.api_port(i)}/health", timeout=0.5).raise_for_status())
        checks.append(lambda: httpx.get(f"http://127.0.0.1:{cfg.lb_port}/health", timeout=0.5).raise_for_status())
    if profile in ("full", "events"):
        def workers():
            n = redis.Redis(port=cfg.events_redis_port).zcard("workers:alive")
            assert n >= cfg.n_workers, n
        checks.append(workers)
    pending = list(checks)
    while pending:
        if time.time() > deadline:
            raise TimeoutError(f"cluster not ready after {timeout}s ({len(pending)} checks failing)")
        still = []
        for c in pending:
            try:
                c()
            except Exception:
                still.append(c)
        pending = still
        if pending:
            time.sleep(0.25)


@contextmanager
def cluster(profile: str = "full", env: dict[str, str] | None = None, run_dir: Path | None = None,
            timeout: float = 120.0, extra_args: list[str] | None = None):
    """Start a supervisor subprocess with the given env overrides; yields its ClusterConfig."""
    full_env = {**os.environ, **(env or {})}
    if run_dir is not None:
        full_env["ECHO_RUN_DIR"] = str(run_dir)
    # ClusterConfig reads env at import time, so build one from the overrides explicitly
    cfg = ClusterConfig(**{k: type(getattr(ClusterConfig(), k))(full_env[v]) for k, v in {
        "port_base": "ECHO_PORT_BASE", "n_shards": "ECHO_SHARDS", "n_replicas": "ECHO_REPLICAS",
        "n_partitions": "ECHO_PARTITIONS", "n_workers": "ECHO_WORKERS", "n_feature_nodes": "ECHO_FEATURE_NODES",
        "n_api": "ECHO_API_REPLICAS", "bundle_dir": "ECHO_BUNDLE"}.items() if v in full_env})
    proc = subprocess.Popen([sys.executable, "-m", "echostream.cluster.supervisor", "--profile", profile,
                             *(extra_args or [])], env=full_env, start_new_session=True)
    try:
        wait_ready(cfg, profile, timeout)
        yield cfg
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def control(cfg: ClusterConfig, action: str, name: str, **params) -> dict:
    r = httpx.post(f"http://127.0.0.1:{cfg.supervisor_port}/{action}/{name}", params=params, timeout=5.0)
    r.raise_for_status()
    return r.json()


def status(cfg: ClusterConfig) -> dict:
    return httpx.get(f"http://127.0.0.1:{cfg.supervisor_port}/status", timeout=2.0).json()
