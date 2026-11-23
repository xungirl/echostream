import os
import shutil
from pathlib import Path

import pytest

from echostream.cluster.client import cluster

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts" / "serving" / "current"
needs_bundle = pytest.mark.skipif(not (BUNDLE / "manifest.json").exists(),
                                  reason="needs a serving bundle (make bundle)")


def run_dir(name: str) -> Path:
    d = ROOT / ".run" / "test" / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


@pytest.fixture(scope="module")
def events_cluster():
    env = {"ECHO_PORT_BASE": "27000", "ECHO_WORKERS": "3", "ECHO_PARTITIONS": "6", "ECHO_FEATURE_NODES": "3",
           "ECHO_LEASE_MS": "1500", "ECHO_CLAIM_IDLE_MS": "500", "ECHO_WORKER_BATCH": "200"}
    with cluster("events", env=env, run_dir=run_dir("events")) as cfg:
        yield cfg


@pytest.fixture(scope="module")
def full_cluster():
    env = {"ECHO_PORT_BASE": "28000", "ECHO_WORKERS": "2", "ECHO_PARTITIONS": "4", "ECHO_SHARDS": "2",
           "ECHO_REPLICAS": "2", "ECHO_FEATURE_NODES": "2", "ECHO_API_REPLICAS": "2",
           "ECHO_LEASE_MS": "1500", "ECHO_CLAIM_IDLE_MS": "500", "ECHO_BUNDLE": str(BUNDLE.resolve()),
           "ECHO_DEMO_MODE": "1"}
    with cluster("full", env=env, run_dir=run_dir("full"), timeout=180) as cfg:
        yield cfg
