import os
import shutil
import subprocess
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


GO_WORKER = ROOT / "go" / "worker" / "bin" / "echostream-worker"


def build_go_worker() -> Path:
    """Build the Go worker (cached by Go, ~1 s when unchanged); skip the test if Go is missing."""
    go = shutil.which("go")
    if go is None:
        pytest.skip("go toolchain not installed")
    subprocess.run([go, "build", "-o", str(GO_WORKER), "./cmd/echostream-worker"], cwd=ROOT / "go" / "worker",
                   check=True)
    return GO_WORKER


# the same reliability tests run against the Python workers, the Go workers, and a mixed
# cluster (w0, w2 Python; w1 Go) where both implementations share leases and streams
@pytest.fixture(scope="module", params=["python", "go", "mixed"])
def events_cluster(request):
    impl = request.param
    if impl != "python":
        build_go_worker()
    port = {"python": "27000", "go": "25000", "mixed": "24000"}[impl]
    env = {"ECHO_PORT_BASE": port, "ECHO_WORKERS": "3", "ECHO_PARTITIONS": "6", "ECHO_FEATURE_NODES": "3",
           "ECHO_LEASE_MS": "1500", "ECHO_CLAIM_IDLE_MS": "500", "ECHO_WORKER_BATCH": "200",
           "ECHO_WORKER_IMPL": impl}
    with cluster("events", env=env, run_dir=run_dir(f"events-{impl}")) as cfg:
        yield cfg


@pytest.fixture(scope="module")
def full_cluster():
    env = {"ECHO_PORT_BASE": "28000", "ECHO_WORKERS": "2", "ECHO_PARTITIONS": "4", "ECHO_SHARDS": "2",
           "ECHO_REPLICAS": "2", "ECHO_FEATURE_NODES": "2", "ECHO_API_REPLICAS": "2",
           "ECHO_LEASE_MS": "1500", "ECHO_CLAIM_IDLE_MS": "500", "ECHO_BUNDLE": str(BUNDLE.resolve()),
           "ECHO_DEMO_MODE": "1"}
    with cluster("full", env=env, run_dir=run_dir("full"), timeout=180) as cfg:
        yield cfg


@pytest.fixture(scope="module")
def pg_cluster():
    env = {"ECHO_PORT_BASE": "29000", "ECHO_WORKERS": "2", "ECHO_PARTITIONS": "4", "ECHO_SHARDS": "2",
           "ECHO_REPLICAS": "1", "ECHO_FEATURE_NODES": "2", "ECHO_API_REPLICAS": "2", "ECHO_PG": "1",
           "ECHO_ARCHIVE": "1", "ECHO_LEASE_MS": "1500", "ECHO_CLAIM_IDLE_MS": "500",
           "ECHO_BUNDLE": str(BUNDLE.resolve()), "ECHO_PG_DSN": "postgresql://echostream@127.0.0.1:29040/echostream"}
    old = os.environ.get("ECHO_PG_DSN")
    os.environ["ECHO_PG_DSN"] = env["ECHO_PG_DSN"]
    try:
        with cluster("full", env=env, run_dir=run_dir("pg"), timeout=180) as cfg:
            yield cfg
    finally:
        if old is None:
            os.environ.pop("ECHO_PG_DSN", None)
        else:
            os.environ["ECHO_PG_DSN"] = old
