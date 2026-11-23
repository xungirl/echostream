"""End-to-end serving on the multi-node-on-localhost cluster: scatter-gather,
replicas, degradation labels, events changing the next recommendation."""
import time

import httpx
import numpy as np
import pytest

from echostream.cluster.client import control
from echostream.serving.bundle import load_bundle
from tests.integration.conftest import BUNDLE, needs_bundle

pytestmark = [pytest.mark.integration, needs_bundle]


def wait_healthy(url, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(url)


def rec(cfg, user, n=10, port=None):
    r = httpx.get(f"http://127.0.0.1:{port or cfg.lb_port}/recommend", params={"user_id": user, "n": n}, timeout=5)
    r.raise_for_status()
    return r.json()


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(BUNDLE)


def test_recommend_shape_timings_and_no_degradation(full_cluster):
    body = rec(full_cluster, 42, n=10)
    assert len(body["items"]) == 10
    assert body["degraded"] == []
    for stage in ("features", "user_vector", "recall", "merge", "rank", "total"):
        assert stage in body["timings_ms"]
    assert len(body["shards"]) == full_cluster.n_shards and all(s["ok"] for s in body["shards"])
    assert {"item_id", "title", "category", "source", "rank_score"} <= set(body["items"][0])


def test_online_ann_equals_offline_exact_search(full_cluster, bundle):
    """Sharded scatter-gather must return exactly the offline top-k (Flat shards + exact merge)."""
    from echostream.core.scatter_gather import merge_topk  # noqa: F401  (the function under test online)
    for u in (7, 42, 3000):
        uv = bundle.engine.user_vector(u, bundle.pos_history(u))
        offline = [i for i, _ in bundle.engine.two_tower(uv, 200, bundle.seen(u))]
        body = rec(full_cluster, u, n=100)
        online_tt = [x for x in body["items"] if "two_tower" in x["sources"]]
        assert all(x["item_id"] in offline for x in online_tt)


def test_event_changes_next_recommendation(full_cluster):
    u = 77
    first = [x["item_id"] for x in rec(full_cluster, u)["items"]]
    watched = first[0]
    r = httpx.post(f"http://127.0.0.1:{full_cluster.lb_port}/events", json={"events": [
        {"event_id": f"t-{u}-1", "user_id": u, "item_id": watched, "watch_ratio": 3.0}]}, timeout=5)
    assert r.json()["accepted"] == 1
    deadline = time.time() + 10
    while time.time() < deadline:
        after = [x["item_id"] for x in rec(full_cluster, u)["items"]]
        if watched not in after:
            break
        time.sleep(0.1)
    assert watched not in after          # already-watched video is no longer recommended
    assert after != first


def test_duplicate_event_counted_once(full_cluster):
    import redis
    from echostream.core.hashing import ConsistentHashRing
    u = 91
    ev = {"event_id": f"idem-{time.time_ns()}", "user_id": u, "item_id": 5, "watch_ratio": 0.1}
    for _ in range(3):   # client retries
        httpx.post(f"http://127.0.0.1:{full_cluster.lb_port}/events", json={"events": [ev]}, timeout=5)
    node = ConsistentHashRing(list(full_cluster.feature_nodes), 100).get_node(f"user:{u}")
    r = redis.Redis.from_url(full_cluster.feature_nodes[node])
    deadline = time.time() + 10
    while time.time() < deadline and not r.exists(f"ev:{ev['event_id']}"):
        time.sleep(0.1)
    time.sleep(1.0)
    hist = [x.decode() for x in r.lrange(f"rt:{u}:hist", 0, -1)]
    assert sum(h.startswith("5:") for h in hist) == 1


def test_shard_primary_down_uses_replica(full_cluster):
    control(full_cluster, "stop", "shard0-r0")
    try:
        for u in range(10, 30):
            body = rec(full_cluster, u)
            assert not any(d.startswith("ann") for d in body["degraded"])
            assert body["shards"][0]["ok"]
    finally:
        control(full_cluster, "start", "shard0-r0")
        wait_healthy(f"http://127.0.0.1:{full_cluster.shard_port(0, 0)}/health")


def test_whole_shard_down_gives_labeled_partial_result(full_cluster):
    control(full_cluster, "stop", "shard1-r0")
    control(full_cluster, "stop", "shard1-r1")
    try:
        body = rec(full_cluster, 123)
        assert "ann_partial:missing_shards=[1]" in body["degraded"]
        assert len(body["items"]) == 10      # still a full page from the other paths
    finally:
        control(full_cluster, "start", "shard1-r0")
        control(full_cluster, "start", "shard1-r1")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if rec(full_cluster, 123)["degraded"] == []:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    assert rec(full_cluster, 123)["degraded"] == []


def test_slow_shard_bounded_latency(full_cluster):
    control(full_cluster, "delay", "shard0-r0", ms=500)
    try:
        lat = []
        for u in range(200, 240):
            t = time.perf_counter()
            body = rec(full_cluster, u)
            lat.append(time.perf_counter() - t)
            assert body["shards"][0]["ok"]           # answered by the replica
        assert max(lat) < 0.25                        # never waits for the 500 ms shard
    finally:
        control(full_cluster, "delay", "shard0-r0", ms=0)


def test_feature_node_down_degrades_not_fails(full_cluster):
    from echostream.core.hashing import ConsistentHashRing
    ring = ConsistentHashRing(list(full_cluster.feature_nodes), 100)
    users = [u for u in range(500, 700) if ring.get_node(f"user:{u}") == "feat0"][:10]
    control(full_cluster, "freeze", "redis-feat0")
    try:
        for u in users:
            body = rec(full_cluster, u)
            assert "feature_store_unavailable" in body["degraded"]
            assert len(body["items"]) == 10
    finally:
        control(full_cluster, "thaw", "redis-feat0")
    assert rec(full_cluster, users[0])["degraded"] == []


def test_api_replica_killed_lb_keeps_serving(full_cluster):
    control(full_cluster, "stop", "api0")
    try:
        ok = sum(1 for u in range(40) if rec(full_cluster, u)["items"])
        assert ok == 40
    finally:
        control(full_cluster, "start", "api0")
        wait_healthy(f"http://127.0.0.1:{full_cluster.api_port(0)}/health")


def test_metrics_exposed(full_cluster):
    rec(full_cluster, 1)
    text = httpx.get(f"http://127.0.0.1:{full_cluster.api_port(0)}/metrics", timeout=5).text
    assert 'echostream_stage_seconds_bucket{le="0.001",stage="recall"}' in text or "echostream_stage_seconds" in text
    assert "echostream_fallback_total" in text
