"""Stage 8: Postgres as source of truth, rebuild of the feature store, hot reload
via the registry, per-user cache + stampede protection."""
import asyncio
import time

import httpx
import pytest
import redis

from echostream.store import db
from tests.integration.conftest import BUNDLE, needs_bundle

pytestmark = [pytest.mark.integration, needs_bundle]


def post(cfg, events):
    r = httpx.post(f"http://127.0.0.1:{cfg.lb_port}/events", json={"events": events}, timeout=5)
    r.raise_for_status()


def snapshot(cfg, users):
    from echostream.core.hashing import ConsistentHashRing
    ring = ConsistentHashRing(list(cfg.feature_nodes), 100)
    out = {}
    for u in users:
        c = redis.Redis.from_url(cfg.feature_nodes[ring.get_node(f"user:{u}")])
        out[u] = (c.hgetall(f"rt:{u}:cnt"), c.lrange(f"rt:{u}:hist", 0, -1))
    return out


def test_archive_and_rebuild_from_postgres(pg_cluster):
    cfg = pg_cluster
    users = list(range(600, 640))
    events = [{"event_id": f"rb-{u}-{k}", "user_id": u, "item_id": (u * 7 + k) % 10728,
               "watch_ratio": [0.1, 1.2, 2.6][k % 3], "ts": 1_600_000_000 + k} for k in range(15) for u in users]
    post(cfg, events)
    deadline = time.time() + 30
    with db.connect() as conn:
        while time.time() < deadline:
            n = conn.execute("SELECT count(*) AS n FROM event_archive WHERE event_id LIKE 'rb-%'").fetchone()["n"]
            if n == len(events):
                break
            time.sleep(0.2)
    assert n == len(events)                               # every event archived exactly once
    time.sleep(0.5)
    before = snapshot(cfg, users)
    from echostream.store.rebuild import rebuild
    out = asyncio.run(rebuild(cfg.feature_nodes, flush=True))   # wipes Redis, re-applies the archive
    assert out["events_replayed"] >= len(events)
    assert snapshot(cfg, users) == before                 # identical counters AND identical per-user order


def test_registry_hot_reload_without_restart(pg_cluster):
    cfg = pg_cluster
    with db.connect() as conn:
        db.register_model(conn, "v1", str(BUNDLE.resolve()))
        db.set_status(conn, "v1", "live")
        db.register_model(conn, "v1b", str(BUNDLE.resolve()))   # same files, new version name
        db.set_status(conn, "v1b", "canary")
        db.set_rollout(conn, "v1", canary="v1b", canary_percent=50)
    pids = {}
    from echostream.cluster.client import status
    for name, st in status(cfg).items():
        pids[name] = st["pid"]
    deadline = time.time() + 60
    seen = set()
    while time.time() < deadline and seen != {"v1", "v1b"}:
        for u in range(40):
            try:
                seen.add(httpx.get(f"http://127.0.0.1:{cfg.lb_port}/recommend", params={"user_id": u},
                                   timeout=5).json()["model_version"])
            except Exception:
                pass
        time.sleep(0.5)
    assert seen == {"v1", "v1b"}                          # both versions served, by user hash
    assert {n: s["pid"] for n, s in status(cfg).items()} == pids   # nothing restarted
    with db.connect() as conn:
        db.set_rollout(conn, "v1")
        db.set_status(conn, "v1b", "retired")


def test_cache_hits_and_single_flight(pg_cluster):
    cfg = pg_cluster
    api = f"http://127.0.0.1:{cfg.api_port(0)}"
    # the previous test ends a 50% canary; replicas reconcile the rollout every 1 s. If the burst below
    # still ran under the canary, its result would be cached under "v1b:10" and the next request
    # (now v1) would miss -> wait until this replica serves v1 only
    deadline = time.time() + 10
    while httpx.get(f"{api}/health", timeout=5).json()["rollout"].get("canary") and time.time() < deadline:
        time.sleep(0.2)
    httpx.post(f"{api}/admin/cache", params={"ttl_s": 30}, timeout=5)
    try:
        async def burst():
            async with httpx.AsyncClient(base_url=api, timeout=5) as c:
                return await asyncio.gather(*(c.get("/recommend", params={"user_id": 4321}) for _ in range(30)))
        rs = asyncio.run(burst())
        assert all(r.status_code == 200 for r in rs)
        comps = httpx.get(f"{api}/admin/rollout/stats", timeout=5).json()["cache"]["computations"]
        assert comps == 1                                 # 30 concurrent misses -> one computation
        for _ in range(3):   # the cache read has a 20 ms budget; on a loaded CI box one read can time out
            again = httpx.get(f"{api}/recommend", params={"user_id": 4321}, timeout=5).json()
            if "cache_hit" in again["degraded"]:
                break
        assert "cache_hit" in again["degraded"], {"burst_degraded": sorted({",".join(r.json()["degraded"]) for r in rs}),
                                                  "again": again["degraded"], "timings": again.get("timings_ms")}
        post(cfg, [{"event_id": f"inv-{time.time_ns()}", "user_id": 4321, "item_id": 1, "watch_ratio": 3.0}])
        time.sleep(1.0)                                   # the worker's Lua deletes rec:{uid}
        fresh = httpx.get(f"{api}/recommend", params={"user_id": 4321}, timeout=5).json()
        assert "cache_hit" not in fresh["degraded"]
    finally:
        httpx.post(f"{api}/admin/cache", params={"ttl_s": 0}, timeout=5)
