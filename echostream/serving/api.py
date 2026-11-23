"""HTTP API (FastAPI). Stateless: all state is in the bundle files (read-only),
the feature store (Redis) and the event streams (Redis), so any number of
replicas can run behind the load balancer.

Run: python -m echostream.serving.api --port 17001
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field

from echostream.domain.kuairec import KuaiRecAdapter
from echostream.serving.ann_client import ShardClient
from echostream.serving.bundle import load_bundle
from echostream.serving.config import CLUSTER, EVENTS
from echostream.serving.events import stream_for_user, stream_name, to_fields
from echostream.serving.feature_store import FeatureStore
from echostream.serving.recommender import Recommender, Rollout

REQUESTS = Counter("echostream_requests_total", "HTTP requests", ["route", "status"])
EVENTS_IN = Counter("echostream_events_ingested_total", "events accepted by POST /events")
ADAPTER = KuaiRecAdapter()


class EventIn(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=64, pattern=r"^[^:\s]+$")
    user_id: int = Field(ge=0)
    item_id: int = Field(ge=0)
    watch_ratio: float = Field(ge=0)
    action: Literal["play", "complete", "skip"] | None = None
    ts: float | None = None


class EventBatch(BaseModel):
    events: list[EventIn] = Field(max_length=5000)


class State:
    rec: Recommender
    events: aioredis.Redis
    replica_name: str
    reconcile_error: str | None = None


S = State()


async def reconcile_rollout() -> None:
    """Stage 8: Postgres holds the desired rollout state (default / canary / shadow
    versions). Every replica polls it, hot-loads missing bundles in a thread (the
    old ones keep serving meanwhile), then switches atomically. A replica that
    restarts mid-rollout converges to the same state."""
    from echostream.store import db
    while True:
        try:
            with db.connect() as conn:
                st = db.get_rollout(conn)
                if st:
                    wanted = [v for v in (st["default_version"], st["canary_version"], st["shadow_version"]) if v]
                    paths = db.bundle_paths(conn, [v for v in wanted if v not in S.rec.bundles])
            if st:
                for v, path in paths.items():
                    b = await asyncio.to_thread(load_bundle, path)
                    b.version = v             # the registry name wins over the manifest name
                    S.rec.bundles[v] = b
                if all(v in S.rec.bundles for v in wanted):
                    S.rec.rollout = Rollout(default=st["default_version"], canary=st["canary_version"],
                                            canary_percent=st["canary_percent"], shadow=st["shadow_version"])
                    for v in [v for v in S.rec.bundles if v not in wanted]:
                        S.rec.bundles.pop(v)          # retire unused versions (free memory)
            S.reconcile_error = None
        except Exception as e:                        # Postgres down: keep serving the current state
            S.reconcile_error = f"{type(e).__name__}: {e}"[:200]
        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    b = load_bundle(CLUSTER.bundle_dir)
    shards = ShardClient(CLUSTER.shard_urls())
    fs = FeatureStore(CLUSTER.feature_nodes, socket_timeout=0.2)
    mmr = os.environ.get("ECHO_MMR_LAMBDA")
    S.rec = Recommender({b.version: b}, Rollout(default=b.version), shards, fs,
                        mmr_lambda=float(mmr) if mmr else None)
    S.events = aioredis.from_url(CLUSTER.events_redis_url, socket_timeout=0.5)
    S.replica_name = os.environ.get("ECHO_REPLICA_NAME", f"api-{os.getpid()}")
    S.rec.cache_ttl_s = float(os.environ.get("ECHO_CACHE_TTL_S", "0"))
    tasks = []
    if os.environ.get("ECHO_PG") == "1":
        tasks.append(asyncio.create_task(reconcile_rollout()))
    yield
    for t in tasks:
        t.cancel()
    await shards.close()
    await fs.close()
    await S.events.aclose()


app = FastAPI(title="echostream", default_response_class=ORJSONResponse, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/recommend")
async def recommend(user_id: int, n: int = 10, mmr_lambda: float | None = None):
    b = next(iter(S.rec.bundles.values()))
    if not 0 <= user_id < b.n_users:
        REQUESTS.labels("recommend", "404").inc()
        raise HTTPException(404, f"unknown user {user_id}")
    n = max(1, min(n, 100))
    if mmr_lambda is None:
        res = await S.rec.cached_recommend(user_id, n)
    else:
        res = await S.rec.recommend(user_id, n, mmr_lambda=mmr_lambda)
    REQUESTS.labels("recommend", "200").inc()
    return {"user_id": res.user_id, "model_version": res.version, "served_by": S.replica_name,
            "items": res.items, "timings_ms": res.timings_ms, "shards": res.shards, "degraded": res.degraded}


@app.post("/events")
async def post_events(batch: EventBatch):
    """Append events to the user's partition stream. The client-supplied
    event_id makes retries safe: the worker applies each id at most once."""
    b = next(iter(S.rec.bundles.values()))
    by_stream: dict[str, list[dict]] = {}
    now = time.time()
    for e in batch.events:
        if e.user_id >= b.n_users or e.item_id >= len(b.item_cats_csv):
            raise HTTPException(422, f"unknown user/item in event {e.event_id}")
        ev = {"event_id": e.event_id, "user_id": e.user_id, "item_id": e.item_id, "watch_ratio": e.watch_ratio,
              "action": e.action or ADAPTER.event_type(e.watch_ratio), "ts": e.ts if e.ts is not None else now}
        by_stream.setdefault(stream_for_user(e.user_id, CLUSTER.n_partitions), []).append(
            to_fields(ev, b.item_cats_csv[e.item_id]))
    try:
        async with S.events.pipeline(transaction=False) as p:
            for stream, rows in by_stream.items():
                for fields in rows:
                    p.xadd(stream, fields)
            await p.execute()
    except Exception as ex:
        REQUESTS.labels("events", "503").inc()
        raise HTTPException(503, f"event bus unavailable: {type(ex).__name__}")
    EVENTS_IN.inc(len(batch.events))
    REQUESTS.labels("events", "200").inc()
    return {"accepted": len(batch.events), "streams": sorted(by_stream)}


@app.get("/health")
async def health():
    return {"ok": True, "replica": S.replica_name, "versions": sorted(S.rec.bundles),
            "rollout": S.rec.rollout.__dict__}


@app.get("/cluster")
async def cluster():
    """Health of every component, for the demo UI and the chaos scripts."""
    shards = await S.rec.shards.health()
    feats = await S.rec.features.ping_all() if S.rec.features else {}
    events: dict = {"up": False}
    try:
        now = time.time() * 1000
        alive = await S.events.zrangebyscore("workers:alive", now - EVENTS.lease_ms, "+inf")
        parts = []
        for p in range(CLUSTER.n_partitions):
            info = {"partition": p, "length": await S.events.xlen(stream_name(p))}
            try:
                groups = await S.events.xinfo_groups(stream_name(p))
                if groups:
                    info.update(pending=groups[0].get("pending"), lag=groups[0].get("lag"))
            except Exception:
                pass
            owner = await S.events.get(f"lease:{p}")
            info["owner"] = owner.decode() if owner else None
            parts.append(info)
        workers = {}
        for w in alive:
            wid = w.decode()
            st = await S.events.hgetall(f"stats:worker:{wid}")
            workers[wid] = {k.decode(): v.decode() for k, v in st.items() if k != b"lat_hist"}
        events = {"up": True, "workers_alive": [w.decode() for w in alive], "partitions": parts, "workers": workers,
                  "dlq_length": await S.events.xlen(EVENTS.dlq_stream)}
    except Exception:
        pass
    return {"shards": shards, "feature_nodes": feats, "events": events, "api_replica": S.replica_name}


@app.post("/chaos/{action}")
async def chaos(action: str, target: str, ms: float | None = None):
    """Demo mode only: forward a chaos action to the process supervisor."""
    if os.environ.get("ECHO_DEMO_MODE") != "1":
        raise HTTPException(403, "chaos actions are only enabled in demo mode")
    if action not in ("kill", "stop", "start", "freeze", "thaw", "delay"):
        raise HTTPException(400, "unknown action")
    params = {"ms": ms} if (action == "delay" and ms is not None) else None
    async with httpx.AsyncClient() as c:
        r = await c.post(f"http://127.0.0.1:{CLUSTER.supervisor_port}/{action}/{target}", params=params, timeout=5.0)
    return r.json()


@app.get("/supervisor/status")
async def supervisor_status():
    """Process table from the supervisor (pids, restarts, held down / frozen) for the demo UI."""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"http://127.0.0.1:{CLUSTER.supervisor_port}/status", timeout=2.0)
        return r.json()
    except httpx.HTTPError:
        return {}


@app.get("/admin/rollout/stats")
async def rollout_stats():
    def lat(d):
        import numpy as np
        a = np.array(d) if d else np.zeros(1)
        return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p99_ms": round(float(np.percentile(a, 99)), 2)}
    per = {v: {"n": st["n"], "errors": st["errors"], "degraded": st["degraded"], **lat(st["lat"])}
           for v, st in S.rec.stats.items()}
    sh = S.rec.shadow_stats
    return {"replica": S.replica_name, "rollout": S.rec.rollout.__dict__, "loaded": sorted(S.rec.bundles),
            "per_version": per, "reconcile_error": S.reconcile_error,
            "shadow": {"n": sh["n"], "errors": sh["errors"], **lat(sh["lat"]),
                       "overlap_at10_mean": round(sum(sh["overlap"]) / len(sh["overlap"]), 4) if sh["overlap"] else None},
            "cache": {"ttl_s": S.rec.cache_ttl_s, "computations": S.rec.computations}}


@app.post("/admin/rollout/reset_stats")
async def reset_stats():
    S.rec.reset_stats()
    return {"ok": True}


@app.post("/admin/cache")
async def set_cache(ttl_s: float = 0.0):
    S.rec.cache_ttl_s = ttl_s
    S.rec.computations = 0
    return {"ttl_s": ttl_s}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False,
                loop="uvloop", http="httptools", timeout_keep_alive=30)


if __name__ == "__main__":
    main()
