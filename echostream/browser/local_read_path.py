"""The production read + write path in ONE process, without a cluster.

Nothing here re-implements the pipeline; it only replaces the network:
  * vector shards: the real shard-server app (faiss Flat, 4 shards by stable_hash) answered through
    the real ShardClient, over an in-memory ASGI transport instead of HTTP
  * feature store: the real FeatureStore + Lua script on a temporary redis-server
  * events: posted the way POST /events does (to_fields -> XADD on the user's partition) and
    applied by the real Worker (process_once: XREADGROUP -> parse -> Lua -> XACK)
  * recommender: the real Recommender with a generous latency budget, so no stage is cut short

Used by scripts/gen_browser_golden.py (the golden file the TypeScript engine must match) and
scripts/stage13_quant_sweep.py. Each recommend() also returns a Trace of what the pipeline passed between
stages (user vector, per-path candidates, merged candidates, ranker features and scores),
recorded by wrapping three methods of this bundle instance - the production code is not changed.
"""
from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import tempfile
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import redis.asyncio as aioredis

from echostream.serving.ann_client import ShardClient
from echostream.serving.bundle import ServingBundle
from echostream.serving.config import Timeouts
from echostream.serving.events import stream_for_user, to_fields
from echostream.serving.feature_store import FeatureStore
from echostream.serving.recommender import Recommender, Result, Rollout
from echostream.serving.shard_server import ShardState, create_app
from echostream.serving.worker import Worker

# the shard server's ORJSONResponse is deprecated in the installed FastAPI; harmless here
warnings.filterwarnings("ignore", message="ORJSONResponse is deprecated")
REDIS = shutil.which("redis-server") or "/opt/homebrew/bin/redis-server"
N_SHARDS = 4
N_PARTITIONS = 8
NO_DEADLINES = Timeouts(total_ms=600_000, features_ms=600_000, shard_ms=600_000, ranker_ms=600_000)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ShardRouter(httpx.AsyncBaseTransport):
    """http://shard{s}/... -> shard s's ASGI app (the real shard_server endpoints)."""

    def __init__(self, apps: dict[str, httpx.ASGITransport]):
        self.apps = apps

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.apps[request.url.host].handle_async_request(request)


@dataclass
class Trace:
    user_vec: np.ndarray | None = None
    paths: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    merged: list = field(default_factory=list)
    features: np.ndarray | None = None
    rank_scores: np.ndarray | None = None


class _RecordingBooster:
    def __init__(self, booster, trace_ref):
        self.booster, self.trace_ref = booster, trace_ref

    def predict(self, X, **kw):
        out = self.booster.predict(X, **kw)
        self.trace_ref[0].features = np.array(X, copy=True)
        self.trace_ref[0].rank_scores = np.array(out, copy=True)
        return out


class LocalReadPath:
    def __init__(self, bundle: ServingBundle, mmr_lambda: float | None, tmp: Path):
        self.b, self.mmr_lambda, self.tmp = bundle, mmr_lambda, tmp
        self.trace_ref: list[Trace] = [Trace()]
        engine_merge = bundle.engine.merge

        def merge(paths, exclude=frozenset()):     # same call, recorded
            self.trace_ref[0].paths = {k: list(v) for k, v in paths.items()}
            out = engine_merge(paths, exclude)
            self.trace_ref[0].merged = list(out)
            return out

        engine_user_vector = bundle.engine.user_vector

        def user_vector(user, hist):
            out = engine_user_vector(user, hist)
            self.trace_ref[0].user_vec = out.copy()
            return out

        bundle.engine.merge = merge
        bundle.engine.user_vector = user_vector
        if bundle.ranker is not None:
            bundle.ranker = _RecordingBooster(bundle.ranker, self.trace_ref)
        self.seq = 0

    async def start(self) -> None:
        port = free_port()
        self.redis = subprocess.Popen([REDIS, "--port", str(port), "--bind", "127.0.0.1", "--save", "",
                                       "--appendonly", "no", "--dir", str(self.tmp)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"redis://127.0.0.1:{port}/0"
        r = aioredis.from_url(url)
        for _ in range(200):
            try:
                await r.ping()
                break
            except Exception:
                await asyncio.sleep(0.05)
        self.events = r
        # vector shards: the shard server loads item_emb.npy from a bundle directory
        emb_dir = self.tmp / "emb"
        emb_dir.mkdir(exist_ok=True)
        np.save(emb_dir / "item_emb.npy", self.b.engine.a.item_vecs)
        apps = {}
        for s in range(N_SHARDS):
            st = ShardState(s, 0, N_SHARDS)
            st.load_version(self.b.version, emb_dir)
            apps[f"shard{s}"] = httpx.ASGITransport(app=create_app(st))
        shards = ShardClient([[f"http://shard{s}"] for s in range(N_SHARDS)], NO_DEADLINES)
        await shards.http.aclose()
        shards.http = httpx.AsyncClient(transport=_ShardRouter(apps))
        self.fs = FeatureStore({"feat0": url})
        self.worker = Worker("golden", events_url=url, feature_nodes={"feat0": url}, n_partitions=N_PARTITIONS)
        await self.worker.ensure_groups()
        self.worker.owned = set(range(N_PARTITIONS))      # a single worker owns every partition
        self.rec = Recommender({self.b.version: self.b}, Rollout(default=self.b.version), shards, self.fs,
                               NO_DEADLINES, mmr_lambda=self.mmr_lambda)

    async def stop(self) -> None:
        await self.rec.shards.close()
        await self.fs.close()
        await self.worker.r.aclose()
        await self.worker.fs.close()
        await self.events.aclose()
        self.redis.terminate()
        self.redis.wait(5)

    async def post_event(self, user_id: int, item_id: int, watch_ratio: float, action: str, ts: float) -> None:
        """What POST /events does for one event, then let the worker apply it."""
        self.seq += 1
        ev = {"event_id": f"golden-{user_id}-{self.seq}", "user_id": user_id, "item_id": item_id,
              "watch_ratio": watch_ratio, "action": action, "ts": ts}
        await self.events.xadd(stream_for_user(user_id, N_PARTITIONS), to_fields(ev, self.b.item_cats_csv[item_id]))
        while await self.worker.process_once():
            pass
        if self.worker.stats["dlq"]:
            raise RuntimeError("event went to the dead-letter queue")

    async def recommend(self, user_id: int, n: int = 10) -> tuple[Result, Trace]:
        self.trace_ref[0] = Trace()
        res = await self.rec.recommend(user_id, n)
        if res.degraded:
            raise RuntimeError(f"degraded response {res.degraded}")
        return res, self.trace_ref[0]


@asynccontextmanager
async def local_read_path(bundle: ServingBundle, mmr_lambda: float | None):
    tmp = Path(tempfile.mkdtemp(prefix="echostream-local-"))
    lp = LocalReadPath(bundle, mmr_lambda, tmp)
    try:
        await lp.start()
        yield lp
    finally:
        try:
            await lp.stop()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

