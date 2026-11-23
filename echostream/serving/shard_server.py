"""Vector-index shard server (one process = one replica of one shard).

Item i belongs to shard `stable_hash(item:i) % n_shards`. Each shard keeps one
faiss index per model version so a new version can be loaded next to the old
one and switched without downtime (Stage 8). Chaos/demo hooks: an injected
delay and a "stale index" switch.

Run: python -m echostream.serving.shard_server --shard 0 --replica 0 --n-shards 4 --port 17100
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import threading
import time
from pathlib import Path

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import ORJSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel

from echostream.core.ann import ann_search, build_index, set_search_params
from echostream.core.hashing import stable_hash

faiss.omp_set_num_threads(1)
SEARCH_SECONDS = Histogram("echostream_shard_search_seconds", "faiss search time inside the shard",
                           buckets=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])


def shard_of(item_id: int, n_shards: int) -> int:
    return stable_hash(f"item:{item_id}") % n_shards


class SearchRequest(BaseModel):
    vector: list[float] | None = None
    vector_b64: str | None = None    # float32 little-endian, base64 (compact; ~10x cheaper to parse than JSON floats)
    k: int = 100
    version: str | None = None
    exclude: list[int] = []          # items the user has already seen: filtered here so top-k stays exact
    exclude_b64: str | None = None   # same as exclude, int32 little-endian base64

    def query(self) -> np.ndarray:
        if self.vector_b64 is not None:
            return np.frombuffer(base64.b64decode(self.vector_b64), dtype="<f4")
        return np.asarray(self.vector, dtype=np.float32)

    def excluded(self) -> set[int]:
        if self.exclude_b64:
            return set(np.frombuffer(base64.b64decode(self.exclude_b64), dtype="<i4").tolist())
        return set(self.exclude)


class LoadRequest(BaseModel):
    version: str
    path: str             # directory with item_emb.npy
    make_default: bool = False


class ShardState:
    def __init__(self, shard: int, replica: int, n_shards: int):
        self.shard, self.replica, self.n_shards = shard, replica, n_shards
        self.indexes: dict[str, tuple[faiss.Index, np.ndarray]] = {}
        self.default_version: str | None = None
        self.delay_ms = 0.0
        self.stale_version: str | None = None   # fault injection: answer with this version instead
        self.lock = threading.Lock()
        self.started = time.time()

    def load_version(self, version: str, emb_path: Path) -> int:
        vecs = np.load(emb_path / "item_emb.npy")
        ids = np.array([i for i in range(len(vecs)) if shard_of(i, self.n_shards) == self.shard], dtype=np.int64)
        index = build_index(vecs[ids], "flat")
        with self.lock:
            self.indexes[version] = (index, ids)
            if self.default_version is None:
                self.default_version = version
        return len(ids)

    def load_synthetic(self, total: int, dim: int, kind: str, seed: int, **params) -> int:
        """Synthetic scale test: this shard's slice of `total` clustered random vectors.
        Generated in-process from a seed (nothing is written to disk)."""
        vecs, ids = synthetic_slice(total, self.n_shards, self.shard, dim, seed)
        faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8) // self.n_shards))   # build in parallel...
        index = build_index(vecs, kind, **params)
        faiss.omp_set_num_threads(1)                                              # ...serve single-threaded
        del vecs
        with self.lock:
            self.indexes["synthetic"] = (index, ids)
            self.default_version = "synthetic"
        return len(ids)


def synthetic_slice(total: int, n_shards: int, shard: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic slice `shard` of a synthetic corpus: 256 Gaussian clusters on the unit sphere."""
    sizes = [total // n_shards + (1 if s < total % n_shards else 0) for s in range(n_shards)]
    per, offset = sizes[shard], sum(sizes[:shard])
    rng = np.random.default_rng(seed * 1000 + shard)
    centers = np.random.default_rng(seed).standard_normal((256, dim)).astype(np.float32)
    vecs = np.empty((per, dim), dtype=np.float32)
    for start in range(0, per, 500_000):
        n = min(500_000, per - start)
        x = centers[rng.integers(0, 256, n)] + 0.6 * rng.standard_normal((n, dim)).astype(np.float32)
        vecs[start:start + n] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return vecs, np.arange(offset, offset + per, dtype=np.int64)


async def reconcile_versions(state: ShardState) -> None:
    """Stage 8: load every model version the registry marks live/canary/shadow."""
    from echostream.store import db
    while True:
        try:
            with db.connect() as conn:
                rows = conn.execute("SELECT version, bundle_path FROM model_registry "
                                    "WHERE status IN ('live', 'canary', 'shadow')").fetchall()
            for r in rows:
                if r["version"] not in state.indexes:
                    await asyncio.to_thread(state.load_version, r["version"], Path(r["bundle_path"]))
        except Exception:
            pass
        await asyncio.sleep(1.0)


def create_app(state: ShardState) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(reconcile_versions(state)) if os.environ.get("ECHO_PG") == "1" else None
        yield
        if task:
            task.cancel()

    app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

    @app.post("/search")
    async def search(req: SearchRequest, x_deadline_ms: float | None = Header(default=None)):
        # deadline propagation: if the caller already gave up, don't burn CPU
        if x_deadline_ms is not None and time.time() * 1000 > x_deadline_ms:
            raise HTTPException(504, "deadline already passed")
        if state.delay_ms > 0:
            await asyncio.sleep(state.delay_ms / 1000)
        version = state.stale_version or req.version or state.default_version
        if version not in state.indexes:
            version = state.default_version
        index, ids = state.indexes[version]
        t = time.perf_counter()
        exclude = req.excluded()
        # over-fetch by the number of excluded items, then filter -> exact top-k of the allowed items
        res_ids, res_scores = ann_search(index, req.query(), req.k + len(exclude), id_map=ids)
        keep = [j for j, i in enumerate(res_ids[0].tolist()) if i not in exclude][:req.k]
        took = time.perf_counter() - t
        SEARCH_SECONDS.observe(took)
        return {"ids": res_ids[0][keep].tolist(), "scores": res_scores[0][keep].tolist(), "shard": state.shard,
                "replica": state.replica, "version": version, "search_ms": took * 1000}

    @app.get("/health")
    async def health():
        return {"ok": True, "shard": state.shard, "replica": state.replica, "versions": sorted(state.indexes),
                "default_version": state.default_version, "delay_ms": state.delay_ms,
                "stale_version": state.stale_version, "pid": os.getpid(),
                "n_items": {v: int(len(ids)) for v, (_, ids) in state.indexes.items()},
                "uptime_s": round(time.time() - state.started, 1)}

    @app.post("/admin/delay")
    async def set_delay(ms: float = 0.0):
        state.delay_ms = ms
        return {"delay_ms": ms}

    @app.post("/admin/stale")
    async def set_stale(version: str | None = None):
        state.stale_version = version
        return {"stale_version": version}

    @app.post("/admin/load")
    async def load(req: LoadRequest):
        # hot reload: build the new index in a thread; searches keep using the old one meanwhile
        n = await asyncio.to_thread(state.load_version, req.version, Path(req.path))
        if req.make_default:
            state.default_version = req.version
        return {"loaded": req.version, "n_items": n}

    @app.post("/admin/search_params")
    async def search_params(ef_search: int | None = None, nprobe: int | None = None):
        for index, _ in state.indexes.values():
            set_search_params(index, ef_search=ef_search, nprobe=nprobe)
        return {"ef_search": ef_search, "nprobe": nprobe}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--bundle", default=None, help="serving bundle dir (contains versions.json)")
    ap.add_argument("--synthetic", type=int, default=0, help="synthetic scale test: total vectors")
    ap.add_argument("--synthetic-kind", default="ivf")
    ap.add_argument("--synthetic-params", default="{}")
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()
    state = ShardState(args.shard, args.replica, args.n_shards)
    if args.synthetic:
        state.load_synthetic(args.synthetic, args.dim, args.synthetic_kind, seed=7,
                             **json.loads(args.synthetic_params))
    elif args.bundle:
        bundle = Path(args.bundle)
        manifest = json.loads((bundle / "manifest.json").read_text())
        state.load_version(manifest["version"], bundle)
    uvicorn.run(create_app(state), host="127.0.0.1", port=args.port, log_level="warning", access_log=False,
                loop="uvloop", http="httptools")


if __name__ == "__main__":
    main()
