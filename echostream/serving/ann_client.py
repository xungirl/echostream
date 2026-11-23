"""Scatter-gather client for the sharded vector index.

For every shard (in parallel): try a replica with a per-attempt timeout that
never exceeds the request's remaining budget; on error/timeout try the other
replica. A replica that fails `eject_after_failures` times in a row is ejected
(skipped as first choice) for `eject_ms` — simple outlier detection so a slow
or dead primary stops costing every request a timeout. If no replica of a shard
answers in time, the merge continues without it and the response is labeled
partial (core.scatter_gather.merge_topk).
"""
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from echostream.core.scatter_gather import MergeResult, merge_topk
from echostream.serving.config import TIMEOUTS, Timeouts


@dataclass
class ReplicaState:
    url: str
    consecutive_failures: int = 0
    ejected_until: float = 0.0

    def healthy(self, now: float) -> bool:
        return now >= self.ejected_until


@dataclass
class ShardCall:
    shard: int
    ok: bool
    replica: int | None = None
    attempts: list[dict] = field(default_factory=list)
    ms: float = 0.0


class ShardClient:
    def __init__(self, shard_urls: list[list[str]], timeouts: Timeouts = TIMEOUTS):
        self.shards = [[ReplicaState(u) for u in replicas] for replicas in shard_urls]
        self.t = timeouts
        self.http = httpx.AsyncClient(limits=httpx.Limits(max_connections=512, max_keepalive_connections=256))
        self._rr = itertools.count()

    def _order(self, s: int) -> list[int]:
        now = time.monotonic()
        reps = self.shards[s]
        start = next(self._rr) % len(reps)              # spread load over replicas
        order = [(start + i) % len(reps) for i in range(len(reps))]
        return sorted(order, key=lambda r: not reps[r].healthy(now))  # healthy first (stable)

    def _record(self, s: int, r: int, ok: bool) -> None:
        rep = self.shards[s][r]
        if ok:
            rep.consecutive_failures = 0
            return
        rep.consecutive_failures += 1
        if rep.consecutive_failures >= self.t.eject_after_failures:
            rep.ejected_until = time.monotonic() + self.t.eject_ms / 1000
            rep.consecutive_failures = 0

    async def search_shard(self, s: int, vec: list[float], k: int, deadline: float, version: str | None,
                           exclude: list[int]) -> tuple[tuple[list[int], list[float]] | None, ShardCall]:
        call = ShardCall(shard=s, ok=False)
        t0 = time.monotonic()
        for r in self._order(s):
            remaining = deadline - time.monotonic()
            if remaining <= 0.001:
                call.attempts.append({"replica": r, "status": "no_budget"})
                break
            timeout = min(self.t.shard_ms / 1000, remaining)
            a0 = time.monotonic()
            try:
                resp = await self.http.post(
                    f"{self.shards[s][r].url}/search", json={"vector": vec, "k": k, "version": version, "exclude": exclude},
                    headers={"X-Deadline-Ms": str((time.time() + remaining) * 1000)}, timeout=timeout)
                resp.raise_for_status()
                body = resp.json()
                self._record(s, r, True)
                call.attempts.append({"replica": r, "status": "ok", "ms": round((time.monotonic() - a0) * 1000, 2)})
                call.ok, call.replica = True, r
                call.ms = (time.monotonic() - t0) * 1000
                return (body["ids"], body["scores"]), call
            except (httpx.TimeoutException, asyncio.TimeoutError):
                status = "timeout"
            except (httpx.HTTPError, ValueError, KeyError):
                status = "error"
            self._record(s, r, False)
            call.attempts.append({"replica": r, "status": status, "ms": round((time.monotonic() - a0) * 1000, 2)})
        call.ms = (time.monotonic() - t0) * 1000
        return None, call

    async def search(self, user_vec: np.ndarray, k: int, deadline: float, version: str | None = None,
                     exclude: list[int] | None = None) -> tuple[MergeResult, list[ShardCall]]:
        vec = [float(x) for x in user_vec]
        ex = exclude or []
        res = await asyncio.gather(*(self.search_shard(s, vec, k, deadline, version, ex)
                                     for s in range(len(self.shards))))
        merged = merge_topk([r for r, _ in res], k)
        return merged, [c for _, c in res]

    async def health(self) -> list[list[dict]]:
        async def one(s, r, rep):
            try:
                resp = await self.http.get(f"{rep.url}/health", timeout=0.5)
                return {**resp.json(), "up": True, "url": rep.url, "ejected": not rep.healthy(time.monotonic())}
            except Exception:
                return {"shard": s, "replica": r, "up": False, "url": rep.url}
        tasks = [[one(s, r, rep) for r, rep in enumerate(reps)] for s, reps in enumerate(self.shards)]
        return [list(await asyncio.gather(*row)) for row in tasks]

    async def close(self) -> None:
        await self.http.aclose()
