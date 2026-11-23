"""Drive the LIVE cluster with simulated users (small-matrix ground truth).

Each simulated user runs a short session over HTTP: GET /recommend -> look up
the true watch_ratio of every shown video -> POST /events -> next slate. The
response says which model version served it, so per-version online metrics
(positives per impression) come straight from real traffic. Used by the
canary controller (Stage 8).
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx
import numpy as np


@dataclass
class SessionResult:
    user_id: int
    version: str
    shown: int = 0
    positives: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    degraded: int = 0


async def run_user(client: httpx.AsyncClient, user: int, responses: dict[int, float], slates: int, n: int,
                   tag: str) -> SessionResult:
    res = SessionResult(user, "")
    for s in range(slates):
        t = time.perf_counter()
        try:
            r = await client.get("/recommend", params={"user_id": user, "n": n})
            r.raise_for_status()
            body = r.json()
        except (httpx.HTTPError, ValueError):
            res.errors += 1
            continue
        res.latencies_ms.append((time.perf_counter() - t) * 1000)
        res.version = body["model_version"]
        res.degraded += bool([d for d in body["degraded"] if d != "cache_hit"])
        events = []
        for x in body["items"]:
            wr = responses.get(x["item_id"])
            if wr is None:          # never observed in the small matrix: the user "doesn't react"
                continue
            res.shown += 1
            res.positives += wr > 2.0
            events.append({"event_id": f"{tag}-{user}-{s}-{x['item_id']}", "user_id": user,
                           "item_id": x["item_id"], "watch_ratio": wr})
        if events:
            try:
                await client.post("/events", json={"events": events})
            except httpx.HTTPError:
                res.errors += 1
        await asyncio.sleep(0.05)   # wait for the event to land before the next slate
    return res


async def drive(base_url: str, users: list[int], responses: dict[int, dict[int, float]], slates: int = 3,
                n: int = 10, concurrency: int = 16, tag: str | None = None) -> list[SessionResult]:
    tag = tag or f"sim{int(time.time())}"
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        async def one(u):
            async with sem:
                return await run_user(client, u, responses[u], slates, n, tag)
        return list(await asyncio.gather(*(one(u) for u in users)))


def per_version(results: list[SessionResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for v in sorted({r.version for r in results if r.version}):
        rs = [r for r in results if r.version == v and r.shown]
        rates = np.array([r.positives / r.shown for r in rs])
        lat = np.concatenate([r.latencies_ms for r in rs]) if rs else np.zeros(1)
        out[v] = {"users": len(rs), "pos_rate": float(rates.mean()) if len(rates) else 0.0, "rates": rates,
                  "p99_ms": float(np.percentile(lat, 99)), "p50_ms": float(np.percentile(lat, 50)),
                  "errors": sum(r.errors for r in rs), "requests": sum(len(r.latencies_ms) + r.errors for r in rs),
                  "degraded": sum(r.degraded for r in rs)}
    return out


def small_matrix_responses(adapter) -> dict[int, dict[int, float]]:
    small = adapter.fully_observed()
    return {int(u): dict(zip(g.item_id.tolist(), g.value.tolist())) for u, g in small.groupby("user_id")}


def shuffled(users, seed=0) -> list[int]:
    u = list(users)
    random.Random(seed).shuffle(u)
    return u
