"""Open-loop load generator.

Requests are *scheduled* at fixed times (t0 + i / rps) whether or not earlier
ones finished — like real users, who don't wait for each other. Latency is
measured from the scheduled start, so a stalled server shows up as latency
instead of silently lowering the request rate ("coordinated omission").
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx
import numpy as np


@dataclass
class Sample:
    t: float            # scheduled offset from start (s)
    latency_ms: float   # from scheduled time to response
    status: int         # HTTP status, 0 = transport error / timeout
    degraded: list[str] = field(default_factory=list)
    served_by: str = ""


async def open_loop(base_url: str, rps: float, duration_s: float, users: np.ndarray, n: int = 10,
                    timeout_s: float = 2.0, seed: int = 0, on_tick=None) -> list[Sample]:
    rng = np.random.default_rng(seed)
    total = int(rps * duration_s)
    chosen = rng.choice(users, size=total)
    samples: list[Sample] = []
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=500)
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=timeout_s) as client:
        async def one(i: int, scheduled: float) -> None:
            try:
                r = await client.get("/recommend", params={"user_id": int(chosen[i]), "n": n})
                body = r.json() if r.status_code == 200 else {}
                s = Sample(scheduled - t0, (time.perf_counter() - scheduled) * 1000, r.status_code,
                           body.get("degraded", []), body.get("served_by", ""))
            except (httpx.HTTPError, ValueError):
                s = Sample(scheduled - t0, (time.perf_counter() - scheduled) * 1000, 0)
            samples.append(s)

        t0 = time.perf_counter()
        tasks = []
        next_tick = 1.0
        for i in range(total):
            scheduled = t0 + i / rps
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i, scheduled)))
            if on_tick and scheduled - t0 >= next_tick:
                on_tick(next_tick)
                next_tick += 1.0
        await asyncio.gather(*tasks)
    return samples


def summarize(samples: list[Sample], duration_s: float) -> dict:
    lat = np.array([s.latency_ms for s in samples])
    ok = np.array([s.status == 200 for s in samples])
    deg = np.array([bool(s.degraded) for s in samples])
    labels: dict[str, int] = {}
    for s in samples:
        for d in s.degraded:
            key = d.split(":")[0]
            labels[key] = labels.get(key, 0) + 1
    return {"degraded_labels": labels, "requests": len(samples), "achieved_rps": round(len(samples) / duration_s, 1),
            "error_rate": round(float(1 - ok.mean()), 5), "degraded_rate": round(float(deg[ok].mean()) if ok.any() else 0, 5),
            "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "p99_ms": round(float(np.percentile(lat, 99)), 2), "max_ms": round(float(lat.max()), 2)}


def per_second(samples: list[Sample], duration_s: float) -> list[dict]:
    rows = []
    for sec in range(int(np.ceil(duration_s))):
        win = [s for s in samples if sec <= s.t < sec + 1]
        if not win:
            continue
        lat = np.array([s.latency_ms for s in win])
        rows.append({"t": sec, "n": len(win), "availability": round(float(np.mean([s.status == 200 for s in win])), 4),
                     "degraded": round(float(np.mean([bool(s.degraded) for s in win])), 4),
                     "p50_ms": round(float(np.percentile(lat, 50)), 2), "p99_ms": round(float(np.percentile(lat, 99)), 2)})
    return rows
