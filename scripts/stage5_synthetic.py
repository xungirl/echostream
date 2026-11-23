"""Stage 5: SYNTHETIC scale test of the sharded vector index.

The real catalog is only 10,728 videos, so to see how sharding behaves at
scale we fill the shards with synthetic clustered 64-d vectors (generated
inside each shard process from a seed; nothing on disk) and measure
scatter-gather query latency and per-shard memory vs shard count.
These numbers say NOTHING about recommendation quality.

    uv run python scripts/stage5_synthetic.py -> results/stage5_synthetic.json
"""
import argparse
import asyncio
import json
import math
import platform
import time

import numpy as np
import psutil

from echostream.cluster.client import cluster, status
from echostream.offline.evaluate import write_json
from echostream.serving.ann_client import ShardClient
from echostream.serving.config import Timeouts

DIM = 64


def queries(n, seed=123):
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(7).standard_normal((256, DIM)).astype(np.float32)  # same centers as shards
    q = centers[rng.integers(0, 256, n)] + 0.6 * rng.standard_normal((n, DIM)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


async def measure(cfg, qs, k=100):
    client = ShardClient(cfg.shard_urls(), Timeouts(total_ms=5000, shard_ms=5000))
    lat, partial = [], 0
    for q in qs[:20]:   # warm-up
        await client.search(q, k, time.monotonic() + 5)
    for q in qs:
        t = time.perf_counter()
        merged, calls = await client.search(q, k, time.monotonic() + 5)
        lat.append((time.perf_counter() - t) * 1000)
        partial += merged.partial
    await client.close()
    lat = np.array(lat)
    return {"p50_ms": round(float(np.percentile(lat, 50)), 3), "p99_ms": round(float(np.percentile(lat, 99)), 3),
            "partial": partial}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--totals", default="1000000,10000000")
    ap.add_argument("--shards", default="1,2,4,8")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    qs = queries(args.n_queries)
    rows = []
    for total in [int(x) for x in args.totals.split(",")]:
        for n_shards in [int(x) for x in args.shards.split(",")]:
            per = total // n_shards
            if total <= 1_000_000:
                kind, params = "flat", {}
            else:
                kind, params = "ivf", {"nlist": int(4 * math.sqrt(per))}
            env = {"ECHO_PORT_BASE": "19000", "ECHO_SHARDS": str(n_shards), "ECHO_REPLICAS": "1"}
            t0 = time.time()
            with cluster("shards", env=env, timeout=1800,
                         extra_args=["--synthetic", str(total), "--synthetic-kind", kind,
                                     "--synthetic-params", json.dumps(params)]) as cfg:
                ready_s = time.time() - t0
                if kind == "ivf":
                    import httpx
                    for s in range(n_shards):
                        httpx.post(f"http://127.0.0.1:{cfg.shard_port(s, 0)}/admin/search_params",
                                   params={"nprobe": 16}, timeout=5)
                st = status(cfg)
                rss = [psutil.Process(v["pid"]).memory_info().rss / 2**20 for v in st.values() if v["group"] == "shards"]
                reps = [asyncio.run(measure(cfg, qs)) for _ in range(args.repeats)]
            row = {"total_vectors": total, "n_shards": n_shards, "vectors_per_shard": per, "index": kind,
                   **params, **({"nprobe": 16} if kind == "ivf" else {}),
                   "startup_build_s": round(ready_s, 1),
                   "rss_mb_per_shard_max": round(max(rss), 1), "rss_mb_total": round(sum(rss), 1),
                   "p50_ms_mean": round(float(np.mean([r["p50_ms"] for r in reps])), 3),
                   "p50_ms_std": round(float(np.std([r["p50_ms"] for r in reps])), 3),
                   "p99_ms_mean": round(float(np.mean([r["p99_ms"] for r in reps])), 3),
                   "p99_ms_std": round(float(np.std([r["p99_ms"] for r in reps])), 3),
                   "partial_responses": sum(r["partial"] for r in reps)}
            rows.append(row)
            print(row, flush=True)
    write_json("results/stage5_synthetic.json", {
        "kind": "SYNTHETIC scale test (random clustered vectors; latency/memory only, no quality)",
        "machine": platform.platform(), "dim": DIM, "k": 100, "n_queries": args.n_queries,
        "repeats": args.repeats, "client": "sequential queries through the real ShardClient scatter-gather (HTTP)",
        "date": time.strftime("%Y-%m-%d %H:%M"), "rows": rows})


if __name__ == "__main__":
    main()
