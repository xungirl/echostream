"""Stage 5: replay REAL KuaiRec events (big matrix, original time order) through
the full pipeline: client -> nginx -> API (POST /events, batches) -> Redis
Streams partitions -> workers -> sharded feature store.

  --rate R     paced at R events/s (e2e latency without a backlog)
  --rate 0     as fast as the system accepts (sustained throughput)
  --limit N    only the first N events (time order); default all 12,530,806

After the run every user's counters in the feature store are compared with the
ground truth computed directly from the parquet file (exact match required).

    uv run python scripts/stage5_replay.py --rate 0 --tag full
"""
import argparse
import asyncio
import platform
import threading
import time

import httpx
import numpy as np
import orjson
import redis

from echostream.cluster.client import cluster
from echostream.core.hashing import ConsistentHashRing
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import write_json
from echostream.serving.worker import LAT_BUCKETS_MS

BATCH = 1000
MAX_BACKLOG = 400_000   # producer backpressure


def worker_totals(r, n_workers):
    applied, dup, hist = 0.0, 0.0, np.zeros(len(LAT_BUCKETS_MS))
    for i in range(n_workers):
        h = r.hgetall(f"stats:worker:w{i}")
        applied += float(h.get(b"applied", 0))
        dup += float(h.get(b"duplicates", 0))
        if b"lat_hist" in h:
            hist += np.array([float(x) for x in h[b"lat_hist"].decode().split(",")])
    return applied, dup, hist


def backlog(r, n_partitions):
    total = 0
    for p in range(n_partitions):
        g = r.xinfo_groups(f"events:{p}")
        total += ((g[0].get("lag") or 0) + (g[0].get("pending") or 0)) if g else 0
    return total


def hist_quantile(hist, q):
    c = np.cumsum(hist)
    if c[-1] == 0:
        return None
    idx = int(np.searchsorted(c, q * c[-1]))
    return LAT_BUCKETS_MS[idx]


async def produce(base, users, items, wrs, tss, rate, state, r, n_partitions, concurrency=12):
    n = len(users)
    batches = [(s, min(s + BATCH, n)) for s in range(0, n, BATCH)]
    queue: asyncio.Queue = asyncio.Queue()
    for b in batches:
        queue.put_nowait(b)
    t0 = time.perf_counter()

    async def sender(client):
        while not queue.empty():
            s, e = queue.get_nowait()
            if rate:
                due = t0 + s / rate
                if due > time.perf_counter():
                    await asyncio.sleep(due - time.perf_counter())
            while state["backlog"] > MAX_BACKLOG:
                await asyncio.sleep(0.05)
            body = orjson.dumps({"events": [
                {"event_id": f"kr{j}", "user_id": int(users[j]), "item_id": int(items[j]),
                 "watch_ratio": float(wrs[j]), "ts": float(tss[j])} for j in range(s, e)]})
            for attempt in range(5):
                try:
                    resp = await client.post("/events", content=body, headers={"content-type": "application/json"})
                    if resp.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2 * (attempt + 1))
            else:
                state["failed_batches"] += 1
            state["produced"] += e - s

    async with httpx.AsyncClient(base_url=base, timeout=30.0,
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        await asyncio.gather(*(sender(client) for _ in range(concurrency)))
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    a = KuaiRecAdapter()
    df = a.interactions()
    if args.limit:
        df = df.iloc[:args.limit]
    users, items = df.user_id.to_numpy(), df.item_id.to_numpy()
    wrs, tss = df.value.to_numpy(np.float64), df.ts.to_numpy()
    n = len(df)
    print(f"replaying {n:,} events, rate={args.rate or 'max'}", flush=True)

    env = {"ECHO_PORT_BASE": "17000", "ECHO_DEDUPE_TTL": "3600"}
    with cluster("full", env=env) as cfg:
        r = redis.Redis(port=cfg.events_redis_port)
        state = {"produced": 0, "backlog": 0, "failed_batches": 0}
        timeline = []
        stop = threading.Event()
        t_start = time.time()

        def sampler():
            while not stop.is_set():
                try:
                    state["backlog"] = backlog(r, cfg.n_partitions)
                    applied, dup, _ = worker_totals(r, cfg.n_workers)
                    timeline.append({"t": round(time.time() - t_start, 2), "produced": state["produced"],
                                     "applied": int(applied), "backlog": state["backlog"]})
                except Exception:
                    pass
                time.sleep(1.0)
        th = threading.Thread(target=sampler)
        th.start()
        produce_s = asyncio.run(produce(f"http://127.0.0.1:{cfg.lb_port}", users, items, wrs, tss,
                                        args.rate, state, r, cfg.n_partitions))
        while True:
            applied, dup, hist = worker_totals(r, cfg.n_workers)
            if applied >= n or time.time() - t_start > produce_s + 1800:
                break
            time.sleep(0.5)
        total_s = time.time() - t_start
        stop.set()
        th.join()
        applied, dup, hist = worker_totals(r, cfg.n_workers)

        # ---- correctness vs ground truth --------------------------------------
        gt = {}
        etype = np.where(wrs < 0.3, "skip", np.where(wrs >= 1.0, "complete", "play"))
        for key, mask in (("n", np.ones(n, bool)), ("complete", etype == "complete"), ("skip", etype == "skip"),
                          ("play", etype == "play"), ("pos", wrs > 2.0)):
            gt[key] = np.bincount(users[mask], minlength=a.n_users)
        ring = ConsistentHashRing(list(cfg.feature_nodes), 100)
        clients = {k: redis.Redis.from_url(u) for k, u in cfg.feature_nodes.items()}
        mismatched = 0
        for u in range(a.n_users):
            h = clients[ring.get_node(f"user:{u}")].hgetall(f"rt:{u}:cnt")
            got = {k: int(float(h.get(k.encode(), 0))) for k in gt}
            if any(got[k] != gt[k][u] for k in gt):
                mismatched += 1
        dlq = r.xlen("events:dlq")
        node_mem = {k: c.info("memory")["used_memory_human"] for k, c in clients.items()}

    steady = [x for x in timeline if x["t"] > 5]
    rates = np.diff([x["applied"] for x in steady]) / np.maximum(np.diff([x["t"] for x in steady]), 1e-9) if len(steady) > 2 else []
    out = {"kind": "real-event replay (KuaiRec big_matrix, original time order)", "machine": platform.platform(),
           "events": n, "target_rate": args.rate or "max", "batch": BATCH,
           "cluster": f"{cfg.n_api} API replicas + nginx, {cfg.n_partitions} partitions, {cfg.n_workers} workers, "
                      f"{cfg.n_feature_nodes} feature nodes",
           "produce_seconds": round(produce_s, 1), "total_seconds_until_all_applied": round(total_s, 1),
           "throughput_events_per_s_end_to_end": round(n / total_s, 1),
           "applied_rate_per_s_median": round(float(np.median(rates)), 1) if len(rates) else None,
           "applied_rate_per_s_p10": round(float(np.percentile(rates, 10)), 1) if len(rates) else None,
           "applied": int(applied), "duplicates_skipped": int(dup), "dlq": int(dlq),
           "failed_batches": state["failed_batches"],
           "e2e_latency_ms_bucket_upper_bounds": {"p50": hist_quantile(hist, 0.5), "p95": hist_quantile(hist, 0.95),
                                                  "p99": hist_quantile(hist, 0.99)},
           "max_backlog": max((x["backlog"] for x in timeline), default=0),
           "users_checked": a.n_users, "users_with_mismatched_counters": mismatched,
           "feature_node_memory": node_mem, "timeline": timeline,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    write_json(f"results/stage5_replay_{args.tag}.json", out)
    print({k: v for k, v in out.items() if k != "timeline"}, flush=True)


if __name__ == "__main__":
    main()
