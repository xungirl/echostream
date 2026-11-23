"""Stage 5: open-loop load test of GET /recommend through the nginx LB.

Starts its own cluster (4 shards x 2 replicas, 3 feature nodes, 4 workers, 4 API
replicas). Several RPS levels x 3 repeats x 30 s. All processes AND the load
generator share the same 8-core laptop -> this measures latency under load on
one machine, not horizontal scaling.

    uv run python scripts/stage5_load.py [--levels 50,100,200] [--duration 30] [--repeats 3]
"""
import argparse
import asyncio
import json
import platform
import time

import numpy as np

from echostream.bench.loadgen import open_loop, summarize
from echostream.cluster.client import cluster, fresh_run_dir
from echostream.offline.evaluate import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="25,50,100,150,200,300")
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/stage5_load.json")
    args = ap.parse_args()
    levels = [float(x) for x in args.levels.split(",")]
    users = np.arange(7176)
    rows = []
    with cluster("full", env={"ECHO_PORT_BASE": "17000"}, run_dir=fresh_run_dir("load")) as cfg:
        base = f"http://127.0.0.1:{cfg.lb_port}"
        asyncio.run(open_loop(base, 50, 5, users, seed=999))   # warm-up (not recorded)
        for rps in levels:
            for rep in range(args.repeats):
                s = asyncio.run(open_loop(base, rps, args.duration, users, seed=rep))
                row = {"target_rps": rps, "repeat": rep, **summarize(s, args.duration)}
                rows.append(row)
                print(row, flush=True)
                time.sleep(2)
    agg = []
    for rps in levels:
        rs = [r for r in rows if r["target_rps"] == rps]
        agg.append({"target_rps": rps, **{f"{k}_mean": round(float(np.mean([r[k] for r in rs])), 3)
                                         for k in ("achieved_rps", "p50_ms", "p95_ms", "p99_ms", "error_rate", "degraded_rate")},
                    **{f"{k}_std": round(float(np.std([r[k] for r in rs])), 3) for k in ("p50_ms", "p99_ms")}})
    write_json(args.out, {"kind": "systems (real KuaiRec users/model, single laptop)", "machine": platform.platform(),
                          "cluster": "4 shards x 2 replicas, 3 feature nodes, 4 workers, 4 API replicas, nginx",
                          "duration_s": args.duration, "repeats": args.repeats, "runs": rows, "summary": agg,
                          "date": time.strftime("%Y-%m-%d %H:%M")})
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
