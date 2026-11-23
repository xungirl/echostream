"""Stage 5: chaos runs under steady open-loop load (100 rps of GET /recommend +
~100 events/s of POST /events). Fault at t=10 s, heal at t=30 s (if the fault
needs healing), 45 s per run, 3 repeats per scenario.

    uv run python scripts/stage5_chaos.py  -> results/stage5_chaos.json

Recovery time = seconds from the fault (auto-recovering faults) or from the heal
until the last "bad" second ends; a bad second has availability < 100%, degraded
share > baseline + 1 pp, or p99 > 2x the pre-fault p99.
"""
import argparse
import asyncio
import platform
import threading
import time

import httpx
import numpy as np
import redis

from echostream.bench.loadgen import open_loop, per_second, summarize
from echostream.cluster.client import cluster, control, wait_ready
from echostream.offline.evaluate import write_json

SCENARIOS = {
    "baseline":          {"fault": None},
    "kill_shard_primary": {"fault": ("kill", "shard0-r0", {}), "heal": None},
    "slow_shard":        {"fault": ("delay", "shard1-r0", {"ms": 500}), "heal": ("delay", "shard1-r0", {"ms": 0})},
    "kill_worker":       {"fault": ("stop", "worker0", {}), "heal": ("start", "worker0", {})},
    "stop_feature_node": {"fault": ("freeze", "redis-feat1", {}), "heal": ("thaw", "redis-feat1", {})},
    "kill_api_replica":  {"fault": ("kill", "api0", {}), "heal": None},
}
FAULT_T, HEAL_T, DURATION, RPS, EVENTS_PER_S = 10.0, 30.0, 45.0, 100.0, 100


def event_poster(cfg, stop: threading.Event, t0: float, log: list):
    rng = np.random.default_rng(1)
    k = 0
    with httpx.Client(base_url=f"http://127.0.0.1:{cfg.lb_port}", timeout=2.0) as c:
        while not stop.is_set():
            batch = [{"event_id": f"chaos-{t0:.0f}-{k + j}", "user_id": int(rng.integers(0, 7176)),
                      "item_id": int(rng.integers(0, 10728)), "watch_ratio": float(rng.random() * 3)} for j in range(10)]
            k += 10
            try:
                ok = c.post("/events", json={"events": batch}).status_code == 200
            except httpx.HTTPError:
                ok = False
            log.append((time.time() - t0, ok))
            time.sleep(10 / EVENTS_PER_S)


def backlog_sampler(cfg, stop: threading.Event, t0: float, out: list):
    r = redis.Redis(port=cfg.events_redis_port, socket_timeout=1)
    while not stop.is_set():
        total = 0
        for p in range(cfg.n_partitions):
            try:
                g = r.xinfo_groups(f"events:{p}")
                total += (g[0].get("lag") or 0) + (g[0].get("pending") or 0) if g else 0
            except Exception:
                pass
        out.append((time.time() - t0, total))
        time.sleep(0.25)


def recovery(seconds, base_deg, base_p99, start):
    bad = [s["t"] for s in seconds if s["t"] >= start and s["t"] < start + 20 and
           (s["availability"] < 1.0 or s["degraded"] > base_deg + 0.01 or s["p99_ms"] > 2 * base_p99)]
    return 0.0 if not bad else float(max(bad) + 1 - start)


def run_scenario(cfg, name, spec, rep):
    t0 = time.time()
    stop = threading.Event()
    ev_log, backlog = [], []
    threads = [threading.Thread(target=event_poster, args=(cfg, stop, t0, ev_log)),
               threading.Thread(target=backlog_sampler, args=(cfg, stop, t0, backlog))]

    def injector():
        if spec["fault"] is None:
            return
        time.sleep(max(0, t0 + FAULT_T - time.time()))
        a, target, params = spec["fault"]
        control(cfg, a, target, **params)
        if spec.get("heal"):
            time.sleep(max(0, t0 + HEAL_T - time.time()))
            a, target, params = spec["heal"]
            control(cfg, a, target, **params)
    threads.append(threading.Thread(target=injector))
    for t in threads:
        t.start()
    samples = asyncio.run(open_loop(f"http://127.0.0.1:{cfg.lb_port}", RPS, DURATION, np.arange(7176), seed=rep))
    stop.set()
    for t in threads:
        t.join()
    secs = per_second(samples, DURATION)
    pre = [s for s in samples if s.t < FAULT_T]
    base = summarize(pre, FAULT_T)
    win = [s for s in samples if FAULT_T <= s.t < (HEAL_T if spec.get("heal") else FAULT_T + 20)]
    start = HEAL_T if spec.get("heal") else FAULT_T
    bl = np.array(backlog) if backlog else np.zeros((1, 2))
    after = bl[bl[:, 0] >= FAULT_T]
    drained = [t for t, v in after if t >= start and v < 50]
    row = {"scenario": name, "repeat": rep, "overall": summarize(samples, DURATION), "pre_fault": base,
           "fault_window": summarize(win, (HEAL_T - FAULT_T) if spec.get("heal") else 20) if win else None,
           "recovery_s": recovery(secs, base["degraded_rate"], base["p99_ms"], start) if spec["fault"] else 0.0,
           "events_post_success": round(float(np.mean([ok for _, ok in ev_log])), 5) if ev_log else None,
           "event_backlog_max": int(after[:, 1].max()) if len(after) else 0,
           "event_backlog_drained_s_after": round(drained[0] - start, 2) if drained else None,
           "per_second": secs}
    print(name, rep, {k: row[k] for k in ("recovery_s", "events_post_success", "event_backlog_max",
                                         "event_backlog_drained_s_after")},
          "fault_window:", {k: row["fault_window"][k] for k in ("error_rate", "degraded_rate", "p99_ms")}
          if row["fault_window"] else None, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    rows = []
    names = [n for n in SCENARIOS if not args.only or n in args.only.split(",")]
    with cluster("full", env={"ECHO_PORT_BASE": "17000"}) as cfg:
        asyncio.run(open_loop(f"http://127.0.0.1:{cfg.lb_port}", 50, 5, np.arange(7176), seed=99))  # warm-up
        for name in names:
            for rep in range(args.repeats):
                rows.append(run_scenario(cfg, name, SCENARIOS[name], rep))
                wait_ready(cfg, "full", timeout=60)   # everything back before the next run
                time.sleep(3)
    summary = {}
    for name in names:
        rs = [r for r in rows if r["scenario"] == name]
        fw = [r["fault_window"] for r in rs if r["fault_window"]]
        summary[name] = {
            "availability_overall_mean": round(float(np.mean([1 - r["overall"]["error_rate"] for r in rs])), 5),
            "degraded_in_fault_window_mean": round(float(np.mean([f["degraded_rate"] for f in fw])), 4) if fw else None,
            "p99_in_fault_window_ms": [f["p99_ms"] for f in fw] if fw else None,
            "p99_pre_fault_ms": [r["pre_fault"]["p99_ms"] for r in rs],
            "recovery_s": [r["recovery_s"] for r in rs],
            "event_backlog_max": [r["event_backlog_max"] for r in rs],
            "event_backlog_drained_s_after": [r["event_backlog_drained_s_after"] for r in rs],
            "events_post_success": [r["events_post_success"] for r in rs]}
    write_json("results/stage5_chaos.json", {
        "kind": "systems / chaos (real model + users, single laptop, multi-node on localhost)",
        "machine": platform.platform(), "rps": RPS, "events_per_s": EVENTS_PER_S, "fault_at_s": FAULT_T,
        "heal_at_s": HEAL_T, "duration_s": DURATION, "date": time.strftime("%Y-%m-%d %H:%M"),
        "summary": summary, "runs": rows})


if __name__ == "__main__":
    main()
