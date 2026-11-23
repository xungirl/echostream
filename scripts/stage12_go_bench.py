"""Stage 12: Python worker vs Go worker on all 12,530,806 real KuaiRec events.

Two harnesses:

  drain   worker-isolated capacity. The 8 partition streams are pre-filled with
          every event (exactly the entries POST /events would write), the event
          bus is restored from that snapshot, then N workers start and the clock
          runs until every stream and pending list is empty. Nothing else runs,
          so the number is the workers' (+ Redis') capacity, not the API's.
  replay  the Stage 5 harness unchanged (client -> nginx -> 4 API replicas ->
          streams -> workers), see scripts/stage5_replay.py --worker-impl.

Each drain run records throughput, the timeline, CPU seconds and RSS of every
worker and every Redis process, and exact correctness (per-user counters vs
ground truth from the parquet, duplicates, DLQ). Raw runs are appended to
results/stage12_go_worker_runs.jsonl; `summarize` folds them into
results/stage12_go_worker.json.

    uv run python scripts/stage12_go_bench.py preload
    uv run python scripts/stage12_go_bench.py drain --impl go --workers 4 --repeats 3
    uv run python scripts/stage12_go_bench.py drain --impl go --workers 4 --kill-at 20   # chaos
    uv run python scripts/stage12_go_bench.py drain --impl go --workers 4 --pprof 30     # profile
    uv run python scripts/stage12_go_bench.py summarize
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import redis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echostream.bench.procstats import ProcSampler  # noqa: E402
from echostream.core.hashing import ConsistentHashRing, partition_for  # noqa: E402
from echostream.serving.worker import LAT_BUCKETS_MS  # noqa: E402

REDIS = shutil.which("redis-server") or "/opt/homebrew/bin/redis-server"
GO_WORKER = ROOT / "go" / "worker" / "bin" / "echostream-worker"
BENCH = ROOT / ".run" / "bench12"
RUNS = ROOT / "results" / "stage12_go_worker_runs.jsonl"
BASE = 36000                       # event bus 36020, feature nodes 36021-36023, worker metrics 36060+
N_PARTITIONS, N_FEAT = 8, 3
CONTENDERS = "echostream|nginx|uvicorn|redis-server|vite"


# ----------------------------------------------------------------- helpers
def other_clusters() -> list[str]:
    """Processes of another echostream cluster (CPU contention check). Ours run on port 36xxx."""
    out = subprocess.run(["pgrep", "-fl", CONTENDERS], capture_output=True, text=True).stdout.splitlines()
    me = {str(os.getpid()), str(os.getppid())}
    return [l[:140] for l in out if l.split()[0] not in me and ":360" not in l and "stage12_go_bench" not in l
            and "pgrep" not in l]


def wait(cond, timeout: float, step: float = 0.1) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            if cond():
                return
        except Exception:
            pass
        if time.time() > deadline:
            raise TimeoutError(f"not ready after {timeout}s")
        time.sleep(step)


def load_events(limit: int = 0):
    from echostream.domain.kuairec import KuaiRecAdapter
    a = KuaiRecAdapter()
    df = a.interactions()
    if limit:
        df = df.iloc[:limit]
    return (df.user_id.to_numpy(), df.item_id.to_numpy(), df.value.to_numpy(np.float64), df.ts.to_numpy(),
            a.n_users)


# ----------------------------------------------------------------- preload
def _fill_partition(args) -> int:
    """XADD one partition's events in time order, exactly as POST /events would write them."""
    port, p, idx, users, items, wrs, tss, cats = args
    from echostream.domain.kuairec import KuaiRecAdapter
    etype = KuaiRecAdapter().event_type
    r = redis.Redis(port=port)
    pipe = r.pipeline(transaction=False)
    now = repr(time.time() * 1000.0)
    for k in range(len(idx)):
        j, wr = int(idx[k]), float(wrs[k])
        pipe.xadd(f"events:{p}", {"event_id": f"kr{j}", "user_id": str(int(users[k])), "item_id": str(int(items[k])),
                                  "action": etype(wr), "watch_ratio": repr(wr), "ts": repr(float(tss[k])),
                                  "cats": cats[int(items[k])], "produced_ms": now})
        if (k + 1) % 5000 == 0:
            pipe.execute()
    pipe.execute()
    return len(idx)


def preload(limit: int = 0) -> Path:
    """Build (once) an RDB snapshot of the event bus holding all events; returns its path."""
    from echostream.serving.bundle import load_bundle
    d = BENCH / f"preload-{limit or 'full'}"
    rdb = d / "dump.rdb"
    if rdb.exists():
        return rdb
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    users, items, wrs, tss, _ = load_events(limit)
    cats = load_bundle(ROOT / "artifacts" / "serving" / "current").item_cats_csv
    part_of_user = {u: partition_for(int(u), N_PARTITIONS) for u in np.unique(users)}
    parts = np.array([part_of_user[u] for u in users])
    port = BASE + 90
    srv = subprocess.Popen([REDIS, "--port", str(port), "--bind", "127.0.0.1", "--dir", str(d), "--save", "",
                            "--appendonly", "no", "--protected-mode", "no"], stdout=subprocess.DEVNULL)
    try:
        r = redis.Redis(port=port)
        wait(r.ping, 10)
        t0 = time.time()
        jobs = []
        for p in range(N_PARTITIONS):
            idx = np.flatnonzero(parts == p)
            jobs.append((port, p, idx, users[idx], items[idx], wrs[idx], tss[idx], cats))
        with mp.get_context("spawn").Pool(N_PARTITIONS) as pool:
            n = sum(pool.map(_fill_partition, jobs))
        total = sum(r.xlen(f"events:{p}") for p in range(N_PARTITIONS))
        assert n == total == len(users), (n, total, len(users))
        r.save()
        print(f"preloaded {total:,} entries in {time.time() - t0:.0f}s; "
              f"event bus memory {r.info('memory')['used_memory_human']}", flush=True)
    finally:
        srv.terminate()
        srv.wait()
    return rdb


# ----------------------------------------------------------------- one drain run
def worker_cmd(impl: str, wid: str, metrics_port: int | None) -> list[str]:
    extra = ["--metrics-port", str(metrics_port)] if metrics_port else []
    if impl == "go":
        return [str(GO_WORKER), "--id", wid, *extra]
    return [sys.executable, "-m", "echostream.serving.worker", "--id", wid, *extra]


def drain(impl: str, n_workers: int, limit: int = 0, kill_at: float = 0, pprof_s: int = 0,
          truth=None, tag: str = "") -> dict:
    contention_before = other_clusters()
    rdb = preload(limit)
    run = BENCH / f"run-{impl}-{n_workers}"
    shutil.rmtree(run, ignore_errors=True)
    (run / "events").mkdir(parents=True)
    subprocess.run(["cp", "-c", str(rdb), str(run / "events" / "dump.rdb")], check=True)  # APFS clone: instant
    env = {**os.environ, "ECHO_PORT_BASE": str(BASE), "ECHO_PARTITIONS": str(N_PARTITIONS),
           "ECHO_FEATURE_NODES": str(N_FEAT), "ECHO_DEDUPE_TTL": "3600", "OMP_NUM_THREADS": "1"}
    procs: list[subprocess.Popen] = []
    workers: dict[str, subprocess.Popen] = {}
    logs = run / "logs"
    logs.mkdir()
    sampler = ProcSampler()
    try:
        # the event bus as in the cluster: AOF on, fsync every second. Redis ignores dump.rdb when it
        # starts with AOF on, so it starts with AOF off, loads the snapshot, then AOF is switched on
        procs.append(subprocess.Popen([REDIS, "--port", str(BASE + 20), "--bind", "127.0.0.1", "--dir",
                                       str(run / "events"), "--appendonly", "no", "--appendfsync", "everysec",
                                       "--save", "", "--protected-mode", "no"],
                                      stdout=open(logs / "redis-events.log", "wb"), stderr=subprocess.STDOUT))
        for i in range(N_FEAT):
            procs.append(subprocess.Popen([REDIS, "--port", str(BASE + 21 + i), "--bind", "127.0.0.1", "--save", "",
                                           "--appendonly", "no", "--protected-mode", "no"],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        r = redis.Redis(port=BASE + 20, socket_timeout=5)
        feats = {f"feat{i}": redis.Redis(port=BASE + 21 + i) for i in range(N_FEAT)}
        wait(lambda: r.ping() and all(f.ping() for f in feats.values()), 300, 0.5)
        r.config_set("appendonly", "yes")       # writes the AOF base file in the background: let it finish
        wait(lambda: (lambda i: i["aof_enabled"] == 1 and i["aof_rewrite_in_progress"] == 0
                      and i["aof_rewrite_scheduled"] == 0)(r.info("persistence")), 600, 0.5)
        assert sum(r.xlen(f"events:{p}") for p in range(N_PARTITIONS)) > 0, "snapshot not loaded"
        n = sum(r.xlen(f"events:{p}") for p in range(N_PARTITIONS))
        sampler.add("redis-events", procs[0].pid)
        for i in range(N_FEAT):
            sampler.add(f"redis-feat{i}", procs[1 + i].pid)
        sampler.start()

        t0 = time.time()
        for i in range(n_workers):
            wid = f"w{i}"
            workers[wid] = subprocess.Popen(worker_cmd(impl, wid, BASE + 60 + i), env=env, cwd=ROOT,
                                            stdout=open(logs / f"{wid}.log", "ab"), stderr=subprocess.STDOUT)
            sampler.add(wid, workers[wid].pid)
        timeline, killed, profile = [], None, None
        prof_thread = None
        if pprof_s and impl == "go":
            prof_path = ROOT / "results" / f"stage12_go_worker_pprof_{n_workers}w.pb.gz"

            def grab():
                time.sleep(5)
                subprocess.run(["curl", "-s", "-o", str(prof_path),
                                f"http://127.0.0.1:{BASE + 60}/debug/pprof/profile?seconds={pprof_s}"], check=False)
            prof_thread = threading.Thread(target=grab, daemon=True)
            prof_thread.start()
            profile = str(prof_path.relative_to(ROOT))

        def remaining() -> tuple[int, int]:
            pipe = r.pipeline(transaction=False)
            for p in range(N_PARTITIONS):
                pipe.xlen(f"events:{p}")
                pipe.xpending(f"events:{p}", "workers")
            res = pipe.execute()
            return sum(res[0::2]), sum(x["pending"] for x in res[1::2])

        last_t = 0.0
        while True:
            try:
                left, pending = remaining()
            except (redis.ResponseError, redis.ConnectionError):   # group not created yet
                left, pending = n, 0
            t = time.time() - t0
            if t - last_t >= 1.0:
                timeline.append({"t": round(t, 2), "left": left, "pending": pending})
                last_t = t
            if kill_at and killed is None and t >= kill_at:
                killed = kill_worker(r, workers, sampler, impl, env, logs, t0)
            if left == 0 and pending == 0:
                break
            if t > 7200:
                raise TimeoutError("drain did not finish in 2 h")
            time.sleep(0.1)
        total_s = time.time() - t0
        if prof_thread:
            prof_thread.join(timeout=pprof_s + 30)
        time.sleep(1.2)       # final heartbeat publishes the stats
        res_workers = sampler.stop()
        stats = worker_stats(r, list(workers))
        correctness = check_correctness(feats, truth) if truth is not None else None
        dlq = r.xlen("events:dlq")
    finally:
        sampler.halt.set()
        for p in workers.values():
            p.send_signal(signal.SIGTERM)
        for p in workers.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=60)
        shutil.rmtree(run / "events", ignore_errors=True)

    rates = np.diff([n - x["left"] for x in timeline]) / np.maximum(np.diff([x["t"] for x in timeline]), 1e-9) \
        if len(timeline) > 3 else np.array([])
    w_names = [k for k in res_workers if k.startswith("w")]
    out = {
        "harness": "drain", "impl": impl, "workers": n_workers, "events": n, "tag": tag,
        "machine": f"{platform.platform()} / {psutil.cpu_count()} cores / {psutil.virtual_memory().total // 2**30} GB",
        "seconds": round(total_s, 2), "throughput_events_per_s": round(n / total_s, 1),
        "applied_rate_per_s_median": round(float(np.median(rates)), 1) if len(rates) else None,
        "applied": stats["applied"], "duplicates": stats["duplicates"], "claimed": stats["claimed"], "dlq": dlq,
        "correctness": correctness, "killed": killed, "pprof": profile,
        "worker_cpu_s_total": round(sum(res_workers[w]["cpu_s"] for w in w_names), 1),
        "worker_cpu_s_per_1M_events": round(sum(res_workers[w]["cpu_s"] for w in w_names) / n * 1e6, 2),
        "worker_rss_mb_max": max(res_workers[w]["rss_mb_max"] for w in w_names),
        "redis_cpu_s": {k: v["cpu_s"] for k, v in res_workers.items() if k.startswith("redis")},
        "processes": res_workers, "timeline": timeline,
        "contention_before": contention_before, "contention_after": other_clusters(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    return out


def kill_worker(r, workers, sampler, impl, env, logs, t0) -> dict:
    """kill -9 the worker that holds the most un-acked entries, restart it 0.3 s later
    (what the supervisor does), and record what was pending at that moment."""
    per = {}
    for p in range(N_PARTITIONS):
        for c in r.xpending(f"events:{p}", "workers")["consumers"]:
            per[c["name"].decode()] = per.get(c["name"].decode(), 0) + c["pending"]
    victim = max(per, key=per.get) if per else "w0"
    workers[victim].send_signal(signal.SIGKILL)
    workers[victim].wait()
    t_kill = time.time() - t0
    time.sleep(0.3)
    workers[victim] = subprocess.Popen(worker_cmd(impl, victim, None), env=env, cwd=ROOT,
                                       stdout=open(logs / f"{victim}.log", "ab"), stderr=subprocess.STDOUT)
    sampler.add(victim, workers[victim].pid)
    return {"worker": victim, "t": round(t_kill, 2), "pending_at_kill": per.get(victim, 0), "pending_all": per}


def worker_stats(r, wids) -> dict:
    tot = {"applied": 0, "duplicates": 0, "claimed": 0}
    for w in wids:
        h = r.hgetall(f"stats:worker:{w}")
        for k in tot:
            tot[k] += int(float(h.get(k.encode(), 0)))
    return tot


def ground_truth(limit: int = 0) -> dict:
    users, _, wrs, _, n_users = load_events(limit)
    etype = np.where(wrs < 0.3, "skip", np.where(wrs >= 1.0, "complete", "play"))
    gt = {}
    for key, mask in (("n", np.ones(len(users), bool)), ("complete", etype == "complete"), ("skip", etype == "skip"),
                      ("play", etype == "play"), ("pos", wrs > 2.0)):
        gt[key] = np.bincount(users[mask], minlength=n_users)
    return {"gt": gt, "n_users": n_users}


def check_correctness(feats: dict, truth: dict) -> dict:
    """Same exact check as Stage 5: every user's counters vs the parquet ground truth."""
    ring = ConsistentHashRing(list(feats), 100)
    gt = truth["gt"]
    mismatched = 0
    hist_len_bad = 0
    for u in range(truth["n_users"]):
        c = feats[ring.get_node(f"user:{u}")]
        h = c.hgetall(f"rt:{u}:cnt")
        got = {k: int(float(h.get(k.encode(), 0))) for k in gt}
        if any(got[k] != gt[k][u] for k in gt):
            mismatched += 1
        if c.llen(f"rt:{u}:hist") != min(100, int(gt["n"][u])):
            hist_len_bad += 1
    return {"users_checked": truth["n_users"], "users_with_mismatched_counters": mismatched,
            "users_with_wrong_history_length": hist_len_bad}


def append_run(row: dict) -> None:
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------- summary
def summarize() -> dict:
    rows = [json.loads(l) for l in RUNS.read_text().splitlines() if l.strip()]
    groups: dict[tuple, list] = {}
    for r in rows:
        if r.get("tag") in ("pilot", "profile"):
            continue
        key = (r["harness"], r["impl"], r["workers"], "chaos" if r.get("killed") else r.get("mode", "max"))
        groups.setdefault(key, []).append(r)

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.median(xs)), 1) if xs else None

    table = []
    for (harness, impl, nw, mode), rs in sorted(groups.items()):
        thr = [r.get("throughput_events_per_s") or r.get("throughput_events_per_s_end_to_end") for r in rs]
        table.append({
            "harness": harness, "mode": mode, "impl": impl, "workers": nw, "repeats": len(rs),
            "throughput_median": med(thr), "throughput_min": min(thr), "throughput_max": max(thr),
            "seconds_median": med([r.get("seconds") or r.get("total_seconds_until_all_applied") for r in rs]),
            "worker_cpu_s_per_1M_events_median": med([r.get("worker_cpu_s_per_1M_events") for r in rs]),
            "worker_rss_mb_max": max((r.get("worker_rss_mb_max") or 0) for r in rs),
            "e2e_latency_ms_p50_median": med([(r.get("e2e_latency_ms") or {}).get("p50") for r in rs]),
            "e2e_latency_ms_p99_median": med([(r.get("e2e_latency_ms") or {}).get("p99") for r in rs]),
            "mismatched_users_max": max(((r.get("correctness") or {}).get("users_with_mismatched_counters",
                                          r.get("users_with_mismatched_counters", 0)) for r in rs)),
            "duplicates_max": max((r.get("duplicates", r.get("duplicates_skipped", 0)) or 0) for r in rs),
            "contended_runs": sum(bool(r.get("contention_before") or r.get("contention_after")) for r in rs),
        })
    out = {"kind": "Stage 12: Python vs Go event worker", "source": str(RUNS.relative_to(ROOT)),
           "machine": rows[0]["machine"] if rows else None, "table": table,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "results" / "stage12_go_worker.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def wait_quiet(log) -> None:
    """Benchmarks only run while no other echostream cluster is up (CPU contention)."""
    while True:
        others = other_clusters()
        if not others:
            return
        log(f"waiting: another cluster is running ({len(others)} processes, e.g. {others[0][:80]})")
        time.sleep(120)


def matrix(repeats: int = 3, only: str = "") -> None:
    """The whole Stage 12 benchmark, sequential, impls interleaved inside each repeat."""
    logf = open(ROOT / "results" / "logs_stage12_go_worker.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    py = [sys.executable, str(ROOT / "scripts" / "stage5_replay.py")]
    truth = None
    if not only or "drain" in only:
        preload(0)
        truth = ground_truth(0)
        for rep in range(repeats):
            for nw in (1, 2, 4, 8):
                for impl in ("python", "go"):
                    wait_quiet(log)
                    row = drain(impl, nw, truth=truth, tag=f"sweep-r{rep}")
                    append_run(row)
                    log(f"drain {impl} x{nw} r{rep}: {row['throughput_events_per_s']:,.0f} ev/s, {row['seconds']} s, "
                        f"worker cpu {row['worker_cpu_s_total']} s, rss max {row['worker_rss_mb_max']} MB, "
                        f"mismatched {row['correctness']['users_with_mismatched_counters']}, dup {row['duplicates']}")
    if not only or "chaos" in only:
        truth = truth or ground_truth(0)
        for rep in range(repeats):
            for impl in ("go", "python"):
                if impl == "python" and rep > 0:
                    continue          # one Python chaos run for reference; the Go one is the claim
                wait_quiet(log)
                row = drain(impl, 4, kill_at=20, truth=truth, tag=f"chaos-r{rep}")
                append_run(row)
                log(f"chaos {impl} x4 r{rep}: killed {row['killed']['worker']} with {row['killed']['pending_at_kill']} "
                    f"pending; applied {row['applied']}, dup {row['duplicates']}, claimed {row['claimed']}, "
                    f"mismatched {row['correctness']['users_with_mismatched_counters']}")
    if not only or "replay" in only:
        for rep in range(repeats):
            for impl in ("python", "go"):
                wait_quiet(log)
                out = f"results/stage12_go_worker_replay_{impl}_r{rep}.json"
                subprocess.run([*py, "--rate", "0", "--worker-impl", impl, "--port-base", "37000", "--tag",
                                f"s12-{impl}-r{rep}", "--out", out, "--runs-jsonl", str(RUNS)], cwd=ROOT, check=True)
                r = json.loads((ROOT / out).read_text())
                log(f"replay(API) {impl} x4 r{rep}: {r['throughput_events_per_s_end_to_end']:,.0f} ev/s e2e, "
                    f"p50/p99 <= {r['e2e_latency_ms_bucket_upper_bounds']['p50']}/"
                    f"{r['e2e_latency_ms_bucket_upper_bounds']['p99']} ms, mismatched "
                    f"{r['users_with_mismatched_counters']}, dup {r['duplicates_skipped']}")
    if not only or "paced" in only:
        for rate, limit in ((10000, 1_200_000), (25000, 2_500_000)):
            for rep in range(repeats):
                for impl in ("python", "go"):
                    wait_quiet(log)
                    out = f"results/stage12_go_worker_paced{rate // 1000}k_{impl}_r{rep}.json"
                    subprocess.run([*py, "--rate", str(rate), "--limit", str(limit), "--worker-impl", impl,
                                    "--port-base", "37000", "--tag", f"s12p-{impl}-r{rep}", "--out", out,
                                    "--runs-jsonl", str(RUNS)], cwd=ROOT, check=True)
                    r = json.loads((ROOT / out).read_text())
                    log(f"paced {rate}/s {impl} r{rep}: p50 in {r['e2e_latency_ms_bucket']['p50']} ms, "
                        f"p99 in {r['e2e_latency_ms_bucket']['p99']} ms, worker cpu {r['worker_cpu_s_total']} s")
    if not only or "profile" in only:
        truth = truth or ground_truth(0)
        wait_quiet(log)
        row = drain("go", 4, pprof_s=30, truth=truth, tag="profile")
        append_run(row)
        log(f"profile run: {row['pprof']}")
    summarize()
    log("matrix done")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preload").add_argument("--limit", type=int, default=0)
    d = sub.add_parser("drain")
    d.add_argument("--impl", choices=["python", "go"], required=True)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--repeats", type=int, default=1)
    d.add_argument("--limit", type=int, default=0)
    d.add_argument("--kill-at", type=float, default=0, help="kill -9 the busiest worker at t seconds (chaos)")
    d.add_argument("--pprof", type=int, default=0, help="grab a CPU profile of w0 for N seconds (Go)")
    d.add_argument("--tag", default="")
    sub.add_parser("summarize")
    m = sub.add_parser("matrix")
    m.add_argument("--repeats", type=int, default=3)
    m.add_argument("--only", default="", help="comma list of: drain, chaos, replay, paced, profile")
    args = ap.parse_args()
    if args.cmd == "preload":
        print(preload(args.limit))
    elif args.cmd == "drain":
        if args.impl == "go" and not GO_WORKER.exists():
            sys.exit("run `make go-build` first")
        truth = ground_truth(args.limit)
        for k in range(args.repeats):
            row = drain(args.impl, args.workers, args.limit, args.kill_at, args.pprof, truth, args.tag)
            append_run(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("timeline", "processes")}), flush=True)
    elif args.cmd == "matrix":
        matrix(args.repeats, args.only)
    else:
        print(json.dumps(summarize()["table"], indent=1))


if __name__ == "__main__":
    main()
