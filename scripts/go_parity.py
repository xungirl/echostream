"""Parity test: the same events through the Python worker and through the Go worker.

Each implementation gets FRESH redis-server processes (event bus + 3 feature
nodes) and, if Postgres is installed, a fresh Postgres for the event archive.
The event set and every stream entry id are identical for both runs, then the
full resulting state is diffed:

  * every key on every feature node (history lists, counter hashes, dedupe keys)
  * the dead-letter stream (all fields, incl. the reason text; minus `_worker`)
  * the stats counters of the phases without a crash
  * the Postgres event_archive table (if Postgres is available)

Phases (each one exercises a different path of the worker):
  1. one worker; a "ghost" consumer has read half the entries and died
     (XAUTOCLAIM takeover, incl. entries deleted while pending); duplicates
     and every kind of poison entry
  2. three workers with balanced leases; out-of-order re-sends of phase-1
     events, more poison, NaN/inf watch ratios (Lua error after a partial write)
  3. three workers; one is SIGKILLed mid-stream and restarted (like the
     supervisor does); its pending entries are reclaimed by the others

    uv run python scripts/go_parity.py [--impls python,go] [--no-archive]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import redis

from echostream.core.hashing import ConsistentHashRing, partition_for

ROOT = Path(__file__).resolve().parents[1]
GO_WORKER = ROOT / "go" / "worker" / "bin" / "echostream-worker"
REDIS = shutil.which("redis-server") or "/opt/homebrew/bin/redis-server"
PG_BIN = Path(shutil.which("postgres") or "/opt/homebrew/bin/postgres").parent
SCHEMA = ROOT / "echostream" / "store" / "schema.sql"

N_PARTITIONS, N_FEAT = 6, 3
PRODUCED_MS = "1700000000000.0"     # fixed: poison entries are copied verbatim into the DLQ
STAT_KEYS = ("applied", "duplicates", "dlq", "claimed", "retries", "archived", "archive_failures")


# ------------------------------------------------------------------ event set
class EventSet:
    """Deterministic entries with explicit, globally increasing stream ids."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.seq = 0
        self.sent: list[dict] = []          # good events (for re-sends)

    def _id(self) -> str:
        self.seq += 1
        return f"{self.seq}-0"

    def good(self, uid: int, k: int, prefix: str) -> list[tuple[bytes, bytes]]:
        wr = self.rng.choice([0.05, 0.2, 0.6, 1.0, 1.7, 2.5, 3.25])
        action = "skip" if wr < 0.3 else ("complete" if wr >= 1.0 else "play")
        cats = self.rng.choice(["", "3", "1,4", "2,5,7"])
        ev = [("event_id", f"{prefix}-{uid}-{k}"), ("user_id", str(uid)), ("item_id", str(self.rng.randrange(3000))),
              ("action", action), ("watch_ratio", repr(wr)), ("ts", repr(float(1000 * k + uid))),
              ("cats", cats), ("produced_ms", PRODUCED_MS)]
        self.sent.append(dict(ev))
        return [(k_.encode(), v.encode()) for k_, v in ev]

    def entry(self, uid: int, fields: list[tuple[bytes, bytes]]) -> tuple[str, str, list]:
        return f"events:{partition_for(uid, N_PARTITIONS)}", self._id(), fields

    def stream(self, users: list[int], per_user: int, prefix: str, dup_every: int = 0) -> list:
        out = []
        for k in range(per_user):
            for u in users:
                f = self.good(u, k, prefix)
                out.append(self.entry(u, f))
                if dup_every and k % dup_every == 0:
                    out.append(self.entry(u, f))           # client retry right after
        return out

    def resend(self, n: int) -> list:
        """Re-send earlier events later in the stream (out-of-order redelivery)."""
        out = []
        for ev in self.rng.sample(self.sent, n):
            out.append(self.entry(int(ev["user_id"]), [(k.encode(), v.encode()) for k, v in ev.items()]))
        return out

    def poison(self, tag: str) -> list:
        g = dict(event_id=f"p{tag}", user_id="77", item_id="5", action="play", watch_ratio="0.5", ts="1.0",
                 cats="1", produced_ms=PRODUCED_MS)

        def f(**kw):
            d = {**g, **kw}
            return [(k.encode(), v if isinstance(v, bytes) else v.encode()) for k, v in d.items()]
        rows = [
            [(b"event_id", f"bad-{tag}".encode()), (b"user_id", b"not-a-number")],         # missing fields
            [(b"garbage", b"x")],
            f(event_id=f"p{tag}-int", user_id="12a"),
            f(event_id=f"p{tag}-float", watch_ratio="abc"),
            f(event_id=f"p{tag}-neg", watch_ratio="-0.5"),
            f(event_id=f"p{tag}-act", action="like"),
            f(event_id=f"p{tag}:colon"),
            f(event_id=""),
            f(event_id=b"\xff\xfe" + tag.encode()),                                         # invalid UTF-8
            f(event_id=f"p{tag}-q", action="ДА'\"\n​\x7f"),                            # repr() quoting
            f(event_id=f"p{tag}-hex", ts="0x1p3"),
            f(event_id=f"p{tag}-spoof", action="bad", _reason="spoofed", _source="x"),      # overwritten in DLQ
            f(event_id=f"p{tag}-nan", watch_ratio="nan"),        # parses; Redis rejects it -> script error
            f(event_id=f"p{tag}-inf", watch_ratio="1e400"),      # float -> inf -> script error
            # odd but valid: Python's int()/float() accept these, so must Go
            f(event_id=f"p{tag}-ws", user_id=" 0_77 ", ts="١٢.٥", watch_ratio="1_0.5"),
            [(b"event_id", f"p{tag}-dupfield".encode()), *f(event_id="ignored")[1:],
             (b"event_id", f"p{tag}-dupfield-last".encode())],                              # repeated field
        ]
        return [self.entry(77, r) for r in rows]


def build_phases() -> list[list]:
    es = EventSet()
    p1 = es.stream(list(range(100)), 10, "a", dup_every=5) + es.poison("1")
    p2 = es.stream(list(range(50, 200)), 8, "b", dup_every=3) + es.resend(300) + es.poison("2")
    p2 += es.resend(50)
    p3 = es.stream(list(range(300)), 40, "c", dup_every=7) + es.resend(500)
    es.rng.shuffle(p3)   # interleave the re-sends; then renumber so stream ids increase again
    return [p1, p2, _renumber(p3, es)]


def _renumber(entries: list, es: EventSet) -> list:
    """Give shuffled entries fresh increasing ids (stream ids must increase)."""
    out = []
    for stream, _, fields in entries:
        es.seq += 1
        out.append((stream, f"{es.seq}-0", fields))
    return out


# ------------------------------------------------------------------ processes
class Env:
    def __init__(self, impl: str, base: int, archive: bool, workdir: Path):
        self.impl, self.base, self.archive, self.dir = impl, base, archive, workdir
        self.procs: list[subprocess.Popen] = []
        self.workers: dict[str, subprocess.Popen] = {}
        self.ev = redis.Redis(port=base + 20)
        self.feat = {f"feat{i}": redis.Redis(port=base + 21 + i) for i in range(N_FEAT)}
        self.env = {**os.environ, "ECHO_PORT_BASE": str(base), "ECHO_PARTITIONS": str(N_PARTITIONS),
                    "ECHO_FEATURE_NODES": str(N_FEAT), "ECHO_LEASE_MS": "3000", "ECHO_CLAIM_IDLE_MS": "500",
                    "ECHO_WORKER_BATCH": "100", "ECHO_DEDUPE_TTL": "3600",
                    "ECHO_ARCHIVE": "1" if archive else "0",
                    "ECHO_PG_DSN": f"postgresql://echostream@127.0.0.1:{base + 40}/echostream"}

    def start(self) -> None:
        for port in [self.base + 20] + [self.base + 21 + i for i in range(N_FEAT)]:
            self.procs.append(subprocess.Popen([REDIS, "--port", str(port), "--bind", "127.0.0.1", "--save", "",
                                                "--appendonly", "no", "--protected-mode", "no"],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if self.archive:
            self._start_postgres()
        for c in [self.ev, *self.feat.values()]:
            wait(lambda c=c: c.ping(), 10)

    def _start_postgres(self) -> None:
        pg = self.dir / "pg"
        subprocess.run([str(PG_BIN / "initdb"), "-D", str(pg), "-U", "echostream", "--auth=trust", "-E", "UTF8"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(subprocess.Popen([str(PG_BIN / "postgres"), "-D", str(pg), "-p", str(self.base + 40),
                                            "-k", str(self.dir), "-c", "listen_addresses=127.0.0.1"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        import psycopg
        wait(lambda: psycopg.connect(f"postgresql://echostream@127.0.0.1:{self.base + 40}/postgres",
                                     connect_timeout=1).close() or True, 30)
        with psycopg.connect(f"postgresql://echostream@127.0.0.1:{self.base + 40}/postgres", autocommit=True) as c:
            c.execute("CREATE DATABASE echostream")
        with psycopg.connect(self.env["ECHO_PG_DSN"], autocommit=True) as c:
            c.execute(SCHEMA.read_text())

    def start_worker(self, wid: str) -> None:
        if self.impl == "go":
            cmd = [str(GO_WORKER), "--id", wid]
        else:
            cmd = [sys.executable, "-m", "echostream.serving.worker", "--id", wid]
        log = open(self.dir / f"{wid}.log", "ab")
        self.workers[wid] = subprocess.Popen(cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)

    def stop_worker(self, wid: str, sig=signal.SIGTERM) -> None:
        p = self.workers.pop(wid)
        p.send_signal(sig)
        p.wait(timeout=15)

    def close(self) -> None:
        for wid in list(self.workers):
            self.stop_worker(wid, signal.SIGKILL)
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    # ------------------------------------------------------------- helpers
    def produce(self, entries: list, rate: float = 0) -> None:
        pipe = self.ev.pipeline(transaction=False)
        for k, (stream, eid, fields) in enumerate(entries):
            pipe.execute_command("XADD", stream, eid, *[x for kv in fields for x in kv])
            if (k + 1) % 100 == 0:
                pipe.execute()
                if rate:
                    time.sleep(100 / rate)
        pipe.execute()

    def drained(self) -> bool:
        for p in range(N_PARTITIONS):
            if self.ev.xlen(f"events:{p}") or self.ev.xpending(f"events:{p}", "workers")["pending"]:
                return False
        return True

    def balanced(self, n_workers: int) -> bool:
        owners = Counter((self.ev.get(f"lease:{p}") or b"").decode() for p in range(N_PARTITIONS))
        fair = -(-N_PARTITIONS // n_workers)
        return "" not in owners and len(owners) == n_workers and max(owners.values()) <= fair

    def stats(self, wids: list[str]) -> dict:
        tot = Counter()
        for w in wids:
            h = self.ev.hgetall(f"stats:worker:{w}")
            for k in STAT_KEYS:
                tot[k] += int(float(h.get(k.encode(), 0)))
        return dict(tot)

    def settle_stats(self) -> None:
        time.sleep(3000 / 6000 * 3)   # three heartbeats: every worker has published its final stats


def freeze_with_pending(env: Env, wid: str, tries: int = 400) -> int:
    """SIGSTOP the worker at a moment when it holds un-acked entries (read, not yet
    acked), so the SIGKILL that follows really interrupts a batch."""
    p = env.workers[wid]
    for _ in range(tries):
        p.send_signal(signal.SIGSTOP)
        pending = Counter()
        for s in range(N_PARTITIONS):
            for c in env.ev.xpending(f"events:{s}", "workers")["consumers"]:
                pending[c["name"].decode()] += c["pending"]
        if pending.get(wid, 0) > 0:
            return pending[wid]
        p.send_signal(signal.SIGCONT)
        time.sleep(0.003)
    return 0


def wait(cond, timeout: float, step: float = 0.05):
    deadline = time.time() + timeout
    while True:
        try:
            if cond():
                return
        except Exception:
            pass
        if time.time() > deadline:
            raise TimeoutError(f"condition not met after {timeout}s")
        time.sleep(step)


# ------------------------------------------------------------------ run + snapshot
def run(impl: str, base: int, archive: bool, phases: list[list], log=print) -> dict:
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix=f"parity-{impl}-"))
    env = Env(impl, base, archive, work)
    out: dict = {"impl": impl, "counters": {}}
    try:
        env.start()
        p1, p2, p3 = phases
        # ---- phase 1: ghost consumer + single worker ---------------------------
        for p in range(N_PARTITIONS):
            env.ev.xgroup_create(f"events:{p}", "workers", id="0", mkstream=True)
        env.produce(p1)
        ghost, deleted = 0, []
        for p in range(N_PARTITIONS):
            n = env.ev.xlen(f"events:{p}")
            res = env.ev.xreadgroup("workers", "ghost", {f"events:{p}": ">"}, count=n // 2)
            ids = [e[0] for _, es in res for e in es]
            ghost += len(ids)
            if len(ids) > 4:
                env.ev.xdel(f"events:{p}", *ids[:2])      # deleted while pending: never applied
                deleted += [i.decode() for i in ids[:2]]
        out["ghost_pending"] = ghost
        out["deleted_while_pending"] = deleted
        env.start_worker("w0")
        wait(env.drained, 60)
        env.settle_stats()
        out["counters"]["phase1"] = env.stats(["w0"])
        env.stop_worker("w0")                               # SIGTERM: graceful
        out["after_graceful_shutdown"] = {
            "leases_held": sum(env.ev.exists(f"lease:{p}") for p in range(N_PARTITIONS)),
            "w0_alive": env.ev.zscore("workers:alive", "w0") is not None}
        log(f"  [{impl}] phase 1 done ({time.time() - t0:.1f}s) {out['counters']['phase1']}")

        # ---- phase 2: three balanced workers -----------------------------------
        env.ev.delete("stats:worker:w0")   # a restarted worker only overwrites the stats fields it touched
        for w in ("w0", "w1", "w2"):
            env.start_worker(w)
        wait(lambda: env.balanced(3), 30)
        time.sleep(1.0)
        assert env.balanced(3)
        env.produce(p2)
        wait(env.drained, 60)
        env.settle_stats()
        out["counters"]["phase2"] = env.stats(["w0", "w1", "w2"])
        log(f"  [{impl}] phase 2 done ({time.time() - t0:.1f}s) {out['counters']['phase2']}")

        # ---- phase 3: kill -9 a worker mid-stream --------------------------------
        prod = threading.Thread(target=env.produce, args=(p3,), kwargs={"rate": 8000})
        prod.start()
        time.sleep(0.5)
        out["killed_with_pending"] = freeze_with_pending(env, "w1")
        env.stop_worker("w1", signal.SIGKILL)               # frozen mid-batch -> its entries stay pending
        time.sleep(0.3)                                     # the supervisor's restart backoff
        env.start_worker("w1")
        prod.join()
        wait(env.drained, 90)
        env.settle_stats()
        out["counters"]["phase3_informational"] = env.stats(["w0", "w1", "w2"])
        log(f"  [{impl}] phase 3 done ({time.time() - t0:.1f}s)")

        out["features"] = snapshot_features(env)
        out["dlq"] = snapshot_dlq(env)
        if archive:
            out["archive"] = snapshot_archive(env)
        out["pel_empty"] = env.drained()
        out["seconds"] = round(time.time() - t0, 1)
    finally:
        env.close()
        shutil.rmtree(work, ignore_errors=True)
    return out


def snapshot_features(env: Env) -> dict:
    state = {}
    for name, c in env.feat.items():
        node = {}
        for key in c.scan_iter(count=1000):
            k = key.decode()
            t = c.type(key).decode()
            if t == "list":
                node[k] = [x.decode() for x in c.lrange(key, 0, -1)]
            elif t == "hash":
                node[k] = {f.decode(): v.decode() for f, v in sorted(c.hgetall(key).items())}
            elif t == "string":
                node[k] = {"value": c.get(key).decode(), "has_ttl": c.ttl(key) > 0}
            else:
                node[k] = f"<{t}>"
        state[name] = dict(sorted(node.items()))
    return state


def _text(b: bytes) -> str:
    try:
        return b.decode()
    except UnicodeDecodeError:
        return "bytes:" + b.hex()


def snapshot_dlq(env: Env) -> list:
    rows = []
    for _, fields in env.ev.xrange("events:dlq"):
        rows.append([[_text(k), _text(v)] for k, v in fields.items() if k != b"_worker"])
    return sorted(rows, key=lambda r: (dict(r).get("_source", ""), int(dict(r).get("_entry", "0-0").split("-")[0])))


def snapshot_archive(env: Env) -> list:
    import psycopg
    with psycopg.connect(env.env["ECHO_PG_DSN"]) as c:
        rows = c.execute("SELECT event_id, user_id, item_id, action, watch_ratio, ts, cats, partition, stream_id "
                         "FROM event_archive ORDER BY event_id").fetchall()
    return [[str(x) for x in r] for r in rows]


def expected_counts(phases: list[list], deleted: set[str]) -> dict:
    """Ground truth independent of both workers: unique valid event ids per user, minus
    entries deleted before any worker saw them (NaN/inf events count: the Lua script
    increments n before HINCRBYFLOAT fails)."""
    from echostream.serving.events import PoisonEvent, parse_fields
    seen, n = set(), Counter()
    for entries in phases:
        for _, eid, fields in entries:
            if eid in deleted:
                continue
            try:
                ev = parse_fields(dict(fields))
            except PoisonEvent:
                continue
            if ev["event_id"] not in seen:
                seen.add(ev["event_id"])
                n[ev["user_id"]] += 1
    return dict(n)


def diff(a: dict, b: dict) -> dict:
    res = {}
    fa, fb = a["features"], b["features"]
    keys_a = {(n, k) for n in fa for k in fa[n]}
    keys_b = {(n, k) for n in fb for k in fb[n]}
    differing = [f"{n}/{k}" for n, k in sorted(keys_a & keys_b) if fa[n][k] != fb[n][k]]
    res["feature_keys"] = {"a": len(keys_a), "b": len(keys_b), "only_a": sorted(map("/".join, keys_a - keys_b))[:20],
                           "only_b": sorted(map("/".join, keys_b - keys_a))[:20], "differing": differing[:20],
                           "n_differing": len(differing), "identical": fa == fb}
    res["dlq"] = {"a": len(a["dlq"]), "b": len(b["dlq"]), "identical": a["dlq"] == b["dlq"]}
    if a["dlq"] != b["dlq"]:
        res["dlq"]["first_difference"] = next(([x, y] for x, y in zip(a["dlq"], b["dlq"]) if x != y), None)
    res["counters"] = {ph: {"a": a["counters"][ph], "b": b["counters"][ph],
                            "identical": a["counters"][ph] == b["counters"][ph]} for ph in ("phase1", "phase2")}
    if "archive" in a and "archive" in b:
        res["archive"] = {"a": len(a["archive"]), "b": len(b["archive"]), "identical": a["archive"] == b["archive"]}
    res["identical"] = all(v["identical"] for v in [res["feature_keys"], res["dlq"], *res["counters"].values()]
                           + ([res["archive"]] if "archive" in res else []))
    return res


def feature_counts(snap: dict) -> dict:
    out = {}
    for node in snap["features"].values():
        for k, v in node.items():
            if k.endswith(":cnt"):
                out[int(k.split(":")[1])] = int(v["n"])
    return out


def parity(impls=("python", "go"), archive: bool | None = None, base: int = 26000, log=print) -> dict:
    if archive is None:
        archive = (PG_BIN / "initdb").exists()
    phases = build_phases()
    runs = {}
    for i, impl in enumerate(impls):
        log(f"running {impl} worker ...")
        runs[impl] = run(impl, base + 100 * i, archive, phases, log)
        truth = expected_counts(phases, set(runs[impl]["deleted_while_pending"]))
        got = feature_counts(runs[impl])
        runs[impl]["users_mismatched_vs_ground_truth"] = sum(got.get(u, 0) != c for u, c in truth.items()) + \
            sum(u not in truth for u in got)
    a, b = impls[0], impls[1]
    d = diff(runs[a], runs[b])
    summary = {
        "kind": "parity (same events, fresh Redis/Postgres per implementation)", "impls": list(impls),
        "events_per_phase": [len(p) for p in phases], "users": len(truth),
        "ground_truth_ok": all(r["users_mismatched_vs_ground_truth"] == 0 for r in runs.values()),
        "archive_checked": archive, "identical": d["identical"], "diff": d,
        **{impl: {k: v for k, v in r.items() if k not in ("features", "dlq", "archive")} for impl, r in runs.items()},
        "feature_keys_compared": d["feature_keys"]["a"], "dlq_entries_compared": d["dlq"]["a"],
        "archive_rows_compared": d.get("archive", {}).get("a"),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    summary["identical"] = summary["identical"] and summary["ground_truth_ok"]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default="python,go")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--out", default="results/stage12_go_parity.json")
    args = ap.parse_args()
    impls = tuple(args.impls.split(","))
    if "go" in impls and not GO_WORKER.exists():
        sys.exit(f"{GO_WORKER} missing: run `make go-build`")
    s = parity(impls, archive=False if args.no_archive else None)
    Path(args.out).write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: s[k] for k in ("identical", "events_per_phase", "feature_keys_compared",
                                        "dlq_entries_compared", "archive_rows_compared")}, indent=1))
    print("diff:", json.dumps(s["diff"], indent=1, ensure_ascii=False)[:3000])
    sys.exit(0 if s["identical"] else 1)


if __name__ == "__main__":
    main()
