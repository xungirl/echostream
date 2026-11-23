"""Event worker: Redis Streams consumer with partition leases.

Guarantees and how they are achieved
------------------------------------
* at-least-once: an entry is XACKed only after its effect is stored; a crashed
  worker's un-acked entries stay in the group's pending list (PEL) and are
  taken over with XAUTOCLAIM by whoever gets the partition next.
* no double counting: the effect is applied by one Lua script that first sets
  a dedupe key `ev:{event_id}` (SET NX) -> redelivery is a no-op.
* per-user order: a user always maps to the same stream (partition), and each
  partition has exactly one owner at a time (a lease key with TTL, renewed by
  heartbeat). The owner processes its own pending entries first, then new ones,
  in stream order; if a feature node is down, events for that node stay pending
  and nothing newer for that node is applied before them.
* poison events: malformed entries or deterministic script errors go to the
  dead-letter stream (events:dlq) and are acked, so they cannot block a partition.

Run:  python -m echostream.serving.worker --id w0
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import signal
import time
from collections import defaultdict

import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram, start_http_server
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from echostream.serving.config import CLUSTER, EVENTS
from echostream.serving.events import PoisonEvent, parse_fields, stream_name
from echostream.serving.feature_store import FeatureStore

RENEW_LUA = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('PEXPIRE', KEYS[1], ARGV[2]) else return 0 end"
RELEASE_LUA = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"
LAT_BUCKETS_MS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000, math.inf]

APPLIED = Counter("echostream_worker_events_applied_total", "events applied to the feature store")
DUPLICATES = Counter("echostream_worker_events_duplicate_total", "redeliveries skipped by dedupe")
DEAD = Counter("echostream_worker_events_dlq_total", "events sent to the dead-letter stream")
RETRIES = Counter("echostream_worker_retries_total", "batches left pending because a feature node failed")
E2E = Histogram("echostream_event_to_feature_seconds", "event produced -> feature updated",
                buckets=[b / 1000 for b in LAT_BUCKETS_MS[:-1]])


class Worker:
    def __init__(self, worker_id: str, events_url: str = CLUSTER.events_redis_url,
                 feature_nodes: dict[str, str] | None = None, n_partitions: int = CLUSTER.n_partitions):
        self.id = worker_id
        self.r = aioredis.from_url(events_url, socket_timeout=2.0)
        self.fs = FeatureStore(feature_nodes or CLUSTER.feature_nodes, socket_timeout=0.5)
        self.P = n_partitions
        self.owned: set[int] = set()
        self.needs_claim: set[int] = set()
        self.running = True
        self.stats = defaultdict(float)
        self.lat_hist = [0] * len(LAT_BUCKETS_MS)

    # ---------------------------------------------------------------- setup
    async def ensure_groups(self) -> None:
        for p in range(self.P):
            try:
                await self.r.xgroup_create(stream_name(p), EVENTS.group, id="0", mkstream=True)
            except ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    # --------------------------------------------------------------- leases
    async def lease_loop(self) -> None:
        while self.running:
            try:
                await self.balance_leases()
            except (RedisConnectionError, RedisTimeoutError):
                pass
            await asyncio.sleep(EVENTS.lease_ms / 6000)

    async def balance_leases(self) -> None:
        now = time.time() * 1000
        await self.r.zadd("workers:alive", {self.id: now})
        await self.r.zremrangebyscore("workers:alive", 0, now - EVENTS.lease_ms)
        alive = max(await self.r.zcard("workers:alive"), 1)
        fair = math.ceil(self.P / alive)
        for p in list(self.owned):  # renew; drop leases we lost
            if not await self.r.eval(RENEW_LUA, 1, f"lease:{p}", self.id, EVENTS.lease_ms):
                self.owned.discard(p)
        while len(self.owned) > fair:  # a new worker joined: give partitions back
            p = max(self.owned)
            self.owned.discard(p)
            await self.r.eval(RELEASE_LUA, 1, f"lease:{p}", self.id)
        if len(self.owned) < fair:
            # start at a worker-specific offset so workers don't all race for partition 0
            start = sum(map(ord, self.id)) % self.P
            for k in range(self.P):
                p = (start + k) % self.P
                if len(self.owned) >= fair:
                    break
                if p not in self.owned and await self.r.set(f"lease:{p}", self.id, nx=True, px=EVENTS.lease_ms):
                    self.owned.add(p)
                    self.needs_claim.add(p)
        await self.r.hset(f"stats:worker:{self.id}", mapping={
            **{k: v for k, v in self.stats.items()}, "partitions": ",".join(map(str, sorted(self.owned))),
            "lat_hist": ",".join(map(str, self.lat_hist)), "updated_ms": now, "pid": os.getpid()})

    # ----------------------------------------------------------- processing
    async def claim(self, p: int) -> bool:
        """Take over pending entries left by a previous owner. Returns True once no
        other consumer holds pending entries for this partition. Until then we must
        not read new entries (they would overtake the old ones -> order broken)."""
        cursor = "0-0"
        while True:
            res = await self.r.xautoclaim(stream_name(p), EVENTS.group, self.id,
                                          min_idle_time=EVENTS.claim_min_idle_ms, start_id=cursor, count=1000)
            cursor, claimed = res[0], res[1]
            self.stats["claimed"] += len(claimed)
            if cursor in (b"0-0", "0-0"):
                break
        summary = await self.r.xpending(stream_name(p), EVENTS.group)
        others = [c for c in (summary.get("consumers") or [])
                  if (c["name"].decode() if isinstance(c["name"], bytes) else c["name"]) != self.id
                  and int(c["pending"]) > 0]
        if others:
            return False
        self.needs_claim.discard(p)
        return True

    async def process_loop(self) -> None:
        while self.running:
            try:
                did_work = await self.process_once()
            except (RedisConnectionError, RedisTimeoutError, OSError):
                did_work = False
                await asyncio.sleep(0.2)
            if not did_work:
                await asyncio.sleep(0.005)

    async def process_once(self) -> bool:
        owned = sorted(self.owned)
        if not owned:
            await asyncio.sleep(0.1)
            return False
        for p in list(owned):
            if p in self.needs_claim and not await self.claim(p):
                owned.remove(p)   # previous owner's entries not claimable yet (min idle time)
        if not owned:
            await asyncio.sleep(0.05)
            return False
        blocked: dict[int, set[str]] = defaultdict(set)   # partition -> feature nodes failing this round
        did = False
        # 1) our own pending entries first (retries, claimed entries), in stream order
        pending = await self.r.xreadgroup(EVENTS.group, self.id, {stream_name(p): "0" for p in owned},
                                          count=EVENTS.batch)
        for stream, entries in pending or []:
            if entries:
                did = True
                await self.handle(stream, entries, blocked)
        # 2) new entries
        new = await self.r.xreadgroup(EVENTS.group, self.id, {stream_name(p): ">" for p in owned},
                                      count=EVENTS.batch, block=50)
        for stream, entries in new or []:
            if entries:
                did = True
                await self.handle(stream, entries, blocked)
        return did

    async def handle(self, stream, entries, blocked: dict[int, set[str]]) -> None:
        stream = stream.decode() if isinstance(stream, bytes) else stream
        p = int(stream.rsplit(":", 1)[1])
        if p not in self.owned:   # lease lost meanwhile: leave entries for the new owner
            return
        to_ack: list = []
        by_node: dict[str, list] = defaultdict(list)
        for entry_id, fields in entries:
            if not fields:   # entry was deleted (already processed + XDEL'd)
                to_ack.append(entry_id)
                continue
            try:
                ev = parse_fields(fields)
            except PoisonEvent as e:
                await self.dead_letter(stream, entry_id, fields, str(e))
                to_ack.append(entry_id)
                continue
            node = self.fs.node_for(ev["user_id"])
            if node in blocked[p]:
                continue  # keep pending -> preserves order behind the failed event
            by_node[node].append((entry_id, ev, fields))

        async def run_node(node: str, items: list):
            calls = [FeatureStore.apply_args(ev, ev["cats"], EVENTS.dedupe_ttl_s, EVENTS.hist_max,
                                             EVENTS.positive_threshold) for _, ev, _ in items]
            return node, items, await self.fs.apply_batch(node, calls)

        results = await asyncio.gather(*(run_node(n, it) for n, it in by_node.items()), return_exceptions=True)
        now_ms = time.time() * 1000
        for node, res in zip(by_node, results):
            if isinstance(res, BaseException):
                blocked[p].add(node)       # transient: retry later, keep order
                RETRIES.inc()
                self.stats["retries"] += 1
                continue
            _, items, outs = res
            for (entry_id, ev, fields), out in zip(items, outs):
                if isinstance(out, Exception):
                    await self.dead_letter(stream, entry_id, fields, f"script error: {out}")
                elif out == 1:
                    APPLIED.inc()
                    self.stats["applied"] += 1
                    lat = now_ms - ev["produced_ms"]
                    E2E.observe(lat / 1000)
                    self.lat_hist[next(i for i, b in enumerate(LAT_BUCKETS_MS) if lat <= b)] += 1
                else:
                    DUPLICATES.inc()
                    self.stats["duplicates"] += 1
                to_ack.append(entry_id)
        if to_ack:
            async with self.r.pipeline(transaction=False) as pipe:
                pipe.xack(stream, EVENTS.group, *to_ack)
                pipe.xdel(stream, *to_ack)   # keep the stream bounded: processed entries are gone
                await pipe.execute()

    async def dead_letter(self, stream: str, entry_id, fields: dict, reason: str) -> None:
        clean = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                 for k, v in fields.items()}
        eid = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
        await self.r.xadd(EVENTS.dlq_stream, {**clean, "_source": stream, "_entry": eid, "_reason": reason[:200],
                                              "_worker": self.id})
        DEAD.inc()
        self.stats["dlq"] += 1

    # ---------------------------------------------------------------- main
    async def run(self) -> None:
        await self.ensure_groups()
        await asyncio.gather(self.lease_loop(), self.process_loop())

    async def shutdown(self) -> None:
        self.running = False
        for p in list(self.owned):
            await self.r.eval(RELEASE_LUA, 1, f"lease:{p}", self.id)
        await self.r.zrem("workers:alive", self.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--metrics-port", type=int, default=0)
    args = ap.parse_args()
    if args.metrics_port:
        start_http_server(args.metrics_port, addr="127.0.0.1")
    w = Worker(args.id)

    async def runner():
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        loop.add_signal_handler(signal.SIGTERM, stop.set)
        loop.add_signal_handler(signal.SIGINT, stop.set)
        task = asyncio.create_task(w.run())
        await stop.wait()
        await w.shutdown()   # graceful: release leases so others take over immediately
        task.cancel()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
