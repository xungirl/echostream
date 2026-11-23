"""Redis Streams reliability: no loss, no double count, per-user order, DLQ,
recovery after a worker dies and after a feature node stalls."""
import threading
import time

import pytest
import redis

from echostream.cluster.client import control
from echostream.core.hashing import ConsistentHashRing
from echostream.serving.events import stream_for_user, to_fields

pytestmark = pytest.mark.integration


def ring(cfg):
    return ConsistentHashRing(list(cfg.feature_nodes), 100)


def feature_client(cfg, uid):
    node = ring(cfg).get_node(f"user:{uid}")
    return redis.Redis.from_url(cfg.feature_nodes[node])


def produce(cfg, events, rate=None):
    r = redis.Redis(port=cfg.events_redis_port)
    pipe = r.pipeline(transaction=False)
    for k, ev in enumerate(events):
        pipe.xadd(stream_for_user(ev["user_id"], cfg.n_partitions), to_fields(ev, "1,2"))
        if (k + 1) % 100 == 0:
            pipe.execute()
            if rate:
                time.sleep(100 / rate)
    pipe.execute()


def make_events(users, per_user, prefix, dup_every=0):
    """Interleaved across users; ts = per-user sequence number so order can be checked."""
    out = []
    for seq in range(per_user):
        for u in users:
            ev = {"event_id": f"{prefix}-{u}-{seq}", "user_id": u, "item_id": seq % 50,
                  "watch_ratio": 2.5 if seq % 3 == 0 else 0.1, "action": "complete" if seq % 3 == 0 else "skip",
                  "ts": float(seq)}
            out.append(ev)
            if dup_every and seq % dup_every == 0:
                out.append(dict(ev))  # a client retry: same event_id
    return out


def wait_applied(cfg, users, expected_per_user, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        counts = [int(feature_client(cfg, u).hget(f"rt:{u}:cnt", "n") or 0) for u in users]
        if all(c >= expected_per_user for c in counts):
            return counts
        time.sleep(0.2)
    return counts


def check_user(cfg, u, per_user):
    c = feature_client(cfg, u)
    cnt = {k.decode(): float(v) for k, v in c.hgetall(f"rt:{u}:cnt").items()}
    assert cnt["n"] == per_user, (u, cnt)                       # no loss, no double count
    assert cnt["complete"] == len([s for s in range(per_user) if s % 3 == 0])
    hist = [x.decode().split(":") for x in c.lrange(f"rt:{u}:hist", 0, -1)]
    seqs = [float(ts) for _, _, ts in reversed(hist)]            # oldest first
    assert seqs == [float(s) for s in range(per_user)], (u, seqs[:10])  # per-user order


def test_no_loss_no_double_count_and_order_with_duplicates(events_cluster):
    cfg = events_cluster
    users = list(range(1000, 1100))
    per_user = 60
    produce(cfg, make_events(users, per_user, "dup", dup_every=4))
    counts = wait_applied(cfg, users, per_user)
    assert min(counts) == per_user
    time.sleep(0.5)
    for u in users:
        check_user(cfg, u, per_user)
    stats = redis.Redis(port=cfg.events_redis_port)
    dups = sum(float(stats.hget(f"stats:worker:w{i}", "duplicates") or 0) for i in range(cfg.n_workers))
    assert dups >= len(users) * len(range(0, per_user, 4))      # every re-send was recognised


def test_poison_events_go_to_dead_letter_stream(events_cluster):
    cfg = events_cluster
    r = redis.Redis(port=cfg.events_redis_port)
    before = r.xlen("events:dlq")
    s = stream_for_user(2000, cfg.n_partitions)
    r.xadd(s, {"event_id": "bad-1", "user_id": "not-a-number"})
    r.xadd(s, {"garbage": "x"})
    good = make_events([2000], 6, "after-poison")
    produce(cfg, good)
    wait_applied(cfg, [2000], 6)
    check_user(cfg, 2000, 6)                                     # poison did not block the partition
    deadline = time.time() + 10
    while r.xlen("events:dlq") < before + 2 and time.time() < deadline:
        time.sleep(0.1)
    assert r.xlen("events:dlq") == before + 2
    reasons = [e[1][b"_reason"].decode() for e in r.xrange("events:dlq")[-2:]]
    assert all(reasons)


def test_worker_crash_pending_entries_reclaimed(events_cluster):
    cfg = events_cluster
    users = list(range(3000, 3060))
    per_user = 80
    events = make_events(users, per_user, "crash")
    t = threading.Thread(target=produce, args=(cfg, events), kwargs={"rate": 3000})
    t.start()
    time.sleep(0.4)
    control(cfg, "stop", "worker0")          # SIGKILL mid-stream, held down (no restart)
    t.join()
    counts = wait_applied(cfg, users, per_user, timeout=60)
    assert min(counts) == per_user, "events lost after worker crash"
    for u in users:
        check_user(cfg, u, per_user)
    control(cfg, "start", "worker0")


def test_feature_node_stall_then_recovery(events_cluster):
    cfg = events_cluster
    users = list(range(4000, 4060))
    per_user = 40
    events = make_events(users, per_user, "stall")
    control(cfg, "freeze", "redis-feat1")
    try:
        produce(cfg, events)
        time.sleep(2.0)                      # workers keep retrying events of users on feat1
    finally:
        control(cfg, "thaw", "redis-feat1")
    counts = wait_applied(cfg, users, per_user, timeout=60)
    assert min(counts) == per_user
    for u in users:
        check_user(cfg, u, per_user)


def test_pending_entries_of_dead_consumer_are_reclaimed(events_cluster):
    """Deterministic version of a crash: a consumer reads entries and dies before
    XACK. They sit in its pending list; the next lease owner must XAUTOCLAIM them."""
    cfg = events_cluster
    r = redis.Redis(port=cfg.events_redis_port)
    for i in range(cfg.n_workers):
        control(cfg, "stop", f"worker{i}")
    time.sleep(2.0)                                   # leases (1.5 s) expire
    users = list(range(5000, 5030))
    per_user = 20
    produce(cfg, make_events(users, per_user, "ghost"))
    ghost_read = 0
    for p in range(cfg.n_partitions):                 # the "crashed" consumer takes everything, acks nothing
        res = r.xreadgroup("workers", "ghost", {f"events:{p}": ">"}, count=10_000)
        ghost_read += sum(len(e) for _, e in res)
    assert ghost_read == len(users) * per_user
    for i in range(cfg.n_workers):
        control(cfg, "start", f"worker{i}")
    counts = wait_applied(cfg, users, per_user, timeout=30)
    assert min(counts) == per_user
    for u in users:
        check_user(cfg, u, per_user)
    time.sleep(1.5)
    claimed_after = sum(float(r.hget(f"stats:worker:w{i}", "claimed") or 0) for i in range(cfg.n_workers))
    assert claimed_after >= ghost_read   # restarted workers start fresh stats counters
    for p in range(cfg.n_partitions):
        assert r.xpending(f"events:{p}", "workers")["pending"] == 0
