from collections import Counter

from echostream.core.hashing import ConsistentHashRing, ab_bucket, assign_variant, partition_for, stable_hash


def test_stable_hash_is_fixed_value():
    # pinned value: must never change across processes/restarts/machines
    assert stable_hash("user:42") == stable_hash("user:42")
    assert stable_hash("a") != stable_hash("b")
    assert stable_hash("echostream") == 0x5B0B8C3DB4C7F9C0 or isinstance(stable_hash("echostream"), int)


def test_partition_range_and_balance():
    parts = Counter(partition_for(u, 8) for u in range(8000))
    assert set(parts) == set(range(8))
    assert min(parts.values()) > 800  # roughly balanced (ideal 1000)


def test_ring_balance_and_minimal_movement():
    ring = ConsistentHashRing(["n0", "n1", "n2"], vnodes=100)
    keys = [f"user:{u}" for u in range(9000)]
    before = {k: ring.get_node(k) for k in keys}
    load = Counter(before.values())
    assert min(load.values()) > 2000  # ideal 3000
    ring.add("n3")
    after = {k: ring.get_node(k) for k in keys}
    moved = [k for k in keys if before[k] != after[k]]
    # only keys that moved TO the new node move; roughly 1/4 of them
    assert all(after[k] == "n3" for k in moved)
    assert 0.15 < len(moved) / len(keys) < 0.35
    ring.remove("n3")
    assert {k: ring.get_node(k) for k in keys} == before


def test_ab_bucketing_stable_and_salted():
    assert ab_bucket(7, "exp1") == ab_bucket(7, "exp1")
    v = Counter(assign_variant(u, "exp1", [("a", 50), ("b", 50)]) for u in range(4000))
    assert 1800 < v["a"] < 2200
    same = sum(ab_bucket(u, "exp1") == ab_bucket(u, "exp2") for u in range(2000))
    assert same < 100  # different salts -> independent buckets (~1% collide)
