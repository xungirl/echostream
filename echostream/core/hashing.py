"""Stable hashing: consistent-hash ring, stream partitioning, A/B bucketing.

Python's built-in hash() is randomized per process, so it can't be used for
anything that must agree across processes or restarts. We use blake2b.
"""
from __future__ import annotations

import bisect
import hashlib


def stable_hash(key: str) -> int:
    """64-bit hash that is identical on every machine, process and restart."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def partition_for(user_id: int, n_partitions: int) -> int:
    """Which event stream a user's events go to. Same user -> same stream,
    which is what preserves per-user ordering."""
    return stable_hash(f"user:{user_id}") % n_partitions


def ab_bucket(user_id: int, salt: str, n_buckets: int = 100) -> int:
    """Experiment bucket in [0, n_buckets). The salt makes different experiments independent."""
    return stable_hash(f"{salt}:{user_id}") % n_buckets


def assign_variant(user_id: int, salt: str, allocation: list[tuple[str, int]]) -> str:
    """allocation = [(variant, n_buckets), ...] summing to 100."""
    assert sum(n for _, n in allocation) == 100
    b = ab_bucket(user_id, salt, 100)
    for variant, n in allocation:
        if b < n:
            return variant
        b -= n
    raise AssertionError("unreachable")


class ConsistentHashRing:
    """Each node is placed on the ring `vnodes` times. A key goes to the first
    node clockwise from hash(key). Adding/removing one node only moves ~1/N
    of the keys, unlike `hash % N` which moves almost all of them."""

    def __init__(self, nodes: list[str], vnodes: int = 100):
        self.vnodes = vnodes
        self._ring: list[tuple[int, str]] = []
        for node in nodes:
            self.add(node)

    def add(self, node: str) -> None:
        for v in range(self.vnodes):
            bisect.insort(self._ring, (stable_hash(f"{node}#{v}"), node))

    def remove(self, node: str) -> None:
        self._ring = [(h, n) for h, n in self._ring if n != node]

    @property
    def nodes(self) -> list[str]:
        return sorted({n for _, n in self._ring})

    def get_node(self, key: str) -> str:
        if not self._ring:
            raise LookupError("empty ring")
        h = stable_hash(key)
        idx = bisect.bisect_right(self._ring, (h, "￿"))
        return self._ring[idx % len(self._ring)][1]
