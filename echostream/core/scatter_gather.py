"""Merge per-shard top-K results into a global top-K.

Each shard holds a disjoint slice of the item vectors and returns its own
local top-K. The global top-K is guaranteed to be inside the union of the local
top-Ks, so a k-way merge by score is exact. A shard that failed (None) is
skipped and reported, so the caller can label the response as partial.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass
class MergeResult:
    ids: list[int]
    scores: list[float]
    missing_shards: list[int] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return bool(self.missing_shards)


def merge_topk(shard_results: list[tuple[list[int], list[float]] | None], k: int) -> MergeResult:
    """shard_results[s] = (ids, scores) sorted desc, or None if shard s failed.
    Ties are broken by item id so the merge is deterministic. If the same id
    is returned by two shards (should not happen with disjoint shards, but can
    during a re-shard) the higher score wins."""
    best: dict[int, float] = {}
    missing = []
    for s, res in enumerate(shard_results):
        if res is None:
            missing.append(s)
            continue
        for item, score in zip(*res):
            item = int(item)
            if item not in best or score > best[item]:
                best[item] = float(score)
    top = heapq.nsmallest(k, best.items(), key=lambda kv: (-kv[1], kv[0]))
    return MergeResult([i for i, _ in top], [s for _, s in top], missing)
