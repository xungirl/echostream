"""Recall paths shared by offline evaluation and online serving (numpy only).

Online, the two-tower path is answered by the sharded faiss servers
(scatter-gather) instead of `two_tower()` below; both are exact inner-product
search over the same vectors, so offline numbers describe the online path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from echostream.core.itemcf import itemcf_scores
from echostream.core.recall_merge import Candidate, merge_with_quotas
from echostream.core.two_tower import user_tower_numpy

HIST_LEN = 50          # user-tower history length (same as training)
CF_HIST = 20           # ItemCF uses the 20 most recent positives
CF_DECAY = 0.98
CONTENT_HIST = 10      # content path averages the captions of the 10 most recent positives


@dataclass
class RecallAssets:
    item_vecs: np.ndarray                 # [n_items, d] two-tower item vectors
    user_tower: dict[str, np.ndarray]     # numpy user tower params
    user_cats: np.ndarray                 # [n_users, F]
    cf_neighbors: np.ndarray              # [n_items, topk]
    cf_sims: np.ndarray
    pop_scores: dict[str, np.ndarray]     # name -> [n_items]
    caption_emb: np.ndarray               # [n_items, 512] L2-normalized
    item_hist_count: np.ndarray           # interactions per item before the cutoff (0 = cold item)
    quotas: dict[str, int] = field(default_factory=dict)
    total: int = 200


def topk_excluding(scores: np.ndarray, k: int, exclude: np.ndarray | None,
                   allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
    s = scores.astype(np.float32, copy=True)
    if exclude is not None and len(exclude):
        s[exclude] = -np.inf
    if allowed is not None:
        s[~allowed] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.lexsort((top, -s[top]))]
    return [(int(i), float(s[i])) for i in top]


class RecallEngine:
    def __init__(self, assets: RecallAssets):
        self.a = assets
        self.n_items = len(assets.item_vecs)
        self.cold_mask = assets.item_hist_count == 0

    # ---- individual paths -------------------------------------------------
    def user_vector(self, user: int, hist: np.ndarray) -> np.ndarray:
        return user_tower_numpy(self.a.user_tower, self.a.user_cats[user], list(hist[-HIST_LEN:]))

    def two_tower(self, user_vec: np.ndarray, k: int, exclude=None) -> list[tuple[int, float]]:
        return topk_excluding(self.a.item_vecs @ user_vec, k, exclude)

    def itemcf(self, hist: np.ndarray, k: int, exclude=None) -> list[tuple[int, float]]:
        if len(hist) == 0:
            return []
        s = itemcf_scores(list(hist[-CF_HIST:]), self.a.cf_neighbors, self.a.cf_sims, self.n_items, CF_DECAY)
        res = topk_excluding(s, k, exclude)
        return [(i, sc) for i, sc in res if sc > 0]

    def popularity(self, name: str, k: int, exclude=None) -> list[tuple[int, float]]:
        return topk_excluding(self.a.pop_scores[name], k, exclude)

    def content(self, hist: np.ndarray, k: int, exclude=None, cold_only: bool = True) -> list[tuple[int, float]]:
        """Caption-embedding similarity to the user's recent likes. cold_only=True
        restricts it to items with no interactions yet: the path exists to give
        brand-new videos a way into the feed."""
        if len(hist) == 0:
            return []
        q = self.a.caption_emb[hist[-CONTENT_HIST:]].mean(axis=0)
        return topk_excluding(self.a.caption_emb @ q, k, exclude, self.cold_mask if cold_only else None)

    # ---- all paths + merge --------------------------------------------------
    def all_paths(self, user: int, hist: np.ndarray, exclude: np.ndarray, k: int = 200,
                  user_vec: np.ndarray | None = None) -> dict[str, list[tuple[int, float]]]:
        uv = self.user_vector(user, hist) if user_vec is None else user_vec
        paths = {"two_tower": self.two_tower(uv, k, exclude), "itemcf": self.itemcf(hist, k, exclude)}
        for name in self.a.pop_scores:
            paths[name] = self.popularity(name, k, exclude)
        paths["content"] = self.content(hist, k, exclude)
        return paths

    def merge(self, paths: dict[str, list[tuple[int, float]]], exclude=frozenset()) -> list[Candidate]:
        ordered = {name: paths.get(name, []) for name in self.a.quotas}
        return merge_with_quotas(ordered, self.a.quotas, self.a.total, exclude)
