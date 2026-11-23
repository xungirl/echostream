"""Top-K ranking metrics + bootstrap confidence intervals.

Every metric works on ONE user's ranked list and returns a float, so they are
easy to check by hand. `evaluate_users` applies them to many users and returns
per-user arrays, which is what the bootstrap needs (we resample users).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], truth: set[int], k: int) -> float:
    """|top-k ∩ truth| / |truth|. Users with an empty truth set are skipped upstream."""
    if not truth:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in truth)
    return hits / len(truth)


def hit_rate_at_k(ranked: Sequence[int], truth: set[int], k: int) -> float:
    """1 if at least one relevant item is in the top-k, else 0."""
    return 1.0 if any(item in truth for item in ranked[:k]) else 0.0


def ndcg_at_k(ranked: Sequence[int], truth: set[int], k: int) -> float:
    """Binary-relevance NDCG: DCG = sum 1/log2(rank+1) over hits (rank starts at 1).
    IDCG puts min(k, |truth|) hits at the top."""
    if not truth:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked[:k]) if item in truth)
    ideal_hits = min(k, len(truth))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def mrr_at_k(ranked: Sequence[int], truth: set[int], k: int) -> float:
    """1 / rank of the first relevant item within the top-k, 0 if none."""
    for rank, item in enumerate(ranked[:k]):
        if item in truth:
            return 1.0 / (rank + 1)
    return 0.0


def catalog_coverage(all_ranked: Iterable[Sequence[int]], k: int, n_items: int) -> float:
    """Share of the catalog that appears in at least one user's top-k."""
    seen: set[int] = set()
    for ranked in all_ranked:
        seen.update(ranked[:k])
    return len(seen) / n_items


def intra_list_diversity(categories: Sequence[frozenset[int] | set[int]]) -> float:
    """Mean pairwise (1 - Jaccard) of the items' category sets.
    0 = every item shares exactly the same categories, 1 = no overlap at all."""
    n = len(categories)
    if n < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = categories[i], categories[j]
            union = len(a | b)
            jaccard = len(a & b) / union if union else 1.0
            total += 1.0 - jaccard
            pairs += 1
    return total / pairs


PER_USER_METRICS = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit": hit_rate_at_k,
    "mrr": mrr_at_k,
}


def evaluate_users(
    ranked_lists: Sequence[Sequence[int]],
    truths: Sequence[set[int]],
    ks: Sequence[int] = (10, 50, 100, 200),
    item_categories: dict[int, frozenset[int]] | None = None,
    n_items: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Return per-user arrays like {"ndcg@10": array([...]), ...} plus scalar
    catalog coverage ("coverage@k"). Diversity (per user) needs item_categories."""
    assert len(ranked_lists) == len(truths)
    out: dict[str, np.ndarray | float] = {}
    for name, fn in PER_USER_METRICS.items():
        for k in ks:
            out[f"{name}@{k}"] = np.array([fn(r, t, k) for r, t in zip(ranked_lists, truths)], dtype=np.float64)
    if item_categories is not None:
        for k in ks:
            if k > 50:  # O(k^2) pairs; 10 and 50 are the lists a user actually sees
                continue
            out[f"diversity@{k}"] = np.array(
                [intra_list_diversity([item_categories.get(i, frozenset()) for i in r[:k]]) for r in ranked_lists]
            )
    if n_items is not None:
        for k in ks:
            out[f"coverage@{k}"] = catalog_coverage(ranked_lists, k, n_items)
    return out


def bootstrap_ci(
    per_user: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, low, high) percentile bootstrap CI of the mean, resampling users."""
    per_user = np.asarray(per_user, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(per_user)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = per_user[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(per_user.mean()), float(low), float(high)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """CI of mean(b - a) where a[i], b[i] belong to the same user (paired)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    return bootstrap_ci(b - a, n_boot=n_boot, alpha=alpha, seed=seed)
