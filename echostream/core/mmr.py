"""Maximal Marginal Relevance re-ranking (Carbonell & Goldstein, 1998).

Greedy: at each step pick the candidate maximizing
    lambda * relevance(i) - (1 - lambda) * max_{j already picked} sim(i, j)
lambda = 1 -> pure relevance order, lambda = 0 -> pure diversity.
Relevance scores are min-max normalized first so lambda is scale-free.
"""
from __future__ import annotations

import numpy as np


def mmr_rerank(relevance: np.ndarray, similarity: np.ndarray, lam: float, k: int) -> list[int]:
    """relevance [n]; similarity [n, n] in [0, 1]; returns k positions into the candidate list."""
    n = len(relevance)
    k = min(k, n)
    if n == 0:
        return []
    rel = np.asarray(relevance, dtype=np.float64)
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 0 else np.ones_like(rel)
    chosen: list[int] = []
    max_sim = np.zeros(n)          # max similarity to anything already chosen
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        mmr = lam * rel - (1 - lam) * max_sim
        mmr[~available] = -np.inf
        best = int(np.argmax(mmr))  # argmax returns the first max -> stable ties
        chosen.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, similarity[best])
    return chosen


def category_similarity(cat_sets: list[frozenset[int]]) -> np.ndarray:
    """Jaccard similarity matrix between category sets."""
    n = len(cat_sets)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            union = len(cat_sets[i] | cat_sets[j])
            s = len(cat_sets[i] & cat_sets[j]) / union if union else 1.0
            sim[i, j] = sim[j, i] = s
    return sim
