"""Hand-written item-based collaborative filtering ("people who liked A also liked B").

sim(i, j) = sum_u w_u * x_ui * x_uj / sqrt(n_i * n_j)

x is the binary user×item "positive" matrix, n_i the number of users who liked
item i, and w_u = 1 / log(1 + |items of u|) down-weights super-active users
(the IUF trick from the classic ItemCF papers: a user who likes everything says
little about any pair).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def itemcf_similarity(user_item: sp.csr_matrix, topk: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (neighbors, sims), each shaped [n_items, topk], sorted by sim desc.
    Missing neighbors are padded with index -1 and similarity 0."""
    x = (user_item > 0).astype(np.float32).tocsr()
    n_items = x.shape[1]
    items_per_user = np.asarray(x.sum(axis=1)).ravel()
    w = 1.0 / np.log1p(np.maximum(items_per_user, 1.0))
    weighted = sp.diags(w.astype(np.float32)) @ x
    co = (x.T @ weighted).toarray()  # dense [n_items, n_items]; fine for ~10k items
    np.fill_diagonal(co, 0.0)
    n_i = np.asarray(x.sum(axis=0)).ravel()
    norm = np.sqrt(np.outer(n_i, n_i))
    sim = np.divide(co, norm, out=np.zeros_like(co), where=norm > 0)

    k = min(topk, n_items - 1)
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    part_sims = np.take_along_axis(sim, part, axis=1)
    order = np.argsort(-part_sims, axis=1, kind="stable")
    neighbors = np.take_along_axis(part, order, axis=1)
    sims = np.take_along_axis(part_sims, order, axis=1)
    neighbors = np.where(sims > 0, neighbors, -1)
    sims = np.where(sims > 0, sims, 0.0).astype(np.float32)
    if k < topk:  # pad tiny catalogs
        pad = topk - k
        neighbors = np.pad(neighbors, ((0, 0), (0, pad)), constant_values=-1)
        sims = np.pad(sims, ((0, 0), (0, pad)))
    return neighbors.astype(np.int64), sims


def itemcf_scores(
    history: list[int], neighbors: np.ndarray, sims: np.ndarray, n_items: int,
    recency_decay: float = 1.0,
) -> np.ndarray:
    """Score every item for a user with this (oldest→newest) history.
    score[j] = sum_h decay^(age of h) * sim(h, j). decay=1 ignores recency."""
    scores = np.zeros(n_items, dtype=np.float32)
    n = len(history)
    for pos, h in enumerate(history):
        weight = recency_decay ** (n - 1 - pos)
        nb, s = neighbors[h], sims[h]
        valid = nb >= 0
        np.add.at(scores, nb[valid], weight * s[valid])
    return scores
