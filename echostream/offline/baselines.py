"""Baseline recommenders. Each one: fit(adapter, phase) then score(users) -> [B, n_items].
All fitting uses only events strictly before the phase's target window."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from echostream.core.itemcf import itemcf_similarity
from echostream.domain.base import DomainAdapter
from echostream.offline.phase import Phase, history_frame


class Popularity:
    """Most-liked items. window_days=None -> all history; else only the last N logged days."""

    def __init__(self, window_days: int | None = None):
        self.window_days = window_days
        self.name = "pop_global" if window_days is None else f"pop_recent{window_days}d"

    def fit(self, adapter: DomainAdapter, phase: Phase) -> "Popularity":
        pos = history_frame(adapter, phase.name)
        if self.window_days is not None:
            days = sorted(pos.date.unique())[-self.window_days:]
            pos = pos[pos.date.isin(days)]
        counts = np.bincount(pos.item_id.values, minlength=adapter.n_items).astype(np.float32)
        self.item_scores = counts / max(counts.max(), 1.0)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.item_scores, (len(users), len(self.item_scores)))


class PlatformPopularity:
    """Popularity measured by the platform-wide daily stats (all Kuaishou users,
    item_daily_features) over the last N days before the cutoff. Unlike the
    in-dataset popularity it also knows items no KuaiRec user has touched yet."""

    def __init__(self, days: int = 3, column: str = "like_cnt"):
        self.days, self.column = days, column
        self.name = f"pop_platform{days}d"

    def fit(self, adapter: DomainAdapter, phase: Phase) -> "PlatformPopularity":
        d = pd.read_parquet(adapter.data_dir / "item_daily_features.parquet", columns=["video_id", "date", self.column])
        d = d[d.date < phase.target_start]
        days = sorted(d.date.unique())[-self.days:]
        s = d[d.date.isin(days)].groupby("video_id")[self.column].sum()
        scores = np.zeros(adapter.n_items, np.float32)
        scores[s.index.values] = s.values
        self.item_scores = scores / max(scores.max(), 1.0)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.item_scores, (len(users), len(self.item_scores)))


def positives_matrix(adapter: DomainAdapter, phase: Phase) -> sp.csr_matrix:
    pos = history_frame(adapter, phase.name)
    m = sp.csr_matrix((np.ones(len(pos), np.float32), (pos.user_id.values, pos.item_id.values)),
                      shape=(adapter.n_users, adapter.n_items))
    m.data[:] = 1.0
    return m


def recency_weights(pos_history: list[np.ndarray], n_items: int, last_n: int, decay: float) -> sp.csr_matrix:
    """Sparse [n_users, n_items] matrix of history weights decay^(age), last_n most recent positives."""
    rows, cols, vals = [], [], []
    for u, hist in enumerate(pos_history):
        h = hist[-last_n:]
        w = decay ** np.arange(len(h) - 1, -1, -1, dtype=np.float32)
        rows.append(np.full(len(h), u)); cols.append(h); vals.append(w)
    m = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(pos_history), n_items))
    return m  # duplicates are summed, which is what itemcf_scores does too


class ItemCF:
    """Hand-written ItemCF (echostream.core.itemcf). Batch scoring uses a
    sparse matmul that is equivalent to calling core.itemcf_scores per user
    (checked in tests/unit/test_baselines.py)."""

    def __init__(self, topk: int = 200, last_n: int = 50, decay: float = 0.98):
        self.topk, self.last_n, self.decay = topk, last_n, decay
        self.name = "itemcf"

    def fit(self, adapter: DomainAdapter, phase: Phase) -> "ItemCF":
        n = adapter.n_items
        self.neighbors, self.sims = itemcf_similarity(positives_matrix(adapter, phase), self.topk)
        valid = self.neighbors >= 0
        rows = np.repeat(np.arange(n), valid.sum(1))
        self.sim_matrix = sp.csr_matrix((self.sims[valid], (rows, self.neighbors[valid])), shape=(n, n))
        self.history = recency_weights(phase.pos_history, n, self.last_n, self.decay)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.history[users] @ self.sim_matrix).toarray()


class ALS:
    """Matrix factorization with the `implicit` library (Hu, Koren & Volinsky 2008)."""

    def __init__(self, factors: int = 64, regularization: float = 0.05, alpha: float = 10.0,
                 iterations: int = 15, seed: int = 0):
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha,
                           iterations=iterations, random_state=seed)
        self.name = "als"

    def fit(self, adapter: DomainAdapter, phase: Phase) -> "ALS":
        from implicit.cpu.als import AlternatingLeastSquares
        model = AlternatingLeastSquares(**self.params)
        model.fit(positives_matrix(adapter, phase), show_progress=False)
        self.user_factors = np.asarray(model.user_factors)
        self.item_factors = np.asarray(model.item_factors)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors[users] @ self.item_factors.T
