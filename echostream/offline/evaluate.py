"""Shared offline evaluation: turn model scores into top-K lists, compute
metrics per user, summarize with bootstrap CIs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from echostream.core.metrics import bootstrap_ci, evaluate_users
from echostream.domain.base import DomainAdapter
from echostream.offline.phase import ARTIFACTS, Phase

KS = (10, 50, 100, 200)
ScoreFn = Callable[[np.ndarray], np.ndarray]   # users [B] -> scores [B, n_items]


def rank_topk(score_fn: ScoreFn, users: np.ndarray, phase: Phase | None, k: int = 200,
              candidate_items: np.ndarray | None = None, batch: int = 512) -> list[list[int]]:
    """Top-k item ids per user. Items seen in the phase history are masked;
    if candidate_items is given, only those items can be ranked."""
    out: list[list[int]] = []
    allowed = None
    for start in range(0, len(users), batch):
        ub = users[start:start + batch]
        scores = np.asarray(score_fn(ub), dtype=np.float32).copy()
        if allowed is None:
            allowed = np.ones(scores.shape[1], dtype=bool)
            if candidate_items is not None:
                allowed[:] = False
                allowed[candidate_items] = True
        scores[:, ~allowed] = -np.inf
        if phase is not None:
            seen = phase.seen[ub]
            rows = np.repeat(np.arange(len(ub)), np.diff(seen.indptr))
            scores[rows, seen.indices] = -np.inf
        kk = min(k, int(allowed.sum()))
        top = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.lexsort((top, -top_scores), axis=1)  # score desc, then item id asc
        out.extend(np.take_along_axis(top, order, axis=1).tolist())
    return out


def summarize(per_user: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    summary = {}
    for name, val in per_user.items():
        if isinstance(val, np.ndarray):
            m, lo, hi = bootstrap_ci(val, n_boot=n_boot, seed=seed)
            summary[name] = {"mean": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)]}
        else:
            summary[name] = {"mean": round(float(val), 5)}
    return summary


def evaluate_ranked(ranked: list[list[int]], truths: list[set[int]], adapter: DomainAdapter,
                    n_items: int | None = None) -> dict:
    return evaluate_users(ranked, truths, KS, adapter.item_categories(), n_items or adapter.n_items)


def save_per_user(tag: str, per_user: dict) -> None:
    d = ARTIFACTS / "per_user"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{tag}.npz", **{k: v for k, v in per_user.items() if isinstance(v, np.ndarray)})


def load_per_user(tag: str) -> dict[str, np.ndarray]:
    return dict(np.load(ARTIFACTS / "per_user" / f"{tag}.npz"))


def small_matrix_protocol(adapter: DomainAdapter) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Fully-observed evaluation: (users, candidate items, truth sets).
    Every small-matrix user is scored on the 3,327 small-matrix items only;
    truth = items they actually liked (label rule of the adapter). Because the
    matrix is ~99.6% observed there is no exposure bias in the truth sets."""
    small = adapter.fully_observed()
    users = np.sort(small.user_id.unique()).astype(np.int64)
    items = np.sort(small.item_id.unique()).astype(np.int64)
    pos = small[adapter.is_positive(small.value.values)]
    groups = pos.groupby("user_id").item_id.apply(lambda s: set(s.tolist()))
    truths = [groups.get(u, set()) for u in users]
    keep = np.array([len(t) > 0 for t in truths])
    return users[keep], items, [t for t, k in zip(truths, keep) if k]


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
