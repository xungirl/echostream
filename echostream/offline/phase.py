"""Build the evaluation "phases" from the temporal split.

A phase answers: "standing at the start of a target window, knowing only the
past, which items will this user like inside the window?"

  val phase : history = train window              target = val window
  test phase: history = train + val windows       target = test window

Models are trained on the train window only. Items a user already interacted
with in the history are excluded both from the candidate lists and from the
truth sets (re-watches are not what a For-You feed is for).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from echostream.domain.base import DomainAdapter

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass
class Phase:
    name: str
    target_start: int                 # first date of the target window (YYYYMMDD)
    seen: sp.csr_matrix               # [n_users, n_items] 1 = any interaction in history
    pos_history: list[np.ndarray]     # per user: positive item ids in time order
    all_history: list[np.ndarray]     # per user: every item id in time order
    truth: list[set[int]]             # per user: new positives in the target window
    eval_users: np.ndarray            # users with a non-empty truth set
    history_items: np.ndarray | None = None  # items with >=1 interaction in history (not cold)

    def truths_for(self, users: np.ndarray) -> list[set[int]]:
        return [self.truth[u] for u in users]


def window(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[(df.date >= start) & (df.date <= end)]


def build_phase(adapter: DomainAdapter, name: str) -> Phase:
    df = adapter.interactions()
    spec = adapter.split_spec()
    target_range = spec.val if name == "val" else spec.test
    hist = df[(df.date >= spec.train[0]) & (df.date < target_range[0])]
    target = window(df, *target_range)

    n_u, n_i = adapter.n_users, adapter.n_items
    seen = sp.csr_matrix((np.ones(len(hist), np.float32), (hist.user_id.values, hist.item_id.values)), shape=(n_u, n_i))
    seen.data[:] = 1.0  # duplicates were summed

    def per_user(frame: pd.DataFrame) -> list[np.ndarray]:
        groups = frame.groupby("user_id").item_id.apply(lambda s: s.to_numpy(np.int64))
        return [groups.get(u, np.empty(0, np.int64)) for u in range(n_u)]

    pos_hist = per_user(hist[adapter.is_positive(hist.value.values)])
    all_hist = per_user(hist)
    tpos = target[adapter.is_positive(target.value.values)]
    truth: list[set[int]] = [set() for _ in range(n_u)]
    for u, i in zip(tpos.user_id.values, tpos.item_id.values):
        if seen[u, i] == 0:
            truth[u].add(int(i))
    eval_users = np.array([u for u in range(n_u) if truth[u]], dtype=np.int64)
    return Phase(name, target_range[0], seen, pos_hist, all_hist, truth, eval_users,
                 history_items=np.unique(hist.item_id.values))


def load_phase(adapter: DomainAdapter, name: str) -> Phase:
    """Cached on disk under artifacts/ (keyed by the label threshold)."""
    thr = getattr(adapter, "positive_threshold", "na")
    path = ARTIFACTS / "phases" / f"{adapter.name}_thr{thr}_{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    phase = build_phase(adapter, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(phase))
    return phase


def history_frame(adapter: DomainAdapter, phase_name: str, positives_only: bool = True) -> pd.DataFrame:
    """Everything a model may be fit on for this phase: events strictly before
    the target window (val phase -> train window; test phase -> train + val).

    Why refit on train+val for the test phase? KuaiRec's catalog drifts a lot
    between windows (86% of test-window positive items never appear in the
    train window, see DATA.md), so a model frozen at Aug 10 would be evaluated
    mostly on items it has never seen. Hyper-parameters are still chosen on
    the val phase only."""
    df = adapter.interactions()
    spec = adapter.split_spec()
    start = spec.val[0] if phase_name == "val" else spec.test[0]
    hist = df[(df.date >= spec.train[0]) & (df.date < start)]
    if positives_only:
        hist = hist[adapter.is_positive(hist.value.values)]
    return hist
