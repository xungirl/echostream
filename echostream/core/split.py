"""Temporal train/val/test split.

Why not a random split? A random split puts a user's *future* interactions in
the training set (and item popularity measured on the future), so the model is
graded on predicting the past from the future. On a feed whose popularity moves
day to day that inflates every metric. Splitting by time means "train on what
happened before day X, predict what happens after day X" — the only setting a
real recommender is in.
"""
from __future__ import annotations

import numpy as np


def time_cutoffs(timestamps: np.ndarray, val_frac: float, test_frac: float) -> tuple[float, float]:
    """Pick (val_start, test_start) so the last `test_frac` of events (by time)
    are test and the `val_frac` before them are validation."""
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1
    ts = np.sort(np.asarray(timestamps, dtype=np.float64))
    test_start = float(ts[int(len(ts) * (1 - test_frac))])
    val_start = float(ts[int(len(ts) * (1 - test_frac - val_frac))])
    return val_start, test_start


def temporal_split(timestamps: np.ndarray, val_start: float, test_start: float) -> np.ndarray:
    """Return a label per event: 0 = train (t < val_start), 1 = val, 2 = test (t >= test_start)."""
    assert val_start <= test_start
    ts = np.asarray(timestamps, dtype=np.float64)
    part = np.zeros(len(ts), dtype=np.int8)
    part[ts >= val_start] = 1
    part[ts >= test_start] = 2
    return part


def assert_no_future_leak(feature_times: np.ndarray, cutoff: float) -> None:
    """Guard used by feature builders: every event used for a feature must be
    strictly before the prediction cutoff."""
    ft = np.asarray(feature_times, dtype=np.float64)
    if len(ft) and ft.max() >= cutoff:
        raise ValueError(f"feature uses an event at t={ft.max()} >= cutoff {cutoff}")
