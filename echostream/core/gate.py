"""Regression gate: should a candidate model be promoted?

The noise threshold is not a magic constant. For each guarded metric we
bootstrap the *paired* per-user difference (candidate - last good) and take the
CI half-width as the noise level at this sample size. A drop is a regression
only if it is larger than that noise level (i.e. the whole 95% CI of the
difference is below zero) AND larger than the run-to-run noise we measured by
retraining with different seeds (`seed_noise`, also measured, not assumed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echostream.core.metrics import paired_bootstrap_diff


@dataclass
class GateCheck:
    metric: str
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    threshold: float   # max(bootstrap half-width, seed noise)
    regressed: bool


def noise_threshold(ci_low: float, ci_high: float, seed_noise: float = 0.0) -> float:
    return max((ci_high - ci_low) / 2.0, seed_noise)


def check_metric(name: str, base_per_user: np.ndarray, cand_per_user: np.ndarray,
                 seed_noise: float = 0.0, n_boot: int = 1000, seed: int = 0) -> GateCheck:
    delta, low, high = paired_bootstrap_diff(base_per_user, cand_per_user, n_boot=n_boot, seed=seed)
    thr = noise_threshold(low, high, seed_noise)
    return GateCheck(
        metric=name, baseline=float(np.mean(base_per_user)), candidate=float(np.mean(cand_per_user)),
        delta=delta, ci_low=low, ci_high=high, threshold=thr,
        regressed=bool(delta < -thr and high < 0),
    )


def regression_gate(checks: list[GateCheck]) -> tuple[bool, list[str]]:
    """Returns (promote?, reasons). Any regressed guarded metric blocks promotion."""
    reasons = [
        f"{c.metric}: {c.baseline:.4f} -> {c.candidate:.4f} (delta {c.delta:+.4f}, "
        f"95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}], noise threshold {c.threshold:.4f})"
        for c in checks if c.regressed
    ]
    return (not reasons), reasons
