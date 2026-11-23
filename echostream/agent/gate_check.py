"""Wire the regression gate (core/gate.py) to eval runs: candidate vs last good run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from echostream.agent.eval_run import per_user
from echostream.core.gate import check_metric, regression_gate

ROOT = Path(__file__).resolve().parents[2]
GUARDED = ("ndcg@10", "recall@200")


def seed_noise() -> dict[str, float]:
    """Run-to-run noise measured in Stage 2 by retraining the two-tower with 3 seeds."""
    s = json.load(open(ROOT / "results" / "stage2_two_tower.json"))["seed_std"]
    return {"ndcg@10": s["ndcg@10"], "recall@200": s["recall@200"]}


def gate(base_run: str, cand_run: str) -> dict:
    b, c = per_user(base_run), per_user(cand_run)
    assert np.array_equal(b["users"], c["users"]), "runs must cover the same users (paired test)"
    noise = seed_noise()
    checks = [check_metric(m, b[m], c[m], seed_noise=noise[m]) for m in GUARDED]
    promote, reasons = regression_gate(checks)
    return {"promote": promote, "reasons": reasons, "base_run": base_run, "candidate_run": cand_run,
            "checks": [c.__dict__ for c in checks]}
