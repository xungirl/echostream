import numpy as np

from echostream.core.gate import check_metric, noise_threshold, regression_gate


def test_noise_threshold_is_ci_half_width_or_seed_noise():
    assert noise_threshold(-0.02, 0.04) == 0.03
    assert noise_threshold(-0.02, 0.04, seed_noise=0.05) == 0.05


def test_identical_model_passes():
    rng = np.random.default_rng(0)
    base = rng.random(2000)
    ok, reasons = regression_gate([check_metric("ndcg@10", base, base.copy())])
    assert ok and reasons == []


def test_small_noise_passes_big_drop_blocks():
    rng = np.random.default_rng(0)
    base = rng.random(2000)
    noisy = base + rng.normal(0, 0.01, 2000)
    assert regression_gate([check_metric("ndcg@10", base, noisy)])[0]
    dropped = base * 0.8
    ok, reasons = regression_gate([check_metric("ndcg@10", base, dropped)])
    assert not ok and "ndcg@10" in reasons[0]


def test_seed_noise_raises_bar():
    base = np.full(1000, 0.5)
    cand = base - 0.01  # consistent small drop -> CI is tight
    assert check_metric("m", base, cand).regressed
    assert not check_metric("m", base, cand, seed_noise=0.02).regressed


def test_improvement_never_blocks():
    base = np.full(500, 0.3)
    assert not check_metric("m", base, base + 0.1).regressed
