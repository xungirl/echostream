"""Hand-computed metric examples (each number below was worked out on paper)."""
import math

import numpy as np
import pytest

from echostream.core.metrics import (
    bootstrap_ci, catalog_coverage, evaluate_users, hit_rate_at_k, intra_list_diversity,
    mrr_at_k, ndcg_at_k, paired_bootstrap_diff, recall_at_k,
)

RANKED = [5, 3, 9, 1, 7]
TRUTH = {3, 7, 100}


def test_recall():
    assert recall_at_k(RANKED, TRUTH, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, TRUTH, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_hit_rate():
    assert hit_rate_at_k(RANKED, TRUTH, 1) == 0.0
    assert hit_rate_at_k(RANKED, TRUTH, 2) == 1.0


def test_ndcg_hand_computed():
    # hits at rank 2 and 5: DCG = 1/log2(3) + 1/log2(6); IDCG (3 relevant, k=5) = 1 + 1/log2(3) + 1/2
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(RANKED, TRUTH, 5) == pytest.approx(dcg / idcg)
    # k=2: one hit at rank 2, ideal has 2 hits
    assert ndcg_at_k(RANKED, TRUTH, 2) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_ndcg_perfect_is_one():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2, 3], {1}, 3) == pytest.approx(1.0)


def test_mrr():
    assert mrr_at_k(RANKED, TRUTH, 5) == pytest.approx(0.5)
    assert mrr_at_k(RANKED, {7}, 5) == pytest.approx(0.2)
    assert mrr_at_k(RANKED, {7}, 4) == 0.0


def test_coverage():
    assert catalog_coverage([[1, 2], [2, 3]], 2, 10) == pytest.approx(0.3)
    assert catalog_coverage([[1, 2], [2, 3]], 1, 10) == pytest.approx(0.2)


def test_diversity():
    same = [frozenset({1}), frozenset({1}), frozenset({1})]
    assert intra_list_diversity(same) == 0.0
    disjoint = [frozenset({1}), frozenset({2}), frozenset({3})]
    assert intra_list_diversity(disjoint) == 1.0
    # {1,2} vs {2}: jaccard 1/2 -> diversity 0.5 (single pair)
    assert intra_list_diversity([frozenset({1, 2}), frozenset({2})]) == pytest.approx(0.5)


def test_evaluate_users_shapes():
    out = evaluate_users([RANKED, [100]], [TRUTH, {100}], ks=(1, 5),
                         item_categories={5: frozenset({1})}, n_items=200)
    assert out["hit@1"].tolist() == [0.0, 1.0]
    assert out["recall@5"][0] == pytest.approx(2 / 3)
    assert out["coverage@1"] == pytest.approx(2 / 200)
    assert "diversity@5" in out


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    x = np.random.default_rng(1).random(500)
    m, lo, hi = bootstrap_ci(x, n_boot=500, seed=3)
    assert lo < m < hi
    assert bootstrap_ci(x, n_boot=500, seed=3) == (m, lo, hi)
    m0, lo0, hi0 = bootstrap_ci(np.full(50, 0.25))
    assert m0 == lo0 == hi0 == 0.25


def test_paired_diff():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    d, lo, hi = paired_bootstrap_diff(a, a + 0.1)
    assert d == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
