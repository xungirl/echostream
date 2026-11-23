import numpy as np

from echostream.core.mmr import category_similarity, mmr_rerank


def test_lambda_one_is_relevance_order():
    rel = np.array([0.1, 0.9, 0.5])
    assert mmr_rerank(rel, np.zeros((3, 3)), lam=1.0, k=3) == [1, 2, 0]


def test_diversity_pushes_duplicate_category_down():
    rel = np.array([1.0, 0.95, 0.5])
    cats = [frozenset({1}), frozenset({1}), frozenset({2})]
    sim = category_similarity(cats)
    # hand: after picking 0, item1 = .5*.9 - .5*1 = -0.05 ; item2 = .5*0 - 0 = 0 -> item2 wins
    assert mmr_rerank(rel, sim, lam=0.5, k=3) == [0, 2, 1]
    assert mmr_rerank(rel, sim, lam=1.0, k=3) == [0, 1, 2]


def test_edge_cases():
    assert mmr_rerank(np.array([]), np.zeros((0, 0)), 0.5, 3) == []
    assert mmr_rerank(np.array([0.3, 0.3]), np.zeros((2, 2)), 0.5, 5) == [0, 1]
    assert category_similarity([frozenset({1, 2}), frozenset({2})])[0, 1] == 0.5
