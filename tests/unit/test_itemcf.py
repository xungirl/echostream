import math

import numpy as np
import scipy.sparse as sp

from echostream.core.itemcf import itemcf_scores, itemcf_similarity

# users x items (binary). Items 0 and 1 are co-liked by users 0 and 1.
X = sp.csr_matrix(np.array([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 1, 1],
]))


def test_similarity_hand_computed():
    nb, sims = itemcf_similarity(X, topk=3)
    w0, w1, w2 = 1 / math.log(3), 1 / math.log(4), 1 / math.log(3)  # 1/log(1+|N(u)|)
    # sim(0,1): users 0 and 1 liked both; n0 = n1 = 2
    expected01 = (w0 + w1) / math.sqrt(2 * 2)
    # sim(0,2): only user 1; n0 = 2, n2 = 2
    expected02 = w1 / 2
    assert nb[0, 0] == 1 and np.isclose(sims[0, 0], expected01)
    assert nb[0, 1] == 2 and np.isclose(sims[0, 1], expected02)
    # item 3 is only co-liked with item 2 (user 2)
    assert nb[3, 0] == 2 and np.isclose(sims[3, 0], w2 / math.sqrt(2 * 1))
    # no self-similarity, -1 padding for missing neighbours
    assert 0 not in nb[0].tolist()
    assert nb[3, 1] == -1 and sims[3, 1] == 0


def test_scores_sum_neighbour_sims():
    nb = np.array([[1, 2], [0, -1], [0, -1]])
    s = np.array([[0.5, 0.2], [0.5, 0.0], [0.2, 0.0]], dtype=np.float32)
    scores = itemcf_scores([1, 2], nb, s, n_items=3)
    assert np.allclose(scores, [0.7, 0.0, 0.0])
    # recency decay: older item (1) weighted 0.5
    scores = itemcf_scores([1, 2], nb, s, n_items=3, recency_decay=0.5)
    assert np.allclose(scores, [0.5 * 0.5 + 0.2, 0.0, 0.0])
