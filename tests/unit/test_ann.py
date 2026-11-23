import numpy as np

from echostream.core.ann import ann_search, build_index, recall_vs_exact, set_search_params


def data(n=2000, d=16, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_flat_is_exact_and_id_map_applied():
    x = data()
    idx = build_index(x, "flat")
    q = x[:3]
    ids, scores = ann_search(idx, q, k=5, id_map=np.arange(len(x)) + 1000)
    brute = np.argsort(-(q @ x.T), axis=1)[:, :5] + 1000
    for a, b in zip(ids, brute):
        assert a.tolist() == b.tolist()
    assert scores[0][0] == np.float32(1.0) or abs(scores[0][0] - 1.0) < 1e-5


def test_hnsw_and_ivf_recall_reasonable():
    x = data()
    q = data(50, seed=1)
    exact, _ = ann_search(build_index(x, "flat"), q, 10)
    h = build_index(x, "hnsw", M=16)
    set_search_params(h, ef_search=128)
    assert recall_vs_exact(ann_search(h, q, 10)[0], exact, 10) > 0.9
    ivf = build_index(x, "ivf", nlist=16)
    set_search_params(ivf, nprobe=16)  # probing all lists == exact
    assert recall_vs_exact(ann_search(ivf, q, 10)[0], exact, 10) == 1.0


def test_k_larger_than_index_and_empty():
    x = data(3)
    ids, _ = ann_search(build_index(x, "flat"), x[:1], k=10)
    assert len(ids[0]) == 3
    assert recall_vs_exact([np.array([1, 2])], [np.array([2, 3])], 2) == 0.5
