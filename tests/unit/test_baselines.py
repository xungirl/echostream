"""The batched ItemCF scorer must equal the readable per-user core function."""
import numpy as np
import scipy.sparse as sp

from echostream.core.itemcf import itemcf_scores, itemcf_similarity
from echostream.offline.baselines import recency_weights


def test_sparse_batch_equals_core_function():
    rng = np.random.default_rng(0)
    x = sp.csr_matrix((rng.random((30, 12)) > 0.6).astype(np.float32))
    nb, sims = itemcf_similarity(x, topk=5)
    valid = nb >= 0
    rows = np.repeat(np.arange(12), valid.sum(1))
    sim_m = sp.csr_matrix((sims[valid], (rows, nb[valid])), shape=(12, 12))
    hists = [rng.integers(0, 12, size=rng.integers(1, 8)) for _ in range(5)]
    w = recency_weights(hists, 12, last_n=4, decay=0.9)
    batch = (w @ sim_m).toarray()
    for u, h in enumerate(hists):
        core = itemcf_scores(list(h[-4:]), nb, sims, 12, recency_decay=0.9)
        assert np.allclose(batch[u], core, atol=1e-5)
