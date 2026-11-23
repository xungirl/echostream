"""Thin wrapper around faiss so the rest of the code never touches faiss directly.

All indexes use inner product (vectors are L2-normalized, so IP == cosine).
"""
from __future__ import annotations

import faiss
import numpy as np


def build_index(vectors: np.ndarray, kind: str = "flat", **params) -> faiss.Index:
    """kind: "flat" (exact), "hnsw" (params: M, ef_construction), "ivf" (params: nlist)."""
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    d = x.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, params.get("M", 32), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = params.get("ef_construction", 80)
    elif kind == "ivf":
        nlist = params.get("nlist", 256)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        train_n = min(len(x), max(nlist * 40, 10_000))
        rng = np.random.default_rng(0)
        index.train(x[rng.choice(len(x), size=train_n, replace=False)])
    else:
        raise ValueError(f"unknown index kind {kind!r}")
    index.add(x)
    return index


def set_search_params(index: faiss.Index, ef_search: int | None = None, nprobe: int | None = None) -> None:
    if ef_search is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search
    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = nprobe


def ann_search(index: faiss.Index, queries: np.ndarray, k: int, id_map: np.ndarray | None = None
               ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Search and return per-query (ids, scores) with faiss' -1 padding removed.
    id_map translates row positions in the index back to global item ids."""
    q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
    k = min(k, index.ntotal)
    if k <= 0:
        return [np.empty(0, np.int64)] * len(q), [np.empty(0, np.float32)] * len(q)
    scores, rows = index.search(q, k)
    ids_out, scores_out = [], []
    for r, s in zip(rows, scores):
        keep = r >= 0
        r, s = r[keep], s[keep]
        ids_out.append(id_map[r] if id_map is not None else r.astype(np.int64))
        scores_out.append(s.astype(np.float32))
    return ids_out, scores_out


def recall_vs_exact(ann_ids: list[np.ndarray], exact_ids: list[np.ndarray], k: int) -> float:
    """Average |ANN top-k ∩ exact top-k| / k over queries."""
    vals = [len(set(a[:k].tolist()) & set(e[:k].tolist())) / k for a, e in zip(ann_ids, exact_ids)]
    return float(np.mean(vals))
