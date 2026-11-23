"""Stage 2b: faiss exact Flat-IP vs HNSW vs IVF on the real two-tower item vectors.
Queries are the real test-phase user vectors. Single-thread, one query at a time
(that is what a serving shard does per request). 3 repeats per config.

    uv run python scripts/stage2_ann_bench.py  -> results/stage2_ann.json
"""
import platform
import time

import faiss
import numpy as np

from echostream.core.ann import ann_search, build_index, recall_vs_exact, set_search_params
from echostream.offline.evaluate import write_json
from echostream.offline.phase import ARTIFACTS

K = 100
N_LAT_QUERIES = 2000
REPEATS = 3


def latency(index, queries):
    lat = []
    for q in queries:
        t = time.perf_counter()
        index.search(q[None, :], K)
        lat.append((time.perf_counter() - t) * 1e6)
    lat = np.array(lat)
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 99))


def main():
    faiss.omp_set_num_threads(1)
    d = ARTIFACTS / "two_tower" / "main"
    items = np.load(d / "item_emb.npy")
    users = np.load(d / "user_vecs_test.npy")
    rng = np.random.default_rng(0)
    lat_q = users[rng.choice(len(users), N_LAT_QUERIES, replace=False)]
    flat = build_index(items, "flat")
    exact_ids, _ = ann_search(flat, users, K)
    exact10 = [e[:10] for e in exact_ids]

    configs = [("flat", {}, {})]
    configs += [("hnsw", {"M": m}, {"ef_search": ef}) for m in (16, 32) for ef in (16, 32, 64, 128, 256)]
    configs += [("ivf", {"nlist": nl}, {"nprobe": p}) for nl in (64, 256) for p in (1, 4, 16, 64)]
    rows, built = [], {}
    for kind, bparams, sparams in configs:
        key = (kind, tuple(sorted(bparams.items())))
        if key not in built:
            t = time.perf_counter()
            built[key] = (build_index(items, kind, **bparams), time.perf_counter() - t)
        index, build_sec = built[key]
        set_search_params(index, **sparams)
        ids, _ = ann_search(index, users, K)
        reps = [latency(index, lat_q) for _ in range(REPEATS)]
        p50s, p99s = [r[0] for r in reps], [r[1] for r in reps]
        row = {"index": kind, **bparams, **sparams,
               "recall@100_vs_exact": round(recall_vs_exact(ids, exact_ids, K), 4),
               "recall@10_vs_exact": round(recall_vs_exact([i[:10] for i in ids], exact10, 10), 4),
               "p50_us_mean": round(float(np.mean(p50s)), 1), "p50_us_std": round(float(np.std(p50s)), 1),
               "p99_us_mean": round(float(np.mean(p99s)), 1), "p99_us_std": round(float(np.std(p99s)), 1),
               "build_sec": round(build_sec, 3), "index_bytes": int(len(faiss.serialize_index(index)))}
        rows.append(row)
        print(row, flush=True)
    write_json("results/stage2_ann.json", {
        "kind": "quality+latency on real KuaiRec two-tower vectors",
        "machine": f"{platform.machine()} {platform.platform()}", "faiss": faiss.__version__,
        "n_items": int(len(items)), "dim": int(items.shape[1]), "n_recall_queries": int(len(users)),
        "n_latency_queries": N_LAT_QUERIES, "repeats": REPEATS, "threads": 1, "k": K, "rows": rows})


if __name__ == "__main__":
    main()
