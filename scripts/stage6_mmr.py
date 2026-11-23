"""Stage 6: MMR re-rank sweep on the test phase.

Pool = top-50 of the ranker's list, MMR picks 10. Similarity = mean of category
Jaccard and caption cosine. Selection rule (fixed before running): the default
lambda is the SMALLEST lambda (most diverse) whose NDCG@10 drop vs lambda=1 is
not significant (paired 95% bootstrap CI of the difference contains 0);
if no lambda < 1 qualifies, the default is 1.0 (MMR off).

    uv run python scripts/stage6_mmr.py -> results/stage6_mmr.json
"""
import json
import time

import lightgbm as lgb
import numpy as np

from echostream.core.metrics import evaluate_users, paired_bootstrap_diff
from echostream.core.mmr import mmr_rerank
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import summarize, write_json
from echostream.offline.phase import ARTIFACTS, load_phase
from echostream.rank.offline import blend_similarity, candidates_and_features, rerank, split_scores

LAMBDAS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
POOL, N = 50, 10


def main():
    t0 = time.time()
    a = KuaiRecAdapter()
    test = load_phase(a, "test")
    quotas = {k: v for k, v in json.load(open("results/stage3_multipath.json"))["selected_quotas"].items() if k != "name"}
    users, rows, X, y, cands, eng, ra = candidates_and_features(a, test, "main", quotas)
    model = lgb.Booster(model_file=str(ARTIFACTS / "ranker" / "main.txt"))
    scores = model.predict(X)
    ranked = rerank(cands, scores)
    per_user_scores = [np.sort(s)[::-1] for s in split_scores(cands, scores)]
    truths = test.truths_for(users)
    cats = a.item_categories()
    out = {"pool": POOL, "n": N, "lambdas": {}, "rule": __doc__.split("Selection rule")[1].split("uv run")[0].strip()}
    base = None
    for lam in LAMBDAS:
        lists = []
        for r, s in zip(ranked, per_user_scores):
            pool = r[:POOL]
            sim = blend_similarity(ra.item_cats, eng.a.caption_emb, pool)
            pick = mmr_rerank(s[:len(pool)], sim, lam, N)
            lists.append([pool[j] for j in pick])
        pu = evaluate_users(lists, truths, (10,), cats, a.n_items)
        if base is None:
            base = pu
        d, lo, hi = paired_bootstrap_diff(base["ndcg@10"], pu["ndcg@10"])
        out["lambdas"][str(lam)] = {**{k: v for k, v in summarize(pu).items() if k.endswith("@10")},
                                    "ndcg@10_delta_vs_lambda1": {"delta": round(d, 5), "ci95": [round(lo, 5), round(hi, 5)]}}
        print(lam, out["lambdas"][str(lam)]["ndcg@10"], out["lambdas"][str(lam)]["diversity@10"],
              out["lambdas"][str(lam)]["coverage@10"], round(d, 5), flush=True)
    ok = [lam for lam in LAMBDAS if lam < 1.0 and out["lambdas"][str(lam)]["ndcg@10_delta_vs_lambda1"]["ci95"][0] <= 0
          <= out["lambdas"][str(lam)]["ndcg@10_delta_vs_lambda1"]["ci95"][1]]
    out["selected_lambda"] = min(ok) if ok else 1.0
    out["runtime_sec"] = round(time.time() - t0, 1)
    write_json("results/stage6_mmr.json", out)
    print("selected", out["selected_lambda"])


if __name__ == "__main__":
    main()
