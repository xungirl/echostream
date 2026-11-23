"""Stage 1: baselines on the temporal split (big matrix) + fully-observed small matrix.

    uv run python scripts/stage1_baselines.py   -> results/stage1_baselines.json
"""
import time

import numpy as np

from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.baselines import ALS, ItemCF, PlatformPopularity, Popularity
from echostream.offline.evaluate import (evaluate_ranked, rank_topk, save_per_user, small_matrix_protocol,
                                         summarize, write_json)
from echostream.offline.phase import load_phase

SELECT = "ndcg@10"


def run(model, adapter, phase, users=None, candidate_items=None, truths=None, use_seen=True):
    users = phase.eval_users if users is None else users
    truths = phase.truths_for(users) if truths is None else truths
    ranked = rank_topk(model.score, users, phase if use_seen else None, 200, candidate_items)
    n_cand = len(candidate_items) if candidate_items is not None else adapter.n_items
    return evaluate_ranked(ranked, truths, adapter, n_items=n_cand)


def brief(per_user):
    return {k: round(float(np.mean(v)), 5) for k, v in per_user.items() if k in ("ndcg@10", "recall@100", "recall@200")}


def main():
    t0 = time.time()
    a = KuaiRecAdapter()
    val, test = load_phase(a, "val"), load_phase(a, "test")
    out = {"protocol": {
        "train_dates": a.split_spec().train, "val_dates": a.split_spec().val, "test_dates": a.split_spec().test,
        "label": "watch_ratio > 2.0", "selection_metric": f"val {SELECT}",
        "val_eval_users": int(len(val.eval_users)), "test_eval_users": int(len(test.eval_users)),
        "note": "val phase: fit on train window; test phase: refit same config on train+val (all data before Sep 1)",
    }, "val_tuning": {}, "test": {}, "small_matrix": {}}

    # ---- tune on val ------------------------------------------------------
    grids = {
        "itemcf": [dict(last_n=n, decay=d) for n in (20, 50, 200) for d in (1.0, 0.98)],
        "als": [dict(factors=f, alpha=al, regularization=0.05) for f in (64, 128) for al in (1.0, 10.0)],
        "pop_platform": [dict(days=d, column=c) for d in (1, 3, 7) for c in ("like_cnt", "complete_play_cnt")],
    }
    makers = {"itemcf": ItemCF, "als": ALS, "pop_platform": PlatformPopularity}
    best = {}
    for name, grid in grids.items():
        rows = []
        for params in grid:
            m = makers[name](**params).fit(a, val)
            r = brief(run(m, a, val))
            rows.append({"params": params, **r})
            print(name, params, r, flush=True)
        out["val_tuning"][name] = rows
        best[name] = max(rows, key=lambda r: r[SELECT])["params"]
    for days in (None, 1, 3, 7):
        m = Popularity(days).fit(a, val)
        out["val_tuning"].setdefault("pop", []).append({"window_days": days, **brief(run(m, a, val))})
    best_pop_recent = max((r for r in out["val_tuning"]["pop"] if r["window_days"]), key=lambda r: r[SELECT])["window_days"]
    out["selected"] = {**best, "pop_recent_days": best_pop_recent}

    # ---- test + small matrix with the selected configs ----------------------
    models = [Popularity(None), Popularity(best_pop_recent), PlatformPopularity(**best["pop_platform"]),
              ItemCF(**best["itemcf"]), ALS(**best["als"])]
    s_users, s_items, s_truths = small_matrix_protocol(a)
    out["protocol"]["small_users"], out["protocol"]["small_items"] = int(len(s_users)), int(len(s_items))
    for m in models:
        m.fit(a, test)
        per_user = run(m, a, test)
        save_per_user(f"stage1_test_{m.name}", per_user)
        out["test"][m.name] = summarize(per_user)
        sm = run(m, a, test, users=s_users, candidate_items=s_items, truths=s_truths, use_seen=False)
        save_per_user(f"stage1_small_{m.name}", sm)
        out["small_matrix"][m.name] = summarize(sm)
        print("TEST", m.name, out["test"][m.name]["ndcg@10"], "SMALL", out["small_matrix"][m.name]["ndcg@10"], flush=True)

    # ---- label sensitivity: does the model ranking survive a different threshold? ---
    out["label_sensitivity"] = {}
    for thr in (1.0, 1.5, 2.0, 3.0):
        ad = KuaiRecAdapter(positive_threshold=thr)
        ph = load_phase(ad, "test")
        row = {}
        for m in (Popularity(None), PlatformPopularity(**best["pop_platform"]), ItemCF(**best["itemcf"]), ALS(**best["als"])):
            m.fit(ad, ph)
            row[m.name] = brief(run(m, ad, ph))
        out["label_sensitivity"][str(thr)] = row
        print("thr", thr, row, flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    write_json("results/stage1_baselines.json", out)


if __name__ == "__main__":
    main()
