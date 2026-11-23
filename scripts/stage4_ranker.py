"""Stage 4: LightGBM ranker over recall candidates.

Train on the RANK phase (recall models refit on everything before Aug 30,
features from before Aug 30, labels = new positives on Aug 30-31), evaluate on
the TEST phase (everything refit / recomputed with data before Sep 1).

History: the first version trained the ranker on the VAL phase (recall models
fit on the train window only, 3 weeks before the labels). On test it LOST to
recall-only ordering, because the two-tower score it learned from was far
weaker in that phase than at test time (train/serve skew). That run is kept
in results/stage4_ranker_v1_valphase.json; `--phase val` reproduces it.

    uv run python scripts/stage4_ranker.py  -> results/stage4_ranker.json, artifacts/ranker/
"""
import json
import time

import lightgbm as lgb
import numpy as np

from echostream.core.hashing import stable_hash
from echostream.core.metrics import auc, bootstrap_ci, evaluate_users, gauc, paired_bootstrap_diff
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import load_per_user, save_per_user, summarize, write_json
from echostream.offline.phase import ARTIFACTS, load_phase
from echostream.rank.features import FEATURE_NAMES, GROUPS
from echostream.rank.offline import candidates_and_features, rerank

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "num_threads": 8, "seed": 0}


def train(X, y, groups_u, valid_mask, feature_idx, params, rounds=600):
    dtr = lgb.Dataset(X[~valid_mask][:, feature_idx], y[~valid_mask])
    dva = lgb.Dataset(X[valid_mask][:, feature_idx], y[valid_mask])
    return lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def ensure_two_tower(tag: str, phase_name: str):
    """Refit the selected two-tower config on the given phase's history (torch only here)."""
    if (ARTIFACTS / "two_tower" / tag).exists():
        return
    # separate process: torch and LightGBM each bundle libomp and crash when loaded together
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "echostream.offline.train_two_tower", "--phase", phase_name,
                    "--tag", tag], check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="rank", choices=["rank", "val"])
    args = ap.parse_args()
    train_phase, tt_tag = (args.phase, "rank_main") if args.phase == "rank" else ("val", "val_main")
    out_json = "results/stage4_ranker.json" if args.phase == "rank" else "results/stage4_ranker_v1_valphase.json"
    ensure_two_tower(tt_tag, train_phase)
    t0 = time.time()
    a = KuaiRecAdapter()
    quotas = {k: v for k, v in json.loads(open("results/stage3_multipath.json").read())["selected_quotas"].items()
              if k != "name"}
    val, test = load_phase(a, train_phase), load_phase(a, "test")
    out = {"params": PARAMS, "features": FEATURE_NAMES, "groups": GROUPS, "train_phase": train_phase,
           "train_target_dates": list(a.split_spec().rank if train_phase == "rank" else a.split_spec().val)}

    vu, vrows, vX, vy, vcands, _, _ = candidates_and_features(a, val, tt_tag, quotas)
    print("val rows", vX.shape, "pos rate", vy.mean(), f"{time.time()-t0:.0f}s", flush=True)
    valid_mask = np.array([stable_hash(f"rk:{u}") % 5 == 0 for u in vrows])   # 20% of val users for early stopping
    all_idx = np.arange(len(FEATURE_NAMES))
    model = train(vX, vy, vrows, valid_mask, all_idx, PARAMS)
    out["best_iteration"] = model.best_iteration

    tu, trows, tX, ty, tcands, _, _ = candidates_and_features(a, test, "main", quotas)
    print("test rows", tX.shape, "pos rate", ty.mean(), f"{time.time()-t0:.0f}s", flush=True)
    out["rows"] = {"val": int(len(vy)), "val_pos_rate": round(float(vy.mean()), 5),
                   "test": int(len(ty)), "test_pos_rate": round(float(ty.mean()), 5),
                   "test_users_with_candidates": int(len(tu))}
    truths = test.truths_for(tu)

    def eval_scores(scores):
        ranked = rerank(tcands, scores)
        pu = evaluate_users(ranked, truths, (10, 50, 100, 200), a.item_categories(), a.n_items)
        return pu, {"auc": round(auc(ty, scores), 5), "gauc": round(gauc(trows, ty, scores), 5)}

    rank_scores = model.predict(tX[:, all_idx], num_iteration=model.best_iteration)
    pu_rank, cls = eval_scores(rank_scores)
    save_per_user("stage4_test_recall_rank", pu_rank)
    out["test_recall_rank"] = {**summarize(pu_rank), **cls}

    # recall-only orderings: (a) merge order, (b) by two-tower score
    merge_order = np.concatenate([-np.arange(len(c), dtype=np.float32) for c in tcands])
    pu_merge, cls_m = eval_scores(merge_order)
    save_per_user("stage4_test_recall_only", pu_merge)
    out["test_recall_only_merge_order"] = {**summarize(pu_merge), **cls_m}
    tt_col = FEATURE_NAMES.index("score_two_tower")
    pu_tt, cls_tt = eval_scores(tX[:, tt_col])
    out["test_recall_only_tt_order"] = {**summarize(pu_tt), **cls_tt}

    # paired: rank vs recall-only, rank vs best Stage-1 baseline (same users only)
    def paired(base_pu, users_base, metric="ndcg@10"):
        idx = {u: j for j, u in enumerate(users_base)}
        common = [j for j, u in enumerate(tu) if u in idx]
        b = base_pu[metric][[idx[tu[j]] for j in common]]
        r = pu_rank[metric][common]
        d, lo, hi = paired_bootstrap_diff(b, r)
        return {"delta": round(d, 5), "ci95": [round(lo, 5), round(hi, 5)], "n_users": len(common)}
    out["paired"] = {"rank_vs_recall_only_merge": paired(pu_merge, tu),
                     "rank_vs_pop_recent7d": paired(load_per_user("stage1_test_pop_recent7d"), test.eval_users),
                     "rank_vs_two_tower_alone": paired(load_per_user("stage2_test_" + json.loads(
                         open("results/stage2_two_tower.json").read())["selected_variant"]), test.eval_users)}

    # feature-group ablation (retrain without the group on val, evaluate on test)
    out["ablation"] = {}
    for g, cols in GROUPS.items():
        idx = np.array([j for j, f in enumerate(FEATURE_NAMES) if f not in cols])
        m = train(vX, vy, vrows, valid_mask, idx, PARAMS)
        s = m.predict(tX[:, idx], num_iteration=m.best_iteration)
        pu, c = eval_scores(s)
        d, lo, hi = paired_bootstrap_diff(pu_rank["ndcg@10"], pu["ndcg@10"])
        out["ablation"][f"without_{g}"] = {"ndcg@10": round(float(pu["ndcg@10"].mean()), 5), **c,
                                           "delta_vs_full": round(d, 5), "ci95": [round(lo, 5), round(hi, 5)]}
        print("ablation", g, out["ablation"][f"without_{g}"], flush=True)

    imp = model.feature_importance("gain")
    out["feature_importance_gain_top15"] = sorted(
        [(f, round(float(v), 1)) for f, v in zip(FEATURE_NAMES, imp)], key=lambda x: -x[1])[:15]
    d = ARTIFACTS / "ranker"
    d.mkdir(parents=True, exist_ok=True)
    name = "main" if args.phase == "rank" else "v1_valphase"
    model.save_model(str(d / f"{name}.txt"), num_iteration=model.best_iteration)
    (d / "features.json").write_text(json.dumps(FEATURE_NAMES))
    out["runtime_sec"] = round(time.time() - t0, 1)
    write_json(out_json, out)
    print(json.dumps({k: out[k] for k in ("test_recall_rank", "test_recall_only_merge_order", "paired")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
