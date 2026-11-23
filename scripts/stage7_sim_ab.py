"""Stage 7: pre-registered simulated A/B (see PREREG.md, committed before this ran).

    uv run python scripts/stage7_sim_ab.py -> results/stage7_ab.json
"""
import json
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
from scipy import stats

from echostream.core.hashing import assign_variant
from echostream.core.metrics import bootstrap_ci, intra_list_diversity, ndcg_at_k
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import write_json
from echostream.offline.phase import ARTIFACTS, history_frame, load_phase
from echostream.rank.features import build_rank_assets, recent_events_by_user
from echostream.recall.assets import build_assets
from echostream.recall.engine import RecallEngine
from echostream.sim.simulator import (PopularityPolicy, RankPolicy, StaticScorePolicy, TwoTowerPolicy, UserState,
                                      run_session)

ALLOC = [("V1_popularity", 20), ("V2_als", 20), ("V3_two_tower", 20), ("V4_recall_rank", 20), ("V5_recall_rank_mmr", 20)]
COMPARISONS = [("C1", "V4_recall_rank", "V3_two_tower"), ("C2", "V5_recall_rank_mmr", "V4_recall_rank"),
               ("C3", "V3_two_tower", "V2_als"), ("C4", "V2_als", "V1_popularity")]
ALS_SCORES = ARTIFACTS / "sim_als_scores.npz"

ALS_SNIPPET = """
import json, numpy as np
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.baselines import ALS
from echostream.offline.phase import load_phase, ARTIFACTS
a = KuaiRecAdapter(); ph = load_phase(a, "test")
cfg = json.load(open("results/stage1_baselines.json"))["selected"]["als"]
m = ALS(**cfg).fit(a, ph)
users = np.sort(a.fully_observed().user_id.unique())
np.savez(ARTIFACTS / "sim_als_scores.npz", users=users, scores=m.score(users))
"""


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    t0 = time.time()
    if not ALS_SCORES.exists():   # implicit's OpenMP and LightGBM's must not share a process
        subprocess.run([sys.executable, "-c", ALS_SNIPPET], check=True)
    a = KuaiRecAdapter()
    test = load_phase(a, "test")
    quotas = {k: v for k, v in json.load(open("results/stage3_multipath.json"))["selected_quotas"].items() if k != "name"}
    eng = RecallEngine(build_assets(a, test, "main", quotas=quotas))
    hist_all = history_frame(a, "test", positives_only=False)
    ra = build_rank_assets(a, hist_all, test.target_start)
    recent = recent_events_by_user(hist_all, a.n_users)
    model = lgb.Booster(model_file=str(ARTIFACTS / "ranker" / "main.txt"))
    lam = json.load(open("results/stage6_mmr.json"))["selected_lambda"]
    als = np.load(ALS_SCORES)
    policies = [PopularityPolicy(eng), StaticScorePolicy(dict(zip(als["users"].tolist(), als["scores"])), a.n_items),
                TwoTowerPolicy(eng), RankPolicy(eng, ra, model), RankPolicy(eng, ra, model, mmr_lambda=lam)]

    small = a.fully_observed()
    catalog = np.sort(small.item_id.unique())
    not_catalog = np.ones(a.n_items, bool)
    not_catalog[catalog] = False
    responses = {u: dict(zip(g.item_id.tolist(), g.value.tolist())) for u, g in small.groupby("user_id")}
    users = np.array(sorted(responses))
    cats = a.item_categories()

    within = {p.name: {"pos_rate": [], "mean_wr": [], "diversity": [], "ndcg10_slate1": []} for p in policies}
    coverage = {p.name: set() for p in policies}
    for p in policies:
        for u in users:
            st = UserState(int(u), test.pos_history[u].copy(), list(recent[u]))
            shown = run_session(p, st, responses[u], not_catalog, test.seen[u].indices)
            wrs = np.array([w for _, w in shown])
            within[p.name]["pos_rate"].append(float((wrs > 2.0).mean()) if len(wrs) else 0.0)
            within[p.name]["mean_wr"].append(float(np.minimum(wrs, 5).mean()) if len(wrs) else 0.0)
            within[p.name]["diversity"].append(intra_list_diversity([cats.get(i, frozenset()) for i, _ in shown]))
            truth = {i for i, w in responses[u].items() if w > 2.0}
            within[p.name]["ndcg10_slate1"].append(ndcg_at_k([i for i, _ in shown[:10]], truth, 10))
            coverage[p.name].update(i for i, _ in shown)
        print(p.name, round(float(np.mean(within[p.name]["pos_rate"])), 5), f"{time.time()-t0:.0f}s", flush=True)
    for k in within:
        for m in within[k]:
            within[k][m] = np.array(within[k][m])

    # ---- primary: between-subjects A/B by consistent-hash bucketing --------------------
    arm = {u: assign_variant(int(u), "sim-ab-2026", ALLOC) for u in users}
    idx = {name: np.array([j for j, u in enumerate(users) if arm[u] == name]) for name, _ in ALLOC}
    out = {"population": int(len(users)), "catalog": int(len(catalog)), "mmr_lambda": lam, "arms": {}, "primary": {},
           "within_subjects": {}, "offline_vs_online": {}, "deviations": []}
    for name, _ in ALLOC:
        v = within[name]["pos_rate"][idx[name]]
        m, lo, hi = bootstrap_ci(v, n_boot=2000, seed=1)
        out["arms"][name] = {"n_users": int(len(v)), "pos_rate": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)],
                             "mean_wr": round(float(within[name]["mean_wr"][idx[name]].mean()), 4),
                             "diversity": round(float(within[name]["diversity"][idx[name]].mean()), 4)}
    pvals = []
    for cid, x, y in COMPARISONS:
        a_, b_ = within[x]["pos_rate"][idx[x]], within[y]["pos_rate"][idx[y]]
        rng = np.random.default_rng(2)
        diffs = [rng.choice(a_, len(a_)).mean() - rng.choice(b_, len(b_)).mean() for _ in range(2000)]
        p = stats.ttest_ind(a_, b_, equal_var=False).pvalue
        pvals.append(p)
        out["primary"][cid] = {"compare": f"{x} - {y}", "diff": round(float(a_.mean() - b_.mean()), 5),
                               "ci95": [round(float(np.percentile(diffs, 2.5)), 5), round(float(np.percentile(diffs, 97.5)), 5)],
                               "welch_p": float(p)}
    for (cid, _, _), padj in zip(COMPARISONS, holm(np.array(pvals))):
        out["primary"][cid]["holm_p"] = float(padj)
        out["primary"][cid]["significant_at_0.05"] = bool(padj < 0.05)

    # ---- secondary: within-subjects (every variant on every user) --------------------
    for name in within:
        m, lo, hi = bootstrap_ci(within[name]["pos_rate"], n_boot=2000, seed=1)
        out["within_subjects"][name] = {"pos_rate": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)],
                                        "mean_wr": round(float(within[name]["mean_wr"].mean()), 4),
                                        "diversity": round(float(within[name]["diversity"].mean()), 4),
                                        "coverage": round(len(coverage[name]) / len(catalog), 4),
                                        "small_matrix_ndcg10_slate1": round(float(within[name]["ndcg10_slate1"].mean()), 5)}
    for cid, x, y in COMPARISONS:
        m, lo, hi = bootstrap_ci(within[x]["pos_rate"] - within[y]["pos_rate"], n_boot=2000, seed=3)
        out["within_subjects"][f"{cid}_paired"] = {"compare": f"{x} - {y}", "diff": round(m, 5),
                                                   "ci95": [round(lo, 5), round(hi, 5)]}

    # ---- offline vs online ranking -------------------------------------------------
    s1 = json.load(open("results/stage1_baselines.json"))["test"]
    s2 = json.load(open("results/stage2_two_tower.json"))
    s4 = json.load(open("results/stage4_ranker.json"))
    s6 = json.load(open("results/stage6_mmr.json"))
    offline = {"V1_popularity": s1["pop_recent7d"]["ndcg@10"]["mean"], "V2_als": s1["als"]["ndcg@10"]["mean"],
               "V3_two_tower": s2["test"][s2["selected_variant"]]["ndcg@10"]["mean"],
               "V4_recall_rank": s4["test_recall_rank"]["ndcg@10"]["mean"],
               "V5_recall_rank_mmr": s6["lambdas"][str(lam)]["ndcg@10"]["mean"]}
    names = [n for n, _ in ALLOC]
    online = [out["within_subjects"][n]["pos_rate"] for n in names]
    for label, off in (("big_matrix_test_ndcg10", [offline[n] for n in names]),
                       ("small_matrix_slate1_ndcg10", [out["within_subjects"][n]["small_matrix_ndcg10_slate1"] for n in names])):
        tau = stats.kendalltau(off, online).statistic
        rho = stats.spearmanr(off, online).statistic
        dis = [f"{names[i]} vs {names[j]}" for i in range(5) for j in range(i + 1, 5)
               if np.sign(off[i] - off[j]) != np.sign(online[i] - online[j])]
        out["offline_vs_online"][label] = {"offline": dict(zip(names, off)), "online_pos_rate": dict(zip(names, online)),
                                           "kendall_tau": round(float(tau), 4), "spearman_rho": round(float(rho), 4),
                                           "disagreeing_pairs": dis}
    out["runtime_sec"] = round(time.time() - t0, 1)
    write_json("results/stage7_ab.json", out)
    print(json.dumps({k: out[k] for k in ("arms", "primary", "offline_vs_online")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
