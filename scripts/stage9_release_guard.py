"""Stage 9: regression gate + release-guard agent evaluation.

Builds 5 REAL regressions and 3 harmless changes, evaluates each with `make eval`
(appends results/history.jsonl), gates each against the baseline run, runs the
agent (FakeProvider) on every blocked candidate, and scores gate precision/recall
and whether the agent named the right root cause.

    uv run python scripts/stage9_release_guard.py -> results/stage9_release_guard.json
"""
import json
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np

from echostream.agent.eval_run import Candidate, evaluate_candidate
from echostream.agent.gate_check import gate
from echostream.agent.loop import run_agent
from echostream.agent.providers import FakeProvider
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import write_json
from echostream.offline.phase import ARTIFACTS, load_phase
from echostream.rank.offline import candidates_and_features

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "num_threads": 8, "seed": 0}
QUOTAS = {k: v for k, v in json.load(open("results/stage3_multipath.json"))["selected_quotas"].items() if k != "name"}


def ensure_tt(tag, *sets):
    if not (ARTIFACTS / "two_tower" / tag).exists():
        subprocess.run([sys.executable, "-m", "echostream.offline.train_two_tower", "--phase", "test", "--tag", tag,
                        "--set", *sets], check=True)


def ensure_ranker(name, label_threshold=2.0, seed=0):
    path = ARTIFACTS / "ranker" / f"{name}.txt"
    if path.exists():
        return str(path)
    a = KuaiRecAdapter(positive_threshold=label_threshold)
    ph = load_phase(a, "rank")
    users, rows, X, y, _, _, _ = candidates_and_features(a, ph, "rank_main", QUOTAS)
    from echostream.core.hashing import stable_hash
    valid = np.array([stable_hash(f"rk:{u}") % 5 == 0 for u in rows])
    m = lgb.train({**PARAMS, "seed": seed, "bagging_seed": seed + 3}, lgb.Dataset(X[~valid], y[~valid]), 600,
                  valid_sets=[lgb.Dataset(X[valid], y[valid])], callbacks=[lgb.early_stopping(50, verbose=False)])
    m.save_model(str(path), num_iteration=m.best_iteration)
    return str(path)


def main():
    t0 = time.time()
    ensure_tt("main_nologq", "logq=false")
    ensure_tt("main_seed1", "seed=1")
    label_bug = ensure_ranker("label_bug_thr0.5", label_threshold=0.5)
    reseed = ensure_ranker("reseed1", seed=1)

    cases = [
        ("baseline", None, Candidate("baseline", notes="v1 pipeline")),
        ("regression", "retrieval_model_regression", Candidate("logq_removed", two_tower="main_nologq",
                                                               notes="two-tower retrained without logQ correction")),
        ("regression", "stale_features", Candidate("stale_item_features", item_stats_cutoff=20200815,
                                                   notes="feature job stuck: item/user stats frozen at Aug 15")),
        ("regression", "dropped_recall_path", Candidate("dropped_two_tower_path", quotas={**QUOTAS, "two_tower": 0},
                                                        notes="config push dropped the two-tower recall path")),
        ("regression", "ranker_regression", Candidate("label_threshold_bug", ranker=label_bug,
                                                      notes="ranker trained with positive = watch_ratio > 0.5")),
        ("regression", "stale_shard_index", Candidate("stale_shard2_index", stale_shard=2,
                                                      notes="shard 2 kept serving last month's item vectors")),
        ("harmless", None, Candidate("two_tower_seed1", two_tower="main_seed1", notes="retrain, different seed")),
        ("harmless", None, Candidate("ranker_reseed", ranker=reseed, notes="ranker retrained, different bagging seed")),
        ("harmless", None, Candidate("quota_tweak", quotas={**QUOTAS, "itemcf": 35, "pop_recent": 40},
                                     notes="small recall quota change")),
    ]
    runs = {}
    for kind, expected, cand in cases:
        rec = evaluate_candidate(cand)
        runs[cand.name] = (kind, expected, rec)
        print(cand.name, rec["metrics"]["ndcg@10"], rec["metrics"]["recall@200"], f"{time.time()-t0:.0f}s", flush=True)

    base_id = runs["baseline"][2]["run_id"]
    rows = []
    for name, (kind, expected, rec) in runs.items():
        if name == "baseline":
            continue
        g = gate(base_id, rec["run_id"])
        row = {"candidate": name, "kind": kind, "expected_root_cause": expected, "blocked": not g["promote"],
               "gate": g, "metrics": rec["metrics"]}
        if not g["promote"]:
            res = run_agent(FakeProvider(), rec["run_id"], base_id, g)
            row["agent"] = res
            row["agent_correct"] = res["root_cause"] == expected
        rows.append(row)
        print(name, "blocked" if row["blocked"] else "passed", row.get("agent", {}).get("root_cause"), flush=True)

    reg = [r for r in rows if r["kind"] == "regression"]
    harm = [r for r in rows if r["kind"] == "harmless"]
    tp = sum(r["blocked"] for r in reg)
    fp = sum(r["blocked"] for r in harm)
    out = {"baseline_run": base_id, "rows": rows,
           "gate": {"regressions": len(reg), "harmless": len(harm), "true_positives": tp, "false_positives": fp,
                    "recall": round(tp / len(reg), 3), "precision": round(tp / max(tp + fp, 1), 3)},
           "agent": {"diagnosed": sum(1 for r in reg if r.get("agent")),
                     "correct_root_cause": sum(1 for r in reg if r.get("agent_correct"))},
           "runtime_sec": round(time.time() - t0, 1)}
    write_json("results/stage9_release_guard.json", out)
    print(json.dumps({"gate": out["gate"], "agent": out["agent"]}, indent=1))


if __name__ == "__main__":
    main()
