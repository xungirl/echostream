"""Stage 2a: two-tower training, temperature tuning (val only), logQ / mixed-negative
ablations, seed variance, test + small-matrix evaluation with CIs.

    uv run python scripts/stage2_two_tower.py  -> results/stage2_two_tower.json
"""
import time

import numpy as np

from echostream.core.metrics import paired_bootstrap_diff
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import (evaluate_ranked, load_per_user, rank_topk, save_per_user,
                                         small_matrix_protocol, summarize, write_json)
from echostream.offline.phase import ARTIFACTS, load_phase
from echostream.offline.train_two_tower import TTConfig, save, train, user_vectors

SELECT = "ndcg@10"


def evaluate(result, a, phase, users, truths, candidate_items=None, use_seen=True):
    uv = user_vectors(result, phase, users)
    iv = result["item_vecs"]
    lookup = {u: i for i, u in enumerate(users)}
    score = lambda ub: uv[[lookup[u] for u in ub]] @ iv.T  # noqa: E731
    ranked = rank_topk(score, users, phase if use_seen else None, 200, candidate_items)
    return evaluate_ranked(ranked, truths, a, n_items=len(candidate_items) if candidate_items is not None else None)


def brief(pu):
    return {k: round(float(np.mean(pu[k])), 5) for k in ("ndcg@10", "recall@50", "recall@200")}


def main():
    t0 = time.time()
    a = KuaiRecAdapter()
    cap = np.load(ARTIFACTS / "caption_emb.npy")
    val, test = load_phase(a, "val"), load_phase(a, "test")
    out = {"val_temperature": [], "val_ablation": [], "test": {}, "small_matrix": {}, "seeds": []}

    def run_val(cfg):
        r = train(a, val, cfg, cap)
        return brief(evaluate(r, a, val, val.eval_users, val.truths_for(val.eval_users))), r

    for temp in (0.05, 0.1, 0.2):
        cfg = TTConfig(temperature=temp)
        res, r = run_val(cfg)
        out["val_temperature"].append({"temperature": temp, **res, "train_sec": round(r["train_sec"], 1)})
        print("val temp", temp, res, flush=True)
    best_t = max(out["val_temperature"], key=lambda r: r[SELECT])["temperature"]
    out["selected_temperature"] = best_t

    variants = {
        "tt_logq_mixed": TTConfig(temperature=best_t, logq=True, mixed_negatives=1024),
        "tt_logq": TTConfig(temperature=best_t, logq=True, mixed_negatives=0),
        "tt_mixed": TTConfig(temperature=best_t, logq=False, mixed_negatives=1024),
        "tt_plain": TTConfig(temperature=best_t, logq=False, mixed_negatives=0),
    }
    for name, cfg in variants.items():
        res, _ = run_val(cfg)
        out["val_ablation"].append({"variant": name, **res})
        print("val", name, res, flush=True)
    main_name = max(out["val_ablation"], key=lambda r: r[SELECT])["variant"]
    out["selected_variant"] = main_name

    s_users, s_items, s_truths = small_matrix_protocol(a)
    for name, cfg in variants.items():
        r = train(a, test, cfg, cap)
        pu = evaluate(r, a, test, test.eval_users, test.truths_for(test.eval_users))
        save_per_user(f"stage2_test_{name}", pu)
        out["test"][name] = summarize(pu)
        sm = evaluate(r, a, test, s_users, s_truths, s_items, use_seen=False)
        save_per_user(f"stage2_small_{name}", sm)
        out["small_matrix"][name] = summarize(sm)
        if name == main_name:
            d = save(r, cfg, "main")
            np.save(d / "user_vecs_test.npy", user_vectors(r, test, np.arange(a.n_users)))
            out["main_model_artifact"] = "artifacts/two_tower/main"
        print("test", name, brief(pu), "small", brief(sm), flush=True)

    # seed-to-seed noise of the selected config (used later by the regression gate)
    for seed in (1, 2):
        cfg = variants[main_name]
        cfg = TTConfig(**{**cfg.__dict__, "seed": seed})
        r = train(a, test, cfg, cap)
        pu = evaluate(r, a, test, test.eval_users, test.truths_for(test.eval_users))
        out["seeds"].append({"seed": seed, **brief(pu)})
    out["seeds"].insert(0, {"seed": 0, **{k: v["mean"] for k, v in out["test"][main_name].items()
                                           if k in ("ndcg@10", "recall@50", "recall@200")}})
    for k in ("ndcg@10", "recall@50", "recall@200"):
        vals = [s[k] for s in out["seeds"]]
        out.setdefault("seed_std", {})[k] = round(float(np.std(vals, ddof=1)), 5)

    # paired comparison vs the baselines on the same test users
    base = {n: load_per_user(f"stage1_test_{n}") for n in ("pop_recent7d", "pop_platform1d", "itemcf", "als")}
    main_pu = load_per_user(f"stage2_test_{main_name}")
    out["paired_vs_baselines"] = {}
    for n, b in base.items():
        out["paired_vs_baselines"][n] = {
            m: dict(zip(("delta", "ci_low", "ci_high"), [round(x, 5) for x in paired_bootstrap_diff(b[m], main_pu[m])]))
            for m in ("ndcg@10", "recall@200")}
    for n in ("tt_logq", "tt_mixed", "tt_plain"):
        o = load_per_user(f"stage2_test_{n}")
        out["paired_vs_baselines"][f"ablation_{n}"] = {
            m: dict(zip(("delta", "ci_low", "ci_high"), [round(x, 5) for x in paired_bootstrap_diff(o[m], main_pu[m])]))
            for m in ("ndcg@10", "recall@200")}
    out["runtime_sec"] = round(time.time() - t0, 1)
    write_json("results/stage2_two_tower.json", out)


if __name__ == "__main__":
    main()
