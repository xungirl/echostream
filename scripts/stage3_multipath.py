"""Stage 3: multi-path recall. Per-path recall, quota tuning on val, merged recall on
test, per-path contribution, cold-start study (low-activity users, never-seen items).

    uv run python scripts/stage3_multipath.py  -> results/stage3_multipath.json
"""
import json
import time

import numpy as np

from echostream.core.metrics import bootstrap_ci
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import write_json
from echostream.offline.phase import ARTIFACTS, load_phase
from echostream.recall.assets import build_assets
from echostream.recall.engine import RecallEngine

QUOTA_GRID = {
    "tt_heavy":   {"two_tower": 100, "itemcf": 40, "pop_recent": 20, "pop_platform": 20, "content": 20},
    "balanced":   {"two_tower": 80, "itemcf": 40, "pop_recent": 30, "pop_platform": 30, "content": 20},
    "pop_heavy":  {"two_tower": 60, "itemcf": 30, "pop_recent": 50, "pop_platform": 40, "content": 20},
    "equal":      {"two_tower": 40, "itemcf": 40, "pop_recent": 40, "pop_platform": 40, "content": 40},
    "no_content": {"two_tower": 90, "itemcf": 40, "pop_recent": 35, "pop_platform": 35, "content": 0},
}
PATHS = ["two_tower", "itemcf", "pop_recent", "pop_platform", "content"]


def ensure_val_two_tower():
    """The val-phase two-tower (fit on the train window only) with the Stage-2 config."""
    d = ARTIFACTS / "two_tower" / "val_main"
    if d.exists():
        return
    from echostream.offline.train_two_tower import TTConfig, save, train  # torch only here
    cfg_d = json.loads((ARTIFACTS / "two_tower" / "main" / "config.json").read_text())
    cfg = TTConfig(**{k: v for k, v in cfg_d.items() if k in TTConfig.__dataclass_fields__})
    a = KuaiRecAdapter()
    r = train(a, load_phase(a, "val"), cfg, np.load(ARTIFACTS / "caption_emb.npy"))
    save(r, cfg, "val_main")


def collect_paths(engine, phase, users):
    out = []
    for u in users:
        hist = phase.pos_history[u]
        exclude = phase.seen[u].indices
        out.append(engine.all_paths(int(u), hist, exclude, k=200))
    return out


def recall_of(items, truth):
    return len(set(items) & truth) / len(truth)


def main():
    t0 = time.time()
    ensure_val_two_tower()
    a = KuaiRecAdapter()
    out = {"val_quota_tuning": {}, "test": {}}

    # ---- quota tuning on val ---------------------------------------------------
    val = load_phase(a, "val")
    eng = RecallEngine(build_assets(a, val, "val_main"))
    users = val.eval_users
    paths_val = collect_paths(eng, val, users)
    truths = val.truths_for(users)
    for name, q in QUOTA_GRID.items():
        eng.a.quotas = q
        r = [recall_of([c.item for c in eng.merge(p)], t) for p, t in zip(paths_val, truths)]
        out["val_quota_tuning"][name] = {"quotas": q, "merged_recall@200": round(float(np.mean(r)), 5)}
        print("val", name, out["val_quota_tuning"][name], flush=True)
    best = max(out["val_quota_tuning"], key=lambda n: out["val_quota_tuning"][n]["merged_recall@200"])
    out["selected_quotas"] = {"name": best, **QUOTA_GRID[best]}

    # ---- test --------------------------------------------------------------------
    test = load_phase(a, "test")
    eng = RecallEngine(build_assets(a, test, "main", quotas=QUOTA_GRID[best]))
    users = test.eval_users
    truths = test.truths_for(users)
    paths_test = collect_paths(eng, test, users)
    merged = [eng.merge(p) for p in paths_test]

    per_path = {}
    for name in PATHS:
        for k in (50, 100, 200):
            vals = np.array([recall_of([i for i, _ in p[name][:k]], t) for p, t in zip(paths_test, truths)])
            m, lo, hi = bootstrap_ci(vals)
            per_path.setdefault(name, {})[f"recall@{k}"] = {"mean": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)]}
    mr = np.array([recall_of([c.item for c in m], t) for m, t in zip(merged, truths)])
    m, lo, hi = bootstrap_ci(mr)
    out["test"]["per_path_alone"] = per_path
    out["test"]["merged_recall@200"] = {"mean": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)]}
    out["test"]["avg_merged_size"] = round(float(np.mean([len(x) for x in merged])), 1)

    # contribution: of the truth items found by the merged list, who claimed them / who found them uniquely
    claimed = {p: 0 for p in PATHS}
    unique = {p: 0 for p in PATHS}
    total_hits = 0
    for cand, t in zip(merged, truths):
        for c in cand:
            if c.item in t:
                total_hits += 1
                claimed[c.source] += 1
                if len(c.all_sources) == 1:
                    unique[c.source] += 1
    out["test"]["contribution"] = {p: {"share_of_hits_claimed": round(claimed[p] / total_hits, 4),
                                       "share_of_hits_unique": round(unique[p] / total_hits, 4)} for p in PATHS}
    out["test"]["total_hits"] = total_hits

    # leave-one-path-out: merged recall when a path's quota is redistributed
    loo = {}
    for name in PATHS:
        q = dict(QUOTA_GRID[best])
        q[name] = 0
        eng.a.quotas = q
        r = [recall_of([c.item for c in eng.merge(p)], t) for p, t in zip(paths_test, truths)]
        loo[name] = round(float(np.mean(r)), 5)
    eng.a.quotas = QUOTA_GRID[best]
    out["test"]["merged_recall@200_without_path"] = loo

    # ---- cold-start study ---------------------------------------------------
    hist_len = np.array([len(test.pos_history[u]) for u in users])
    buckets = {"0-4": (0, 5), "5-19": (5, 20), "20-99": (20, 100), "100+": (100, 10**9)}
    cold_users = {}
    for b, (lo_, hi_) in buckets.items():
        sel = np.where((hist_len >= lo_) & (hist_len < hi_))[0]
        row = {"n_users": int(len(sel))}
        if len(sel):
            row["merged_recall@200"] = round(float(np.mean(mr[sel])), 5)
            for name in PATHS:
                row[name] = round(float(np.mean([recall_of([i for i, _ in paths_test[j][name]], truths[j]) for j in sel])), 5)
        cold_users[b] = row
    out["test"]["by_user_history_positives"] = cold_users

    cold_item = eng.cold_mask
    found = {p: 0 for p in PATHS + ["merged"]}
    n_cold_truth = 0
    for p, cand, t in zip(paths_test, merged, truths):
        ct = {i for i in t if cold_item[i]}
        n_cold_truth += len(ct)
        for name in PATHS:
            found[name] += len(ct & {i for i, _ in p[name]})
        found["merged"] += len(ct & {c.item for c in cand})
    out["test"]["cold_items"] = {"n_cold_items_in_catalog": int(cold_item.sum()), "n_cold_truth_pairs": n_cold_truth,
                                 "recall_of_cold_truth": {k: round(v / max(n_cold_truth, 1), 5) for k, v in found.items()}}
    out["runtime_sec"] = round(time.time() - t0, 1)
    write_json("results/stage3_multipath.json", out)
    print(json.dumps(out["test"], indent=1)[:3000])


if __name__ == "__main__":
    main()
