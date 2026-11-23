"""`make eval`: offline evaluation of a candidate pipeline configuration on the
test phase, appended to results/history.jsonl, with per-user arrays and
slice data saved for the regression gate and the release-guard agent.

A candidate = which two-tower model, which ranker, which recall quotas, and
(for fault injection) feature cutoff / stale shard overrides.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from echostream.core.metrics import evaluate_users
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import summarize
from echostream.offline.phase import ARTIFACTS, history_frame, load_phase
from echostream.rank.features import build_features, build_rank_assets, recent_events_by_user
from echostream.recall.assets import build_assets, load_two_tower
from echostream.recall.engine import RecallEngine
from echostream.serving.shard_server import shard_of

ROOT = Path(__file__).resolve().parents[2]
def history_path() -> Path:
    return Path(os.environ.get("ECHO_EVAL_HISTORY", ROOT / "results" / "history.jsonl"))
EVAL_DIR = ARTIFACTS / "eval"
N_SHARDS = 4


@dataclass
class Candidate:
    name: str
    two_tower: str = "main"
    ranker: str = str(ARTIFACTS / "ranker" / "main.txt")
    quotas: dict = field(default_factory=lambda: {k: v for k, v in json.load(
        open(ROOT / "results" / "stage3_multipath.json"))["selected_quotas"].items() if k != "name"})
    item_stats_cutoff: int | None = None       # None = the phase cutoff (fresh features)
    stale_shard: int | None = None             # this shard answers with an OLD model's vectors
    stale_shard_model: str = "val_main"
    notes: str = ""


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "unknown"


def evaluate_candidate(c: Candidate, users_limit: int | None = None) -> dict:
    t0 = time.time()
    a = KuaiRecAdapter()
    test = load_phase(a, "test")
    assets = build_assets(a, test, c.two_tower, quotas=c.quotas)
    if c.stale_shard is not None:
        old_vecs, _, _ = load_two_tower(c.stale_shard_model)
        mask = np.array([shard_of(i, N_SHARDS) == c.stale_shard for i in range(a.n_items)])
        assets.item_vecs = assets.item_vecs.copy()
        assets.item_vecs[mask] = old_vecs[mask]
    eng = RecallEngine(assets)
    hist_all = history_frame(a, "test", positives_only=False)
    cutoff = c.item_stats_cutoff or test.target_start
    ra = build_rank_assets(a, hist_all if cutoff == test.target_start else hist_all[hist_all.date < cutoff], cutoff)
    recent = recent_events_by_user(hist_all, a.n_users)
    model = lgb.Booster(model_file=c.ranker)
    users = test.eval_users[:users_limit] if users_limit else test.eval_users
    ranked, path_hits = [], {p: [] for p in ("two_tower", "itemcf", "pop_recent", "pop_platform", "content")}
    claimed = {p: 0 for p in path_hits}
    for u in users:
        hist = test.pos_history[u]
        uvec = eng.user_vector(int(u), hist)
        paths = eng.all_paths(int(u), hist, test.seen[u].indices, k=200, user_vec=uvec)
        cands = eng.merge(paths)
        truth = test.truth[u]
        for p in path_hits:
            path_hits[p].append(len({i for i, _ in paths.get(p, [])} & truth) / len(truth))
        for cd in cands:
            if cd.item in truth:
                claimed[cd.source] = claimed.get(cd.source, 0) + 1
        X = build_features(ra, eng, int(u), uvec, hist, recent[u], cands)
        s = model.predict(X, num_threads=8)
        order = np.lexsort((np.array([cd.item for cd in cands]), -s))
        ranked.append([cands[j].item for j in order])
    truths = test.truths_for(users)
    pu = evaluate_users(ranked, truths, (10, 50, 200), a.item_categories(), a.n_items)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{c.name}"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    item_cold = (assets.item_hist_count == 0)
    np.savez_compressed(EVAL_DIR / f"{run_id}.npz", users=users,
                        **{k: v for k, v in pu.items() if isinstance(v, np.ndarray)},
                        **{f"path_recall_{p}": np.array(v) for p, v in path_hits.items()},
                        cold_truth_share=np.array([np.mean([item_cold[i] for i in t]) for t in truths]),
                        top10=np.array([r[:10] + [-1] * (10 - len(r[:10])) for r in ranked]))
    summary = {k: v for k, v in summarize(pu).items() if k in ("ndcg@10", "recall@10", "recall@200", "hit@10",
                                                                "coverage@10", "diversity@10")}
    rec = {"run_id": run_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_sha(), "candidate": asdict(c),
           "n_users": int(len(users)), "metrics": summary,
           "path_recall@200": {p: round(float(np.mean(v)), 5) for p, v in path_hits.items()},
           "claimed_hits": claimed, "feature_cutoff": cutoff, "history_cutoff": test.target_start,
           "seconds": round(time.time() - t0, 1)}
    with open(history_path(), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_history() -> list[dict]:
    h = history_path()
    if not h.exists():
        return []
    return [json.loads(line) for line in h.read_text().splitlines() if line.strip()]


def per_user(run_id: str) -> dict[str, np.ndarray]:
    return dict(np.load(EVAL_DIR / f"{run_id}.npz"))
