"""Tools the release-guard agent can call. Each returns a small JSON-able dict
(the agent's context window is precious, so tools summarize; they never dump arrays)."""
from __future__ import annotations

import numpy as np

from echostream.agent.eval_run import load_history, per_user
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.phase import load_phase
from echostream.serving.shard_server import shard_of

_cache: dict = {}


def _run(run_id: str) -> dict:
    for r in load_history():
        if r["run_id"] == run_id:
            return r
    raise KeyError(run_id)


def _phase():
    if "phase" not in _cache:
        a = KuaiRecAdapter()
        _cache["adapter"], _cache["phase"] = a, load_phase(a, "test")
    return _cache["adapter"], _cache["phase"]


def diff_runs(run_id: str, base_run_id: str) -> dict:
    """Metric deltas and configuration differences between two eval runs."""
    r, b = _run(run_id), _run(base_run_id)
    metrics = {m: {"base": b["metrics"][m]["mean"], "candidate": r["metrics"][m]["mean"],
                   "delta": round(r["metrics"][m]["mean"] - b["metrics"][m]["mean"], 5)} for m in r["metrics"]}
    cfg = {k: {"base": b["candidate"].get(k), "candidate": v} for k, v in r["candidate"].items()
           if k not in ("name", "notes") and b["candidate"].get(k) != v}
    return {"metrics": metrics, "config_changes": cfg}


def slice_metrics(run_id: str, base_run_id: str, by: str) -> dict:
    """NDCG@10 per slice (candidate vs base). by: activity_bucket | cold_items | category."""
    a, ph = _phase()
    r, b = per_user(run_id), per_user(base_run_id)
    users = r["users"]
    if by == "activity_bucket":
        n = np.array([len(ph.pos_history[u]) for u in users])
        groups = {"0-19 positives": n < 20, "20-99": (n >= 20) & (n < 100), "100+": n >= 100}
    elif by == "cold_items":
        share = r["cold_truth_share"]
        groups = {"truth mostly cold items (>50%)": share > 0.5, "truth mostly warm items": share <= 0.5}
    elif by == "category":
        cats = a.item_categories()
        top = {}
        for j, u in enumerate(users):
            cs = [c for i in ph.truth[u] for c in cats.get(i, ())]
            top.setdefault(max(set(cs), key=cs.count) if cs else -1, []).append(j)
        big = sorted(top.items(), key=lambda kv: -len(kv[1]))[:6]
        groups = {f"category {c}": np.isin(np.arange(len(users)), idx) for c, idx in big}
    else:
        return {"error": f"unknown slice {by}"}
    out = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        out[name] = {"users": int(mask.sum()), "base_ndcg@10": round(float(b["ndcg@10"][mask].mean()), 5),
                     "candidate_ndcg@10": round(float(r["ndcg@10"][mask].mean()), 5),
                     "delta": round(float(r["ndcg@10"][mask].mean() - b["ndcg@10"][mask].mean()), 5)}
    worst = min(out, key=lambda k: out[k]["delta"]) if out else None
    return {"by": by, "slices": out, "worst_slice": worst}


def path_contribution(run_id: str, base_run_id: str) -> dict:
    """Recall@200 of each recall path alone, candidate vs base."""
    r, b = _run(run_id), _run(base_run_id)
    out = {p: {"base": b["path_recall@200"][p], "candidate": r["path_recall@200"][p],
               "delta": round(r["path_recall@200"][p] - b["path_recall@200"][p], 5),
               "claimed_hits_base": b["claimed_hits"].get(p, 0), "claimed_hits_candidate": r["claimed_hits"].get(p, 0)}
           for p in r["path_recall@200"]}
    worst = min(out, key=lambda p: min(out[p]["delta"], (out[p]["claimed_hits_candidate"] - out[p]["claimed_hits_base"])
                                       / max(out[p]["claimed_hits_base"], 1)))
    return {"paths": out, "most_degraded_path": worst}


def data_freshness(run_id: str) -> dict:
    """Which data cutoffs the candidate's features were computed with."""
    r = _run(run_id)
    lag_days = (np.datetime64(f"{str(r['history_cutoff'])[:4]}-{str(r['history_cutoff'])[4:6]}-{str(r['history_cutoff'])[6:]}")
                - np.datetime64(f"{str(r['feature_cutoff'])[:4]}-{str(r['feature_cutoff'])[4:6]}-{str(r['feature_cutoff'])[6:]}"))
    lag = int(lag_days.astype(int))
    return {"history_cutoff": r["history_cutoff"], "item_feature_cutoff": r["feature_cutoff"], "feature_lag_days": lag,
            "stale": lag > 1}


def shard_health(run_id: str, base_run_id: str, n_shards: int = 4) -> dict:
    """Two-tower contribution per index shard: did items of one shard stop showing up?"""
    r, b = per_user(run_id), per_user(base_run_id)
    a, ph = _phase()
    def share(top10):
        cnt = np.zeros(n_shards)
        for row in top10:
            for i in row:
                if i >= 0:
                    cnt[shard_of(int(i), n_shards)] += 1
        return cnt / max(cnt.sum(), 1)
    sr, sb = share(r["top10"]), share(b["top10"])
    per = {f"shard{s}": {"base_share_of_top10": round(float(sb[s]), 4), "candidate_share_of_top10": round(float(sr[s]), 4),
                         "relative_change": round(float((sr[s] - sb[s]) / max(sb[s], 1e-9)), 4)} for s in range(n_shards)}
    worst = max(per, key=lambda k: abs(per[k]["relative_change"]))
    return {"shards": per, "suspect_shard": worst if abs(per[worst]["relative_change"]) > 0.25 else None,
            "index_versions": _run(run_id)["candidate"].get("stale_shard") is not None and
            {f"shard{_run(run_id)['candidate']['stale_shard']}": _run(run_id)["candidate"]["stale_shard_model"]} or "all shards on candidate version"}


TOOLS = {
    "diff_runs": (diff_runs, "Metric deltas and config differences vs the last good run", ["run_id", "base_run_id"]),
    "slice_metrics": (slice_metrics, "NDCG@10 per slice; by = activity_bucket | cold_items | category",
                      ["run_id", "base_run_id", "by"]),
    "path_contribution": (path_contribution, "Recall@200 per recall path, candidate vs base", ["run_id", "base_run_id"]),
    "data_freshness": (data_freshness, "Feature and history cutoff dates of the candidate run", ["run_id"]),
    "shard_health": (shard_health, "Per-shard share of two-tower results and index versions", ["run_id", "base_run_id"]),
}


def openai_tool_specs() -> list[dict]:
    specs = []
    for name, (_, desc, params) in TOOLS.items():
        specs.append({"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": {p: {"type": "string"} for p in params}, "required": params}}})
    specs.append({"type": "function", "function": {"name": "write_report", "description":
                  "Finish: write the incident report (markdown) with root-cause hypothesis and evidence",
                  "parameters": {"type": "object", "properties": {"markdown": {"type": "string"},
                                                                  "root_cause": {"type": "string"}},
                                 "required": ["markdown", "root_cause"]}}})
    return specs


def call(name: str, args: dict) -> dict:
    fn = TOOLS[name][0]
    return fn(**args)
