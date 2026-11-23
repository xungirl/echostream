"""LLM providers for the release-guard agent.

FakeProvider: deterministic, rule-based "LLM" used in tests/CI. It reads the
tool results in the conversation and decides the next call, so the tool trace
is real even though no model is involved.

DashScopeProvider: real model (OpenAI-compatible endpoint, model qwen3.8-flash),
key from DASHSCOPE_API_KEY at runtime only, every call charged to a token-based
cost ledger with a hard cap (default ¥2). Prices are env-configurable; the
defaults are deliberately pessimistic placeholders so the cap trips early, not
late -- set ECHO_PRICE_IN_PER_1K / ECHO_PRICE_OUT_PER_1K from DashScope's price page.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostLedger:
    cap_yuan: float = 2.0
    price_in_per_1k: float = float(os.environ.get("ECHO_PRICE_IN_PER_1K", "0.01"))
    price_out_per_1k: float = float(os.environ.get("ECHO_PRICE_OUT_PER_1K", "0.03"))
    path: Path = ROOT / "results" / "agent_cost_ledger.jsonl"
    spent: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path.exists():   # the cap is cumulative across runs
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.spent += json.loads(line)["cost_yuan"]

    def check(self, est_in_tokens: int, max_out_tokens: int) -> None:
        worst = est_in_tokens / 1000 * self.price_in_per_1k + max_out_tokens / 1000 * self.price_out_per_1k
        if self.spent + worst > self.cap_yuan:
            raise BudgetExceeded(f"next call could cost up to ¥{worst:.4f}; spent ¥{self.spent:.4f} of ¥{self.cap_yuan}")

    def charge(self, model: str, usage: dict) -> float:
        cost = usage.get("prompt_tokens", 0) / 1000 * self.price_in_per_1k + \
            usage.get("completion_tokens", 0) / 1000 * self.price_out_per_1k
        self.spent += cost
        rec = {"model": model, **usage, "cost_yuan": round(cost, 6), "total_spent_yuan": round(self.spent, 6)}
        self.calls.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return cost


class DashScopeProvider:
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, model: str = "qwen3.8-flash", ledger: CostLedger | None = None, max_tokens: int = 1200):
        self.key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set (make agent-real reads it from the environment)")
        self.model, self.ledger, self.max_tokens = model, ledger or CostLedger(), max_tokens

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        est_in = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 2 + 500  # rough upper bound
        self.ledger.check(est_in, self.max_tokens)
        r = httpx.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, timeout=60, json={
            "model": self.model, "messages": messages, "tools": tools, "max_tokens": self.max_tokens,
            "temperature": 0})
        r.raise_for_status()
        body = r.json()
        self.ledger.charge(self.model, body.get("usage", {}))
        return body["choices"][0]["message"]


class FakeProvider:
    """Deterministic investigator. Policy: gather evidence with every tool once
    (diff -> paths -> freshness -> shards -> slices), then apply simple rules to
    the evidence to name the most likely root cause."""

    PLAN = [("diff_runs", {}), ("path_contribution", {}), ("data_freshness", {}), ("shard_health", {}),
            ("slice_metrics", {"by": "activity_bucket"}), ("slice_metrics", {"by": "cold_items"})]

    def __init__(self):
        self.step = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        ctx = json.loads(messages[1]["content"])           # {"run_id", "base_run_id", "gate"}
        if self.step < len(self.PLAN):
            name, extra = self.PLAN[self.step]
            self.step += 1
            args = {"run_id": ctx["run_id"], **extra}
            if name != "data_freshness":
                args["base_run_id"] = ctx["base_run_id"]
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        ev = {}
        for m in messages:
            if m["role"] == "tool":
                ev.setdefault(m["name"], []).append(json.loads(m["content"]))
        cause, evidence = self.diagnose(ev)
        md = self.render(ctx, cause, evidence, ev)
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "final", "type": "function", "function": {
            "name": "write_report", "arguments": json.dumps({"markdown": md, "root_cause": cause})}}]}

    @staticmethod
    def diagnose(ev: dict) -> tuple[str, list[str]]:
        diff = ev["diff_runs"][0]
        paths = ev["path_contribution"][0]
        fresh = ev["data_freshness"][0]
        shards = ev["shard_health"][0]
        cfg = diff["config_changes"]
        if shards["suspect_shard"]:
            s = shards["suspect_shard"]
            return "stale_shard_index", [f"{s} share of top-10 changed {shards['shards'][s]['relative_change']:+.1%}",
                                         f"index versions: {shards['index_versions']}"]
        if fresh["stale"]:
            return "stale_features", [f"item features computed at {fresh['item_feature_cutoff']}, "
                                      f"{fresh['feature_lag_days']} days before the history cutoff {fresh['history_cutoff']}"]
        worst = paths["most_degraded_path"]
        p = paths["paths"][worst]
        if p["claimed_hits_candidate"] == 0 and p["claimed_hits_base"] > 0:
            return "dropped_recall_path", [f"path {worst} contributed {p['claimed_hits_base']} hits before, 0 now",
                                           f"quota change: {cfg.get('quotas')}"]
        if worst == "two_tower" and p["delta"] < -0.01:
            return "retrieval_model_regression", [f"two_tower recall@200 {p['base']:.4f} -> {p['candidate']:.4f}",
                                                  f"model change: {cfg.get('two_tower')}"]
        if "ranker" in cfg:
            return "ranker_regression", [f"recall paths unchanged (worst delta {p['delta']:+.4f}) but ranking metrics fell",
                                         f"ranker changed: {cfg['ranker']}"]
        return "unknown", ["no single tool isolates the cause"]

    @staticmethod
    def render(ctx, cause, evidence, ev) -> str:
        diff = ev["diff_runs"][0]["metrics"]
        lines = [f"# Incident report: promotion of `{ctx['run_id']}` blocked", "",
                 f"**Root-cause hypothesis:** `{cause}`", "", "## Gate", ""]
        lines += [f"- {r}" for r in ctx["gate"]["reasons"]]
        lines += ["", "## Evidence", ""] + [f"- {e}" for e in evidence]
        lines += ["", "## Metric deltas vs last good run", "", "| metric | base | candidate | delta |", "|---|---|---|---|"]
        lines += [f"| {m} | {v['base']:.4f} | {v['candidate']:.4f} | {v['delta']:+.4f} |" for m, v in diff.items()]
        for s in ev.get("slice_metrics", []):
            lines += ["", f"## Slices by {s['by']} (worst: {s['worst_slice']})", "", "| slice | users | delta NDCG@10 |", "|---|---|---|"]
            lines += [f"| {k} | {v['users']} | {v['delta']:+.4f} |" for k, v in s["slices"].items()]
        lines += ["", "_Generated by the release-guard agent (FakeProvider: deterministic rules over tool results)._"]
        return "\n".join(lines)
