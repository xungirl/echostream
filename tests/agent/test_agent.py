"""Agent loop + providers. The loop test uses stub tools (no data); the end-to-end
test runs real eval runs on a user subset (needs data + artifacts)."""
import json

import pytest

from echostream.agent import tools as T
from echostream.agent.loop import run_agent
from echostream.agent.providers import BudgetExceeded, CostLedger, FakeProvider


def test_loop_calls_tools_then_writes_report(monkeypatch, tmp_path):
    calls = []

    def stub(name, result):
        def fn(**kw):
            calls.append(name)
            return result
        return fn
    fake = {
        "diff_runs": (stub("diff_runs", {"metrics": {"ndcg@10": {"base": 0.1, "candidate": 0.05, "delta": -0.05}},
                                        "config_changes": {"quotas": {"base": {"two_tower": 90}, "candidate": {"two_tower": 0}}}}), "", []),
        "path_contribution": (stub("path_contribution", {"paths": {"two_tower": {"base": 0.27, "candidate": 0.27, "delta": 0.0,
                                   "claimed_hits_base": 100, "claimed_hits_candidate": 0}}, "most_degraded_path": "two_tower"}), "", []),
        "data_freshness": (stub("data_freshness", {"stale": False, "feature_lag_days": 0, "item_feature_cutoff": 1,
                                                    "history_cutoff": 1}), "", []),
        "shard_health": (stub("shard_health", {"suspect_shard": None, "shards": {}, "index_versions": "ok"}), "", []),
        "slice_metrics": (stub("slice_metrics", {"by": "x", "slices": {}, "worst_slice": None}), "", []),
    }
    monkeypatch.setattr(T, "TOOLS", fake)
    gate = {"reasons": ["ndcg@10 dropped"]}
    res = run_agent(FakeProvider(), "cand", "base", gate, out_dir=tmp_path)
    assert res["root_cause"] == "dropped_recall_path"
    assert calls[:4] == ["diff_runs", "path_contribution", "data_freshness", "shard_health"]
    report = open(res["report_path"]).read()
    assert "dropped_recall_path" in report and "ndcg@10 dropped" in report


def test_bad_tool_call_is_reported_not_fatal(monkeypatch, tmp_path):
    class Bad:
        n = 0
        def chat(self, messages, tools):
            self.n += 1
            if self.n == 1:
                return {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {
                    "name": "diff_runs", "arguments": json.dumps({"run_id": "missing", "base_run_id": "x"})}}]}
            assert "error" in json.loads(messages[-1]["content"])
            return {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {
                "name": "write_report", "arguments": json.dumps({"markdown": "# r", "root_cause": "unknown"})}}]}
    monkeypatch.setenv("ECHO_EVAL_HISTORY", str(tmp_path / "h.jsonl"))
    res = run_agent(Bad(), "c", "b", {"reasons": []}, out_dir=tmp_path)
    assert res["root_cause"] == "unknown"


def test_cost_ledger_hard_cap(tmp_path):
    led = CostLedger(cap_yuan=0.01, price_in_per_1k=1.0, price_out_per_1k=1.0, path=tmp_path / "ledger.jsonl")
    led.check(1, 1)                                   # ¥0.002 worst case: fine
    led.charge("m", {"prompt_tokens": 5, "completion_tokens": 3})
    assert abs(led.spent - 0.008) < 1e-9
    with pytest.raises(BudgetExceeded):
        led.check(1000, 1000)
    # the cap is cumulative across runs (re-reads the ledger file)
    assert abs(CostLedger(cap_yuan=0.01, path=tmp_path / "ledger.jsonl").spent - 0.008) < 1e-9


def test_dashscope_provider_needs_key(monkeypatch):
    from echostream.agent.providers import DashScopeProvider
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        DashScopeProvider()
