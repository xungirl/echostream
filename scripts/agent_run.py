"""Run the release-guard agent on the blocked candidates from Stage 9.

    uv run python scripts/agent_run.py --provider fake      # deterministic, no network (CI)
    DASHSCOPE_API_KEY=... uv run python scripts/agent_run.py --provider dashscope   # make agent-real (¥2 hard cap)
"""
import argparse
import json

from echostream.agent.loop import run_agent
from echostream.agent.providers import CostLedger, DashScopeProvider, FakeProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["fake", "dashscope"], default="fake")
    ap.add_argument("--cap-yuan", type=float, default=2.0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    s9 = json.load(open("results/stage9_release_guard.json"))
    ledger = CostLedger(cap_yuan=args.cap_yuan)
    results = []
    for row in s9["rows"]:
        if not row["blocked"] or (args.only and row["candidate"] != args.only):
            continue
        provider = FakeProvider() if args.provider == "fake" else DashScopeProvider(ledger=ledger)
        try:
            res = run_agent(provider, row["gate"]["candidate_run"], s9["baseline_run"], row["gate"])
        except Exception as e:   # includes BudgetExceeded: stop cleanly, keep the ledger
            print(f"{row['candidate']}: stopped ({type(e).__name__}: {e})")
            break
        results.append({"candidate": row["candidate"], "expected": row["expected_root_cause"], **res})
        print(row["candidate"], "->", res["root_cause"], res["report_path"])
    if args.provider == "dashscope":
        print(f"spent ¥{ledger.spent:.4f} of ¥{ledger.cap_yuan} (ledger: results/agent_cost_ledger.jsonl)")
        with open("results/agent_real_runs.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
