"""Hand-written tool-using agent loop (no frameworks).

    messages = [system, user(context)]
    loop: provider.chat(messages, tools) -> tool_calls? run them, append results : stop
until the model calls `write_report` or the step limit is hit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from echostream.agent import tools as T

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = ("You are a release guard for a recommender system. A candidate model was blocked by the offline "
          "regression gate. Use the tools to find which part of the pipeline regressed (model, recall path, "
          "features, data freshness, index shards) and which users/items are hurt. Then call write_report with a "
          "markdown incident report: root-cause hypothesis, evidence (numbers from tools), and next steps.")


def run_agent(provider, run_id: str, base_run_id: str, gate: dict, max_steps: int = 12,
              out_dir: Path = ROOT / "results" / "incidents") -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"run_id": run_id, "base_run_id": base_run_id, "gate": gate})}]
    trace = []
    report = None
    for _ in range(max_steps):
        msg = provider.chat(messages, T.openai_tool_specs())
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"]["arguments"] or "{}")
            if name == "write_report":
                report = args
                trace.append({"tool": name})
                break
            try:
                result = T.call(name, args)
            except Exception as e:      # a bad tool call is reported back to the model, not fatal
                result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": name,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
        if report:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    path = None
    if report:
        path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}.md"
        path.write_text(report["markdown"])
    return {"root_cause": report["root_cause"] if report else None, "report_path": str(path) if path else None,
            "trace": trace, "steps": len(trace)}
