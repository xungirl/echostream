"""Re-derive the chaos summary from the raw per-second data in results/stage5_chaos.json.

Two recovery definitions, both computed from the same raw data:
  recovery_s_p99  : last second (within 20 s after fault/heal) with availability < 100%, degraded share
                    > pre-fault + 1 pp, or per-second p99 > 2x pre-fault p99. At 100 rps a per-second p99 is
                    ~the slowest single request, so one GC pause marks a whole second "bad" (noisy).
  recovery_s      : same, but using per-second p50 > 2x pre-fault p50 instead of p99 (robust to single spikes).
"""
import json

import numpy as np

FAULT_T, HEAL_T = 10, 30


def recovery(secs, base, start, key, factor=2.0):
    bad = [s["t"] for s in secs if start <= s["t"] < start + 20 and
           (s["availability"] < 1.0 or s["degraded"] > base["degraded_rate"] + 0.01 or s[key] > factor * base[key])]
    return 0.0 if not bad else float(max(bad) + 1 - start)


def main():
    d = json.load(open("results/stage5_chaos.json"))
    heal = {"slow_shard", "kill_worker", "stop_feature_node"}
    summary = {}
    for r in d["runs"]:
        start = HEAL_T if r["scenario"] in heal else FAULT_T
        if r["scenario"] == "baseline":
            r["recovery_s"] = r["recovery_s_p99"] = 0.0
        else:
            r["recovery_s_p99"] = recovery(r["per_second"], r["pre_fault"], start, "p99_ms")
            r["recovery_s"] = recovery(r["per_second"], r["pre_fault"], start, "p50_ms")
    for name in dict.fromkeys(r["scenario"] for r in d["runs"]):
        rs = [r for r in d["runs"] if r["scenario"] == name]
        fw = [r["fault_window"] for r in rs if r["fault_window"]]
        summary[name] = {
            "availability_overall": [round(1 - r["overall"]["error_rate"], 5) for r in rs],
            "errors_overall": [int(round(r["overall"]["error_rate"] * r["overall"]["requests"])) for r in rs],
            "degraded_share_fault_window": [f["degraded_rate"] for f in fw],
            "degraded_labels_fault_window": [f.get("degraded_labels") for f in fw],
            "p50_ms_fault_window": [f["p50_ms"] for f in fw], "p99_ms_fault_window": [f["p99_ms"] for f in fw],
            "p99_ms_pre_fault": [r["pre_fault"]["p99_ms"] for r in rs],
            "recovery_s": [r["recovery_s"] for r in rs], "recovery_s_p99": [r["recovery_s_p99"] for r in rs],
            "event_backlog_max": [r["event_backlog_max"] for r in rs],
            "events_post_success": [r["events_post_success"] for r in rs],
        }
    d["summary"] = summary
    d["recovery_definitions"] = __doc__.split("Two recovery definitions")[1].strip()
    json.dump(d, open("results/stage5_chaos.json", "w"), indent=2)
    for k, v in summary.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "degraded_labels_fault_window"})


if __name__ == "__main__":
    main()
