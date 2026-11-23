"""Safe model rollout: register -> (offline gate) -> shadow -> canary -> promote or auto-rollback.

Desired state lives in Postgres (model_registry + rollout_state); API replicas
and shard servers reconcile to it (hot reload, no restart). The controller only
writes desired state, drives/observes traffic and decides.

Decision rules (all measured, none hand-waved):
  shadow : shadow scoring error rate must be <= 1 %.
  canary : roll back if ANY of
             - canary error rate > 1 %
             - canary p99 latency > max(SLO_P99_MS, 2 x control p99)
             - canary degraded-response share > control share + 5 pp
             - online quality: 95% bootstrap CI of (canary - control) positives-per-impression
               lies entirely below 0 (significantly worse for simulated users)
           otherwise promote.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from echostream.core.metrics import bootstrap_ci
from echostream.sim.online_driver import drive, per_version
from echostream.store import db

SLO_P99_MS = 150.0


@dataclass
class RolloutReport:
    version: str
    decision: str = "pending"
    reasons: list[str] = field(default_factory=list)
    phases: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self.log.append(line)
        print(line, flush=True)


class RolloutController:
    def __init__(self, lb_url: str, api_urls: list[str], responses: dict, users: list[int],
                 canary_percent: int = 10, shadow_users: int = 150, slates: int = 3):
        self.lb, self.apis, self.responses, self.users = lb_url, api_urls, responses, users
        self.canary_percent, self.shadow_users, self.slates = canary_percent, shadow_users, slates

    # ------------------------------------------------------------ helpers
    def _stats(self) -> list[dict]:
        return [httpx.get(f"{u}/admin/rollout/stats", timeout=3).json() for u in self.apis]

    def _reset(self) -> None:
        for u in self.apis:
            httpx.post(f"{u}/admin/rollout/reset_stats", timeout=3)

    def _wait_converged(self, want: dict, timeout: float = 120) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if all(s["rollout"]["default"] == want["default"] and s["rollout"]["canary"] == want.get("canary")
                       and s["rollout"]["shadow"] == want.get("shadow") for s in self._stats()):
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"replicas did not converge to {want}")

    def _set(self, conn, default, canary=None, pct=0, shadow=None) -> None:
        db.set_rollout(conn, default, canary, pct, shadow)
        self._wait_converged({"default": default, "canary": canary, "shadow": shadow})

    # ------------------------------------------------------------- main
    def rollout(self, version: str, bundle_path: str, offline_metrics: dict | None = None,
                gate: dict | None = None) -> RolloutReport:
        rep = RolloutReport(version)
        with db.connect() as conn:
            current = db.get_rollout(conn)["default_version"]
            db.register_model(conn, version, bundle_path, offline_metrics)
            db.log(conn, version, "registered", {"bundle": bundle_path})
            rep.note(f"registered {version} (current live: {current})")
            if gate is not None and not gate.get("promote", True):
                db.set_status(conn, version, "blocked")
                db.log(conn, version, "blocked_by_gate", gate)
                rep.decision, rep.reasons = "blocked_by_offline_gate", gate.get("reasons", [])
                rep.note(f"offline regression gate BLOCKED {version}: {rep.reasons}")
                return rep

            # ---- shadow ------------------------------------------------------------------
            db.set_status(conn, version, "shadow")
            self._set(conn, current, shadow=version)
            db.log(conn, version, "shadow_start")
            rep.note(f"shadow: {version} scores every request in parallel, results are not returned")
            self._reset()
            asyncio.run(drive(self.lb, self.users[:self.shadow_users], self.responses, slates=1, tag=f"shadow-{version}"))
            time.sleep(1.0)
            sh = [s["shadow"] for s in self._stats()]
            n, errs = sum(x["n"] for x in sh), sum(x["errors"] for x in sh)
            overlap = [x["overlap_at10_mean"] for x in sh if x["overlap_at10_mean"] is not None]
            rep.phases["shadow"] = {"scored": n, "errors": errs, "overlap_at10_mean": float(np.mean(overlap)) if overlap else None,
                                    "shadow_p99_ms_per_replica": [x["p99_ms"] for x in sh]}
            rep.note(f"shadow result: {rep.phases['shadow']}")
            if n == 0 or errs / max(n + errs, 1) > 0.01:
                return self._rollback(conn, rep, current, [f"shadow errors {errs}/{n + errs}"])

            # ---- canary ------------------------------------------------------------------
            db.set_status(conn, version, "canary")
            self._set(conn, current, canary=version, pct=self.canary_percent)
            db.log(conn, version, "canary_start", {"percent": self.canary_percent})
            rep.note(f"canary: {self.canary_percent}% of users (consistent hash) now get {version}")
            self._reset()
            results = asyncio.run(drive(self.lb, self.users, self.responses, slates=self.slates, tag=f"canary-{version}"))
            pv = per_version(results)
            can, ctl = pv.get(version), pv.get(current)
            if can is None or ctl is None:
                return self._rollback(conn, rep, current, ["canary received no traffic"])
            diffs = []
            rng = np.random.default_rng(0)
            for _ in range(2000):
                diffs.append(rng.choice(can["rates"], len(can["rates"])).mean() - rng.choice(ctl["rates"], len(ctl["rates"])).mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            phase = {"control": {k: v for k, v in ctl.items() if k != "rates"},
                     "canary": {k: v for k, v in can.items() if k != "rates"},
                     "pos_rate_diff": float(can["pos_rate"] - ctl["pos_rate"]), "pos_rate_diff_ci95": [float(lo), float(hi)]}
            rep.phases["canary"] = phase
            rep.note(f"canary result: control pos_rate {ctl['pos_rate']:.4f} p99 {ctl['p99_ms']:.1f}ms | "
                     f"canary pos_rate {can['pos_rate']:.4f} p99 {can['p99_ms']:.1f}ms | diff CI [{lo:+.4f}, {hi:+.4f}]")
            reasons = []
            if can["errors"] / max(can["requests"], 1) > 0.01:
                reasons.append(f"canary error rate {can['errors']}/{can['requests']}")
            if can["p99_ms"] > max(SLO_P99_MS, 2 * ctl["p99_ms"]):
                reasons.append(f"canary p99 {can['p99_ms']:.1f} ms > SLO max({SLO_P99_MS:.0f}, 2x control {ctl['p99_ms']:.1f})")
            if can["degraded"] / max(can["requests"], 1) > ctl["degraded"] / max(ctl["requests"], 1) + 0.05:
                reasons.append(f"canary degraded share {can['degraded']}/{can['requests']} vs control "
                               f"{ctl['degraded']}/{ctl['requests']}")
            if hi < 0:
                reasons.append(f"online quality: positives/impression diff CI [{lo:+.4f}, {hi:+.4f}] < 0")
            if reasons:
                return self._rollback(conn, rep, current, reasons)

            # ---- promote -------------------------------------------------------------------
            self._set(conn, version)
            db.set_status(conn, version, "live")
            db.set_status(conn, current, "retired")
            db.log(conn, version, "promoted", phase)
            rep.decision = "promoted"
            rep.note(f"PROMOTED {version}; {current} retired")
            return rep

    def _rollback(self, conn, rep: RolloutReport, current: str, reasons: list[str]) -> RolloutReport:
        self._set(conn, current)
        db.set_status(conn, rep.version, "rolled_back")
        db.log(conn, rep.version, "auto_rollback", {"reasons": reasons})
        rep.decision, rep.reasons = "rolled_back", reasons
        rep.note(f"AUTO-ROLLBACK {rep.version}: {reasons}; all traffic back on {current}")
        return rep


def report_json(rep: RolloutReport) -> dict:
    return json.loads(json.dumps(rep.__dict__, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
