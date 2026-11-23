"""Stage 8: Postgres-backed safe rollout demo + cache experiment + rebuild timing.

Cluster: full profile + Postgres (ECHO_PG=1), workers archive events (ECHO_ARCHIVE=1).
  1. v1 live.  2. roll out v2 (refresh)       -> expect PROMOTE
  3. roll out v3-slow (2000-tree ranker)       -> expect AUTO-ROLLBACK (latency SLO)
  4. roll out v4-plain (no logQ two-tower)     -> expect AUTO-ROLLBACK (online quality)
  5. per-user cache on/off under Zipf traffic; stampede burst
  6. rebuild the feature store from the Postgres archive (timed)

    uv run python scripts/stage8_rollout.py -> results/stage8_rollout.json
"""
import asyncio
import os
import platform
import time
from pathlib import Path

import httpx
import numpy as np

from echostream.bench.loadgen import open_loop, summarize
from echostream.cluster.client import cluster
from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.evaluate import write_json
from echostream.offline.phase import ARTIFACTS
from echostream.rollout.controller import RolloutController, report_json
from echostream.sim.online_driver import shuffled, small_matrix_responses
from echostream.store import db

PORT = 17000


def main():
    os.environ["ECHO_PG_DSN"] = f"postgresql://echostream@127.0.0.1:{PORT + 40}/echostream"
    run = Path(".run/stage8")
    if run.exists():
        import shutil
        shutil.rmtree(run)
    env = {"ECHO_PORT_BASE": str(PORT), "ECHO_PG": "1", "ECHO_ARCHIVE": "1",
           "ECHO_BUNDLE": str((ARTIFACTS / "serving" / "v1").resolve())}
    a = KuaiRecAdapter()
    responses = small_matrix_responses(a)
    users = shuffled(responses.keys(), seed=8)
    out = {"kind": "systems + simulated online metrics (small-matrix users through the live cluster)",
           "machine": platform.platform(), "date": time.strftime("%Y-%m-%d %H:%M"), "rollouts": []}
    with cluster("full", env=env, run_dir=run, timeout=240) as cfg:
        lb = f"http://127.0.0.1:{cfg.lb_port}"
        apis = [f"http://127.0.0.1:{cfg.api_port(i)}" for i in range(cfg.n_api)]
        with db.connect() as conn:
            db.register_model(conn, "v1", str((ARTIFACTS / "serving" / "v1").resolve()), notes="baseline")
            db.set_status(conn, "v1", "live")
            db.set_rollout(conn, "v1")
        ctl = RolloutController(lb, apis, responses, users, canary_percent=10, shadow_users=150, slates=3)
        ctl._wait_converged({"default": "v1"})
        for version, note in (("v2", "refresh: two-tower retrained with another seed"),
                              ("v3-slow", "ranker with 2,000 trees x 255 leaves"),
                              ("v4-plain", "two-tower without logQ / mixed negatives (gate bypassed on purpose)")):
            path = str((ARTIFACTS / "serving" / version).resolve())
            rep = ctl.rollout(version, path)
            out["rollouts"].append({"version": version, "note": note, **report_json(rep)})
        with db.connect() as conn:
            out["rollout_log"] = [dict(r) for r in conn.execute(
                "SELECT ts::text, version, action, details FROM rollout_log ORDER BY id").fetchall()]
            out["registry"] = [dict(r) for r in conn.execute(
                "SELECT version, status, created_at::text FROM model_registry ORDER BY created_at").fetchall()]
            out["archived_events"] = conn.execute("SELECT count(*) AS n FROM event_archive").fetchone()["n"]

        # ---- cache experiment: Zipf(1.1) user popularity, 100 rps, 30 s, cache off vs on -------------
        rng = np.random.default_rng(0)
        ranks = np.arange(1, a.n_users + 1)
        p = ranks ** -1.1
        p /= p.sum()
        zipf_users = rng.choice(rng.permutation(a.n_users), size=200_000, p=p)
        cache = {}
        for ttl in (0, 30):
            for api in apis:
                httpx.post(f"{api}/admin/cache", params={"ttl_s": ttl}, timeout=5)
            time.sleep(1)
            s = asyncio.run(open_loop(lb, 100, 30, zipf_users, seed=11))
            stats = [httpx.get(f"{api}/admin/rollout/stats", timeout=5).json()["cache"]["computations"] for api in apis]
            row = summarize(s, 30)
            hits = sum(1 for x in s if "cache_hit" in x.degraded)
            cache[f"ttl_{ttl}s"] = {**row, "cache_hits": hits, "hit_rate": round(hits / len(s), 4),
                                    "pipeline_computations": sum(stats) if ttl else len(s)}
        # stampede: 200 concurrent requests for one uncached user, spread over all replicas via the LB
        async def burst():
            async with httpx.AsyncClient(base_url=lb, timeout=5) as c:
                return await asyncio.gather(*(c.get("/recommend", params={"user_id": 1234}) for _ in range(200)))
        before = [httpx.get(f"{api}/admin/rollout/stats", timeout=5).json()["cache"]["computations"] for api in apis]
        rs = asyncio.run(burst())
        after = [httpx.get(f"{api}/admin/rollout/stats", timeout=5).json()["cache"]["computations"] for api in apis]
        cache["stampede_200_concurrent_same_user"] = {"ok": sum(r.status_code == 200 for r in rs),
                                                     "pipeline_computations": sum(after) - sum(before)}
        out["cache"] = cache
        for api in apis:
            httpx.post(f"{api}/admin/cache", params={"ttl_s": 0}, timeout=5)

        # ---- rebuild the feature store from the archive --------------------------------
        from echostream.store.rebuild import rebuild
        out["rebuild"] = asyncio.run(rebuild(cfg.feature_nodes, flush=True))
    write_json("results/stage8_rollout.json", out)
    for r in out["rollouts"]:
        print(r["version"], r["decision"], r["reasons"])
    print(out["cache"])
    print(out["rebuild"])


if __name__ == "__main__":
    main()
