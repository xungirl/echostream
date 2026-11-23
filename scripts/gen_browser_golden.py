"""Golden file for the browser engine's parity test (Stage 13).

Python is the reference. The shipped browser bundle (web/demo-public/model/) is decoded with
echostream/browser/bundle_format.py into the production ServingBundle object, and the real read
path runs in-process (echostream/browser/local_read_path.py: real shard servers over ASGI, real
Worker + Lua on a temporary redis-server, real Recommender, MMR lambda 0.7). For 50 users
(43 sampled + the 6 demo users + the brand-new user) it plays a scripted session
(first screen, then like / skip / watch / like / skip on the current screen) and records, for
every screen: the user vector, each recall path's candidates and scores, the merged 200, the
LightGBM probability of every candidate, the 10 shown videos, and the ranker features of some
screens. web/src/demo/engine/parity.test.ts replays the same events in TypeScript and compares.

    uv run python scripts/gen_browser_golden.py           # -> web/demo-golden/golden.json.gz
    uv run python scripts/gen_browser_golden.py --check   # exit 1 if the golden file is stale
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

from echostream.browser.bundle_format import MMR_LAMBDA, read_browser_bundle, sha256_file, to_serving_bundle
from echostream.browser.export import user_record
from echostream.browser.local_read_path import local_read_path
from echostream.browser.sessions import SCRIPT, run_session
from echostream.rank.features import PATHS
from echostream.serving.bundle import load_bundle

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "web" / "demo-public" / "model"
SERVING = ROOT / "artifacts" / "serving" / "current"
OUT = ROOT / "web" / "demo-golden" / "golden.json.gz"
N_SAMPLED = 43
SEED = 13
FEATURE_USERS = 10        # ranker features are stored for screens 0 and 3 of the first 10 users


def b64(x: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(x, dtype=dtype).tobytes()).decode()


def array_digest(x: np.ndarray) -> str:
    """sha256 of the decoded array as the browser holds it (float32 or int32, little-endian)."""
    dt = "<f4" if np.issubdtype(x.dtype, np.floating) else "<i4"
    return hashlib.sha256(np.ascontiguousarray(x, dtype=dt).tobytes()).hexdigest()


def golden_users(bb, n_users: int) -> list[int]:
    demo = [u["uid"] for u in bb.users if u["kind"] == "real"]
    pool = np.setdiff1d(np.arange(n_users), demo)
    sampled = np.random.default_rng(SEED).choice(pool, size=N_SAMPLED, replace=False)
    new = [u["uid"] for u in bb.users if u["kind"] == "new"]
    return sorted(sampled.tolist()) + demo + new


def screen_record(k: int, s: dict, with_features: bool) -> dict:
    tr, res = s["trace"], s["result"]
    paths = {p: {"ids": b64([i for i, _ in tr.paths.get(p, [])], "<u2"),
                 "scores": b64([sc for _, sc in tr.paths.get(p, [])], "<f4")} for p in PATHS}
    merged = tr.merged
    rec = {
        "step": k, "event": s["event"],
        "user_vec": b64(tr.user_vec, "<f4"),
        "paths": paths,
        "merged_ids": b64([c.item for c in merged], "<u2"),
        "merged_source": b64([PATHS.index(c.source) for c in merged], "<u1"),
        "merged_members": b64([sum(1 << j for j, p in enumerate(PATHS) if p in c.all_sources) for c in merged], "<u1"),
        "rank_scores": b64(tr.rank_scores, "<f8"),
        "shown_ids": [x["item_id"] for x in res.items],
        "shown_scores": b64([float(tr.rank_scores[[c.item for c in merged].index(x["item_id"])]) for x in res.items], "<f8"),
    }
    if with_features:
        rec["features"] = b64(tr.features, "<f4")
    return rec


async def run(bb, records: list[dict], users: list[int], n_users: int) -> list[dict]:
    sb = to_serving_bundle(bb, records, lgb.Booster(model_file=str(SERVING / "ranker.txt")), n_users)
    sessions = []
    async with local_read_path(sb, MMR_LAMBDA) as lp:
        for n, u in enumerate(users):
            screens = await run_session(lp, u)
            sessions.append({"uid": u, "screens": [screen_record(k, s, n < FEATURE_USERS and k in (0, 3))
                                                   for k, s in enumerate(screens)]})
    return sessions


def build() -> dict:
    bb = read_browser_bundle(BUNDLE)
    stale = [f for f, h in bb.manifest["source"]["sha256"].items() if sha256_file(SERVING / f) != h]
    if stale:
        sys.exit(f"the browser bundle was exported from other artifacts ({stale}); "
                 "run scripts/export_browser_model.py first")
    sb = load_bundle(SERVING)
    users = golden_users(bb, sb.n_users)
    in_bundle = {u["uid"]: u for u in bb.users}
    records = [in_bundle.get(u) or user_record(sb, u) for u in users]
    n_users = bb.manifest["n_users"] + 1                        # + the brand-new user
    sessions = asyncio.run(run(bb, records, users, n_users))
    return {
        "what": "Python read path on the shipped browser bundle; the TypeScript engine must reproduce it",
        "bundle_files": {f: v["sha256"] for f, v in bb.manifest["files"].items()},
        "decoded_sha256": {k: array_digest(v) for k, v in bb.arrays.items()},
        "script": SCRIPT, "mmr_lambda": MMR_LAMBDA, "paths": PATHS,
        "users": [r for r in records if r["uid"] not in in_bundle],
        "sessions": sessions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data = json.dumps(build(), separators=(",", ":")).encode()
    if args.check:
        old = gzip.decompress(OUT.read_bytes()) if OUT.exists() else b""
        if old != data:
            sys.exit(f"{OUT.relative_to(ROOT)} is stale: run scripts/gen_browser_golden.py")
        print("golden file up to date")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(gzip.compress(data, 9, mtime=0))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
