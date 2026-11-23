"""Stage 13 experiment: how small can the browser bundle get before the recommendations change?

For each storage config: export the browser bundle, read it back, run the real Python read path
(echostream/browser/local_read_path.py) on the scripted sessions of the parity users, and compare
every screen with the float32 reference on identical inputs (the reference run's events are
replayed). First checks that the float32 browser bundle reproduces the serving bundle exactly.

    uv run python scripts/stage13_quant_sweep.py            # -> results/stage13_quant_sweep.json
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from echostream.browser.bundle_format import MMR_LAMBDA, QuantConfig, read_browser_bundle, to_serving_bundle
from echostream.browser.export import user_record, write_browser_bundle
from echostream.browser.local_read_path import local_read_path
from echostream.browser.sessions import run_session, screen_ids
from echostream.serving.bundle import load_bundle

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "artifacts" / "serving" / "current"
OUT = ROOT / "results" / "stage13_quant_sweep.json"

CONFIGS = [   # the first must be float32 (the reference); PRESETS in echostream/browser/export.py are among these
    QuantConfig(),
    QuantConfig(caption="f16"),
    QuantConfig(caption="i8row"),
    QuantConfig(item_emb="f16", id_emb="f16", caption="f16", cf_sims="f16"),
    QuantConfig(item_emb="i8row", id_emb="i8row", caption="i8row", cf_sims="u8code"),
    QuantConfig(item_emb="f16", id_emb="f16", caption="i8row", cf_sims="u8code"),
    QuantConfig(item_emb="f32", id_emb="f32", caption="i8row", cf_sims="u8code"),
    QuantConfig(item_emb="f16", id_emb="f16", caption="i8row", cf_sims="u8code", cf_topk=100),
    QuantConfig(item_emb="f16", id_emb="f16", caption="i8row", cf_sims="u8code", cf_topk=50),
]


def parity_users(n_users: int, k: int = 50, seed: int = 13) -> list[int]:
    return sorted(np.random.default_rng(seed).choice(n_users, size=k, replace=False).tolist())


def bundle_size(d: Path) -> dict:
    """on disk = the files as committed (model.bin is stored gzipped); transfer = what a host that
    gzips JSON sends (model.bin.gz as is, the JSON files gzipped)."""
    m = json.loads((d / "manifest.json").read_text())
    files = list(m["files"]) + ["manifest.json"]
    disk = {f: (d / f).stat().st_size for f in files}
    transfer = {f: disk[f] if f.endswith(".gz") else len(gzip.compress((d / f).read_bytes(), 9)) for f in files}
    return {"on_disk_bytes": sum(disk.values()), "transfer_bytes": sum(transfer.values()),
            "model_bin_uncompressed_bytes": m["model_bytes"], "per_file_on_disk": disk, "per_file_transfer": transfer}


async def run_all(bundle, users: list[int], events: dict | None) -> dict:
    out = {}
    async with local_read_path(bundle, MMR_LAMBDA) as lp:
        for u in users:
            out[u] = await run_session(lp, u, events=None if events is None else [s["event"] for s in events[u][1:]])
    return out


def compare(ref: dict, got: dict) -> dict:
    screens = exact = 0
    overlap, top1 = [], 0
    for u in ref:
        for a, b in zip(ref[u], got[u]):
            ia, ib = screen_ids(a["result"]), screen_ids(b["result"])
            screens += 1
            exact += ia == ib
            top1 += ia[:1] == ib[:1]
            overlap.append(len(set(ia) & set(ib)) / max(len(ia), 1))
    return {"screens": screens, "identical_screens": exact, "identical_share": round(exact / screens, 4),
            "mean_overlap_at10": round(float(np.mean(overlap)), 4), "min_overlap_at10": round(float(np.min(overlap)), 2),
            "same_top1_share": round(top1 / screens, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--configs", type=int, default=len(CONFIGS), help="only the first N configs (smoke runs)")
    args = ap.parse_args()
    t0 = time.time()
    sb = load_bundle(SERVING)
    users = parity_users(sb.n_users, args.users)
    records = [user_record(sb, u) for u in users]
    tmp = Path(tempfile.mkdtemp(prefix="echostream-sweep-"))
    rows = []
    try:
        # 1) the serving bundle itself (real users only: it has no row for a brand-new user)
        direct = asyncio.run(run_all(sb, users, None))
        ref = None
        for q in CONFIGS[:args.configs]:
            d = tmp / q.name
            write_browser_bundle(d, SERVING, q)
            bb = read_browser_bundle(d)
            new_uid = bb.manifest["n_users"]
            sbq = to_serving_bundle(bb, records + [u for u in bb.users if u["uid"] == new_uid], lgb.Booster(model_file=str(SERVING / "ranker.txt")),
                                    new_uid + 1)
            run_users = users + [new_uid]
            if ref is None:      # float32 config: must equal the serving bundle exactly, then it is the reference
                got = asyncio.run(run_all(sbq, run_users, None))
                same = compare({u: direct[u] for u in users}, {u: got[u] for u in users})
                scores_equal = all(
                    [x["rank_score"] for x in a["result"].items] == [x["rank_score"] for x in b["result"].items]
                    for u in users for a, b in zip(direct[u], got[u]))
                rows.append({"config": q.__dict__ | {"name": q.name}, "size": bundle_size(d),
                             "vs_serving_bundle": same | {"rank_scores_identical": scores_equal}})
                ref = got
                print(q.name, rows[-1]["size"]["transfer_bytes"], same, scores_equal, flush=True)
                continue
            got = asyncio.run(run_all(sbq, run_users, ref))
            rows.append({"config": q.__dict__ | {"name": q.name}, "size": bundle_size(d), "vs_float32": compare(ref, got)})
            print(q.name, rows[-1]["size"]["transfer_bytes"], rows[-1]["vs_float32"], flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    OUT.write_text(json.dumps({
        "what": "browser bundle size vs agreement with the float32 bundle, real Python read path, same events",
        "users": users, "n_users": len(users) + 1, "screens_per_user": 6, "mmr_lambda": MMR_LAMBDA,
        "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0]}",
        "date": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0, 1), "configs": rows}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
