"""Export a serving bundle (test-phase state: all data before 2020-09-01).

    uv run python scripts/export_bundle.py --version v1 [--two-tower main] [--ranker artifacts/ranker/main.txt]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.phase import ARTIFACTS, history_frame, load_phase
from echostream.rank.features import build_rank_assets, recent_events_by_user
from echostream.recall.assets import build_assets
from echostream.serving.bundle import pack_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--two-tower", default="main")
    ap.add_argument("--ranker", default=str(ARTIFACTS / "ranker" / "main.txt"))
    ap.add_argument("--quotas", default="results/stage3_multipath.json")
    ap.add_argument("--set-current", action="store_true")
    args = ap.parse_args()
    a = KuaiRecAdapter()
    phase = load_phase(a, "test")
    quotas = {k: v for k, v in json.loads(Path(args.quotas).read_text())["selected_quotas"].items() if k != "name"}
    out = ARTIFACTS / "serving" / args.version
    out.mkdir(parents=True, exist_ok=True)
    assets = build_assets(a, phase, args.two_tower, quotas=quotas)
    np.save(out / "item_emb.npy", assets.item_vecs)
    np.savez(out / "user_tower.npz", **assets.user_tower)
    np.save(out / "user_cats.npy", assets.user_cats)
    np.save(out / "cf_neighbors.npy", assets.cf_neighbors)
    np.save(out / "cf_sims.npy", assets.cf_sims)
    np.save(out / "pop_recent.npy", assets.pop_scores["pop_recent"])
    np.save(out / "pop_platform.npy", assets.pop_scores["pop_platform"])
    np.save(out / "caption_emb.npy", assets.caption_emb)
    np.save(out / "item_hist_count.npy", assets.item_hist_count)

    hist_all = history_frame(a, "test", positives_only=False)
    ra = build_rank_assets(a, hist_all, phase.target_start)
    np.savez(out / "rank_assets.npz", **ra.__dict__)
    if args.ranker and Path(args.ranker).exists():
        shutil.copy(args.ranker, out / "ranker.txt")

    pos_flat, pos_off = pack_lists(phase.pos_history, np.int32)
    seen_lists = [phase.seen[u].indices for u in range(a.n_users)]
    seen_flat, seen_off = pack_lists(seen_lists, np.int32)
    recent = recent_events_by_user(hist_all, a.n_users)
    r_items, r_off = pack_lists([np.array([i for i, _ in r], np.int32) for r in recent], np.int32)
    r_wr, _ = pack_lists([np.array([w for _, w in r], np.float32) for r in recent], np.float32)
    np.savez(out / "history.npz", pos_flat=pos_flat, pos_off=pos_off, seen_flat=seen_flat, seen_off=seen_off,
             recent_items=r_items, recent_wr=r_wr, recent_off=r_off)

    meta = a.item_display()
    cats = a.item_categories()
    meta["cats"] = [",".join(str(c) for c in sorted(cats.get(i, ()))) for i in meta.item_id]
    meta.to_parquet(out / "item_meta.parquet", index=False)
    manifest = {"version": args.version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "domain": a.name,
                "cutoff_date": phase.target_start, "two_tower": args.two_tower, "ranker": args.ranker,
                "quotas": quotas, "total": 200, "n_users": a.n_users, "n_items": a.n_items}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.set_current:
        cur = ARTIFACTS / "serving" / "current"
        if cur.is_symlink() or cur.exists():
            cur.unlink()
        cur.symlink_to(args.version)
    print("bundle", out, sum(f.stat().st_size for f in out.iterdir()) / 1e6, "MB")


if __name__ == "__main__":
    main()
