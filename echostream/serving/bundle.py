"""A serving bundle = everything one model version needs at request time.

Written by scripts/export_bundle.py into artifacts/serving/<version>/ and loaded
by the API (and, for item vectors, by the shard servers). Contents:

  manifest.json            version, quotas, source artifacts, created time
  item_emb.npy             two-tower item vectors (also loaded by shards)
  user_tower.npz           numpy user tower + user_cats.npy
  cf_neighbors/sims.npy    ItemCF table
  pop_recent/platform.npy  popularity scores
  caption_emb.npy          content path
  item_hist_count.npy      interactions per item before the cutoff (cold items = 0)
  rank_assets.npz, ranker.txt
  history.npz              batch user state as of the cutoff: positive history,
                           seen items, last-100 events (CSR-style offsets)
  item_meta.parquet        title / category name / category ids for the UI and events
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from echostream.rank.features import RankAssets
from echostream.recall.engine import RecallAssets, RecallEngine


def pack_lists(lists: list[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(lists) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(x) for x in lists])
    flat = np.concatenate([np.asarray(x, dtype=dtype) for x in lists]) if lists else np.empty(0, dtype)
    return flat, offsets


@dataclass
class ServingBundle:
    version: str
    manifest: dict
    engine: RecallEngine
    rank_assets: RankAssets
    ranker: lgb.Booster | None
    pos_flat: np.ndarray
    pos_off: np.ndarray
    seen_flat: np.ndarray
    seen_off: np.ndarray
    recent_items: np.ndarray
    recent_wr: np.ndarray
    recent_off: np.ndarray
    item_meta: pd.DataFrame
    item_cats_csv: list[str]
    path: Path

    @property
    def n_users(self) -> int:
        return len(self.pos_off) - 1

    def pos_history(self, u: int) -> np.ndarray:
        return self.pos_flat[self.pos_off[u]:self.pos_off[u + 1]]

    def seen(self, u: int) -> np.ndarray:
        return self.seen_flat[self.seen_off[u]:self.seen_off[u + 1]]

    def recent(self, u: int) -> list[tuple[int, float]]:
        lo, hi = self.recent_off[u], self.recent_off[u + 1]
        return list(zip(self.recent_items[lo:hi].tolist(), self.recent_wr[lo:hi].tolist()))


def load_bundle(path: str | Path) -> ServingBundle:
    p = Path(path).resolve()
    m = json.loads((p / "manifest.json").read_text())
    ut = dict(np.load(p / "user_tower.npz"))
    assets = RecallAssets(
        item_vecs=np.load(p / "item_emb.npy"), user_tower=ut, user_cats=np.load(p / "user_cats.npy"),
        cf_neighbors=np.load(p / "cf_neighbors.npy"), cf_sims=np.load(p / "cf_sims.npy"),
        pop_scores={"pop_recent": np.load(p / "pop_recent.npy"), "pop_platform": np.load(p / "pop_platform.npy")},
        caption_emb=np.load(p / "caption_emb.npy"), item_hist_count=np.load(p / "item_hist_count.npy"),
        quotas=m["quotas"], total=m.get("total", 200))
    ra = RankAssets(**dict(np.load(p / "rank_assets.npz")))
    ranker = lgb.Booster(model_file=str(p / "ranker.txt")) if (p / "ranker.txt").exists() else None
    h = np.load(p / "history.npz")
    meta = pd.read_parquet(p / "item_meta.parquet")
    return ServingBundle(m["version"], m, RecallEngine(assets), ra, ranker, h["pos_flat"], h["pos_off"],
                         h["seen_flat"], h["seen_off"], h["recent_items"], h["recent_wr"], h["recent_off"],
                         meta, meta.cats.tolist(), p)
