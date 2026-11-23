"""Build RecallAssets for a phase from trained artifacts (no torch needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from echostream.core.itemcf import itemcf_similarity
from echostream.domain.base import DomainAdapter
from echostream.offline.baselines import PlatformPopularity, Popularity, positives_matrix
from echostream.offline.phase import ARTIFACTS, Phase
from echostream.recall.engine import RecallAssets

DEFAULT_QUOTAS = {"two_tower": 80, "itemcf": 40, "pop_recent": 30, "pop_platform": 30, "content": 20}


def load_two_tower(tag: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    d = ARTIFACTS / "two_tower" / tag
    params = dict(np.load(d / "user_tower.npz"))
    return np.load(d / "item_emb.npy"), params, np.load(d / "user_cats.npy")


def build_assets(adapter: DomainAdapter, phase: Phase, two_tower_tag: str,
                 quotas: dict[str, int] | None = None, total: int = 200,
                 caption_path: Path = ARTIFACTS / "caption_emb.npy") -> RecallAssets:
    item_vecs, params, user_cats = load_two_tower(two_tower_tag)
    nb, sims = itemcf_similarity(positives_matrix(adapter, phase), topk=200)
    pops = {"pop_recent": Popularity(7).fit(adapter, phase).item_scores,
            "pop_platform": PlatformPopularity(1, "complete_play_cnt").fit(adapter, phase).item_scores}
    item_hist_count = np.asarray(phase.seen.sum(axis=0)).ravel()
    return RecallAssets(item_vecs=item_vecs, user_tower=params, user_cats=user_cats, cf_neighbors=nb, cf_sims=sims,
                        pop_scores=pops, caption_emb=np.load(caption_path), item_hist_count=item_hist_count,
                        quotas=dict(quotas or DEFAULT_QUOTAS), total=total)
