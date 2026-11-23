"""Session simulator on the fully-observed small matrix (see PREREG.md).

The "user" is the real KuaiRec small-matrix row: when a policy shows video i,
the response is the watch_ratio the user actually had on i. Positives update
the user's history between slates, like the event pipeline would.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from echostream.core.mmr import mmr_rerank
from echostream.rank.features import RankAssets, build_features
from echostream.rank.offline import blend_similarity
from echostream.recall.engine import RecallEngine

POS_THR = 2.0


@dataclass
class UserState:
    user: int
    hist_pos: np.ndarray
    recent: list[tuple[int, float]]
    shown: set[int] = field(default_factory=set)


class Policy:
    name = "base"

    def slate(self, st: UserState, exclude: np.ndarray, n: int) -> list[int]:
        raise NotImplementedError


class PopularityPolicy(Policy):
    name = "V1_popularity"

    def __init__(self, eng: RecallEngine):
        self.eng = eng

    def slate(self, st, exclude, n):
        return [i for i, _ in self.eng.popularity("pop_recent", n, exclude)]


class StaticScorePolicy(Policy):
    """Fixed per-user scores (ALS: user factors not updated inside a session)."""
    name = "V2_als"

    def __init__(self, scores: dict[int, np.ndarray], n_items: int):
        self.scores, self.n_items = scores, n_items

    def slate(self, st, exclude, n):
        s = self.scores[st.user].astype(np.float32).copy()
        s[exclude] = -np.inf
        top = np.argsort(-s, kind="stable")[:n]
        return [int(i) for i in top if np.isfinite(s[i])]


class TwoTowerPolicy(Policy):
    name = "V3_two_tower"

    def __init__(self, eng: RecallEngine):
        self.eng = eng

    def slate(self, st, exclude, n):
        uv = self.eng.user_vector(st.user, st.hist_pos)
        return [i for i, _ in self.eng.two_tower(uv, n, exclude)]


class RankPolicy(Policy):
    name = "V4_recall_rank"

    def __init__(self, eng: RecallEngine, ra: RankAssets, model: lgb.Booster, mmr_lambda: float | None = None):
        self.eng, self.ra, self.model, self.lam = eng, ra, model, mmr_lambda
        if mmr_lambda is not None:
            self.name = "V5_recall_rank_mmr"

    def slate(self, st, exclude, n):
        uv = self.eng.user_vector(st.user, st.hist_pos)
        paths = self.eng.all_paths(st.user, st.hist_pos, exclude, k=200, user_vec=uv)
        cands = self.eng.merge(paths)
        if not cands:
            return []
        X = build_features(self.ra, self.eng, st.user, uv, st.hist_pos, st.recent, cands)
        s = self.model.predict(X, num_threads=1)
        order = np.lexsort((np.array([c.item for c in cands]), -s))
        items = [cands[j].item for j in order]
        s = s[order]
        if self.lam is None or self.lam >= 1.0:
            return items[:n]
        pool = items[:50]
        pick = mmr_rerank(s[:len(pool)], blend_similarity(self.ra.item_cats, self.eng.a.caption_emb, pool), self.lam, n)
        return [pool[j] for j in pick]


def run_session(policy: Policy, st: UserState, responses: dict[int, float], not_catalog: np.ndarray,
                seen: np.ndarray, slates: int = 5, n: int = 10) -> list[tuple[int, float]]:
    """Returns the (item, watch_ratio) pairs shown, in order."""
    shown: list[tuple[int, float]] = []
    catalog = np.flatnonzero(~not_catalog)
    unobserved = np.setdiff1d(catalog, np.fromiter(responses.keys(), np.int64, len(responses)))
    base_exclude = np.union1d(np.union1d(np.flatnonzero(not_catalog), seen), unobserved)
    for _ in range(slates):
        exclude = np.union1d(base_exclude, np.fromiter(st.shown, np.int64, len(st.shown)))
        items = policy.slate(st, exclude, n)
        for i in items:
            wr = responses[i]
            shown.append((i, wr))
            st.shown.add(i)
            st.recent.append((i, wr))
            if wr > POS_THR:
                st.hist_pos = np.append(st.hist_pos, i)
    return shown
