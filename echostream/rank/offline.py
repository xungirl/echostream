"""Offline recall -> rank pipeline for a phase (used by Stages 4, 6, 7, 9).
Same RecallEngine + build_features as the online API."""
from __future__ import annotations

import numpy as np

from echostream.domain.base import DomainAdapter
from echostream.offline.phase import Phase, history_frame
from echostream.rank.features import build_features, build_rank_assets, recent_events_by_user
from echostream.recall.assets import build_assets
from echostream.recall.engine import RecallEngine


def candidates_and_features(a: DomainAdapter, phase: Phase, tt_tag: str, quotas: dict[str, int],
                            users: np.ndarray | None = None, with_labels: bool = True):
    """Returns (users, row_user_ids, X, y, candidate_lists, engine, rank_assets)."""
    eng = RecallEngine(build_assets(a, phase, tt_tag, quotas=quotas))
    hist_all = history_frame(a, phase.name, positives_only=False)
    ra = build_rank_assets(a, hist_all, phase.target_start)
    recent = recent_events_by_user(hist_all, a.n_users)
    users = phase.eval_users if users is None else users
    out_users, rows_u, X, y, cand_lists = [], [], [], [], []
    for u in users:
        hist = phase.pos_history[u]
        uvec = eng.user_vector(int(u), hist)
        paths = eng.all_paths(int(u), hist, phase.seen[u].indices, k=200, user_vec=uvec)
        cands = eng.merge(paths)
        if not cands:
            continue
        X.append(build_features(ra, eng, int(u), uvec, hist, recent[u], cands))
        items = [c.item for c in cands]
        if with_labels:
            y.append(np.array([i in phase.truth[u] for i in items], np.float32))
        rows_u.append(np.full(len(items), u))
        out_users.append(u)
        cand_lists.append(items)
    yy = np.concatenate(y) if y else None
    return np.array(out_users), np.concatenate(rows_u), np.concatenate(X), yy, cand_lists, eng, ra


def rerank(cand_lists: list[list[int]], scores: np.ndarray) -> list[list[int]]:
    """Sort each user's candidates by score (desc), ties by item id."""
    out, pos = [], 0
    for items in cand_lists:
        s = scores[pos:pos + len(items)]
        order = np.lexsort((np.array(items), -s))
        out.append([items[j] for j in order])
        pos += len(items)
    return out


def split_scores(cand_lists: list[list[int]], scores: np.ndarray) -> list[np.ndarray]:
    out, pos = [], 0
    for items in cand_lists:
        out.append(scores[pos:pos + len(items)])
        pos += len(items)
    return out


def blend_similarity(item_cats: np.ndarray, caption_emb: np.ndarray, items: list[int]) -> np.ndarray:
    """MMR similarity: mean of category Jaccard and caption cosine (clipped to [0, 1])."""
    cats = item_cats[items]
    inter = cats @ cats.T
    sizes = cats.sum(1)
    union = sizes[:, None] + sizes[None, :] - inter
    jac = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    emb = caption_emb[items]
    return 0.5 * jac + 0.5 * np.clip(emb @ emb.T, 0.0, 1.0)
