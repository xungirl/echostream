"""Ranking features — ONE implementation used by offline training and by the
online API, so the model sees the same features in both places.

Feature groups (used for the ablation in Stage 4):
  user      batch stats of the user (as of the cutoff)
  item      batch stats of the video: platform-wide daily stats (past days only)
            + in-dataset history counts
  cross     user x category affinity, duration match
  recall    each recall path's score for the candidate + which paths proposed it
  realtime  features from the user's most recent events (the feature-store window)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from echostream.core.itemcf import itemcf_scores
from echostream.core.recall_merge import Candidate
from echostream.domain.base import DomainAdapter
from echostream.recall.engine import CF_DECAY, CF_HIST, CONTENT_HIST, RecallEngine

RECENT_N = 100          # size of the "recent events" window (same as the feature store list)
POS_THR = 2.0
SKIP_THR = 0.3
PATHS = ["two_tower", "itemcf", "pop_recent", "pop_platform", "content"]

ITEM_STAT_COLS = ["all_like_rate", "all_complete_rate", "all_play_rate", "r7_like_rate", "r7_complete_rate",
                  "r7_play_rate", "log_all_show", "log_r7_show", "log_r7_like", "video_duration_s", "age_days",
                  "hist_n", "hist_pos", "hist_pos_rate"]
USER_STAT_COLS = ["u_n", "u_pos", "u_pos_rate", "u_mean_wr", "u_active_degree", "u_days_active"]

GROUPS = {
    "user": USER_STAT_COLS,
    "item": ITEM_STAT_COLS,
    "cross": ["cat_affinity", "dur_gap"],
    "recall": [f"score_{p}" for p in PATHS] + [f"in_{p}" for p in PATHS] + ["n_paths"],
    "realtime": ["rt_n", "rt_pos_rate", "rt_skip_rate", "rt_cat_pos", "rt_cat_skip", "rt_max_sim", "rt_mean_sim"],
}
FEATURE_NAMES = [f for g in GROUPS.values() for f in g]


@dataclass
class RankAssets:
    item_stats: np.ndarray      # [n_items, len(ITEM_STAT_COLS)]
    user_stats: np.ndarray      # [n_users, len(USER_STAT_COLS)]
    user_cat_aff: np.ndarray    # [n_users, n_cats] share of the user's positives per category
    item_cats: np.ndarray       # [n_items, n_cats] 0/1
    item_log_dur: np.ndarray    # [n_items]
    user_log_dur: np.ndarray    # [n_users] mean log duration of the user's positives


def build_rank_assets(adapter: DomainAdapter, history: pd.DataFrame, cutoff_date: int) -> RankAssets:
    """history = ALL events (not only positives) strictly before cutoff_date."""
    n_u, n_i, n_c = adapter.n_users, adapter.n_items, adapter.n_categories
    st = adapter.item_stat_features(cutoff_date)
    pos = adapter.is_positive(history.value.values)
    hn = np.bincount(history.item_id.values, minlength=n_i).astype(np.float32)
    hp = np.bincount(history.item_id.values[pos], minlength=n_i).astype(np.float32)
    item = pd.DataFrame({
        "all_like_rate": st.all_like_rate, "all_complete_rate": st.all_complete_rate,
        "all_play_rate": st.all_play_rate, "r7_like_rate": st.r7_like_rate,
        "r7_complete_rate": st.r7_complete_rate, "r7_play_rate": st.r7_play_rate,
        "log_all_show": np.log1p(st.all_show_cnt), "log_r7_show": np.log1p(st.r7_show_cnt),
        "log_r7_like": np.log1p(st.r7_like_cnt), "video_duration_s": st.video_duration / 1000.0,
        "age_days": st.age_days, "hist_n": np.log1p(hn), "hist_pos": np.log1p(hp),
        "hist_pos_rate": hp / np.maximum(hn, 1.0)})[ITEM_STAT_COLS].to_numpy(np.float32)

    un = np.bincount(history.user_id.values, minlength=n_u).astype(np.float32)
    up = np.bincount(history.user_id.values[pos], minlength=n_u).astype(np.float32)
    wr_sum = np.bincount(history.user_id.values, weights=np.minimum(history.value.values, 5.0), minlength=n_u)
    days = history.groupby("user_id").date.nunique().reindex(range(n_u)).fillna(0).to_numpy()
    user_cats, _ = adapter.user_categorical_features()
    user = np.stack([np.log1p(un), np.log1p(up), up / np.maximum(un, 1), wr_sum / np.maximum(un, 1),
                     user_cats[:, 0], days], axis=1).astype(np.float32)

    item_cats = np.zeros((n_i, n_c), np.float32)
    for i, cs in adapter.item_categories().items():
        item_cats[i, list(cs)] = 1.0
    pos_items = history.item_id.values[pos]
    pos_users = history.user_id.values[pos]
    aff = np.zeros((n_u, n_c), np.float32)
    np.add.at(aff, pos_users, item_cats[pos_items])
    aff /= np.maximum(aff.sum(1, keepdims=True), 1.0)

    dur = np.maximum(st.video_duration.to_numpy(np.float32), 1000.0)
    # fall back to the interaction log's duration for videos without daily stats
    hist_dur = history.groupby("item_id").video_duration.median()
    missing = st.video_duration.to_numpy() <= 0
    fill = hist_dur.reindex(np.arange(n_i)).to_numpy()
    dur[missing & ~np.isnan(fill)] = np.maximum(fill[missing & ~np.isnan(fill)], 1000.0)
    item_log_dur = np.log(dur)
    s = np.bincount(pos_users, weights=item_log_dur[pos_items], minlength=n_u)
    c = np.bincount(pos_users, minlength=n_u)
    user_log_dur = np.where(c > 0, s / np.maximum(c, 1), np.median(item_log_dur)).astype(np.float32)
    return RankAssets(item, user, aff, item_cats, item_log_dur.astype(np.float32), user_log_dur)


def build_features(ra: RankAssets, eng: RecallEngine, user: int, user_vec: np.ndarray, hist_pos: np.ndarray,
                   recent: list[tuple[int, float]], cands: list[Candidate]) -> np.ndarray:
    """Feature matrix [len(cands), len(FEATURE_NAMES)] for one request.
    recent = the user's last RECENT_N events (item, watch_ratio), oldest first."""
    items = np.array([c.item for c in cands], dtype=np.int64)
    n = len(items)
    f_user = np.broadcast_to(ra.user_stats[user], (n, ra.user_stats.shape[1]))
    f_item = ra.item_stats[items]

    cat_aff = ra.item_cats[items] @ ra.user_cat_aff[user]
    dur_gap = np.abs(ra.item_log_dur[items] - ra.user_log_dur[user])

    # recall-path scores for EVERY candidate (not only the path that proposed it)
    a = eng.a
    s_tt = a.item_vecs[items] @ user_vec
    if len(hist_pos):
        cf = itemcf_scores(list(hist_pos[-CF_HIST:]), a.cf_neighbors, a.cf_sims, eng.n_items, CF_DECAY)[items]
        q = a.caption_emb[hist_pos[-CONTENT_HIST:]].mean(axis=0)
        s_content = a.caption_emb[items] @ q
    else:
        cf = np.zeros(n, np.float32)
        s_content = np.zeros(n, np.float32)
    scores = [s_tt, cf, a.pop_scores["pop_recent"][items], a.pop_scores["pop_platform"][items], s_content]
    member = np.array([[1.0 if p in c.all_sources else 0.0 for p in PATHS] for c in cands], np.float32).reshape(n, len(PATHS))

    # real-time window
    recent = recent[-RECENT_N:]
    if recent:
        r_items = np.array([i for i, _ in recent], np.int64)
        r_wr = np.array([w for _, w in recent], np.float32)
        r_pos = r_items[r_wr > POS_THR]
        r_skip = r_items[r_wr < SKIP_THR]
        rt = [len(recent), float((r_wr > POS_THR).mean()), float((r_wr < SKIP_THR).mean())]
        cat_pos = ra.item_cats[items] @ ra.item_cats[r_pos].sum(0) if len(r_pos) else np.zeros(n)
        cat_skip = ra.item_cats[items] @ ra.item_cats[r_skip].sum(0) if len(r_skip) else np.zeros(n)
        last = r_pos[-5:]
        if len(last):
            sims = a.caption_emb[items] @ a.caption_emb[last].T
            max_sim, mean_sim = sims.max(1), sims.mean(1)
        else:
            max_sim = mean_sim = np.zeros(n)
    else:
        rt = [0, 0.0, 0.0]
        cat_pos = cat_skip = max_sim = mean_sim = np.zeros(n)
    f_rt = np.column_stack([np.full(n, rt[0]), np.full(n, rt[1]), np.full(n, rt[2]), cat_pos, cat_skip, max_sim, mean_sim])

    return np.column_stack([f_user, f_item, cat_aff, dur_gap, *scores, member, member.sum(1), f_rt]).astype(np.float32)


def recent_events_by_user(history: pd.DataFrame, n_users: int, n: int = RECENT_N) -> list[list[tuple[int, float]]]:
    """Each user's last n events (item, watch_ratio), oldest first, from a time-ordered frame."""
    tail = history.groupby("user_id").tail(n)
    out: list[list[tuple[int, float]]] = [[] for _ in range(n_users)]
    for u, i, v in zip(tail.user_id.values, tail.item_id.values, tail.value.values):
        out[u].append((int(i), float(v)))
    return out
