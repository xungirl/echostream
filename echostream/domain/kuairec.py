"""KuaiRec (Gao et al., CIKM 2022; CC BY-SA 4.0) adapter.

Label: KuaiRec has no like signal. Following the dataset authors' suggestion we
call an interaction positive when watch_ratio > 2.0 (the user watched the video
more than twice through). DATA.md shows the sensitivity to this threshold.
"""
from __future__ import annotations

import ast
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from echostream.domain.base import DomainAdapter, SplitSpec

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

POSITIVE_WATCH_RATIO = 2.0   # "like" proxy (official suggestion), see DATA.md
COMPLETE_WATCH_RATIO = 1.0   # watched to the end at least once
SKIP_WATCH_RATIO = 0.3       # left within the first ~30% of the video

USER_FIELDS = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

DAILY_COUNT_COLS = ["show_cnt", "play_cnt", "complete_play_cnt", "valid_play_cnt", "like_cnt",
                    "comment_cnt", "share_cnt", "follow_cnt", "collect_cnt", "reduce_similar_cnt"]


class KuaiRecAdapter(DomainAdapter):
    name = "kuairec"
    n_users = 7176
    n_items = 10728

    def __init__(self, positive_threshold: float = POSITIVE_WATCH_RATIO, data_dir: Path = PROCESSED):
        self.positive_threshold = positive_threshold
        self.data_dir = data_dir

    # ---- interactions + labels ------------------------------------------
    @cached_property
    def _big(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_dir / "big_matrix.parquet")
        df = df.rename(columns={"watch_ratio": "value"})
        # order by the event time; date is the logged partition (used for windows)
        return df.sort_values(["ts", "user_id", "item_id"], kind="stable").reset_index(drop=True)

    def interactions(self) -> pd.DataFrame:
        return self._big

    def is_positive(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value) > self.positive_threshold

    def event_type(self, value: float) -> str:
        if value < SKIP_WATCH_RATIO:
            return "skip"
        if value >= COMPLETE_WATCH_RATIO:
            return "complete"
        return "play"

    def split_spec(self) -> SplitSpec:
        # The big matrix covers three date windows: Jul 5-12, Aug 1-10, Aug 27-Sep 5.
        return SplitSpec(train=(20200705, 20200810), val=(20200827, 20200831), test=(20200901, 20200905),
                         rank=(20200830, 20200831))

    # ---- side information -------------------------------------------------
    @cached_property
    def _captions(self) -> pd.DataFrame:
        c = pd.read_parquet(self.data_dir / "kuairec_caption_category.parquet")
        # 4 rows of the CSV are fragments of captions that contain raw newlines
        c = c[c.video_id.astype(str).str.isdigit()].copy()
        c["video_id"] = c.video_id.astype(int)
        return c.drop_duplicates("video_id").set_index("video_id").sort_index()

    @cached_property
    def _item_cats(self) -> dict[int, frozenset[int]]:
        df = pd.read_parquet(self.data_dir / "item_categories.parquet")
        return {int(v): frozenset(ast.literal_eval(f)) for v, f in zip(df.video_id, df.feat)}

    def item_categories(self) -> dict[int, frozenset[int]]:
        return self._item_cats

    @property
    def n_categories(self) -> int:
        return max(max(s) for s in self._item_cats.values() if s) + 1

    def item_text(self) -> dict[int, str]:
        def clean(x) -> str:
            x = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
            return "" if x in ("UNKNOWN", "[]", "nan") else x
        out = {}
        for vid, row in self._captions.iterrows():
            parts = [clean(row.caption), clean(row.manual_cover_text), clean(row.topic_tag),
                     clean(row.first_level_category_name), clean(row.second_level_category_name)]
            out[int(vid)] = " ".join(p for p in parts if p) or "未知"
        for vid in range(self.n_items):
            out.setdefault(vid, "未知")
        return out

    def item_display(self) -> pd.DataFrame:
        c = self._captions.reindex(range(self.n_items))
        title = c.caption.fillna("").astype(str).str.slice(0, 80)
        return pd.DataFrame({"item_id": range(self.n_items), "title": title.values,
                             "category_name": c.first_level_category_name.fillna("UNKNOWN").values})

    def user_categorical_features(self) -> tuple[np.ndarray, list[int]]:
        u = pd.read_parquet(self.data_dir / "user_features.parquet").set_index("user_id").sort_index()
        assert len(u) == self.n_users and u.index[0] == 0 and u.index[-1] == self.n_users - 1
        mats, vocab = [], []
        for col in USER_FIELDS:
            codes, uniques = pd.factorize(u[col], sort=True)  # missing -> -1
            mats.append(codes + 1)                            # 0 = missing
            vocab.append(len(uniques) + 1)
        return np.stack(mats, axis=1).astype(np.int64), vocab

    def item_stat_features(self, cutoff_date: int) -> pd.DataFrame:
        """Platform-wide daily stats (all Kuaishou users) summed over dates < cutoff,
        plus the last 7 logged days, plus a few static attributes."""
        d = pd.read_parquet(self.data_dir / "item_daily_features.parquet",
                            columns=["video_id", "date", "video_duration", "upload_dt"] + DAILY_COUNT_COLS)
        past = d[d.date < cutoff_date]
        recent_dates = sorted(past.date.unique())[-7:]
        recent = past[past.date.isin(recent_dates)]
        total = past.groupby("video_id")[DAILY_COUNT_COLS].sum().add_prefix("all_")
        last7 = recent.groupby("video_id")[DAILY_COUNT_COLS].sum().add_prefix("r7_")
        static = past.sort_values("date").groupby("video_id")[["video_duration", "upload_dt"]].last()
        cutoff_ts = pd.to_datetime(str(cutoff_date))
        static["age_days"] = (cutoff_ts - pd.to_datetime(static.upload_dt, errors="coerce")).dt.days
        out = total.join(last7, how="outer").join(static[["video_duration", "age_days"]], how="outer")
        out = out.reindex(range(self.n_items)).fillna(0.0)
        for prefix in ("all_", "r7_"):
            shows = out[f"{prefix}show_cnt"].clip(lower=1.0)
            out[f"{prefix}like_rate"] = out[f"{prefix}like_cnt"] / shows
            out[f"{prefix}complete_rate"] = out[f"{prefix}complete_play_cnt"] / shows
            out[f"{prefix}play_rate"] = out[f"{prefix}play_cnt"] / shows
        out.index.name = "item_id"
        return out.astype(np.float32)

    def fully_observed(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_dir / "small_matrix.parquet")
        return df.rename(columns={"watch_ratio": "value"})
