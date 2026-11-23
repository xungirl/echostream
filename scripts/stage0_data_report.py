"""Stage 0: verify the dataset and write results/stage0_data_stats.json (feeds DATA.md)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.phase import load_phase, window

OUT = Path("results/stage0_data_stats.json")


def wr_stats(wr: pd.Series) -> dict:
    qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {
        "quantiles": {str(q): round(float(wr.quantile(q)), 4) for q in qs},
        "max": round(float(wr.max()), 2), "mean": round(float(wr.mean()), 4),
        "share_zero": round(float((wr == 0).mean()), 4),
        "positive_rate_by_threshold": {str(t): round(float((wr > t).mean()), 4) for t in (1.0, 1.5, 2.0, 3.0)},
    }


def main() -> None:
    a = KuaiRecAdapter()
    big = a.interactions()
    small = a.fully_observed()
    s: dict = {}
    for name, df in (("big_matrix", big), ("small_matrix", small)):
        nu, ni = df.user_id.nunique(), df.item_id.nunique()
        s[name] = {
            "rows": len(df), "users": int(nu), "items": int(ni),
            "density": round(len(df) / (nu * ni), 4),
            "distinct_user_item_pairs": int(len(df.drop_duplicates(["user_id", "item_id"]))),
            "rows_missing_timestamp": int(df.ts.isna().sum()),
            "date_range": [int(df.date[df.date > 0].min()), int(df.date.max())],
            "watch_ratio": wr_stats(df.value),
        }
    s["small_matrix"]["observation_rate"] = round(
        s["small_matrix"]["distinct_user_item_pairs"] / (s["small_matrix"]["users"] * s["small_matrix"]["items"]), 4)
    pairs_big = pd.MultiIndex.from_arrays([big.user_id, big.item_id])
    pairs_small = pd.MultiIndex.from_arrays([small.user_id, small.item_id])
    s["small_big_overlap_pairs"] = int(pairs_small.isin(pairs_big).sum())
    s["big_rows_per_date"] = {int(k): int(v) for k, v in big.groupby("date").size().items()}
    ts_date = pd.to_datetime(big.ts, unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").astype(int)
    s["big_ts_date_mismatch_share"] = round(float((ts_date != big.date).mean()), 5)

    # Is "watch_ratio > 2" biased toward short videos? positive rate by duration bucket
    dur_s = big.video_duration / 1000.0
    buckets = pd.cut(dur_s, [0, 7, 15, 30, 60, 1e9], labels=["<7s", "7-15s", "15-30s", "30-60s", ">60s"])
    s["positive_rate_by_duration"] = {
        str(k): {"rows": int(len(g)), "pos_rate_wr>2": round(float((g.value > 2).mean()), 4)}
        for k, g in big.groupby(buckets, observed=True)
    }

    spec = a.split_spec()
    s["split"] = {}
    for part, rng in (("train", spec.train), ("val", spec.val), ("test", spec.test)):
        w = window(big, *rng)
        s["split"][part] = {"dates": list(rng), "rows": len(w), "users": int(w.user_id.nunique()),
                            "items": int(w.item_id.nunique()),
                            "positives_wr>2": int((w.value > 2).sum())}

    # label sensitivity: how many users have something to predict at each threshold
    s["label_sensitivity"] = {}
    for thr in (1.0, 1.5, 2.0, 3.0):
        ad = KuaiRecAdapter(positive_threshold=thr)
        ph = load_phase(ad, "test")
        sizes = np.array([len(ph.truth[u]) for u in ph.eval_users])
        s["label_sensitivity"][str(thr)] = {
            "test_eval_users": int(len(ph.eval_users)),
            "median_truth_items_per_user": float(np.median(sizes)),
            "mean_train_positives_per_user": round(float(np.mean([len(h) for h in ph.pos_history])), 1),
        }
    # how many test positives are re-watches of items seen in history (excluded)
    test_w = window(big, *spec.test)
    tpos = test_w[test_w.value > 2]
    ph = load_phase(a, "test")
    s["test_positive_rewatch_share"] = round(float(np.mean(
        [ph.seen[u, i] > 0 for u, i in zip(tpos.user_id.values, tpos.item_id.values)])), 4)

    cats = a.item_categories()
    s["items"] = {"n_categories": a.n_categories,
                  "items_with_multiple_categories": int(sum(len(c) > 1 for c in cats.values())),
                  "caption_missing_share": round(float(a._captions.caption.isna().mean()), 4)}
    OUT.write_text(json.dumps(s, indent=2, ensure_ascii=False))
    print(json.dumps(s, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
