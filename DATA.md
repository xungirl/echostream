# DATA.md — KuaiRec, verified on this machine

All numbers below are **quality-side facts about the real dataset**, computed by
`uv run python scripts/stage0_data_report.py` → `results/stage0_data_stats.json`
(run 2026-11-22 on an Apple M2 / 24 GB). Nothing here is estimated.

## Source, license, integrity

- KuaiRec 2.0, Gao et al., *"KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems"*, CIKM 2022.
  Download: `https://zenodo.org/records/18164998/files/KuaiRec.zip` (431,964,858 bytes). Project page: https://github.com/chongminggao/KuaiRec
- License: **CC BY-SA 4.0**. This repo does **not** redistribute the data; `make data` downloads it,
  verifies SHA-256 for the zip and every CSV against `data_checksums.json`, and converts the CSVs to parquet under
  `data/processed/` (gitignored, ~330 MB; raw unzip ~1.9 GB).

## Files and shapes (verified)

| file | rows | notes |
|---|---:|---|
| big_matrix.csv | 12,530,806 | 7,176 users × 10,728 videos, density 0.1628; 10,300,969 distinct (user, video) pairs → 2.2 M rows are repeat exposures |
| small_matrix.csv | 4,676,570 | 1,411 users × 3,327 videos, **observation rate 0.9962** (fully observed); 181,992 rows have no timestamp |
| kuairec_caption_category.csv | 10,728 videos (+4 junk rows) | 4 CSV rows are fragments of captions containing raw newlines; the adapter drops them. 12.6% of captions are empty → the text we embed is caption + cover text + topic tags + category names |
| item_categories.csv | 10,728 | 31 categories, 1,758 videos have >1 category |
| item_daily_features.csv | 343,341 | per video per day, **platform-wide** stats (shows, plays, likes, completes, …) for 2020-07-05 … 09-05 |
| user_features.csv | 7,176 | activity level, follower/fan/friend ranges, 18 anonymous one-hot features |
| social_network.csv | 472 | not used |

**Small/big overlap:** 0 (user, video) pairs appear in both matrices (checked), so evaluating on the
small matrix cannot leak through training pairs. Small-matrix users and videos all also appear in the big matrix.

## Time

The big matrix is **not continuous**: it covers three logged-date windows — 2020-07-05…07-12, 08-01…08-10, 08-27…09-05
(rows per date are in the JSON). 0.124% of rows have a Unix timestamp whose Beijing-time date differs from the logged
`date` (client clock skew); windows are defined by `date`, ordering inside a window by timestamp.

## The label: what counts as a "like"?

KuaiRec has no like/follow signal per interaction, only `watch_ratio = play_duration / video_duration`.

| | p10 | p25 | p50 | p75 | p90 | p95 | p99 | share = 0 |
|---|---|---|---|---|---|---|---|---|
| big watch_ratio | 0.113 | 0.315 | 0.724 | 1.178 | 1.784 | 2.335 | 4.793 | 1.9% |
| small watch_ratio | 0.197 | 0.468 | 0.769 | 1.121 | 1.559 | 1.954 | 3.585 | 0.6% |

Positive rate by threshold (big / small): `>1.0` 33.8% / 32.4%, `>1.5` 15.0% / 11.2%, **`>2.0` 7.5% / 4.6%**, `>3.0` 2.7% / 1.5%.

**Decision: positive = watch_ratio > 2.0** (the dataset authors' suggestion). Justification: `>1.0` marks a third of all
plays as "liked" — it mostly means "the video was short and looped once"; `>3.0` leaves 1,263 fewer evaluable test users
than `>2.0` (5,344 vs 6,607) and a median of 4 truth items. `>2.0` keeps ~7% positives (close to a typical like/long-view rate)
and a median of 9 new positive videos per test user.

**Known bias of this label (measured):** watch_ratio > 2 is much easier for short videos. Positive rate by video length:
<7 s **18.1%**, 7–15 s 5.8%, 15–30 s 2.1%, 30–60 s 0.6%, >60 s 0.2%. Models will learn "short videos are liked"; this
is a property of the proxy label, not of users, and is listed under Limitations in the README.

Sensitivity (test phase, `results/stage0_data_stats.json → label_sensitivity`):

| threshold | evaluable test users | median truth items/user | mean history positives/user |
|---|---:|---:|---:|
| 1.0 | 6,854 | 34 | 539.4 |
| 1.5 | 6,794 | 16 | 239.2 |
| **2.0** | **6,607** | **9** | **118.4** |
| 3.0 | 5,344 | 4 | 43.5 |

Model-ranking stability across thresholds is in `results/stage1_baselines.json → label_sensitivity` (Stage 1).

Serving event types (for the event pipeline) also come from watch_ratio via the adapter: `skip` < 0.3 ≤ `play` < 1.0 ≤ `complete`.

## Split

| part | dates | rows | users | videos | positives (wr>2) |
|---|---|---:|---:|---:|---:|
| train | 07-05 … 08-10 | 9,199,872 | 7,174 | 7,655 | 688,226 |
| val | 08-27 … 08-31 | 2,192,739 | 6,953 | 4,798 | 161,664 |
| test | 09-01 … 09-05 | 1,138,195 | 6,871 | 4,740 | 86,500 |

**Catalog drift (important, measured):** of the 79,041 (user, new positive video) pairs in the test window,
**86.3% involve a video that never appears in the train window** and **38.0% a video that appears nowhere before Sep 1**.
Only 7.6% of test positives are re-watches of something already seen (these are excluded from truth and candidates).
Consequences for the protocol:
- Hyper-parameters are tuned on the **val phase** (fit on train window, predict val window).
- For the **test phase** every model is **refit with the chosen settings on all data before Sep 1** (train+val).
- Pure-ID models cannot recommend the 38% never-seen videos; content features (captions, categories) and platform-wide
  daily stats (`item_daily_features`, which exist for every video from day 1) are what can. This motivates the
  content cold-start recall path and the platform-popularity path.
- The candidate catalog is the static 10,728 videos (standard offline protocol). A real feed would only rank videos
  that already exist at request time; that "catalog knowledge" is a mild optimism shared by all models here.

## Fully-observed evaluation (small matrix)

1,411 users × 3,327 videos, 99.6% observed. For each small-matrix user we rank the 3,327 small-matrix videos using the
model state of the test phase (fit on all big-matrix data before Sep 1) and compare against the videos they actually
watched with watch_ratio > 2. No small-matrix interaction is ever used for training (and the two matrices share 0 pairs).
Because every user saw every video, this measures preference without exposure bias — but on a 3,327-video sub-catalog.
