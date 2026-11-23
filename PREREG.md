# PREREG — Stage 7 simulated online A/B (written and committed BEFORE running)

## Question
Does the **offline** ranking of recommendation variants agree with their ranking in a **simulated online**
setting with session dynamics (history updates between slates)?

## Simulator (fixed before running)
- Population: every user of KuaiRec's fully-observed `small_matrix` (1,411 users). Catalog: its 3,327 videos.
- Starting state of a user: the serving state as of 2020-09-01 (all big-matrix data before the test window).
  The small and big matrices share 0 (user, video) pairs, so no simulated response was ever trained on.
- Session: 5 slates × 10 videos. A video shown once is never shown again in the session.
- Response: the user's true `watch_ratio` for that video from `small_matrix` (the matrix is 99.6% observed;
  unobserved pairs are never shown). Positive = watch_ratio > 2.0 (same label as offline).
- After each slate: positives are appended to the user's positive history; every shown video is appended to the
  recent-events window (watch_ratio included) — exactly what the event pipeline would write to the feature store.
- The simulator is deterministic (no response noise); a user's outcome under a variant does not depend on
  which other users are in the experiment.

## Variants
| id | variant |
|---|---|
| V1 | popularity (recent in-dataset popularity, 7 logged days) |
| V2 | ALS (implicit, Stage-1 config; user factors are not updated within a session) |
| V3 | two-tower recall only (user vector recomputed after every slate) |
| V4 | multi-path recall + LightGBM ranker (Stage 4 model) |
| V5 | V4 + MMR re-rank (lambda chosen in Stage 6) |
All variants rank only the 3,327 catalog videos.

## Assignment
Consistent-hash bucketing: `bucket = blake2b("sim-ab-2026:" + user_id) mod 100`; buckets 0–19 → V1, 20–39 → V2,
40–59 → V3, 60–79 → V4, 80–99 → V5. Stable across restarts and machines (`echostream/core/hashing.py`).

## Metrics
- **Primary:** positives per impression = (# shown videos with watch_ratio > 2) / (# shown videos), averaged over users.
- Secondary: mean watch_ratio of shown videos (clipped at 5), intra-session category diversity, catalog coverage.

## Sample size
Fixed by the data: 1,411 users, ~282 per arm under bucketing. No power analysis beyond reporting CIs; a
difference whose 95% CI includes 0 is reported as "no detectable difference", not as "equal".

## Analysis plan
1. **Primary (between-subjects, the A/B as it would run):** per-arm mean of the primary metric with 95% percentile
   bootstrap CI (2,000 resamples of users). Pre-specified comparisons, Holm–Bonferroni corrected at α = 0.05:
   C1 V4 vs V3 (does ranking help?), C2 V5 vs V4 (does MMR cost engagement?), C3 V3 vs V2, C4 V2 vs V1.
   Test: bootstrap CI of the difference of means + two-sided Welch t-test p-value (for Holm).
2. **Secondary (within-subjects):** because the simulator is deterministic, also run every variant on every user
   and report paired bootstrap CIs for the same comparisons (much tighter; not what a real A/B can do).
3. **Offline vs online agreement:** offline score = big-matrix test NDCG@10 of each variant (Stages 1, 2, 4, 6).
   Also the small-matrix single-slate NDCG@10 (no session dynamics). Report Kendall's tau and Spearman's rho
   between the offline ranking and the online (primary metric, within-subjects) ranking of the 5 variants,
   and list every pair whose order disagrees.

## Hypotheses
- H1: V4 > V3 on the primary metric.
- H2: V5 is not significantly worse than V4 on the primary metric.
- H3: offline (big-matrix NDCG@10) and online orderings agree (Kendall tau > 0); we expect at least one disagreement
  involving popularity, because the small matrix is free of exposure bias while the big-matrix test set is not.

## Stopping rule
One fixed-horizon run with seed-free deterministic simulation. No interim looks, no re-runs with changed settings.
Any deviation from this document is reported in `results/stage7_ab.json → deviations` and in the README.
