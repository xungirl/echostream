# PROGRESS_LOG

How to resume: `git status`, `git log --oneline`, read the newest entry below, then `make data` (if data/ is
missing) and continue with "Next". Artifacts (models, indexes, per-user metric arrays) live in gitignored
`artifacts/` and can be rebuilt with the stage scripts listed in each entry.

## Stage 0 — data (done, 2026-11-22)
- `make data` → `echostream/data/download.py`: download from Zenodo, SHA-256 verify (data_checksums.json), parquet.
- `scripts/stage0_data_report.py` → `results/stage0_data_stats.json`; DATA.md written from it.
- Findings that changed the plan: big matrix has 3 date windows; heavy catalog drift (86% of test positives are
  videos absent from the train window) → refit on train+val for the test phase, tune on val only.
- Domain adapter: `echostream/domain/base.py` (interface) + `kuairec.py` (only implementation).

## Stage 1 — offline eval + baselines (done)
- `scripts/stage1_baselines.py` → `results/stage1_baselines.json` (val tuning grid, test + small-matrix results with
  95% bootstrap CIs, label sensitivity). Per-user arrays: `artifacts/per_user/stage1_*.npz` (rebuild by rerunning).
- Headline (quality, big-matrix test NDCG@10): pop_recent7d 0.0430, pop_platform1d 0.0418, itemcf 0.0181, als 0.0162,
  pop_global 0.0087. Small matrix flips the order (itemcf 0.618, pop_global 0.561) — exposure/catalog drift matters.

## Stage 2 — two-tower + ANN (done)
- `scripts/stage2_two_tower.py` → `results/stage2_two_tower.json`; `scripts/stage2_ann_bench.py` → `results/stage2_ann.json`.
- Selected: temperature 0.1 (val), variant `tt_logq` (logQ on, no mixed negatives; val NDCG@10 0.01361 vs 0.01359).
  Test NDCG@10 0.0803 vs pop_recent7d 0.0430 (paired +0.0372 [0.0349, 0.0397]); removing logQ AND mixed negs → 0.0071.
- Artifacts: `artifacts/two_tower/{main,val_main,rank_main}` (+ `artifacts/caption_emb.npy`, bge-small-zh in `.cache/hf`).
- macOS gotcha: torch, faiss, LightGBM and implicit each bundle libomp. A process imports torch OR the others.

## Stage 3 — multi-path recall (done)
- `results/stage3_multipath.json`. Val-selected quotas = `no_content` (tt 90, cf 40, pop_recent 35, pop_platform 35).
- Honest finding: merged recall@200 (0.233) < two-tower alone (0.271) on test — val (where tt was weak because of
  catalog drift) favoured a mix. Content path recalls 21% of cold-item truth but gets quota 0. Not re-tuned on test.

## Stage 4 — ranker (done)
- `results/stage4_ranker.json` (rank phase: recall refit < Aug 30, labels Aug 30–31) and
  `results/stage4_ranker_v1_valphase.json` (first attempt, lost to recall-only: train/serve skew). Model: `artifacts/ranker/main.txt`.
- Test: recall+rank NDCG@10 0.1100 vs recall-only 0.0803 (+0.0297 [0.0267, 0.0325]); AUC 0.861, GAUC 0.813.
- Caveat: the rank-phase protocol was designed after seeing the v1 test result.

## Stage 5 — distributed serving core (done) → MVP (tag `mvp`)
- Results: `results/stage5_load.json` (200 rps p50 12.1 / p99 26.3 ms), `stage5_chaos.json` (100% availability, 15 fault
  runs; run `scripts/stage5_chaos_summarize.py` after a new chaos run), `stage5_replay_full.json` (12.53 M events,
  37,221 ev/s, 0 mismatched users), `stage5_replay_paced10k.json` (p99 ≤ 50 ms), `stage5_synthetic.json`.
- Docs: README (bilingual header from the brand draft; logo `<img>` commented out until `docs/brand/logo.svg` exists),
  BENCHMARK.md, MVP_REPORT.md, DATA.md, INTERVIEW.md (draft, finalize in Stage 11), DRILLS.md.
- Code for Stages 6–9 is committed but NOT yet run (scripts/stage6_mmr.py, stage7_sim_ab.py, stage8_*.py,
  stage9_release_guard.py). PREREG.md is committed before Stage 7 runs.
- Web demo (Stage 10) is being built in `web/` (not yet committed at the mvp tag).
- How to resume: `make data && make train-all` (if artifacts/ missing), then the next stage script; `make ci` must stay green.

## Stage 6 — MMR (done)
- `results/stage6_mmr.json`: λ=0.7 selected by the pre-written rule (NDCG@10 0.1099 vs 0.1100, Δ CI contains 0;
  category diversity@10 0.746 → 0.896; coverage@10 0.0355 → 0.0390). Serving: `ECHO_MMR_LAMBDA=0.7` or `?mmr_lambda=`.

## Stage 7 — pre-registered simulated A/B (done)
- PREREG.md committed (at the `mvp` commit) before `scripts/stage7_sim_ab.py` ran → `results/stage7_ab.json`. No deviations.
- Between-subjects A/B (~280 users/arm): only C4 (ALS > popularity, +0.066) is significant after Holm. H1 (rank > two-tower)
  NOT supported: −0.021 [−0.043, +0.001]; within-subjects the ranker is significantly worse (−0.034 [−0.037, −0.030]).
- Offline (big-matrix NDCG@10) vs simulated-online ranking: Kendall τ = −0.4; small-matrix single-slate vs online τ = +0.6.

## Stage 8 prep + Stage 9 — release guard (done)
- `scripts/stage8_prepare_bundles.py` built bundles v2 (two-tower seed 1), v3-slow (2000×255 ranker), v4-plain (no logQ).
- `results/stage9_release_guard.json`, `results/history.jsonl`, `results/incidents/*.md`. Gate recall 0.8 (4/5 regressions
  blocked; stale shard-2 index had no measurable offline effect and passed), precision 0.8 (ranker reseed blocked: the
  seed-noise threshold comes from two-tower seeds only). Fake-provider agent: 2/4 correct root causes
  (stale features misread as a stale shard; "dropped" two-tower path still fills leftover slots → "unknown").
- `make agent-real` (DashScope, ¥2 cap) left for the user.

## Stage 8 — Postgres + safe rollout (done)
- `results/stage8_rollout.json` (+ `logs_stage8.txt`): v2 (seed refresh) PROMOTED; v3-slow (2000×255 ranker) AUTO-ROLLED BACK
  (canary degraded share 185/312 from ranker timeouts vs 7/1800 control); v4-plain (no logQ, gate bypassed on purpose) was
  PROMOTED — the simulated users liked it more (canary 0.349 vs control 0.208 positives/impression), the opposite of the
  offline gate (Stage 9 blocks the same change on recall@200). Cache (Zipf 1.1, 100 rps): hit rate 71.7%, p50 10.3 → 3.1 ms,
  p99 21 → 81 ms; 200 concurrent misses → 1 computation. Rebuild from archive: 11,381 events in 0.22 s.
- Fix during the run: supervisor now uses an absolute run dir (Postgres chdirs into its data dir; relative socket path failed).
