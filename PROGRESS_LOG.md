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
