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
