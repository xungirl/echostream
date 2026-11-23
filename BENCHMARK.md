# BENCHMARK.md — every number, how it was measured, where the raw output is

**Machine:** Apple M2 (8 cores: 4 performance + 4 efficiency), 24 GB RAM, macOS (Darwin 27.2), Python 3.12.14 (uv),
PyTorch 2.15 (MPS for training), faiss-cpu 1.16, LightGBM 4.7, implicit 0.7.3, Redis (Homebrew), nginx (Homebrew),
PostgreSQL 16. **Date:** 2026-11-22. Everything (load generator included) runs on this one laptop.

**Two kinds of numbers, never mixed:**
- **quality** — recommendation metrics on the real KuaiRec data (temporal split; or the fully-observed small matrix).
- **systems** — latency / availability / throughput: either *real-event replay* (KuaiRec's 12.5 M interactions) or
  *synthetic scale-up* (random vectors; latency/memory only, never quality). "Multi-node" always means
  **multiple processes on localhost**.

Bootstrap CIs: 1,000 resamples of users (2,000 for Stage 7), percentile method, seed fixed. "Paired" = same users,
CI of the per-user difference.

---

## Quality

### Stage 1 — baselines (`uv run python scripts/stage1_baselines.py` → `results/stage1_baselines.json`)
Test phase = predict new positives (watch_ratio > 2) in 2020-09-01…05 for 6,607 users, catalog 10,728 videos,
already-seen videos excluded. Hyper-parameters chosen on the val phase only.

| model | test NDCG@10 [95% CI] | test Recall@200 | small-matrix NDCG@10 (1,411 users × 3,327 videos) |
|---|---|---|---|
| popularity (all history) | 0.0087 [0.0080, 0.0095] | 0.1078 | 0.5611 |
| popularity (last 7 logged days) | 0.0430 [0.0410, 0.0451] | 0.2280 | 0.1520 |
| platform popularity (yesterday's completes) | 0.0418 [0.0400, 0.0438] | 0.1418 | 0.0479 |
| ItemCF (hand-written, IUF, last 20 positives) | 0.0181 [0.0167, 0.0194] | 0.1246 | 0.6181 |
| ALS (implicit, 64 factors) | 0.0162 [0.0150, 0.0175] | 0.1057 | 0.3849 |

Label-threshold sensitivity (same file, `label_sensitivity`): platform popularity has the highest test NDCG@10 among
these four at every threshold (1.0 / 1.5 / 2.0 / 3.0); ItemCF and ALS swap places at 3.0.

### Stage 2 — two-tower (`scripts/stage2_two_tower.py` → `results/stage2_two_tower.json`)
Temperature on val: 0.05 → 0.01242, **0.1 → 0.01356**, 0.2 → 0.01330 (val NDCG@10). Training: 4 epochs, ~50 s on MPS.

| variant (test phase, refit on train+val) | test NDCG@10 [95% CI] | test Recall@200 | small-matrix NDCG@10 |
|---|---|---|---|
| **logQ, in-batch only (selected on val)** | **0.0803 [0.0775, 0.0831]** | 0.2705 | 0.4024 |
| logQ + 1,024 mixed uniform negatives | 0.0795 [0.0764, 0.0824] | 0.2599 | 0.3566 |
| no logQ + mixed negatives | 0.0505 [0.0482, 0.0528] | 0.2315 | 0.3604 |
| no logQ, in-batch only | 0.0071 [0.0063, 0.0080] | 0.1054 | 0.1574 |

Paired vs strongest baseline (popularity 7 d): NDCG@10 +0.0372 [0.0349, 0.0397]. Seed variance of the selected config
(seeds 0/1/2): NDCG@10 0.0803 / 0.0833 / 0.0802 (std 0.0018), Recall@200 std 0.0035.

### Stage 2 — ANN on the real vectors (`scripts/stage2_ann_bench.py` → `results/stage2_ann.json`)
10,728 item vectors (64-d), 6,607 real test-user queries for recall, 2,000 for latency; 1 thread, one query at a time,
3 repeats (mean ± std of the per-repeat percentile).

| index | params | recall@100 vs exact | p50 µs | p99 µs |
|---|---|---|---|---|
| Flat (exact) | – | 1.000 | 92.2 ± 0.0 | 103.5 ± 0.7 |
| HNSW | M=32, efSearch=64 | 0.866 | 28.9 ± 0.0 | 37.6 ± 0.3 |
| HNSW | M=32, efSearch=128 | 0.955 | 47.5 ± 0.1 | 58.8 ± 1.1 |
| HNSW | M=32, efSearch=256 | 0.994 | 103.7 ± 0.2 | 117.4 ± 5.3 |
| IVF | nlist=64, nprobe=16 | 0.993 | 25.2 ± 0.0 | 29.1 ± 0.1 |
| IVF | nlist=256, nprobe=16 | 0.935 | 15.7 ± 0.1 | 20.3 ± 1.2 |

At this catalog size exact search costs ~0.1 ms, so the serving shards use Flat (exact; offline == online).

### Stage 3 — multi-path recall (`scripts/stage3_multipath.py` → `results/stage3_multipath.json`)
Quotas chosen on val (merged recall@200): `no_content` {two_tower 90, itemcf 40, pop_recent 35, pop_platform 35}.
Test: merged recall@200 **0.2327 [0.2282, 0.2374]** vs two-tower alone 0.2705 — the val-selected mix is *worse* than
the best single path on test (val's two-tower was weak because of catalog drift). Content path alone recalls 21.3% of
the cold-item truth pairs (30,058 of 79,041 test pairs involve never-seen videos) but its val-selected quota is 0.

### Stage 4 — ranker (`scripts/stage4_ranker.py --phase rank` → `results/stage4_ranker.json`)
LightGBM (binary, 210 trees after early stopping), 40 features, trained on the rank phase (labels 08-30…31, 1,175,000 rows).

| test (6,607 users) | NDCG@10 [95% CI] | AUC | GAUC |
|---|---|---|---|
| recall only (merge order = two-tower order) | 0.0803 [0.0775, 0.0831] | 0.710 | 0.732 |
| **recall + rank** | **0.1100 [0.1064, 0.1136]** | **0.861** | **0.813** |
| first attempt: ranker trained on the val phase (`stage4_ranker_v1_valphase.json`) | 0.0696 [0.0670, 0.0724] | 0.814 | 0.748 |

Paired: rank vs recall-only +0.0297 [0.0267, 0.0325]; vs popularity-7d +0.0670 [0.0642, 0.0700].
Feature-group ablation (retrain without the group, ΔNDCG@10 vs full, paired CI): item −0.0415 [−0.0446, −0.0382],
user −0.0153, recall scores −0.0138, cross −0.0071, real-time −0.0025 [−0.0043, −0.0008].

---

## Systems (all on one laptop; multi-node on localhost)

Cluster for every systems run unless stated: nginx → 4 API replicas (FastAPI/uvicorn, uvloop) · 4 vector shards × 2 replicas
(faiss Flat) · 3 Redis feature nodes (consistent hashing) · 1 Redis event bus (8 stream partitions, AOF) · 4 workers ·
process supervisor. The load generator runs on the same machine and competes for the same 8 cores.

### Stage 5 — load (`uv run python scripts/stage5_load.py` → `results/stage5_load.json`)
Open-loop GET /recommend (latency from the *scheduled* send time), real users drawn uniformly from all 7,176, full pipeline
(features, 5-path recall, LightGBM over 200 candidates). 30 s per level, 3 repeats, mean (± std across repeats).

| target rps | achieved | p50 ms | p95 ms | p99 ms | errors | degraded |
|---|---|---|---|---|---|---|
| 50 | 50.0 | 11.93 ± 0.08 | 13.25 | 14.85 ± 0.71 | 0 | 0.1% |
| 100 | 100.0 | 11.99 ± 0.03 | 13.65 | 25.71 ± 11.54 | 0 | 0.1% |
| 150 | 150.0 | 11.33 ± 0.06 | 12.89 | 26.90 ± 13.78 | 0 | 0.1% |
| 200 | 200.0 | 12.10 ± 0.47 | 14.58 | 26.34 ± 3.29 | 0 | 0.1% |
| 250 | 250.0 | 3,980 ± 3,267 | 24,223 | 29,736 | 0 | 4.5% |

Saturation is between 200 and 250 rps: each request costs ~5 ms of single-threaded Python CPU in an API replica
(LightGBM ~2.4 ms of it, measured with cProfile in-process) plus shard and nginx work, on 8 shared cores. More API
replicas would not help much on this machine; this is **not** a horizontal-scaling result.

### Stage 5 — chaos (`uv run python scripts/stage5_chaos.py && uv run python scripts/stage5_chaos_summarize.py` → `results/stage5_chaos.json`)
100 rps of /recommend + ~100 events/s of POST /events, 45 s per run, fault at t = 10 s, heal at t = 30 s when the fault
needs healing, 3 repeats per scenario (4,500 requests per run). Raw per-second rows are in the JSON.

| scenario | how | availability (3 runs) | degraded share during fault | p99 during fault (ms) | recovery_s (p50 rule) |
|---|---|---|---|---|---|
| baseline | – | 100 / 100 / 100% | 0 / 0 / 0 | 13.9 / 12.9 / 18.4 | – |
| kill shard primary | SIGKILL `shard0-r0`, supervisor restarts it | 100 / 100 / 100% | 0 / 0.1% / 0 | 20.0 / 15.3 / 34.6 | 0 / 20 / 0 |
| slow shard | +500 ms on `shard1-r0` for 20 s | 100 / 100 / 100% | 0.15% / 0 / 0 | 26.4 / 14.3 / 28.6 | 0 / 13 / 0 |
| kill worker | stop `worker0` for 20 s | 100 / 100 / 100% | 0.15% / 0 / 0 | 14.1 / 14.4 / 14.0 | 0 / 9 / 0 |
| stall feature node | SIGSTOP `redis-feat1` for 20 s | 100 / 100 / 100% | 33.3% / 37.3% / 34.6% | 34.7 / 35.2 / 34.3 | 0 / 0 / 0 |
| kill API replica | SIGKILL `api0`, supervisor restarts it | 100 / 100 / 100% | 0.1% / 0 / 0 | 17.2 / 14.8 / 14.4 | 0 / 0 / 0 |

- *Degraded* = a full answer carrying a label (e.g. `feature_store_unavailable` for the ~1/3 of users whose features live on
  the stalled node; they get batch-history-only recommendations). No request failed in any run.
- Event backlog (stream lag + pending) peaked at 55–73 entries when a worker died (its partitions are re-leased after the
  3 s lease TTL) and at 636–640 while a feature node was stalled (events for its users wait, in order, until it is back).
  POST /events success: 100% in all runs.
- *recovery_s* = seconds after the fault (auto-recovering faults) or heal until the last "bad" second within 20 s
  (availability < 100%, degraded share > pre-fault + 1 pp, or per-second p50 > 2× pre-fault p50). The non-zero values all come
  from repeat #1 of three different scenarios, i.e. single noisy seconds rather than a systematic slow recovery; a stricter
  p99-based variant is also stored (`recovery_s_p99`), and it is noisier still because at 100 rps a per-second p99 is roughly
  the single slowest request.

### Stage 5 — real-event replay (`make replay` / `make replay-paced` → `results/stage5_replay_*.json`)
All big-matrix interactions in original timestamp order, `event_id = kr<row>`, batches of 1,000 through nginx → POST /events
(4 API replicas) → 8 stream partitions → 4 workers → 3 feature nodes. Producer backpressure at 400k unprocessed entries.
After the run, every user's counters (n / complete / skip / play / positives) in the feature store are compared with counts
computed directly from the parquet file.

| run | events | wall time to all applied | throughput | correctness | event→feature latency |
|---|---|---|---|---|---|
| max rate (`full`) | 12,530,806 | 336.7 s | **37,221 events/s** (median applied rate 36,634/s, p10 28,359/s) | **0 / 7,176 users mismatched**, 0 duplicates, 0 DLQ, 0 failed batches | backlog-dominated (max backlog 460,688; p50 bucket ≤ 30 s) |
| paced 10k/s (`paced10k`, first 1.2 M events) | 1,200,000 | 120.4 s | 9,963 events/s | 0 / 7,176 mismatched | **p50/p95/p99 ≤ 50 ms** (histogram bucket upper bound; buckets …20, 50, 100… ms) |

Feature-node memory after the full replay: 300 / 443 / 444 MB (mostly 12.5 M dedupe keys with a 1 h TTL).

### Stage 5 — SYNTHETIC scale test (`make synthetic` → `results/stage5_synthetic.json`)
**Synthetic data**: 256-cluster Gaussian vectors (64-d) generated inside each shard process; latency/memory only. One sequential
client issues 500 queries (k = 100) through the real scatter-gather client (HTTP), 3 repeats. 1 replica per shard.

| vectors | shards | index | vectors held per shard (computed) | shard RSS max (MB) | p50 ms | p99 ms | build+start s |
|---|---|---|---|---|---|---|---|
| 1 M | 1 | Flat | 244 MB | 946 | 6.17 | 8.05 | 1.1 |
| 1 M | 2 | Flat | 122 MB | 695 | 5.68 | 6.71 | 0.8 |
| 1 M | 4 | Flat | 61 MB | 386 | 6.81 | 8.12 | 0.9 |
| 1 M | 8 | Flat | 31 MB | 231 | 9.30 | 11.07 | 1.0 |
| 10 M | 1 | IVF (nlist 3162, nprobe 63) | 2,519 MB | 1,801* | 2.33 | 5.48 | 53.7 |
| 10 M | 2 | IVF (2236 / 44) | 1,259 MB | 3,183 | 2.42 | 2.90 | 22.1 |
| 10 M | 4 | IVF (1581 / 31) | 630 MB | 1,450 | 3.50 | 4.09 | 14.0 |
| 10 M | 8 | IVF (1118 / 22) | 315 MB | 571 | 5.96 | 7.81 | 12.0 |

\* macOS RSS excludes compressed pages, so it can be below the data actually held; the computed column is the reliable one.
What this shows on one laptop: sharding divides **memory per node** and **build time**; it does **not** reduce query latency
here — each extra shard adds ~0.5–0.8 ms of RPC/fan-out work in a single-threaded Python client, more than the per-shard
search it saves. On this synthetic distribution IVF reaches recall@100 = 1.0 at ~2% of lists probed (100 queries vs exact
brute force), which says nothing about real data — real-vector ANN recall is in Stage 2.

---

## Part 2

### Stage 6 — MMR (`make stage6` → `results/stage6_mmr.json`, quality)
Pool = ranker top-50, pick 10; similarity = mean(category Jaccard, caption cosine). 6,607 test users.

| λ | NDCG@10 | Δ vs λ=1 [95% CI] | category diversity@10 | coverage@10 |
|---|---|---|---|---|
| 1.0 (off) | 0.1100 | – | 0.746 | 0.0355 |
| 0.9 | 0.1108 | +0.0007 | 0.799 | 0.0366 |
| 0.8 | 0.1109 | +0.0009 | 0.851 | 0.0382 |
| **0.7 (selected)** | **0.1099** | −0.0001 (CI contains 0) | **0.896** | 0.0390 |
| 0.5 | 0.1045 | −0.0055 | 0.958 | 0.0414 |
| 0.3 | 0.0953 | −0.0148 | 0.991 | 0.0427 |

### Stage 7 — pre-registered simulated A/B (`make stage7` → `results/stage7_ab.json`, quality/simulated)
1,411 fully-observed users, 5 slates × 10, primary metric = positives per impression; bootstrap 2,000×.

| variant | between-subjects arm (n) | within-subjects (all users) [95% CI] | slate-1 NDCG@10 (small matrix) | big-matrix test NDCG@10 |
|---|---|---|---|---|
| V1 popularity | 0.155 (296) | 0.158 [0.152, 0.164] | 0.156 | 0.0430 |
| V2 ALS | 0.221 (289) | **0.214** [0.208, 0.220] | 0.387 | 0.0162 |
| V3 two-tower | 0.207 (293) | **0.213** [0.206, 0.219] | 0.406 | 0.0803 |
| V4 recall + rank | 0.186 (256) | 0.179 [0.172, 0.186] | 0.325 | 0.1100 |
| V5 + MMR λ=0.7 | 0.182 (277) | 0.180 [0.173, 0.187] | 0.322 | 0.1099 |

Pre-registered comparisons (between subjects, Holm): C1 V4−V3 −0.021 [−0.043, +0.001] p_holm 0.20; C2 V5−V4 −0.004
[−0.027, +0.019]; C3 V3−V2 −0.013 [−0.033, +0.007]; C4 V2−V1 **+0.066 [0.047, 0.085]**, p_holm 7e-11 (only significant one).
Within subjects: V4−V3 −0.034 [−0.037, −0.030]; V5−V4 +0.0013 [0.0004, 0.0021].
Rank agreement with the online order: big-matrix NDCG@10 Kendall τ = −0.4 (7/10 pairs disagree); small-matrix slate-1
NDCG@10 τ = +0.6.

### Stage 8 — rollout, cache, rebuild (`make stage8` → `results/stage8_rollout.json`, `results/logs_stage8.txt`)
Cluster + Postgres; workers archive events. Shadow: 150 simulated users × 1 slate. Canary: 10% of users by hash, all
1,411 small-matrix users × 3 slates (16 concurrent sessions), decision rule in `echostream/rollout/controller.py`.

| version | change | shadow | canary vs control | decision |
|---|---|---|---|---|
| v2 | two-tower retrained, seed 1 | overlap@10 0.69, 0 errors | pos/impr 0.333 vs 0.343, CI [−0.069, +0.044]; p99 44.7 vs 52.1 ms | promoted |
| v3-slow | 2,000 × 255-leaf ranker | p99 207–427 ms per replica | degraded 185/312 vs 7/1,800; p99 199.7 vs 120.4 ms | **auto-rollback** |
| v4-plain | no logQ two-tower (gate bypassed) | overlap@10 0.43 | pos/impr 0.349 vs 0.208, CI [+0.067, +0.219] | promoted (offline gate would block) |

Caveat: each canary arm has ~100–130 simulated users with at least one observed impression (KuaiRec's small matrix covers
3,327 of 10,728 videos, so many recommended videos have no simulated response), and the same users are reused across the
three rollouts.

Cache (Zipf s = 1.1 over users, 100 rps, 30 s, through nginx): TTL 0 → p50 10.27 / p99 21.18 ms; TTL 30 s (±20% jitter) →
hit rate 71.7%, p50 3.06 / p99 81.35 ms, 849 pipeline computations for 3,000 requests. 200 concurrent requests for one
uncached user → 1 computation. Rebuild of the feature store from 11,381 archived events: 0.22 s.

### Stage 9 — regression gate + agent (`make release-guard` → `results/stage9_release_guard.json`, quality)
Baseline run NDCG@10 0.1100, recall@200 0.2327 (6,607 users). Threshold = max(paired CI half-width, seed std from Stage 2:
NDCG@10 0.0018, recall@200 0.0035).

| candidate | kind | ΔNDCG@10 | Δrecall@200 | gate | agent root cause |
|---|---|---|---|---|---|
| logQ removed | regression | −0.0013 | −0.0599 | blocked | retrieval_model_regression ✓ |
| stale item features (Aug 15) | regression | −0.0714 | 0 | blocked | stale_shard_index ✗ |
| two-tower quota 0 | regression | +0.0010 | −0.0158 | blocked | unknown ✗ |
| ranker label bug (wr > 0.5) | regression | −0.0238 | 0 | blocked | ranker_regression ✓ |
| shard 2 stale vectors | regression | −0.0002 | +0.0011 | passed (missed) | – |
| two-tower seed 1 | harmless | +0.0002 | +0.0039 | passed | – |
| ranker reseed | harmless | −0.0023 | 0 | blocked (false alarm) | ranker_regression |
| quota tweak | harmless | 0.0000 | +0.0018 | passed | – |
