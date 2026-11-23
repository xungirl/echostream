# BENCHMARK.md — every number, how it was measured, where the raw output is

**Machine:** Apple M2 (8 cores: 4 performance + 4 efficiency), 24 GB RAM, macOS (Darwin 27.2), Python 3.12.14 (uv),
PyTorch 2.15 (MPS for training), faiss-cpu 1.16, LightGBM 4.7, implicit 0.7.3, Redis (Homebrew), nginx (Homebrew),
PostgreSQL 16, Go 1.27.4 + go-redis v9.23 (Stage 12). **Date:** 2026-11-22 (Stage 12: 2026-11-23). Everything (load generator included) runs on this one laptop.

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
(faiss Flat) · 3 Redis feature nodes (consistent hashing) · 1 Redis event bus (8 stream partitions, AOF) · 4 workers
(Python in Stages 5–11; Stage 12 compares Python and Go and makes Go the default) ·
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

### Stage 12 — Go event worker vs Python worker (`make bench-go` → `results/stage12_go_worker*.json`, systems/real-replay)
Same algorithm in both languages (see `go/worker/`, parity below); only the worker process changes. Every run uses all
**12,530,806** KuaiRec events, 3 repeats per cell, Python and Go interleaved inside each repeat, started only when no
other echostream cluster ran. Raw rows: `results/stage12_go_worker_runs.jsonl` (one JSON line per run, incl. per-process
CPU seconds and RSS from psutil, every Redis' CPU, the timeline); log: `results/logs_stage12_go_worker.txt`; summary
(median, min, max and every run): `results/stage12_go_worker.json`. Every single run: **0 / 7,176 users with a counter
mismatch**, 0 users with a wrong history length, 0 DLQ.

**Two harnesses.** *drain* (`scripts/stage12_go_bench.py drain`): the 8 partition streams are pre-filled with exactly the
entries POST /events writes (snapshot restored into a fresh AOF-on event bus, 1.13 GB), then N workers start and the clock
runs until every stream and pending list is empty — worker + Redis capacity, nothing else running. *replay*: the Stage 5
harness unchanged (`scripts/stage5_replay.py --worker-impl … --port-base 37000`: client → nginx → 4 API replicas → streams
→ 4 workers → 3 feature nodes, backpressure at 400k unprocessed entries).

| harness | workers | Python events/s, median [min–max] | Go events/s, median [min–max] | Go ÷ Python, same repeat | worker CPU s / 1M events, Py → Go (median) | worker RSS max, Py / Go |
|---|---|---|---|---|---|---|
| drain | 1 | 18,277 [16,105–18,295] | 72,241 [58,636–76,054] | 3.6–4.2× | 49.4 → 7.7 | 55 / 35 MB |
| drain | 2 | 29,813 [28,293–35,061] | 97,574 [95,985–122,290] | 3.3–3.5× | 58.8 → 10.0 | 47 / 30 MB |
| drain | 4 | 37,450 [35,952–52,752] | 103,040 [95,836–155,325] | 2.6–2.9× | 91.8 → 12.7 | 53 / 29 MB |
| drain | 8 | 50,490 [37,552–65,208] | 149,539 [94,708–156,298] | 2.4–3.0× | 95.8 → 10.9 | 53 / 27 MB |
| **replay (Stage 5 harness)** | 4 | **42,276 [41,670–42,295]** | **85,306 [83,231–85,370]** | **2.0×** (2.00–2.02) | **74.3 → 12.6** | 43 / 28 MB |

Event → feature latency (worker histogram, 40 buckets ≤ 1.5× apart; the true value lies in the bucket shown; same in all
3 repeats unless a second bucket is listed), Stage 5 replay harness, 4 workers:

| load | Python p50 / p99 | Go p50 / p99 | worker CPU s / 1M events, Py → Go |
|---|---|---|---|
| paced 10,000 events/s (first 1.2 M events) | 30–40 / 40–50 ms | 20–30 / 30–40 ms (one run 40–50) | 62.6 → 15.8 |
| paced 25,000 events/s (first 2.5 M events) | 20–30 / 40–50 ms (one run 50–75) | 15–20 / 30–40 ms (one run 20–30) | 59.9 → 13.9 |
| max rate (full replay, backlog-dominated) | 10–15 s / 15–30 s | 3–5 s / 7.5–15 s | — |

At a paced load both workers keep up, so latency is dominated by the producer's 1,000-event HTTP batches and the 50 ms
blocking read, not by processing; Go is one bucket lower. At max rate the latency is queueing behind the 400k backpressure
window, so it mostly measures throughput again.

**Chaos** (`drain --kill-at 20`, 4 workers): at t = 20 s the worker with the most un-acked entries is `kill -9`ed (1,000
pending in every run) and restarted 0.3 s later, as the supervisor does. Go ×3: 151,834 / 152,080 / 152,272 events/s
(the kill costs about the 3 s lease TTL on two partitions), **0 / 7,176 users mismatched in all 3**; Python ×1: 52,421 events/s, 0
mismatched. `claimed` / `duplicates`: Go 2,000/500, 500/474, 2,000/0; Python 1,000/0 — explained below.

**Where the time goes (pprof, `results/stage12_go_worker_pprof_4w.pb.gz`, 30 s of one of 4 Go workers in a 155k events/s
drain):** the worker was on-CPU 9.95 s of 30 s (≈ 33 % of one core). Of that, syscalls (socket read/write) 26 %, goroutine
wake-ups (`pthread_cond_signal`) 20 %, GC `madvise` 9 %, `kevent` 9 %; parsing, hashing and formatting together are a few
percent. In the same run the single-threaded event-bus Redis used 62 s of CPU in 81 s (77 % busy) and each feature node
≈ 53 %. So the Go worker is **round-trip / Redis bound**, not CPU bound: one worker already does 72k events/s, and 4 → 8
workers adds little because the event bus (≈ 5 µs of Redis CPU per event → a ceiling near 200k events/s) is the next
limit. The Python worker is **CPU bound**: 49–96 CPU-seconds per 1M events, so it scales with processes (18k → 50k from
1 → 8 workers) until the 8 cores (4 performance + 4 efficiency, shared with 4 Redis) run out. In the full Stage 5 harness
Go doubles throughput; there the 4 API replicas (≈ 2.4 cores) and the event bus (83 % busy, it also does the XADDs) share
the machine, and the backlog still sits at the 400k cap, i.e. the event path is still the limit, now on Redis + CPU rather
than on the worker. A wasted 1 % in Go: `ParseInt`/`ParseFloat` build their (Python-compatible) error text even when
parsing succeeds; left as is so the measured binary is the committed code.

**Notes on three observations.**
- *(a) Go chaos `claimed` / `duplicates` vary, Python's were 1,000 / 0.* Not a Go-specific bug, and no entry was ever taken
  from a worker that was still processing it. Two separate effects, both timing-dependent and both present in the shared
  algorithm: (1) `claimed` also counts **start-up rebalancing**. A Go process is up in ~10 ms, so the first worker can
  register alone, take its "fair share" of all 8 partitions, read a 500-entry batch from each, and give 6 back on the next
  heartbeat; `handle()` sees the lost lease and leaves those entries pending, and the new owners wait (XPENDING gate) until
  they are 1 s idle and then XAUTOCLAIM them. Re-running the chaos drain on 3M events with a temporary log line in
  `claim()` showed exactly this: new owners logged "blocked by w0=500" every ~55 ms for ~1 s, then claimed 500 per moved
  partition (3,000 at start-up), and the restarted worker re-claimed only the killed worker's own entries. Python
  sweep runs show the same start-up claims (Python ×4 r0: 3,000; ×8: 2,000 / 500 / 1,000). (2) `duplicates` in a chaos
  run are entries of the killed worker whose Lua script had already run but whose XACK had not been sent when SIGKILL
  landed: 500 = one partition's whole batch, 474 = a batch killed mid-pipeline, 0 = killed between batches. They are
  redelivered and the dedupe key turns them into no-ops — that is at-least-once delivery with an exactly-once effect, and
  the final counts were exact every time. The parity claim is about final state (feature store, DLQ, archive, and the
  counters of crash-free, already-balanced phases), which is identical; how many entries a crash or a start-up race
  happens to strand is timing, not behaviour.
- *(b) Python ×8 r1: 184 duplicates without a kill.* `claimed` = 500 in that run, so one partition changed owner while
  entries were pending. With 8 CPU-bound Python processes plus 4 Redis on 8 cores, a worker that had already passed the
  ownership check in `handle()` can take longer than `claim_min_idle` (1 s) to finish its 500-entry batch; the new owner
  then XAUTOCLAIMs entries the old owner is still applying, and 184 of them were applied twice — the second time a no-op
  thanks to the dedupe key (0 mismatched users). This is the "zombie window" already listed as a known limit (INTERVIEW Q21):
  per-user order is not guaranteed during that window, counts are. It exists in Go too, but a Go batch takes milliseconds,
  so it has not been observed there. A fencing token checked inside the Lua script would close it; not done.
- *(c) Up to ~60 % run-to-run spread in multi-worker drains (e.g. Go ×4: 103k / 96k / 155k).* Not the workers: in the slow
  runs **Redis needed ~50 % more CPU time for exactly the same work** (event bus 7.3–7.7 µs of CPU per event vs 5.0 in the
  fast runs; same for the feature nodes), and Python and Go slowed down in the same time window (runs started 00:41–01:37; every
  run after 02:00 is fast and tight — the three Go chaos runs land within 0.3 % of each other at 152k). Same work taking more
  CPU time means slower cores: the M2 has 4 performance + 4 efficiency cores, and with 4 Redis + N workers + anything else
  on the machine the scheduler sometimes runs a Redis instance on an efficiency core (or the chip clocks down when hot).
  The contention check only looks for other echostream clusters, so unrelated load during that window cannot be excluded (another
  agent was using this machine for a screen recording that night). Hence: multi-worker numbers are given as ranges, and the
  Go ÷ Python ratio is computed within a repeat (the two runs are minutes apart and see the same conditions); that ratio
  varies far less than the absolute numbers.

**Parity** (`make parity` → `results/stage12_go_parity.json`; also `tests/integration/test_go_parity.py` in `make ci`): one
deterministic event set with explicit stream ids, fresh Redis ×4 + fresh Postgres per implementation, three phases
(ghost-consumer takeover incl. entries deleted while pending + 16 kinds of poison; 3 balanced workers + out-of-order
re-sends + NaN/inf partial writes; kill -9 of a worker frozen mid-batch). Python vs Go: **14,802 feature-store keys,
28 DLQ entries (all fields incl. reason text), phase-1/phase-2 counters and 14,198 archive rows identical**, and both equal
an independent ground truth. `tests/integration/test_event_pipeline.py` (no loss / no double count / order / DLQ / crash /
ghost consumer / frozen feature node) runs against Python, Go and a mixed cluster.

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

---

## Stage 13 — serverless browser demo (`web/src/demo/`, systems + correctness)

**Machine:** the same Apple M2 (8 cores, 24 GB), macOS Darwin 27.2; Node v25.9.0; headless Chromium 140.0.7339.16 via
Playwright 1.55.0; Python 3.12.14, numpy 2.2.6, LightGBM 4.7.0, faiss 1.16.0. **Date:** 2026-11-23. Load average ~2 during
the browser runs (no echostream cluster running). All browser times are **computation in the page's main thread**, not
server latency; nothing in this stage touches the cluster.

### Parity: TypeScript engine vs the Python read path (`uv run python scripts/gen_browser_golden.py`, then `cd web && npx vitest run src/demo/engine/parity.test.ts`; numbers: `cd web && npx vite-node scripts/parity-report.ts` → `results/stage13_parity.json`)
Python side: the shipped bundle decoded into the production `ServingBundle`, the real `Recommender` (MMR λ = 0.7), the real
shard-server apps (4 shards, faiss Flat) over an in-memory ASGI transport, the real `Worker.process_once()` applying events
with the Lua script on a temporary redis-server. 50 users (43 sampled with seed 13, the 6 demo users, the brand-new user) ×
6 screens (first screen, then like / skip / watch / like / skip on the current screen) = 300 screens.

| check | result |
|---|---|
| decoded arrays (all 42), sha256 of the float32 / int32 values | identical to the Python reader |
| user vector | max abs diff 8.9e-8 |
| two-tower / pop_platform lists (200 each) | 300 / 300 identical (ids and order) |
| ItemCF / pop_recent / content lists | 282 / 162 / 281 identical; **300 / 300 equivalent** (differ only among tied items, see below) |
| path scores | max abs diff 1.8e-7 (two-tower), 0 (ItemCF, popularity), 8.9e-8 (content) |
| our merge of Python's 5 lists | 300 / 300 identical (ids, claiming path, every proposing path) |
| our features on Python's candidates (4,000 rows × 40) | 93.4% bit-identical, max abs diff 6.0e-7 (`rt_max_sim`); 35 of 40 features always exact |
| our trees on Python's features vs LightGBM's probabilities | max abs diff 1.4e-17 |
| our MMR on Python's ranked list | 300 / 300 identical screens |
| **end to end (our engine alone): the 10 shown videos** | **300 / 300 identical, same order** |
| end to end: merged 200 identical | 246 / 300 (the rest differ only in which tied tail items the paths kept → `in_pop_recent` etc.) |

"Equivalent": position by position the scores agree within 1e-6 and every item scoring more than 1e-6 above the 200th is
in both lists. Two causes, both inspected: numpy's `argpartition` keeps an arbitrary subset of items tied with the 200th
score (popularity scores have only 365 distinct values; quantized ItemCF sims and duplicate captions also tie) while the
browser keeps the lowest ids; and two items whose float32 dot products differ by one rounding step swap places (BLAS
sums in another order). Python was not changed.

### Storage presets vs float32 (`make quant-sweep` = `uv run python scripts/stage13_quant_sweep.py` → `results/stage13_quant_sweep.json`, 4.6 min)
Every config is exported, read back and run through the real Python read path on 50 sampled users + the brand-new user
(6 screens each); the float32 run's events are replayed, so both see identical inputs. Check first: the float32 browser
bundle reproduces the serving bundle itself on 300 / 300 screens with bit-identical ranker scores. Two full runs gave
identical numbers. "transfer" = model.bin.gz as stored + the JSON files gzipped (what a host that gzips JSON sends).

| config (item vectors / id emb / caption / ItemCF sims × neighbours) | transfer | identical screens | mean top-10 overlap | same #1 |
|---|---|---|---|---|
| f32 / f32 / f32 / f32 × 200 (= serving bundle) | 32.9 MB | reference | | |
| f32 / f32 / **f16** / f32 × 200 | 22.7 MB | 98.0% (300/306) | 0.9997 | 100% |
| f32 / f32 / **int8** / f32 × 200 | 16.9 MB | 65.4% | 0.9856 | 99.4% |
| f16 / f16 / f16 / f16 × 200 (`--quant f16`) | 17.8 MB | 96.4% | 0.9993 | 100% |
| **int8 / int8 / int8 / u8 table × 200 (`--quant int8`, shipped)** | **10.0 MB** | **52.3% (160/306)** | **0.9778** | **99.4%** |
| f16 / f16 / int8 / u8 × 200 | 10.8 MB | 59.2% | 0.9830 | 99.4% |
| f32 / f32 / int8 / u8 × 200 | 12.8 MB | 59.5% | 0.9833 | 99.4% |
| f16 / f16 / int8 / u8 × **100** | 9.4 MB | 2.0% | 0.866 | 85.6% |
| f16 / f16 / int8 / u8 × **50** | 8.5 MB | 0.7% | 0.826 | 85.3% |

Caption PCA (measured separately on 200 random 10-video queries, rms error of the caption dot products): int8 at 512
dimensions 4.4e-4; PCA to 256 / 128 / 64 dimensions 2.9e-3 / 9.8e-3 / 1.6e-2 (63–96% variance kept) — not tried in the
pipeline. The 5 MB target is therefore **not met**; the shipped bundle is 10.0 MB as transferred (10.95 MB on disk,
model.bin 11.3 MB before gzip). The bundle is committed so `make demo-browser` needs no data download.

### Browser: time to interactive and per-screen compute (`make bench-browser` = `cd web && npm run build:demo && MEASURE=1 npx playwright test e2e/measure.spec.ts` → `results/stage13_browser_demo.json`, every run in `results/stage13_browser_demo_runs.jsonl`; 3 runs)
The built site is served by `web/e2e/serve.mjs` under `/echostream/demo/` (JSON / JS gzip-encoded, model.bin.gz as is).

| metric | run 1 | run 2 | run 3 |
|---|---|---|---|
| time to interactive, localhost, fresh cache, median of 5 loads | 226 ms | 228 ms | 228 ms |
| … of which download / gunzip + decode (typical load) | 97 / 65 ms | | |
| time to interactive, simulated 20 Mbit/s + 40 ms latency (Chromium throttling), median of 3 | 4.38 s | 4.38 s | 4.38 s |
| per-screen compute, 7 viewers × 120 recomputes (each after one reaction): median / p95 | 11.8 / 12.7 ms | 11.7 / 12.7 ms | 11.8 / 12.7 ms |
| … min / max | 9.6 / 20.4 ms | 9.6 / 20.2 ms | 9.6 / 21.2 ms |

Time to interactive = navigation start → first screen rendered (model fetched, decompressed, decoded, first
recommendation computed). Per viewer (run 3) the median ranges from 11.2 ms (Dance viewer, Anime viewer) to 12.4 ms
(Pets · Comedy viewer); the brand-new viewer is 11.9 ms. On a real phone or a slower network both numbers will be higher;
only this laptop was measured.
