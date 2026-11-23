<p align="center">
  <!-- <img src="docs/brand/logo.svg" alt="echostream logo" width="120" /> -->
</p>

<h1 align="center">echostream · 回响流转</h1>

<p align="center">
  <b>Every swipe echoes.</b><br/>
  每一次划动，都有回响。
</p>

<p align="center">
  A distributed short-video recommender driven by real-time user feedback.<br/>
  基于实时行为反馈的分布式短视频推荐系统
</p>

---

## What is this? · 这是什么

**English**
echostream is a scaled-down replica of the architecture behind a Douyin / TikTok-style "For You" feed. It is a backend service, not a mobile app. A client asks "what should user 42 see next?" and gets a ranked list back within a strict latency budget. When the user watches or skips a video, that event flows back through the system and changes the very next recommendation. That loop is the echo.

It runs on **KuaiRec**, a real short-video interaction dataset released by Kuaishou, not synthetic data. Every number in this repo comes from a run on real hardware, and each one is labeled as either a quality result or a systems test.

**中文**
回响流转是抖音、TikTok「推荐页」背后架构的缩小版。它是一个后端服务，不是手机 App。客户端问「用户 42 下一屏该看什么」，它在严格的延迟预算内返回一个排好序的推荐列表。用户看完或划走一个视频，这个行为会流回系统，马上改变下一次推荐。这个闭环就是「回响」。

数据用的是快手公开的真实短视频数据集 **KuaiRec**，不是模拟数据。仓库里每个数字都来自真实运行，并且标明它是推荐质量结果，还是系统测试结果。

---

## How a request flows · 一次请求怎么走

```
  client                         echostream
 ┌────────┐  GET /recommend   ┌──────────────────────────────────────────────┐
 │  feed  │ ────────────────▶ │  API (stateless, replicated)                 │
 │        │                   │   ├─ recall (parallel, per-path timeouts)    │
 │        │                   │   │    two-tower ANN ── scatter-gather ──▶ index shards (+replicas)
 │        │                   │   │    ItemCF · trending · cold-start content │
 │        │                   │   ├─ features ◀── sharded real-time store    │
 │        │                   │   ├─ rank   (LightGBM)                       │
 │        │ ◀──────────────── │   └─ re-rank (MMR diversity)                 │
 │        │   items + timings │                                              │
 │        │                   │                                              │
 │        │  POST /events     │  Redis Streams (partitioned by user)         │
 │ watch/ │ ────────────────▶ │   └─ consumer group workers ── update ──▶ real-time features
 │  skip  │                   │      (at-least-once · idempotent · DLQ)      │
 └────────┘                   └──────────────────────────────────────────────┘
                                 offline: train · evaluate · regression gate · canary
```

**English**: recall → rank → re-rank on the read path; events → stream → real-time features on the write path. The two paths meet in the feature store, which is what makes the next swipe different.

**中文**：读路径是「召回 → 排序 → 重排」，写路径是「事件 → 事件流 → 实时特征」。两条路径在特征存储里交汇，所以下一屏会变。

---

## What's inside · 包含什么

| Layer · 层 | English | 中文 |
|---|---|---|
| Evaluation | Time-based split, Recall@K / NDCG@K, popularity · ItemCF · ALS baselines, bootstrap CIs | 按时间切分、Recall@K / NDCG@K、热门 / ItemCF / ALS 基线、自助法置信区间 |
| Recall | Two-tower model with logQ correction + ANN, ItemCF, trending, cold-start from captions | 带 logQ 修正的双塔 + 向量检索、ItemCF、热门、基于标题的冷启动 |
| Ranking | LightGBM over user, item, affinity and real-time features | LightGBM，用用户、视频、兴趣匹配和实时特征 |
| Re-ranking | MMR for diversity | MMR 多样性重排 |
| Distributed serving | Sharded vector index with replicas, scatter-gather with deadlines, partitioned event stream, sharded feature store, graceful degradation | 带副本的向量索引分片、带截止时间的并行查询合并、分区事件流、分片特征存储、逐级降级 |
| Release safety | Shadow → canary → automatic rollback, eval regression gate | 影子运行 → 灰度 → 自动回滚，评测回归门禁 |
| Agent | A tool-using agent that investigates gate failures and writes incident reports | 门禁触发时自动排查原因、写事故报告的 agent |
| Experiments | Simulated online A/B on the fully-observed matrix | 在全观测矩阵上做模拟线上 A/B |
| Demo | Web feed with live per-stage latency waterfall | 网页推荐流 demo，实时显示每个阶段的耗时 |

---

## Honest scope · 诚实说明

**English**
- Everything runs as **multiple nodes on one laptop** (localhost). Fault tolerance, tail latency and event reliability are tested for real; horizontal throughput scaling is not claimed.
- KuaiRec is small (~10K videos). Recommendation quality is measured on real data only. Systems tests use either **real event replay** or clearly labeled **synthetic scale-up**.
- KuaiRec has no video files, so the demo shows video cards (caption, category), not playback.

**中文**
- 所有组件都是**在一台笔记本上运行的多个节点**（localhost）。容错、尾延迟和事件可靠性是真实测出来的；不声称吞吐能随机器数量线性扩展。
- KuaiRec 规模不大（约 1 万个视频）。推荐质量只在真实数据上评测；系统测试要么用**真实事件重放**，要么用明确标注的**合成扩容**数据。
- KuaiRec 没有视频文件，所以 demo 显示的是视频卡片（标题、分类），不能播放。

---

## Roadmap · 路线图

- [x] Stage 0 · Data & labels · 数据与标签定义
- [x] Stage 1 · Offline evaluation & baselines · 离线评测与基线
- [x] Stage 2 · Two-tower retrieval + ANN · 双塔召回 + 向量检索
- [x] Stage 3 · Multi-path recall & cold start · 多路召回与冷启动
- [x] Stage 4 · Ranking · 排序
- [x] Stage 5 · Distributed serving, load & chaos tests · 分布式服务、压测与故障注入 — **MVP**
- [ ] Stage 6 · MMR re-ranking · MMR 重排
- [ ] Stage 7 · Simulated online A/B (pre-registered) · 模拟线上 A/B（预注册）
- [ ] Stage 8 · Postgres, caching, shadow / canary / rollback · Postgres、缓存、影子 / 灰度 / 回滚
- [ ] Stage 9 · Release-guard agent · 发布守门 agent
- [ ] Stage 10 · Web demo · 网页 demo
- [ ] Next · E-commerce domain adapter (Amazon Reviews 2023) · 电商领域适配器

---

## Project Journey · 项目历程

Each stage: goal → decision & alternatives → built → verified → measured → problems/fixes. Numbers are copied from the
`results/` file named next to them (BENCHMARK.md has the exact commands). 每个阶段下面有一段中文小结。

### Stage 0 — Data & labels
- **Goal:** get KuaiRec, verify every assumption, define "liked" without a like signal.
- **Decision & alternatives:** positive = `watch_ratio > 2.0` (dataset authors' suggestion). `>1.0` marks 33.8% of plays positive
  (mostly "short video looped once"); `>3.0` drops evaluable test users from 6,607 to 5,344. Sensitivity kept in DATA.md.
- **Built:** `make data` (Zenodo download, SHA-256 for the zip and all 7 CSVs, parquet), `DomainAdapter` interface + KuaiRec adapter.
- **Verified / measured:** 12,530,806 big-matrix rows, 7,176 users × 10,728 videos, density 0.163; small matrix 99.62% observed;
  0 (user, video) pairs shared between the two matrices.
- **Problems → fixes:** the big matrix is three date windows, not one continuous log; **86.3%** of test-window positives are
  videos absent from the train window (catalog drift) → test-phase models are refit on all data before Sep 1, tuning stays on val.
  The label is biased toward short videos (positive rate 18.1% for <7 s vs 0.2% for >60 s) — documented, not hidden.
- 中文：数据下载校验、正样本定义（watch_ratio>2）和敏感性；发现三段日期窗口和严重的目录漂移，改成「val 调参、test 前用全部历史重训」。

### Stage 1 — Offline evaluation & baselines
- **Goal:** a leak-free yardstick before any model.
- **Decision:** temporal split by logged date; metrics Recall/NDCG/HitRate/MRR @ {10, 50, 100, 200}, coverage, category diversity,
  95% bootstrap CIs over users; a second, exposure-bias-free evaluation on the fully-observed small matrix.
- **Built:** `core/metrics.py`, `core/split.py`, hand-written ItemCF, popularity (3 kinds), ALS.
- **Measured (quality, `results/stage1_baselines.json`):** best test NDCG@10 = recent popularity 0.0430 [0.0410, 0.0451];
  ItemCF 0.0181, ALS 0.0162. On the small matrix the order flips: ItemCF 0.618, global popularity 0.561.
- **Lesson:** which baseline "wins" depends on exposure bias and catalog drift; both evaluations are kept.
- 中文：时间切分 + 自助法置信区间；近期热门在时间切分上最强，全观测小矩阵上 ItemCF 最强——评测方式会改变结论。

### Stage 2 — Two-tower retrieval + ANN
- **Goal:** personalized retrieval that scales.
- **Decision & alternatives:** in-batch sampled softmax with logQ correction; ablated logQ and 1,024 mixed uniform negatives;
  temperature tuned on val (0.1). Item tower = id + categories + bge-small-zh caption embedding, with 30% id-dropout for cold items.
  User tower exported to numpy so serving does not need PyTorch.
- **Measured (quality, `results/stage2_two_tower.json`):** test NDCG@10 **0.0803** [0.0775, 0.0831], +0.0372 [0.0349, 0.0397]
  over the best baseline (paired). Without logQ and mixed negatives: 0.0071. Seed std 0.0018.
- **ANN (`results/stage2_ann.json`):** on the real 10,728 vectors exact Flat search is 92 µs p50 (1 thread); HNSW M=32/ef=128
  reaches recall@100 0.955 at 47.5 µs. → serving uses exact Flat shards; ANN trade-offs are shown at synthetic scale (Stage 5).
- **Problems → fixes:** faiss, torch, LightGBM and implicit each bundle `libomp`; loading torch with any of the others in one
  process aborts on macOS → rule "torch never shares a process with them", training runs in subprocesses.
- 中文：带 logQ 的双塔把 NDCG@10 从 0.043 提到 0.080；去掉 logQ 掉到 0.007。1 万级向量精确检索只要 0.09 ms，所以分片用精确索引。

### Stage 3 — Multi-path recall & cold start
- **Goal:** candidates that cover personalization, trends and brand-new videos.
- **Built:** five paths (two-tower, ItemCF, recent popularity, platform popularity, caption-content restricted to cold videos),
  quota merge with dedup that remembers every path that proposed an item.
- **Measured (`results/stage3_multipath.json`):** val-selected quotas give merged recall@200 0.233 on test — *lower* than two-tower
  alone (0.271), because on val (catalog drift) the two-tower was weak and the mix looked better. The content path finds 21.3% of
  the cold-video truth but got quota 0. Not re-tuned on test.
- 中文：多路召回的配额在 val 上选；test 上反而不如双塔单路——如实记录，没有在 test 上重调。

### Stage 4 — Ranking
- **Goal:** order ~200 candidates well.
- **Built:** one feature builder shared by offline training and the online API (40 features in 5 groups), LightGBM, AUC/GAUC.
- **Problem → fix:** v1 trained on the val phase *lost* to recall-only ordering on test (NDCG@10 0.0696 vs 0.0803): its two-tower
  score feature came from a model 3 weeks stale relative to its labels, unlike at serving time (train/serve skew). v2 trains on a
  "rank phase" whose recall models are refit right before the labels (08-30…31). This protocol was designed **after** seeing v1's
  test result; both runs are kept.
- **Measured (`results/stage4_ranker.json`):** recall+rank NDCG@10 **0.1100** [0.1064, 0.1136], +0.0297 [0.0267, 0.0325] over
  recall-only; AUC 0.861, GAUC 0.813. Every feature group helps (item features most: −0.0415 when removed).
- 中文：排序第一版因为训练阶段的召回分数分布和线上不一致而变差；改训练协议后 NDCG@10 0.110（+0.030）。

### Stage 5 — Distributed serving core (MVP)
- **Goal:** serve the Stage-4 pipeline with the mechanisms a real feed backend needs, and measure them under failure.
- **Decisions & alternatives:** a ~400-line process supervisor instead of docker compose (ms-level kill/freeze/delay from
  tests, no VM network inside latency numbers); nginx as the load balancer (retries the other replica on connect errors);
  Redis Streams with **leased partitions** (one owner per partition → per-user order) instead of a plain consumer group
  (which would interleave a user's events across workers); a Lua script that does dedupe + apply atomically on the user's
  feature node (exactly-once effect over at-least-once delivery); exact Flat shards (offline == online, tested).
- **Built:** shard servers (+ replicas, delay/stale hooks, deadline header), scatter-gather client with per-attempt timeouts,
  replica failover and outlier ejection, `/recommend` with a 250 ms budget and labeled degradation, `/events`, workers,
  sharded feature store, Prometheus metrics, open-loop load generator, chaos runner, replay runner, synthetic scale runner.
- **Verified:** integration tests (`tests/integration/`) — no loss / no double count / per-user order with duplicate sends, DLQ, XAUTOCLAIM
  takeover of a dead consumer's 600 pending entries, feature-node stall, online ANN == offline exact search, an event changes
  the next recommendation, shard/replica/API failures, ranker timeout and budget fallbacks.
- **Measured:**
  - *systems* (`results/stage5_load.json`): 200 rps → p50 12.1 ms, p99 26.3 ms, 0 errors; 250 rps saturates the laptop.
  - *systems* (`results/stage5_chaos.json`): 100% availability in all 15 fault runs (kill shard primary, slow shard +500 ms,
    kill worker, stall a feature node, kill an API replica; 3 repeats each); a stalled feature node degrades the ~1/3 of
    users it owns (33–37% of responses carry `feature_store_unavailable`) instead of failing them.
  - *real-replay* (`results/stage5_replay_full.json`): all 12,530,806 KuaiRec events in time order through
    nginx → API → streams → workers → feature store at **37,221 events/s**; **0** of 7,176 users with a counter mismatch,
    0 duplicates, 0 DLQ. Paced at 10k events/s: event → feature latency p99 ≤ 50 ms (`stage5_replay_paced10k.json`).
  - *synthetic* (`results/stage5_synthetic.json`): 1M–10M random vectors over 1–8 shards: per-shard memory drops linearly (10M IVF: 2.5 GB → 315 MB of vectors per shard), but p50 latency does not improve on one laptop (2.3 → 6.0 ms from 1 to 8 shards: fan-out RPC cost dominates).
- **Problems → fixes:** per-request CPU (≈5 ms, 2.4 ms of it LightGBM) capped 2 API replicas at ~150 rps → binary (base64)
  shard payloads, uvloop/httptools, 4 replicas. A worker started while Redis was still loading its AOF died silently and
  sat idle → startup retry + exit-on-crash so the supervisor restarts it; benchmarks now start from a fresh data dir.
  The first chaos summary used per-second p99 for "recovery", which at 100 rps is a single request → added a p50-based rule
  (both kept).
- 中文：自写进程管理器 + nginx + 分片向量检索（副本、超时、摘除）+ 分区事件流（租约保证用户内有序、Lua 幂等）+ 分片特征库。
  200 rps 下 p99 26 ms；15 次故障注入全部 100% 可用；1,253 万条真实事件重放 3.7 万条/秒、逐用户计数零误差。

## Key metrics · 关键指标

| metric | value | kind | how measured | source file |
|---|---|---|---|---|
| NDCG@10, recall + rank (test, 6,607 users) | 0.1100 [0.1064, 0.1136] | quality | temporal split, new positives Sep 1–5, 1,000× user bootstrap | `results/stage4_ranker.json` |
| … gain over recall-only (paired) | +0.0297 [0.0267, 0.0325] | quality | paired bootstrap, same users | `results/stage4_ranker.json` |
| … gain over best baseline (recent popularity 0.0430) | +0.0670 [0.0642, 0.0700] | quality | paired bootstrap | `results/stage4_ranker.json` |
| two-tower NDCG@10 with logQ / without logQ + mixed negs | 0.0803 / 0.0071 | quality | ablation, same protocol | `results/stage2_two_tower.json` |
| ranker AUC / GAUC | 0.861 / 0.813 | quality | test candidates (1.32 M rows) | `results/stage4_ranker.json` |
| small-matrix NDCG@10 ItemCF vs two-tower | 0.618 vs 0.402 | quality | fully-observed 1,411 × 3,327 | `results/stage1_baselines.json`, `results/stage2_two_tower.json` |
| exact search, 10,728 real vectors, p50 | 92 µs | systems | faiss Flat, 1 thread, 2,000 queries × 3 | `results/stage2_ann.json` |
| /recommend at 200 rps: p50 / p99 / errors | 12.1 ms / 26.3 ms / 0 | systems | open-loop, 30 s × 3, through nginx | `results/stage5_load.json` |
| availability under 5 fault types × 3 runs | 100% (0 of 67,500 requests failed) | systems | kill/slow/stall at 100 rps | `results/stage5_chaos.json` |
| real-event replay throughput (12,530,806 events) | 37,221 events/s | real-replay | API → streams → workers → feature store | `results/stage5_replay_full.json` |
| replay correctness | 0 / 7,176 users with mismatched counters | real-replay | feature store vs parquet ground truth | `results/stage5_replay_full.json` |
| event → feature latency at 10k events/s | p99 ≤ 50 ms (bucket bound) | real-replay | worker histogram | `results/stage5_replay_paced10k.json` |
| 10M vectors: per-shard vectors at 1 → 8 shards | 2,519 → 315 MB | synthetic | computed size; RSS also stored | `results/stage5_synthetic.json` |

## Next steps · 下一步

- E-commerce `DomainAdapter` (Amazon Reviews 2023: positive = rating ≥ 4 / purchase, text = title) — serving/eval unchanged.
- Re-tune recall quotas per phase (or widen the candidate set) — the val-selected mix underperforms the two-tower alone on test.
- A real ANN index (HNSW/IVF) inside the shards once the catalog is ≥ 10⁶ items; a gRPC/binary scatter-gather client.
- Docker-compose packaging for multi-machine runs (the supervisor is single-host by design).

## Architecture (as built) · 架构

```
                              ┌──────────── supervisor (restart policy + chaos control API :17009) ───────────┐
 client / web UI / loadgen    │                                                                                │
        │  :17000             │   API replicas ×4 (FastAPI, stateless)        vector shards 4 × 2 replicas     │
        ▼                     │  ┌──────────────────────────────────┐  HTTP  ┌──────────┐ ┌──────────┐        │
   ┌─────────┐ round-robin    │  │ 1 features ◀── Redis feat0..2 ───┼──┐     │ shard0-r0│ │ shard0-r1│ ...    │
   │  nginx  │ + retry on     │  │   (consistent hash on user id)   │  │ ┌──▶│ faiss    │ │ faiss    │        │
   │   LB    │ dead replica ──┼─▶│ 2 user vector (numpy user tower) │  │ │   │ Flat, by │ │ (same    │        │
   └─────────┘                │  │ 3 recall: two-tower ANN ─────────┼──┼─┘   │ hash(item│ │  items)  │        │
        ▲                     │  │      scatter-gather + deadlines  │  │     │  ) % 4   │ │          │        │
        │                     │  │    + ItemCF / pop ×2 / content   │  │     └──────────┘ └──────────┘        │
        │                     │  │ 4 merge (quotas)  5 LightGBM rank│  │                                      │
        │ items + per-stage   │  │ 6 MMR (optional)  → degradation  │  │  Redis event bus (AOF)              │
        │ timings + labels    │  │   labels, Prometheus /metrics    │  │  events:0..7 (partition = hash(user))│
        │                     │  └──────────────┬───────────────────┘  │  consumer group "workers", DLQ       │
        │ POST /events ───────┼─────────────────┘ XADD ───────────────▶│        │ leases, XAUTOCLAIM          │
        │                     │                                         │        ▼                             │
        │                     │   workers ×4: Lua(dedupe + history + counters) on the user's feature node,   │
        │                     │   then (Stage 8) batch-archive to Postgres, then XACK + XDEL                 │
        │                     │                                                                              │
        │                     │   Postgres (Stage 8): users, videos, model registry, rollout state, archive  │
        │                     └──────────────────────────────────────────────────────────────────────────────┘
 offline (Python): KuaiRec → temporal split → baselines / two-tower (PyTorch MPS) / ItemCF / ranker (LightGBM)
                   → serving bundle (artifacts/serving/vN) → registry → shadow → canary → promote | rollback
```

Everything above is a separate OS process on **one laptop** (ports 17000–17140). "Node" = process, not machine.

## Design decisions & trade-offs · 设计取舍

| decision | why | alternative rejected / cost |
|---|---|---|
| Temporal split; tune on val, refit on train+val for test | random split leaks the future; catalog drift (86% new videos) makes a frozen model meaningless | test-phase models see 5 more days of data than val-phase models (documented) |
| Exact Flat index per shard | 10,728 vectors: 92 µs/query; offline == online exactly (tested) | HNSW/IVF only pay off at 10⁶+ vectors → measured separately on synthetic data |
| User tower exported to numpy | serving needs no PyTorch; avoids the torch/faiss/LightGBM libomp clash on macOS | a parity test must keep torch and numpy in sync |
| Shard filters the user's seen items | keeps top-k exact without over-fetching thousands of ids | 5.6 KB base64 payload per shard call |
| Per-attempt timeout = min(cap, remaining budget) + replica failover + outlier ejection | bounds tail latency of scatter-gather | a request can return partial ANN results (labeled) |
| Redis Streams, partition = hash(user), one leased owner per partition | per-user order without a global lock; at-least-once with XAUTOCLAIM | head-of-line blocking for a partition while one feature node is down |
| Lua script = dedupe + apply in one atomic step on the user's node | exactly-once *effect* on top of at-least-once delivery | dedupe keys cost memory (TTL) |
| Feature nodes without persistence; Postgres archive + rebuild | Redis holds derived data only; one source of truth | rebuild takes time proportional to the archive |
| Process supervisor instead of docker compose | ms-level kill/freeze from tests; no VM network in latency numbers | not a production orchestrator; single machine |
| One feature builder for offline training and online serving | removes a whole class of train/serve skew | features must be cheap enough for the request path |

## How to run · 运行

```bash
uv sync                      # Python deps (uv); web: cd web && npm install
make data                    # download KuaiRec (Zenodo) + checksums + parquet  (~2 min, ~2.2 GB on disk)
make train-all               # stages 0-4 + serving bundle (~25 min on an M2)
make ci                      # unit + integration + agent tests, web tests (no network, no LLM)
make cluster-up              # multi-node-on-localhost cluster on :17000 ; make cluster-down
curl '127.0.0.1:17000/recommend?user_id=42&n=10'
make demo                    # cluster in demo mode + web UI on http://127.0.0.1:5173 ; make demo-down
make load chaos replay synthetic   # Stage-5 benchmarks (each starts its own cluster)
make stage6 stage7 stage8 release-guard
make agent                   # release-guard agent, deterministic fake provider
DASHSCOPE_API_KEY=... make agent-real   # real LLM (qwen3.8-flash), hard cap ¥2 via a token cost ledger
uv run python scripts/make_drill.py --list   # rewrite-the-core-function drills (DRILLS.md)
```

## Limitations · 局限

- **Multi-node on localhost.** Every "node" is a process on one 8-core laptop; the load generator shares the CPU. Numbers show
  fault tolerance, tail latency and event reliability — **not** horizontal throughput scaling.
- **Dataset size.** 7,176 users × 10,728 videos. A single process could serve it; the distributed parts exist to build and
  measure the mechanisms (see INTERVIEW.md, "why distributed for 10k videos").
- **Synthetic scale tests** (1M / 10M vectors) measure latency and memory only; they say nothing about recommendation quality.
- **Proxy label.** "Liked" = watch_ratio > 2, which favours short videos (18.1% positive rate for <7 s vs 0.2% for >60 s).
- **Static catalog** in offline evaluation (all 10,728 videos are candidates on every day), shared by all models.
- **Protocol changes after looking at test:** the ranker's training phase (Stage 4 v2) was designed after v1 lost on test.
  Both results are reported.
- **Simulator ≠ users.** Stage 7/8 "online" metrics replay true small-matrix watch ratios; real users are stochastic and
  react to the feed itself.
- **No video files** in KuaiRec, so the demo shows cards, not playback.

## Docs · 文档

- [DATA.md](DATA.md) — dataset verification, label definition, sensitivity, split, catalog drift
- [BENCHMARK.md](BENCHMARK.md) — every number with its command, machine and raw output
- [MVP_REPORT.md](MVP_REPORT.md) — Stages 0–5 summary, robust vs caveated, resume bullets
- [INTERVIEW.md](INTERVIEW.md) — 中文：按教学顺序逐文件讲解 + 面试问答
- [DRILLS.md](DRILLS.md) — 中文：把核心函数挖空自己重写的练习
- [PREREG.md](PREREG.md) — pre-registration of the simulated A/B (Stage 7)
- [PROGRESS_LOG.md](PROGRESS_LOG.md) — stage-by-stage log for resuming work

---

## Data & credits · 数据与致谢

KuaiRec — Gao et al., *KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems*, CIKM 2022 ([arXiv 2202.10842](https://arxiv.org/abs/2202.10842)). Data © the KuaiRec authors, licensed [CC BY-SA 4.0](https://github.com/chongminggao/KuaiRec/blob/main/LICENSE). The dataset is downloaded by `make data` and is not redistributed in this repository.

KuaiRec 数据集由快手和中国科学技术大学的作者发布，许可证 CC BY-SA 4.0。数据通过 `make data` 下载，不随本仓库分发。
