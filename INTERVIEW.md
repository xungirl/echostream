# INTERVIEW.md — 按教学顺序读懂 echostream

每个文件三层：**(a) 骨架**（它做什么、输入输出）、**(b) 为什么**（设计取舍，面试官会追问的点）、
**(c) 语法**（读代码时可能卡住的 Python / PyTorch / Redis 写法）。力气花在前两层。
文中数字都来自 `results/` 里的真实运行，引用时注明文件。

---

## 第一部分：数据与评测（先懂「怎么算对」）

### 1. `echostream/domain/base.py` + `echostream/domain/kuairec.py`
- **骨架**：`DomainAdapter` 抽象类定义「一个领域要提供什么」：交互表、正样本定义 `is_positive`、事件类型 `event_type`、
  时间切分 `split_spec`、物品类别/文本/展示信息、用户类别特征、过去才有的物品统计 `item_stat_features(cutoff)`、
  可选的全观测评测集。`KuaiRecAdapter` 是唯一实现。
- **为什么**：推荐、评测、服务代码只调用接口，不知道「watch_ratio」这种快手特有字段。以后接电商（Amazon Reviews 2023，
  正样本 = 评分 ≥ 4 或购买，文本 = 商品标题）只需要写一个新子类。KuaiRec 没有点赞，所以正样本 = watch_ratio > 2.0
  （官方建议）；DATA.md 里有阈值敏感性和「短视频更容易 >2」的偏差测量（<7 秒 18.1% vs >60 秒 0.2%）。
- **语法**：`@cached_property` 第一次访问时计算并缓存；`pd.factorize(sort=True)` 把字符串类别变成 0..k-1 的整数，缺失值是 -1，
  所以代码里 `codes + 1`（0 留给缺失）。

### 2. `echostream/core/split.py` + `echostream/offline/phase.py`
- **骨架**：按日期切 train (07-05…08-10) / val (08-27…31) / test (09-01…05)。`Phase` = 「站在目标窗口开始那一刻，只知道过去」：
  历史正样本序列、已看过的物品（稀疏矩阵）、目标窗口里的新正样本（truth）。
- **为什么**：随机切分会把用户「未来」的行为放进训练集，评测变成「用未来预测过去」，指标虚高。KuaiRec 还有严重的目录漂移：
  测试窗口 86.3% 的正样本视频在训练窗口里根本没出现（DATA.md），所以 test 阶段用 train+val 全部历史重新拟合，超参只在 val 上选。
- **语法**：`scipy.sparse.csr_matrix((data, (rows, cols)))` 重复坐标会相加，所以之后 `seen.data[:] = 1.0`；
  `phase.seen[u].indices` 直接拿到用户 u 看过的物品 id。

### 3. `echostream/core/metrics.py`
- **骨架**：单个用户的 Recall@K、NDCG@K、HitRate@K、MRR；全体的覆盖率、类别多样性（1 − 平均 Jaccard）；AUC、GAUC；
  `bootstrap_ci` 对用户重抽样求 95% 置信区间；`paired_bootstrap_diff` 同一批用户上两个模型差值的区间。
- **为什么**：每个函数只算一个用户，能手算验证（`tests/unit/test_metrics.py` 里有手算例子）。CI 按「用户」重抽样，
  因为用户是独立单位，同一用户的多个物品不独立。比较两个模型用配对差值，比各自 CI 是否重叠更有统计效力。
- **语法**：`rng.integers(0, n, size=(n_boot, n))` 一次生成 1000×n 的下标矩阵，`per_user[idx].mean(axis=1)` 向量化求 1000 个均值。

### 4. `echostream/offline/evaluate.py` + `echostream/offline/baselines.py`
- **骨架**：`rank_topk` 把模型打分变成 top-200（屏蔽看过的物品）；基线：全局热门、近期热门、平台热门
  （`item_daily_features` 里全快手的完播数）、手写 ItemCF、ALS（`implicit` 库）。
- **为什么**：先有基线才知道复杂模型值不值。结果（`results/stage1_baselines.json`，test NDCG@10）：近期热门 0.0430 最强，
  ItemCF 0.0181，ALS 0.0162；但在全观测小矩阵上排序反过来（ItemCF 0.618，全局热门 0.561）——曝光偏差和目录漂移会改变结论。
- **语法**：`np.argpartition(-scores, k-1)[:k]` O(n) 选出前 k 个（无序），再 `np.lexsort((ids, -scores))` 排序（分数降序、id 升序打破平局）。

### 5. `echostream/core/itemcf.py`
- **骨架**：`itemcf_similarity` 用 `Xᵀ·diag(w)·X` 得到物品共现，除以 `sqrt(n_i·n_j)` 得到余弦相似度，每行留 top-200；
  `itemcf_scores` 把用户最近 20 个正样本的邻居相似度累加（带 0.98 的时间衰减）。
- **为什么**：w_u = 1/log(1+|N(u)|) 降低「什么都看」的重度用户的权重（IUF）。1 万个物品的稠密共现矩阵只要 460MB，所以直接稠密算；
  更大规模要用稀疏 + 采样。批量打分用稀疏矩阵乘法，`tests/unit/test_baselines.py` 保证它和逐用户的纯函数结果一致。
- **语法**：`np.add.at(scores, idx, vals)` 能处理 idx 里有重复的累加（普通 `scores[idx] += vals` 重复下标只加一次）。

---

## 第二部分：召回 → 排序（推荐的主干）

### 6. `echostream/core/two_tower.py` + `echostream/core/losses.py` + `echostream/offline/train_two_tower.py`
- **骨架**：用户塔 = 26 个用户类别字段 embedding + 最近 50 个正样本 item embedding 的平均 → MLP → L2 归一化；
  物品塔 = item id + 类别 + bge-small-zh 标题向量 → MLP → L2 归一化；打分 = 内积。损失 = 批内 softmax：
  一个 batch 2048 个 (用户, 正样本)，其他用户的正样本当负样本。
- **为什么**：
  - 两塔互不看对方输入，所以物品向量能离线算好放进向量索引，线上只跑用户塔 —— 这是召回能在毫秒级从全库挑候选的原因。
  - **logQ 修正**：热门物品在 batch 里出现得多，被当负样本惩罚得也多，模型会学成「反热门」。从 logit 里减去 log(期望出现次数)
    抵消这个偏差。实测（`results/stage2_two_tower.json`）：去掉 logQ 和混合负样本，test NDCG@10 从 0.0803 掉到 0.0071。
  - 混合随机负样本（mixed negatives）在这个数据上反而变差（0.0795 vs 0.0803，val 上按 NDCG@10 选了不加的版本）。
  - **id dropout**：训练时 30% 概率屏蔽 item id 向量，逼模型只靠内容打分；推理时没在训练里出现过的物品（冷启动）只用内容。
  - 温度 τ 只在 val 上调（0.05/0.1/0.2 → 0.1）。
  - `user_tower_numpy` 把用户塔导出成 numpy，线上服务不需要 PyTorch；单测保证和 torch 前向误差 < 1e-5。
- **语法**：`masked_fill(mask, -inf)` 让 softmax 忽略某些位置；`F.cross_entropy(logits, arange(B))` 第 i 行的正确类别是第 i 列；
  `nn.Linear` 的权重形状是 [out, in]，所以 numpy 里是 `W @ x + b`。

### 7. `echostream/core/ann.py` + `scripts/stage2_ann_bench.py`
- **骨架**：faiss 的 Flat（精确）、HNSW、IVF 三种索引；`ann_search` 去掉 -1 填充并把行号映射回物品 id。
- **为什么**：在 10,728 个真实向量上 Flat 单线程单查询 p50 只要 92 µs（`results/stage2_ann.json`），所以线上分片用 Flat
  —— 结果精确，线上线下完全一致（集成测试验证）。HNSW(M=32, ef=128) recall@100 0.955、p50 47.5 µs；IVF(nlist=64, nprobe=16) 0.993、25 µs。
  ANN 的价值要在百万级才体现，见第 13 节的合成扩容测试。
- **语法**：faiss 要求 `float32` 且 C 连续：`np.ascontiguousarray(x, dtype=np.float32)`。

### 8. `echostream/recall/engine.py` + `echostream/core/recall_merge.py`
- **骨架**：五路召回：双塔、ItemCF、近期热门、平台热门、内容冷启动（标题向量相似度，只在冷物品里找）。
  `merge_with_quotas` 按配额合并去重，记录每个物品被哪几路提出过。
- **为什么**：召回求「全」，排序求「准」。多路互补：双塔管个性化，热门管时效，内容管新物品。配额在 val 上选。
  诚实的结果（`results/stage3_multipath.json`）：test 上合并后的 recall@200 = 0.233，反而低于双塔单路 0.271——因为 val 上双塔很弱
  （目录漂移），配额偏向了其他路；内容路能召回 21% 的冷物品真值，但被选中的配额是 0。没有为了好看在 test 上重调。
- **语法**：`dict` 在 Python 3.7+ 保持插入顺序，所以 `paths` 的顺序就是优先级。

### 9. `echostream/rank/features.py` + `echostream/rank/offline.py` + `scripts/stage4_ranker.py`
- **骨架**：40 个特征分五组（用户、物品、交叉、召回分、实时），LightGBM 二分类，标签 = 候选是否成为目标窗口里的新正样本。
- **为什么**：
  - 同一个 `build_features` 线上线下共用，避免 train/serve skew。
  - **最重要的一次教训**：第一版用 val 阶段训练排序（召回模型只用到 8 月 10 日的数据，离标签 3 周），test 上反而输给只按召回排序
    （NDCG@10 0.0696 vs 0.0803，`results/stage4_ranker_v1_valphase.json`）。原因：排序学到的「双塔分」分布和线上不一样。
    改成在 rank 阶段训练（召回模型用 8 月 30 日之前全部数据，标签是 8/30–31），test NDCG@10 = 0.1100，比只用召回高 +0.0297
    [0.0267, 0.0325]，AUC 0.861，GAUC 0.813（`results/stage4_ranker.json`）。这个协议是看过 v1 的 test 结果后设计的，README 里如实写了。
  - GAUC 按用户分组求 AUC 再加权：排序只需要在「同一个用户的候选」内部排对，全局 AUC 还会奖励「区分活跃和不活跃用户」。
  - 特征组消融：去掉物品组 −0.041、用户组 −0.015、召回分组 −0.014、交叉组 −0.007、实时组 −0.0025，CI 都不含 0。
- **语法**：`lgb.train(..., callbacks=[lgb.early_stopping(50)])` 在 20% 用户的验证集上早停；按用户哈希划分，避免同一用户同时出现在两边。

### 10. `echostream/core/mmr.py` + `scripts/stage6_mmr.py`
- **骨架**：贪心选 λ·相关性 − (1−λ)·与已选物品的最大相似度；相似度 = 类别 Jaccard 与标题余弦的平均。
- **为什么**：只按分数排，前 10 个常常是同一类，用户会腻。λ 的默认值按事先写好的规则选：λ=0.7 时类别多样性 0.746 → 0.896，
  NDCG@10 不显著变化（`results/stage6_mmr.json`）。在 Stage 7 的模拟里 MMR 还让每次曝光的正样本率略升 +0.0013。
- **语法**：`max_sim = np.maximum(max_sim, similarity[best])` 增量维护「到已选集合的最大相似度」，整体 O(k·n)。

### 10b. `echostream/sim/simulator.py` + `scripts/stage7_sim_ab.py` + `PREREG.md`
- **骨架**：把全观测小矩阵当成「模拟用户」：给每个用户展示 5 屏 × 10 个视频，反应 = 真实 watch_ratio，正样本更新历史。
  5 个方案按一致性哈希分桶（每个方案 20 个桶）。
- **为什么**：先把假设、指标、分析方法、停止规则写进 PREREG.md 并提交，再跑实验，避免「看了结果再挑指标」。
  模拟器是确定性的，所以同时报告组间（真实 A/B 能做的）和组内（每个用户跑所有方案，更灵敏）两种分析。
- **语法**：`scipy.stats.ttest_ind(a, b, equal_var=False)` 是 Welch t 检验；Holm 校正：p 值从小到大乘以 (m − rank)，再取累计最大值。

---

## 第三部分：分布式服务（后端岗重点）

### 11. `echostream/serving/shard_server.py` + `echostream/serving/ann_client.py` + `echostream/core/scatter_gather.py`
- **骨架**：物品按 `hash(item) % 4` 分到 4 个分片，每片 2 个副本。API 并发查询所有分片（scatter），每片返回局部 top-k，
  `merge_topk` 合成全局 top-k（gather）。
- **为什么**：
  - 全局 top-k 一定在各分片局部 top-k 的并集里，所以合并是精确的。
  - 每次尝试的超时 = min(单片上限 40 ms, 请求剩余预算)，截止时间通过 `X-Deadline-Ms` 头传给分片，过期的请求分片直接拒绝不浪费 CPU。
  - 一个副本连续失败 3 次会被「摘除」2 秒（outlier ejection），慢副本不会让每个请求都等一次超时。
  - 一个分片所有副本都挂：返回其他分片的结果，并标注 `ann_partial:missing_shards=[1]`，其余召回路照常。
  - 「看过的物品」以 base64 int32 数组发给分片过滤，分片多取 k+|exclude| 再过滤，保证 top-k 仍然精确。
- **语法**：`asyncio.gather(*coros)` 并发等待；`httpx.AsyncClient` 复用连接池；`base64.b64encode(arr.tobytes())` 二进制压缩传输。

### 12. `echostream/serving/recommender.py` + `echostream/serving/api.py`
- **骨架**：`/recommend` 的流水线：实时特征（Redis）→ 用户向量（numpy）→ 召回（ANN 网络请求与本地路并行）→ 合并 → 排序（线程池 + 超时）
  →（可选 MMR）→ 返回，带每阶段耗时、分片详情、降级标签。
- **为什么**：总预算 250 ms，每阶段拿 min(自己的上限, 剩余时间)。降级而不是失败：特征库挂了用批量历史；ANN 挂了跳过这一路；
  排序超时按召回顺序；整体超预算返回热门。API 无状态（状态都在 bundle 文件、Redis、Postgres），所以能多副本放在 nginx 后面。
- **语法**：`asyncio.wait_for(fut, timeout)` 超时会取消；CPU 密集的排序放 `run_in_executor`，否则会堵住事件循环，超时也触发不了。

### 13. `echostream/serving/events.py` + `echostream/serving/worker.py` + `echostream/serving/feature_store.py`
- **骨架**：`POST /events` → 按用户哈希写入 8 个 Redis Stream 分区之一 → 4 个 worker 组成消费组，每个分区同一时刻只有一个 owner（租约）
  → Lua 脚本原子地「去重 + 写历史 + 计数」到该用户所在的特征节点（一致性哈希，3 个节点）→ XACK + XDEL。
- **为什么**：
  - at-least-once：处理完才 ACK；worker 崩了，没 ACK 的条目留在 PEL，下一个拿到租约的 worker 用 XAUTOCLAIM 接管。
  - 幂等：Lua 先 `SET ev:{id} NX`，重复投递直接跳过，所以「至少一次投递 + 幂等处理 = 恰好一次效果」。
  - 用户内有序：同一用户永远进同一分区，一个分区一个 owner，先处理自己的 pending 再处理新的；特征节点挂了，这个节点的事件留在 pending，
    后面同节点的新事件也不处理，避免「新的先生效」。
  - 毒消息进死信流 `events:dlq` 并 ACK，不会卡住分区。
  - 测试（`tests/integration/test_event_pipeline.py`）：带重复的 6,000 条事件不丢不重、顺序正确；中途杀 worker；模拟「读了不 ACK 就死」
    的消费者，600 条全部被接管；特征节点冻结 2 秒后恢复，全部正确。
- **语法**：`XREADGROUP ... STREAMS s 0` 读自己的 pending，`>` 读新消息；`redis.eval(lua, numkeys, *keys, *args)`；
  租约续期用「比较后再 PEXPIRE」的 Lua，防止续了别人的租约。

### 14. `echostream/cluster/supervisor.py` + `echostream/cluster/client.py`
- **骨架**：一个小型进程管理器：启动 Redis×4、分片×8、worker×4、API×4、nginx（和 Postgres），崩了自动重启，提供 kill/stop/freeze/delay 控制接口。
- **为什么不用 docker compose**：混沌测试要在毫秒级杀/冻结单个进程并测恢复；colima 的虚拟网络会混进延迟测量；启动从分钟变成秒。
  所有东西都在一台 8 核笔记本上，所以全文都写「multi-node on localhost」，不宣称横向扩展吞吐。

### 15. `echostream/store/*` + `echostream/rollout/controller.py`（Stage 8）
- **骨架**：Postgres 存用户、视频、模型注册表、实验配置、事件归档；Redis 只放可重建的派生数据。期望的发布状态写在
  `rollout_state` 一行里，所有 API 副本和分片每秒对齐（热加载 bundle，不重启）。控制器：注册 → 离线门禁 → 影子 → 金丝雀 → 晋升/自动回滚。
- **为什么**：特征节点没有持久化，丢了用 `python -m echostream.store.rebuild` 从归档按分区顺序重放（测试验证重建后计数和顺序完全一致）。
  缓存：用户结果缓存 TTL + 抖动（防止同时过期），进程内 single-flight + 跨副本 Redis 锁防击穿，新事件到达时 Lua 顺手删缓存。

### 16. `echostream/agent/eval_run.py` + `echostream/agent/gate_check.py` + `echostream/core/gate.py`（Stage 9）
- **骨架**：`make eval` 对一个「候选配置」（双塔模型、排序模型、召回配额、故障注入开关）在测试阶段跑完整的召回 → 排序，
  把汇总指标写进 `results/history.jsonl`，逐用户数组存进 `artifacts/eval/`。`gate()` 拿候选和上一个好版本做配对比较。
- **为什么**：门禁的阈值来自数据：配对 bootstrap 的 CI 半宽和 Stage 2 实测的「换随机种子重训」标准差取大者；只有下降超过阈值
  且 CI 整体在 0 以下才拦。这样「换个种子重训」这类无害改动不会误报，真正的退化会被拦。
- **语法**：`np.savez_compressed(path, **arrays)` 一次存多个数组；`dataclasses.asdict` 把配置写进 JSON。

### 17. `echostream/agent/loop.py` + `tools.py` + `providers.py`（Stage 9）
- **骨架**：手写的 agent 循环：`messages = [system, user(上下文)]`，每轮调用 `provider.chat(messages, tools)`，
  模型返回工具调用就执行并把结果作为 `role=tool` 消息追加，直到模型调用 `write_report`。工具：`diff_runs`、`slice_metrics`
  （按用户活跃度 / 冷物品 / 类别切片）、`path_contribution`、`data_freshness`、`shard_health`。
- **为什么**：
  - 不用框架，循环只有 40 行，面试能一行行讲清楚；工具只返回小摘要，不把大数组塞进上下文。
  - `FakeProvider` 是确定性的规则「模型」：按固定顺序调工具、按证据下结论，CI 里不需要网络和密钥，但工具调用链是真的。
  - `DashScopeProvider` 走 OpenAI 兼容接口（qwen3.8-flash），密钥只从环境变量读；每次调用前用「最坏情况 token 数 × 单价」
    检查累计花费，超过 ¥2 直接拒绝（`CostLedger`，跨运行累计）。单价默认是故意偏高的占位值，用户可用环境变量改成官网价格。
  - 工具出错不会让循环崩溃，错误作为工具结果返回给模型（有测试）。
- **语法**：OpenAI 风格的 `tools=[{"type": "function", "function": {"name", "description", "parameters": JSON Schema}}]`；
  模型返回 `tool_calls[i].function.arguments` 是 JSON 字符串，要 `json.loads`。

### 18. `web/`（Stage 10）
- **骨架**：Vite + React + TypeScript。Feed 页（选用户 → 推荐卡片 → 「看完 / 划走」发事件 → 下一屏变化高亮）、
  每阶段耗时瀑布图（含每个分片的耗时和失败重试）、集群面板（分片 / 副本 / 特征节点 / worker / 分区积压，演示模式下可以一键注入故障）、
  A/B 页（Stage 7 结果）、质量页（评测历史 + 项目历程时间轴）。
- **为什么**：后端项目也要能「看见」：瀑布图直接展示 scatter-gather 的尾延迟和降级标签；点「划走」后下一屏变化，就是「回响」。

### 19. `go/worker/`（Stage 12：用 Go 重写的事件 worker，可直接替换 Python 版）

整体：`cmd/echostream-worker/main.go`（入口）+ `internal/` 下 7 个小包。读的顺序：hashing → pycompat → events → featurestore → worker → main。
对照着 `echostream/serving/worker.py` 读，几乎一行对一行——这是故意的：**算法完全不变，只换语言**，这样性能差异只来自语言/运行时。

**19.1 `internal/hashing/hashing.go`（一致性哈希 + 分区）**
- **骨架**：`StableHash(key)` = 8 字节 BLAKE2b 当大端 uint64；`PartitionFor(uid, P)` = hash("user:uid") % P；`Ring` = 每个节点放 100 个虚拟节点，
  按 (hash, 名字) 排序，`GetNode` 找第一个 hash 比 key 大的点，越界绕回 0。
- **为什么**：Go 写的 worker 必须把用户写到 **Python API 去读的同一个特征节点**，差一个用户都会「写了但读不到」。所以不能用语言自带的哈希，
  用一个两边都能算出同样结果的固定函数。`testdata/hashing_golden.json` 是 Python 算出来的 11,005 个用户 × 6 种节点数 + 7 种分区数的答案，Go 测试要求 100% 一致。
  注意：8 字节摘要的 BLAKE2b 是单独的参数变体，**不是**把 64 字节摘要截断——`blake2b.New(8, nil)` 对，`blake2b.Sum512(...)[:8]` 错。
- **语法**：`sort.Slice(xs, less)` 原地排序；`sort.Search(n, f)` 二分找第一个让 f 为真的下标（等价 Python 的 `bisect_right`）；
  `binary.BigEndian.Uint64(b)` 把 8 字节转成整数。

**19.2 `internal/pycompat/pycompat.go`（和 Python 一字不差的解析/格式化）**
- **骨架**：`FormatFloat` = Python 的 `repr(float)`；`ParseInt` / `ParseFloat` = Python 的 `int(str)` / `float(str)`（含报错文字）；`QuoteString` = `repr(str)`。
- **为什么**：worker 写进 Redis 的不只是数字，还有字符串：历史条目 `"item:watch_ratio:ts"` 里的浮点数格式、死信里的错误原因（会引用坏值）。
  Go 的 `%g` 会把 `12.0` 写成 `12`、把时间戳 `1593878903.438` 写成 `1.593878903438e+09`，Python 的 `repr` 不会——不处理的话两种 worker 写出的历史就不一样。
  Python 的 `int()` 还接受 `" 0_42 "`、阿拉伯数字 `"٤٢"`；`float()` 接受 `"1_0.5"`、`"1e400"`（变成 inf）但不接受 `"0x1p3"`。
  这些都用 Python 生成的 ~2,000 个模糊字符串 + ~3,000 个随机浮点数做黄金测试。可打印字符表、十进制数字表直接从 CPython 3.12（Unicode 15）生成成 Go 源码
  （`unicode_py312.go`），因为 Go 标准库的 Unicode 版本更新，有 ~几千个新字符 Python 认为不可打印。
- **语法**：`strconv.FormatFloat(f, 'e', -1, 64)` 给出最短能往返的数字和指数，再按 Python 规则（指数在 [-4, 16) 用定点）拼；
  `strings.Builder` 拼字符串；`for _, r := range s` 按 rune（Unicode 码点）遍历，不是按字节。

**19.3 `internal/events/events.go`（事件格式 + 毒消息判定）**
- **骨架**：`Field{Key, Value}` 有序字段；`Entry{ID, Fields}`（`Fields == nil` = 读到时已被 XDEL）；`Parse(fields)` 返回 `Event` 或 `*PoisonError`。
- **为什么**：检查顺序和 Python `parse_fields` 完全一样（先缺字段、再按字典顺序逐个转换、最后检查取值），这样同一条坏消息两种 worker 给出同一个死信原因。
  字段用切片不用 map：Go 的 map 无序，死信副本和 `invalid values {...}` 原因要保持生产者写入的顺序。
- **语法**：`errors.As(err, &pe)` 判断错误类型（类似 Python 的 `except PoisonEvent`）；返回 `(值, error)` 二元组是 Go 的错误处理方式，没有异常。

**19.4 `internal/featurestore/store.go`（分片特征库）**
- **骨架**：每个特征节点一个 go-redis 客户端；`ApplyArgs` 拼出 Lua 的 KEYS/ARGV；`ApplyBatch` 把一个节点上的一批事件放进一个 pipeline（一次往返），
  每条结果是 已应用 / 重复 / 脚本错误；连接错误整体返回 error（调用方重试）。
- **为什么**：Lua 脚本源码用 `//go:embed apply_event.lua` 嵌进二进制，黄金测试保证它和 Python 的 `APPLY_EVENT_LUA` 一字不差。
  区分「Redis 回了错误」（确定性，比如 NaN 导致 HINCRBYFLOAT 失败 → 进死信）和「连不上 / 超时」（暂时性 → 留在 pending 重试）是正确性的关键。
  `NOSCRIPT`（节点重启丢了脚本缓存）→ 重新 SCRIPT LOAD 再整批重跑，安全是因为脚本幂等。
- **语法**：`pipe := c.Pipeline(); cmds[i] = pipe.EvalSha(...); pipe.Exec(ctx)`；之后逐个看 `cmd.Int64()` 的错误；
  `errors.As(err, &rerr)` 判断是不是 `redis.Error`（服务器返回的错误）。`Protocol: 2` 强制 RESP2，和 redis-py 一致（RESP3 会把 XREADGROUP 的结果变成无序 map）。

**19.5 `internal/worker/worker.go` + `streams.go`（核心）**
- **骨架**：两个 goroutine 共享状态：
  - `leaseLoop`：每 lease_ms/6 一次心跳 —— ZADD 自己到 `workers:alive`、删掉过期的、算公平份额 ceil(P/存活数)、续租（只续自己的）、多了就从最大分区号开始还、少了就 SET NX 抢、把统计写到 `stats:worker:{id}`。
  - `processLoop`：`processOnce` = 对需要接管的分区先 XAUTOCLAIM（直到别的消费者在这个分区没有 pending 才放行）→ 读自己的 pending（ID "0"）→ 读新消息（">"，最多阻塞 50 ms）→ 每个流调 `handle`。
  - `handle`：解析 → 毒消息进死信 → 按特征节点分组 → 每个节点一个 goroutine 跑 pipeline（`sync.WaitGroup` 等全部完成，相当于 `asyncio.gather`）→ 计数 / 归档 / XACK + XDEL。
- **为什么**：
  - 一个分区同一时刻只有一个 owner（租约），owner 先处理旧的再处理新的 → 用户内有序；接管时没等别人的 pending 清空就读新消息，新消息会「插队」。
  - 某个特征节点失败：`blocked[p][node]` 记下来，这一轮同分区同节点的后续事件全部不处理（留在 pending），否则后面的事件会比失败的那条先生效。
  - 共享的 `owned / needsClaim / stats` 用一把 `sync.Mutex` 保护；Python 是单线程事件循环，不需要锁，Go 的两个 goroutine 真的会并行跑 —— `go test -race` 就是查这个的。
  - XREADGROUP 用 `r.Do(...)` 发原始命令自己解析，因为 go-redis 的 `XReadGroup` 把字段放进 map，丢顺序。
  - 关闭：`Shutdown()` 关闭 `stop` channel → 两个循环在当前一步做完后退出 → `ReleaseLeases` 释放租约、从存活集合删掉自己 → 别的 worker 立刻接管，不用等租约过期。
    Python 版是先释放租约再取消任务（可能留下半批已应用未 ACK 的条目，之后作为重复被跳过），Go 版先停循环再释放，最终状态一样。
- **语法**：`go func() {...}()` 起 goroutine；`chan struct{}` + `close(ch)` 广播「停止」；`select { case <-w.stop: ... case <-t.C: ... }` 可被打断的 sleep；
  `sync.Once` 保证只 close 一次；`defer mu.Unlock()`；Go 1.22 起 `for i, node := range ...` 每轮是新变量，闭包里直接用是安全的。

**19.6 `cmd/echostream-worker/main.go`**
- **骨架**：读 `--id`、`--metrics-port`；`/metrics`（Prometheus，指标名和 Python 一样）和 `/debug/pprof`（CPU 剖析）开在同一个端口；
  SIGTERM/SIGINT → 优雅关闭；`Run` 返回错误 → 退出码 1，让 supervisor 重启（和 Python 版「任务崩了就退出」一致）。
- **语法**：`import _ "net/http/pprof"` 只为副作用（注册 handler）；`signal.Notify(ch, syscall.SIGTERM)`。

**19.7 怎么切换、怎么验证**
- 默认就是 Go（`make cluster-up` 会先 `make go-build`）；`ECHO_WORKER_IMPL=python` 切回 Python，`mixed` = 偶数号 Python、奇数号 Go 混在一个集群里，共用租约和流。没装 Go 时自动用 Python。
- `make go-test`：`go vet` + `go test -race -count=3`，worker 测试自己起 redis-server（重复+毒消息、幽灵消费者接管、崩溃接管、冻结特征节点）。
- `make parity` / `tests/integration/test_go_parity.py`：同一批事件（含重复、乱序重发、16 种毒消息、NaN 部分写入、幽灵消费者、kill -9）分别喂给两种 worker，
  每种都用全新的 Redis 和 Postgres，然后逐 key 对比整个特征库、死信流、计数器和归档表。
- `tests/integration/test_event_pipeline.py` 的 5 个可靠性测试对 python / go / mixed 三种集群各跑一遍。
- 实测（`results/stage12_go_worker.json`，全量 12,530,806 条，各 3 次）：Stage 5 端到端重放 85,306 vs 42,276 条/秒（Go vs Python，中位数），
  worker CPU 12.6 vs 74.3 秒 / 百万事件；只测 worker 的 drain：单 worker 72,241 vs 18,277 条/秒；多 worker 的绝对值在几次之间波动可达 60%
  （慢的那几次 Redis 做同样的活多花约 50% CPU 时间，说明是核变慢了，不是 worker），所以多 worker 只报区间、倍数在同一轮里算。
  10k 条/秒下事件→特征延迟 p50 20–30 ms / p99 30–40 ms（Python 30–40 / 40–50 ms）。内存 25–35 MB vs 40–55 MB，差距不大——别夸大。

### 20. 浏览器 demo（Stage 13）：`echostream/browser/` + `scripts/export_browser_model.py` + `web/src/demo/`

一句话：把**同一个训练好的模型**导出成一个约 10 MB 的静态包，用 TypeScript 重写一遍读路径，放进浏览器里跑；HR 点一个链接就能「划」，
每划一次下一屏在浏览器里重新算。读的顺序：导出器 → Python 进程内读路径（造黄金数据）→ TS 引擎 → 对拍测试 → 页面。

**20.1 `echostream/browser/bundle_format.py` + `export.py`（导出器）**
- **骨架**：读 `artifacts/serving/current`（集群正在服务的那个 bundle），写 `web/demo-public/model/`：
  `model.bin.gz`（所有数组拼在一起，每个数组在 `manifest.json` 里记偏移、形状、编码）、`ranker.json`（LightGBM 树摊平成数组）、
  `catalog.json`（10,728 个视频的中文标题 + 一级类目，类目英文名是手写表 `category_en.json`）、`users.json`（6 个真实用户 + 1 个新用户）、`LICENSE.md`（CC BY-SA 4.0）。
  三个预设：`f32`（和服务 bundle 一模一样，约 33 MB）、`f16`、`int8`（发布的这个）。
- **为什么**：
  - **为什么不是 5 MB？** 目标是 ≤ 5 MB，实测做不到还不伤排序：光 10,728 × 512 的标题向量 int8 就 5.5 MB。试过 PCA 降维——点积误差比 int8 大 7–37 倍；
    试过把 ItemCF 每个物品的邻居从 200 砍到 100——完全相同的屏幕从 52% 掉到 2%。所以选 int8（传输约 10 MB），把「和 float32 模型差多少」测出来写清楚
    （`results/stage13_quant_sweep.json`，见 BENCHMARK Stage 13）。
  - **为什么每个数组的解码都要「一次 float32 乘法」？** 浏览器和 Python 读同一个文件，必须解出**逐位相同**的数：int8 × 每行一个 float32 缩放，
    两种语言里「两个 float32 相乘再舍入到 float32」都是 IEEE 精确舍入，所以结果一样；测试里对每个解码后的数组算 sha256，两边相同。
  - **ItemCF 表为什么按 id 排序 + 差分 + varint？** 原表 [10728, 200] 按相似度排序、-1 补齐，174 万个有效邻居。按 id 排序后相邻 id 的差很小，一个字节就够；
    而 ItemCF 打分是「对每个历史物品，把它的每个邻居加一次」，同一行里邻居的顺序不影响任何一个分数（有单测：重排前后 float32 分数逐位相同）。
  - 输出是**确定性**的（没有时间戳，gzip 的 mtime=0），同样的 artifacts 导出同样的字节，所以 bundle 提交进仓库，构建 demo 不需要 1 GB 数据。
- **语法**：`np.rint(x / scale[:, None]).astype(np.int8)` 每行量化；`gzip.compress(data, 9, mtime=0)` 去掉时间戳；
  LightGBM 文本模型里 `decision_type=2` = 数值型、缺失值当 0、`default_left`，`left_child` 为负数时 `~child` 就是叶子编号。

**20.2 `echostream/browser/local_read_path.py` + `scripts/gen_browser_golden.py`（Python 参考答案）**
- **骨架**：不起集群，在**一个进程里**跑生产代码：4 个真的 shard server（FastAPI app）通过内存里的 ASGI transport 接到真的 `ShardClient`；
  临时起一个 redis-server，真的 `Worker.process_once()` 把事件用真的 Lua 脚本写进特征库；真的 `Recommender` 出结果（预算放宽，保证不降级）。
  50 个用户（43 个随机 + 6 个 demo 用户 + 新用户），每人 6 屏：第一屏，然后依次对当前屏的视频 like / skip / watch / like / skip。
  记录每屏的用户向量、5 路召回列表和分数、合并后的 200、每个候选的排序概率、展示的 10 个、部分屏的 40 维特征 → `web/demo-golden/golden.json.gz`。
- **为什么**：「浏览器和 Python 一样」只有在同一批输入下逐阶段对比才有说服力。中间结果靠**包装这个 bundle 实例的三个方法**记录（`engine.merge`、
  `engine.user_vector`、`ranker.predict`），生产代码一行没改。Python 读的是**浏览器同一个 bundle 解出来的数**（`to_serving_bundle`），所以对比的只是「代码」。
- **语法**：`httpx.ASGITransport(app=...)` 让 httpx 直接调 ASGI 应用不走网络；自定义 `AsyncBaseTransport` 按 host 分发到 4 个分片。

**20.3 `web/src/demo/engine/`（TypeScript 读路径）**
- **骨架**：`bundle.ts` 解码 → `recall.ts`（用户塔、精确双塔检索、ItemCF、两种热门、内容召回、配额合并）→ `features.ts`（40 维排序特征）
  → `gbdt.ts`（210 棵树）→ `mmr.ts`（λ=0.7，前 50 里挑 10）→ `recommender.ts` 串起来；`featureStore.ts` 是浏览器里的「实时特征库」：
  和 Lua 脚本同样的效果（event_id 去重、LPUSH 历史、截到 100 条、计数器；读最新 50 条、按时间正序）。每次点按钮 = 发一个事件 + 从头重算一屏。
- **为什么能和 numpy 对上？**
  - 浮点规则：Float32Array 每存一次就舍入到 float32；一次 + − × ÷ √ 先用 float64 算再舍入一次，结果正好等于 float32 运算（双精度位数 ≥ 2×24+2，二次舍入无害）。
  - 只有长的点积对不上：numpy 用 BLAS 在 float32 里按它自己的顺序累加，我们用 float64 累加最后舍入一次——差约 1 个 ulp（实测用户向量最大差 8.9e-8，特征最大差 6e-7）。
  - ItemCF 是 `np.add.at` 按历史顺序逐个 float32 乘加，可以**逐位复现**（实测 score_itemcf 误差 0）；权重 0.98^k 由 Python 算好放进 manifest，不依赖 JS 的 `Math.pow`。
  - numpy 的小细节要照抄：`mean(axis=0)` 是按行顺序累加；`axis=1` 上 5 个数的 float32 求和是「第一个数 + 其余数从 0 开始的顺序和」（pairwise sum 的小数组分支）。
- **语法**：`Math.fround`；`Float32Array.subarray` 不复制；`~next` 是按位取反（−1 → 0）；`(bits >>> c) & 1` 取类目位。

**20.4 `parity.ts` + `parity.test.ts`（对拍）**
- **骨架**：两种视角。**逐阶段**：每个阶段喂 Python 的输入（Python 的召回列表 → 我们的合并；Python 的候选 → 我们的特征和树；Python 的排序 → 我们的 MMR），
  这样一个阶段的差异不会掩盖或引发另一个阶段的差异。**端到端**：只给用户记录和事件，我们的引擎自己跑完。
- **实测**（`results/stage13_parity.json`，300 屏）：端到端展示的 10 个视频 **300/300 完全相同（顺序也相同）**；合并 300/300；MMR 300/300；
  树对 LightGBM 的概率最大差 1.4e-17；特征 93.4% 逐位相同、最大差 6e-7；双塔召回列表 300/300 相同。
  ItemCF 282、热门 162、内容 281 个列表逐位相同，其余**只差在并列的物品上**：numpy 的 `argpartition` 在第 200 名有并列时随便留一部分，
  我们按 id 留；或者两个分数差一个 ulp 的物品顺序互换。测试把这种情况定义为「等价」（逐位置分数差 ≤ 1e-6、明显高于截断线的物品集合相同），并要求 300/300 等价。
- **为什么不改 Python 让并列也按 id？** 那会改变线上和离线的已有结果（虽然只在第 200 名附近），不该为一个 demo 改生产代码；如实记录，留作下一步。

**20.5 页面（`web/src/demo/DemoApp.tsx`、`session.ts`、`components/`）**
- **骨架**：加载（进度条）→ 首次访问 3 步提示 → 选人（6 个真实用户按最喜欢的类目命名 + 新用户）→ 手机形状的 feed，每张卡片 Skip / Watched / Like
  （watch_ratio 0.1 / 1.0 / 3.0；KuaiRec 没有点赞，模型的「喜欢」= 看了两遍以上）→ 右边漏斗「这屏是怎么来的」、回响（新进来的、被挤掉的）、兴趣画像。
- **为什么**：给不看代码的人看：所有数字都是这一屏真实算出来的（每路召回多少、合并留了多少、排序、MMR）；兴趣画像就是用户塔读的那 50 个喜欢的视频的类目，
  只有 Like 会改变它——Skip / Watched 进的是排序模型的实时特征，页面上也写明了。内容召回在这个模型版本里配额是 0（Stage 3 在验证集上选的），页面上如实显示「找到 200、留 0」。
- **语法**：`useLayoutEffect` + `getBoundingClientRect` 做 FLIP 动画（卡片从旧位置滑到新位置）；`DecompressionStream("gzip")` 在浏览器里解压；
  `role="radiogroup"` / `aria-live` 做无障碍；`prefers-reduced-motion` 关掉动画。


---

## 第四部分：面试问答（约 46 题，答案都能在代码里指出来）

### 推荐算法

1. **为什么召回和排序要分开？** 召回要从 1 万（真实系统是上亿）里挑几百个，只能用便宜的模型（双塔内积 + 向量索引，线上 1 次用户塔
   + 4 个分片各 ~0.1 ms 的搜索）；排序只看几百个，可以用很多交叉特征和树模型（每请求 ~2.4 ms 的 LightGBM）。一个模型干两件事，要么慢要么不准。
2. **负采样怎么做的？为什么要 logQ？** 批内负样本：同 batch 其他用户的正样本。代价是热门物品被当负样本的次数正比于热度，
   模型会系统性低估热门。logQ 从 logit 减去 log(Bq_j + M/N)（期望出现次数），去掉这个偏差。去掉 logQ 和混合负样本后 NDCG@10 从 0.0803 掉到 0.0071。
3. **同一个物品在一个 batch 里出现两次怎么办？** 它不能当另一个用户的负样本，`same_item & ~eye` 的位置 mask 成 −inf（`core/losses.py`）。
4. **温度系数的作用？** logits 除以 τ。τ 小，softmax 更尖，更关注难负样本；在 val 上试了 0.05/0.1/0.2，选 0.1。
5. **为什么用时间切分？随机切分哪里漏了？** 随机切分让训练集包含测试时刻之后的行为（以及之后的热度），等于偷看答案。
6. **NDCG 怎么算？和 Recall 区别？** 命中位置越靠前贡献越大（1/log2(rank+1)），再除以理想排列的 DCG；Recall 只看命中个数不看位置。
7. **GAUC 和 AUC 有什么区别？为什么排序看 GAUC？** 见第 9 节：GAUC 按用户分组，只衡量「同一用户内部」排得对不对。
8. **冷启动怎么处理？** 物品：id dropout 让物品塔能只靠内容；内容召回路在「历史里没有交互」的物品中按标题向量找；平台热门路知道还没被
   数据集用户看过的新视频。用户：用户塔的类别特征 + 热门路。数据：冷物品占测试真值的 38%，内容路召回其中 21%（`stage3_multipath.json`）。
9. **离线指标好，线上一定好吗？** 不一定，这个项目里就有两个真实反例。① Stage 7 预注册的模拟 A/B：大矩阵离线 NDCG@10 最好的
   「召回+排序」(0.110)，在全观测小矩阵的模拟用户上反而比双塔差（每次曝光的正样本率 0.179 vs 0.213，配对差 −0.034
   [−0.037, −0.030]）；5 个方案的离线排名和模拟线上排名 Kendall τ = −0.4（`results/stage7_ab.json`）。我的解释（假设）：
   排序模型的标签来自曝光日志，它学会了「平台会推什么」，在没有曝光偏差的用户上不成立。② 排序 v1 训练阶段和线上特征分布不一致导致变差。
10. **MMR 的 λ 怎么选？** 事先写好规则：选 NDCG@10 下降不显著（配对 CI 含 0）的最小 λ。结果 λ=0.7：类别多样性 0.746 → 0.896，
    NDCG@10 0.1100 → 0.1099（差值 CI 含 0）；λ=0.3 时 NDCG@10 掉 0.015（`results/stage6_mmr.json`）。
11. **你的正样本定义有什么问题？** watch_ratio > 2 偏向短视频（<7 秒 18.1% 是正样本，>60 秒只有 0.2%）；阈值敏感性在 DATA.md。
12. **为什么 ItemCF 在大矩阵上很弱、在小矩阵上很强？** 大矩阵测试窗口大部分视频是新视频，ItemCF 只能推荐历史里出现过的；小矩阵是固定的
    3,327 个老视频，且每个用户都看过全部，没有曝光偏差。
13. **排序模型为什么第一版变差了？怎么发现、怎么修？** 见第 9 节。教训：排序的训练数据要模拟线上时刻的召回分布。
14. **多路召回合并后 recall 反而低于单路，怎么解释？** 配额是在 val 上选的，val 上双塔弱；test 上双塔强。这说明配额应该随各路效果
    动态调整（或者干脆加大候选数交给排序），而不是固定。
15. **为什么用 bootstrap CI？** 指标是用户平均，用户之间独立；bootstrap 不需要正态假设，配对版本能抵消用户间差异。

### 分布式 / 后端

16. **scatter-gather 为什么会放大尾延迟？怎么缓解？** 请求要等最慢的分片，N 个分片任何一个慢都拖慢整体（分片越多越容易碰到慢的）。
    缓解：每次尝试设超时、失败换副本、连续失败摘除慢副本、超预算返回部分结果并标注。混沌测试里给一个分片加 500 ms 延迟，看 `results/stage5_chaos.json`。
17. **分片和副本分别解决什么？** 分片解决「一台放不下 / 一次算不完」（数据量和单次计算量）；副本解决可用性和读吞吐。杀掉一个分片主副本，
    请求自动走另一个副本。
18. **一致性哈希为什么比 hash % N 好？** 加一个节点，`% N` 几乎所有 key 都要搬；一致性哈希只搬约 1/N（单测：加第 4 个节点只移动 15–35% 的 key，
    并且只移到新节点）。虚拟节点让负载更均匀。
19. **Redis Streams 消费者组怎么保证 at-least-once？** 见第 13 节：PEL + XACK + XAUTOCLAIM。
20. **怎么保证幂等？** 客户端生成 event_id；Lua 里 `SET ev:{id} NX EX ttl` 和状态更新在同一个原子脚本里，且和该用户的数据在同一个节点上。
21. **怎么保证同一用户的事件有序？** 分区按用户哈希 + 分区单 owner（租约）+ 先处理 pending + 节点故障时阻塞同节点的后续事件。
    已知边界：一个「僵尸」worker（暂停后租约已过期）在恢复的一瞬间可能和新 owner 同时处理，幂等保证不重复计数，但顺序在那个窗口里不保证。
22. **死信队列什么时候用？** 确定性的失败（格式错、脚本报错）进死信，否则一条坏消息会永远卡住分区；暂时性的失败（节点连不上）重试。
23. **缓存击穿怎么防？** 进程内 single-flight（同一个 key 只算一次，其他请求等同一个 Future）+ 跨副本 Redis 短锁 + TTL 抖动防止雪崩。
    测试：30 个并发未命中请求只计算 1 次；实测 200 个并发未命中 → 1 次计算。Zipf 流量下命中率 71.7%，p50 10.3 → 3.1 ms，
    但 p99 21 → 81 ms（未命中路径多了锁和缓存读写）——缓存不是免费的。
24. **影子、灰度、回滚的流程？** 影子：新模型并行打分不返回，比较 overlap 和错误；灰度：按一致性哈希给 10% 用户（稳定，不会每次请求换模型）；
    自动回滚条件见 `echostream/rollout/controller.py`（`canary_reasons` 是纯函数，有单测）。实测（`results/stage8_rollout.json`）：
    换种子重训的 v2 被晋升；2000 棵树的慢排序模型 v3 在灰度里 185/312 个响应降级（排序超时），被自动回滚；
    去掉 logQ 的 v4 在模拟灰度里反而更好被晋升，但离线门禁会拦它——线上信号和离线信号要一起看。
25. **降级策略？** 见第 12 节。每种降级都有 Prometheus 计数器和响应里的标签，方便告警。
26. **为什么 API 要无状态？** 无状态才能随便加副本、随便杀；状态放在 Redis（派生、可重建）和 Postgres（事实来源）。混沌测试杀 API 副本，nginx 重试另一个。
27. **开环压测和闭环压测的区别？** 闭环（发完一个等回复再发下一个）在服务器变慢时会自动降低发送速率，掩盖排队延迟（coordinated omission）。
    开环按固定时刻发请求，从「计划发送时刻」开始计延迟（`echostream/bench/loadgen.py`）。
28. **「1 万个视频为什么要分布式？」（诚实回答）** 这个规模单机单进程完全够：Flat 索引查询 92 µs。分布式在这里是为了练习和验证
    那些在大规模下才必需的机制——分片合并、副本故障转移、分区事件流、一致性哈希、降级——并在真实故障下测量它们。规模上的意义用合成扩容
    （百万、千万向量）单独测，且明确标注为合成数据；所有组件都在一台 8 核笔记本上，所以不声称吞吐能横向扩展。
29. **怎么证明线上召回和离线一致？** 分片用 Flat 精确索引 + 精确合并；集成测试比较线上双塔候选和离线精确搜索结果。
30. **回归门禁的阈值怎么定的？效果如何？** 不是拍脑袋：对同一批用户做配对 bootstrap，阈值 = max(CI 半宽, 换随机种子重训的标准差)。
    5 个真实退化 + 3 个无害改动：召回 0.8、精确 0.8（`results/stage9_release_guard.json`）。漏掉的是「一个分片用旧向量」——离线几乎没影响；
    误报的是「排序模型换种子」——因为种子噪声只测了双塔，没测排序。改进方向：每个会重训的组件都要测种子噪声。

### Go worker（Stage 12）

31. **这个负载为什么 Go 的 goroutine 比 Python 的 asyncio 快？** 两个都是「一个线程里等很多网络往返」的模型，往返本身并不更快。
    差别在两次往返之间的 CPU：Python 每条事件要做字典解码、字符串拼接、`repr`、协程调度，实测 49–96 CPU 秒 / 百万事件，
    单个 worker 被一个核卡死（单 worker 18,277 条/秒）；Go 同样的活 7.7–12.7 CPU 秒 / 百万事件（单 worker 72,241 条/秒）。
    另外 asyncio 只用一个核，要多核只能开多进程；goroutine 由运行时分到所有核上，一个进程里「每个特征节点一个 pipeline 并行跑」是真并行。
    pprof 显示 Go worker 只占约 33% 的一个核，剩下的时间在等 Redis——所以瓶颈从「worker 的 CPU」移到了「Redis 往返 + 单线程事件总线」。
    （`results/stage12_go_worker.json`、`results/stage12_go_worker_pprof_4w.pb.gz`）
32. **消费者组和 PEL 是什么？** 一个 stream 上建一个消费者组 `workers`，组里每个消费者（w0、w1…）用 `XREADGROUP ... >` 领新消息；
    领走但还没 `XACK` 的消息记在 PEL（Pending Entries List）里，记着「谁领的、领了多久、投递了几次」。
    worker 每轮先用 ID `0` 读自己 PEL 里的旧消息（上次没处理完的、刚接管来的），再读新消息，所以同一个用户的事件不会被新的插队。
    处理完才 ACK，并且马上 XDEL，让 stream 不无限变长。
33. **XAUTOCLAIM 怎么用？有什么坑？** `XAUTOCLAIM stream group me min-idle start COUNT n`：把 PEL 里空闲超过 min-idle 的消息转到我名下。
    我们只在刚拿到一个分区租约时调用，并且要等 `XPENDING` 显示别的消费者在这个分区已经没有 pending 才开始读新消息。
    坑 1：min-idle（1 s）必须大于「一批正常处理所需时间」，否则会抢走一个还活着、只是慢的 worker 正在处理的消息——
    实测出现过一次：8 个 Python 进程 + 4 个 Redis 挤 8 个核，一批处理超过 1 秒，被新 owner 接管，184 条被处理了两次（幂等兜底，计数仍然全对）。
    坑 2：worker 启动时第一个进程会暂时拿走全部分区、各读 500 条，下一次心跳再还回去；这些消息要等 1 秒空闲后被别人接管，所以 `claimed` 计数在不同运行之间不一样。
    坑 3：XAUTOCLAIM 会把「已被 XDEL 的 pending 条目」直接从 PEL 删掉（Redis 7+ 在返回值第三项里列出来），不会交给你处理。
34. **幂等是怎么做的？为什么要和更新放在同一个 Lua 脚本里？** event_id 由客户端生成；Lua 第一步 `SET ev:{id} 1 NX EX ttl`，
    设置失败说明处理过，直接返回 0（记为 duplicate）。如果「检查去重」和「更新计数」分成两条命令，中间崩溃就会出现「标记了但没更新」或者「更新了但没标记」。
    放在一个脚本里，Redis 单线程原子执行，而且一个用户的所有 key（去重 key、历史、计数）都在同一个特征节点上（一致性哈希按用户），所以一个脚本就够。
    注意 Lua 脚本出错时前面的写入不会回滚：NaN 的 watch_ratio 会先写进历史和 n，然后 HINCRBYFLOAT 报错——两种 worker 行为一致，进死信。
35. **at-least-once 和 exactly-once 的区别？你们是哪种？** 投递是 at-least-once（崩溃后 PEL 里的消息会被重新投递），效果是 exactly-once
    （去重 key 让重复投递变成空操作）。kill -9 测试里，被杀的 worker 有一批「Lua 已经执行、还没 ACK」的消息，接管后重新投递，
    计成 duplicates（Go 三次分别是 500 / 474 / 0，取决于 SIGKILL 落在批次的哪个位置），最终每个用户的计数都和真值一致。
    真正的端到端 exactly-once 还要求下游（这里是 Postgres 归档）也幂等：`ON CONFLICT (event_id) DO NOTHING`。死信流本身是 at-least-once：
    写完死信、ACK 之前崩溃，死信里会有重复条目（两种 worker 一样）。
36. **背压在哪里？** 三层：(1) 生产端：重放脚本在「stream 积压 + pending > 40 万」时暂停发送；(2) worker 每次每个分区最多读 500 条（COUNT），
    处理完才读下一批，不会把内存撑爆；(3) 某个特征节点失败时，这个分区里属于该节点的后续消息都留在 pending（不 ACK），
    积压体现在 stream 长度上，可以监控告警。Go 版没有改这套机制——如果 Go 版把消息读进无界 channel 再慢慢处理，吞吐看起来更高，但背压和顺序都没了。
37. **为什么只重写 worker，不重写 API 或整个系统？** worker 是实时链路里最热、最独立的一段：输入输出契约很窄（stream 名、Redis key、Lua 脚本、指标名），
    逻辑是纯 I/O + 字符串处理，Python 在这里吃亏最大（实测 Python worker 每百万事件 74 CPU 秒，Go 12.6）。API 的耗时主要在 LightGBM / numpy，
    本来就是 C/C++；离线训练离不开 Python 的 ML 生态。而且算法逐行照搬，对比时唯一变量是语言/运行时。
    结果：同一套端到端重放从 42,276 提到 85,306 条/秒（3 次中位数，区间都很窄），之后的瓶颈变成单线程的事件总线 Redis（83% 忙）和整机 CPU，不再是 worker。
38. **跨语言怎么保证哈希完全一致？** 不能用语言自带的哈希（Python 的 `hash()` 每个进程随机，Go 的 map 哈希也不公开）。
    两边都用 8 字节摘要的 BLAKE2b，按大端转成 uint64；环上排序规则（hash，再按节点名）和查找规则（第一个 hash 大于 key 的点，越界绕回）也一样。
    验证：Python 生成 11,005 个用户 id（含 2^31、2^62 等边界）在 6 种节点数下的归属、7 种分区数的分区，写进 `go/worker/testdata/`，Go 测试要求 100% 相同；
    `tests/unit/test_go_golden.py` 保证 Python 改了以后黄金文件也必须重新生成。最后用对拍测试在真 Redis 上比对整个特征库，14,802 个 key 一个不差。
    踩过的坑：`blake2b.New(8, nil)` 才对，把 64 字节摘要截断成 8 字节是另一个结果。

### 浏览器 demo（Stage 13）

39. **为什么浏览器能和 Python 算得一模一样？** 三件事：(1) 读同一个文件、解出逐位相同的数组（每个解码都是一次 float32 乘法或查表，测试比 sha256）；
    (2) float32 的单次运算在 JS 里用 float64 算再 `Math.fround` 一次，结果和 float32 运算完全一样；(3) 把 numpy 的累加顺序照抄（`np.add.at`、`mean`）。
    唯一对不上的是 BLAS 的长点积（累加顺序不同，差 1 ulp），所以测试对分数用容差（1e-6），对 id 和顺序要求完全相同。结果：300 屏展示的视频 300/300 一样。
40. **LightGBM 的树在 TS 里怎么算？** 从文本模型里取每棵树的 `split_feature / threshold / left_child / right_child / leaf_value`。
    从根开始：特征是 NaN 就当 0（这个模型的 missing_type 是 None），`x <= threshold` 走左边，子节点编号为负就是叶子（`~child`）。
    210 棵树的叶子值按树的顺序用 float64 相加（和 LightGBM 一样），最后 sigmoid。导出器遇到类别型分裂或别的 decision_type 会直接报错，不会悄悄算错。
    对 Python 的 4,000 行特征，概率最大差 1.4e-17。
41. **这个 demo 证明不了什么？** 分布式那一层完全不在页面里：没有分片、副本、scatter-gather、Redis Streams、Go worker、降级、延迟预算。
    页面上的毫秒数是**浏览器里的计算时间**，不是服务端延迟（服务端延迟看 Stage 5 / 12）。另外发布的是 int8 量化的模型：和 float32 模型相比，
    屏幕上的 10 个视频平均有 9.8 个相同，但完全同序的屏只有约一半——这是为了下载体积做的取舍，测出来写清楚了。
42. **为什么不录一段回放 / 为什么不放一个云端后端？** 回放只能看我选好的路径，HR 自己点什么都不会变，「回响」就不成立；
    云端后端要一直开着、要钱、会被刷、几个月后链接就挂了。浏览器里跑真模型：零运维、链接永远能用、每一次点击都是真的重算。代价是要重写一遍读路径，
    所以必须有对拍测试证明两份代码一样。
43. **量化怎么选的？** 做了一个扫描（`scripts/stage13_quant_sweep.py`）：每种存储方式都导出、读回、用真的 Python 读路径跑同样的 300 屏，和 float32 对比。
    先确认 float32 的浏览器 bundle 和服务 bundle 逐屏一样（排序分数也逐位相同）。结论：标题向量的精度影响最大（int8 时完全同序的屏从 98% 掉到 65%），
    ItemCF 不能截断邻居（截到 100 个只剩 2% 同屏）。选 int8：传输约 10 MB；f16 是 18 MB / 96% 同序，留了 `--quant f16` 开关。
44. **为什么用 `argpartition` 会出现并列问题？这算 bug 吗？** `argpartition` 只保证第 k 个位置正确，同分的物品谁进前 k 是算法内部决定的，不是按 id。
    代码后面用 `lexsort` 按 (分数, id) 排序，说明本意是「同分按 id」，但被截断的那一刻已经不是了。它不影响正确性，只影响第 200 名附近的并列物品，
    而且对同一个输入是确定的。修它会改变已有的离线结果，所以这次只在对拍里把它测量出来、定义成「等价」，列为下一步。
45. **实时特征在浏览器里怎么维护？** `featureStore.ts` 复刻 Lua 脚本的效果：event_id 去重（重复投递是空操作）、历史 LPUSH 并截到 100 条、计数器
    （n、各 action、sum_wr、pos、每个类目的 pos / skip）。推荐时读最新 50 条、按时间正序——和 `FeatureStore.get(hist_n=50)` 一样。
    注意这意味着第 51 次以前的反应不再被排除（和线上行为一致）。
46. **新用户（冷启动）怎么处理？** 一个没有任何历史、所有画像字段都是「缺失」（编码 0）的用户：用户塔只有字段向量、没有历史平均，ItemCF 和内容召回为空，
    靠双塔 + 两种热门凑够 200；排序特征里用户统计全 0，时长偏好用全目录中位数（和 `build_rank_assets` 对没有正样本的用户的处理一样）。
    第一次 Like 之后，ItemCF 和内容召回马上有了输入，下一屏就变。
