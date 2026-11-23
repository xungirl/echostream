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
- **为什么**：只按分数排，前 10 个常常是同一类，用户会腻。λ 的默认值按事先写好的规则选（见 `results/stage6_mmr.json`）。

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

---

## 第四部分：面试问答（约 30 题，答案都能在代码里指出来）

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
9. **离线指标好，线上一定好吗？** 不一定。Stage 7 用全观测小矩阵做会话模拟，比较离线和模拟线上的排序是否一致（`results/stage7_ab.json`）。
   另外排序模型 v1 就是一个反例：离线训练阶段和线上特征分布不一致导致变差。
10. **MMR 的 λ 怎么选？** 事先写好规则：选 NDCG@10 下降不显著（配对 CI 含 0）的最小 λ，见 `results/stage6_mmr.json`。
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
    测试：30 个并发未命中请求只计算 1 次。
24. **影子、灰度、回滚的流程？** 影子：新模型并行打分不返回，比较 overlap 和错误；灰度：按一致性哈希给 10% 用户（稳定，不会每次请求换模型）；
    自动回滚条件见 `echostream/rollout/controller.py` 文件头。结果在 `results/stage8_rollout.json`。
25. **降级策略？** 见第 12 节。每种降级都有 Prometheus 计数器和响应里的标签，方便告警。
26. **为什么 API 要无状态？** 无状态才能随便加副本、随便杀；状态放在 Redis（派生、可重建）和 Postgres（事实来源）。混沌测试杀 API 副本，nginx 重试另一个。
27. **开环压测和闭环压测的区别？** 闭环（发完一个等回复再发下一个）在服务器变慢时会自动降低发送速率，掩盖排队延迟（coordinated omission）。
    开环按固定时刻发请求，从「计划发送时刻」开始计延迟（`echostream/bench/loadgen.py`）。
28. **「1 万个视频为什么要分布式？」（诚实回答）** 这个规模单机单进程完全够：Flat 索引查询 92 µs。分布式在这里是为了练习和验证
    那些在大规模下才必需的机制——分片合并、副本故障转移、分区事件流、一致性哈希、降级——并在真实故障下测量它们。规模上的意义用合成扩容
    （百万、千万向量）单独测，且明确标注为合成数据；所有组件都在一台 8 核笔记本上，所以不声称吞吐能横向扩展。
29. **怎么证明线上召回和离线一致？** 分片用 Flat 精确索引 + 精确合并；集成测试比较线上双塔候选和离线精确搜索结果。
30. **回归门禁的阈值怎么定的？** 不是拍脑袋：对同一批用户做配对 bootstrap，阈值 = max(CI 半宽, 换随机种子重训的标准差)。见 `core/gate.py`。
