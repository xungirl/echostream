# DRILLS — 把核心函数删掉，自己重写一遍

`echostream/core/` 里的函数都是「纯函数 + 强单测」：输入输出清楚、不依赖网络和数据。
练法：用脚本把某个函数的函数体换成 `raise NotImplementedError`（文档字符串保留），测试变红，
自己写到测试全绿。每个练习都在单独的 git 分支上，写坏了也不影响 main。

```bash
uv run python scripts/make_drill.py --list        # 看全部练习
uv run python scripts/make_drill.py ndcg          # 建分支 drill/ndcg 并把函数体挖空
OMP_NUM_THREADS=1 uv run pytest tests/unit/test_metrics.py -q   # 红 -> 写 -> 绿
git checkout main                                  # 回主线（你的答案留在 drill 分支）
```

> 前提：工作区干净（先 commit）。脚本会拒绝在有未提交改动时建分支。

## 推荐顺序

| # | 练习名 | 函数 | 必须做到什么 | 检查它的测试 | 难度 | 提示 |
|---|---|---|---|---|---|---|
| 1 | `recall` | `core/metrics.py::recall_at_k` | 前 k 个里命中的个数 / 真值个数；真值为空返回 0 | `tests/unit/test_metrics.py` | 易 | 只看 `ranked[:k]` |
| 2 | `ndcg` | `core/metrics.py::ndcg_at_k` | 二元相关 NDCG：DCG = Σ 1/log2(rank+1)（rank 从 1 开始），IDCG 把 min(k, \|truth\|) 个命中放最前 | 同上（有手算例子） | 易 | `enumerate` 从 0 开始，所以分母是 `log2(i+2)` |
| 3 | `bootstrap` | `core/metrics.py::bootstrap_ci` | 对「用户」有放回重抽样 n_boot 次，返回 (均值, 2.5% 分位, 97.5% 分位)；同一 seed 结果完全一样 | 同上 | 易 | `rng.integers(0, n, size=(n_boot, n))` 一次生成所有下标 |
| 4 | `split` | `core/split.py::temporal_split` | 时间 < val_start → 0，≥ test_start → 2，中间 → 1；输入可以是乱序 | `tests/unit/test_split.py` | 易 | 先全设 0，再两次布尔赋值 |
| 5 | `scatter_gather` | `core/scatter_gather.py::merge_topk` | 多个分片的局部 top-k 合成全局 top-k；失败分片（None）跳过并记到 `missing_shards`；同分按 id 升序；同一 id 取最高分 | `tests/unit/test_scatter_gather.py` | 易 | 先用 dict 去重，再 `heapq.nsmallest(k, ..., key=(-score, id))` |
| 6 | `ann` | `core/ann.py::ann_search` | 调 faiss `index.search`，去掉 -1 填充，用 `id_map` 把行号换回全局 id；k 大于索引大小时自动截断 | `tests/unit/test_ann.py` | 易 | `k = min(k, index.ntotal)` |
| 7 | `hash_ring` | `core/hashing.py::ConsistentHashRing.get_node` | 在排好序的 (hash, node) 环上找第一个 ≥ hash(key) 的位置，越界绕回 0 | `tests/unit/test_hashing.py`（含「加一个节点只搬走约 1/4 的 key」） | 中 | `bisect.bisect_right(self._ring, (h, "￿"))` |
| 8 | `quota_merge` | `core/recall_merge.py::merge_with_quotas` | 按路的优先顺序，每路最多取 quota 个、跳过重复和已看过的；配额用完还没到 total 就轮流补；记录每个物品被哪些路提出过 | `tests/unit/test_recall_merge.py` | 中 | 每路维护一个游标；重复的物品只往 `all_sources` 里加一条 |
| 9 | `itemcf` | `core/itemcf.py::itemcf_similarity` | sim(i,j) = Σ_u w_u x_ui x_uj / sqrt(n_i n_j)，w_u = 1/log(1+\|N(u)\|)；去掉自己；每行 top-k 降序，不足补 -1 / 0 | `tests/unit/test_itemcf.py`、`tests/unit/test_baselines.py` | 中 | `x.T @ diag(w) @ x` 得到共现；`np.argpartition` 取 top-k 再排序 |
| 10 | `mmr` | `core/mmr.py::mmr_rerank` | 贪心：每步选 λ·rel − (1−λ)·max_sim_to_chosen 最大的；rel 先 min-max 归一化；λ=1 等于按相关性排序 | `tests/unit/test_mmr.py`（有手算例子） | 中 | 维护 `max_sim` 数组，每选一个就 `np.maximum(max_sim, sim[best])` |
| 11 | `user_tower` | `core/two_tower.py::user_tower_numpy` | 用 numpy 复现 PyTorch 用户塔前向：字段 embedding 拼接 + 历史 item embedding 平均 → Linear → ReLU → Linear → L2 归一化；空历史用零向量 | `tests/unit/test_two_tower.py::test_numpy_export_matches_torch` | 中 | `nn.Linear` 是 `W @ x + b`，W 形状 [out, in] |
| 12 | `gauc` | `core/metrics.py::gauc` | 每个用户内部算 AUC，按该用户样本数加权平均；只有一类标签的用户跳过 | `tests/unit/test_metrics.py` | 中 | 按 user 排序后用 `np.diff` 找分组边界，复用 `auc()` |
| 13 | `gate` | `core/gate.py::check_metric` | 配对 bootstrap 求 (候选 − 基线) 的 CI；阈值 = max(CI 半宽, 种子噪声)；只有 Δ < −阈值且 CI 上界 < 0 才算退化 | `tests/unit/test_gate.py` | 中 | 先调 `paired_bootstrap_diff`，再调 `noise_threshold` |
| 14 | `logq_loss` | `core/losses.py::in_batch_softmax_loss` | 批内 softmax：logits = U·Vᵀ/τ；可选减去 logQ；同一物品在批里出现两次时不能当彼此的负样本（mask 成 −inf）；可选拼接随机负样本（撞到正样本的也 mask） | `tests/unit/test_losses.py`（手算 + logQ 例子） | 难 | label 就是 `arange(B)`；`masked_fill(same_item & ~eye, -inf)` |

### Go 练习（Stage 12）

同一个脚本，名字以 `go_` 开头：函数体换成 `panic("drill: ...")`（注释保留），测试对照的是 **Python 算出来的黄金数据**（`go/worker/testdata/`）。

```bash
uv run python scripts/make_drill.py go_hash_ring     # 建分支 drill/go_hash_ring 并挖空 Ring.GetNode
cd go/worker && go test ./internal/hashing/          # 红 -> 写 -> 绿（11,005 个用户 × 6 种节点数必须和 Python 一致）
```

| # | 练习名 | 函数 | 必须做到什么 | 检查它的测试 | 难度 | 提示 |
|---|---|---|---|---|---|---|
| G1 | `go_hash_ring` | `go/worker/internal/hashing/hashing.go::Ring.GetNode` | 在按 (hash, 节点名) 排好序的 `r.points` 上找第一个 hash **严格大于** `StableHash(key)` 的点，越界绕回 0；空环 panic | `go test ./internal/hashing/`（黄金数据 + 「加一个节点只搬约 1/4」） | 中 | `sort.Search(len(r.points), func(i int) bool { ... })`，结果 `% len(r.points)`；对应 Python 的 `bisect_right(ring, (h, "\uffff"))` |
| G2 | `go_float_repr` | `go/worker/internal/pycompat/pycompat.go::FormatFloat` | 输出和 Python `repr(float)` 一字不差：nan/inf/-inf；最短往返数字；十进制指数在 [-4, 16) 用定点（整数补 `.0`），否则 `1.5e+16` / `1e-05`（指数至少两位） | `go test ./internal/pycompat/ -run FormatFloat`（~3,000 个随机浮点数） | 中 | 先 `strconv.FormatFloat(f, 'e', -1, 64)` 拿到「数字串 + 指数」，再按规则重新摆小数点 |

### TypeScript 练习（Stage 13 浏览器引擎）

同一个脚本，名字以 `ts_` 开头：函数体换成 `throw new Error("drill: ...")`（注释保留）。检查它的除了单测，还有**对拍测试**
`parity.test.ts`：300 屏 Python 读路径的黄金数据（`web/demo-golden/golden.json.gz`），你写的版本必须让 300 屏展示的视频和 Python 一模一样。

```bash
uv run python scripts/make_drill.py ts_tree          # 建分支 drill/ts_tree 并挖空 leafIndex
cd web && npx vitest run src/demo/engine/gbdt.test.ts src/demo/engine/parity.test.ts   # 红 -> 写 -> 绿
```

| # | 练习名 | 函数 | 必须做到什么 | 检查它的测试 | 难度 | 提示 |
|---|---|---|---|---|---|---|
| T1 | `ts_tree` | `web/src/demo/engine/gbdt.ts::leafIndex` | 从根节点走到叶子：特征是 NaN 当 0；`x <= threshold` 走左；子节点编号 < 0 表示叶子，返回 `~child`；只有一个叶子的树返回 0 | `gbdt.test.ts` + `parity.test.ts`（树对 LightGBM 概率误差 < 1e-12） | 易 | `for (;;)` 循环，`node = next` |
| T2 | `ts_mmr` | `web/src/demo/engine/mmr.ts::mmrRerank` | 和 `core/mmr.py` 完全一样：先 min-max 归一化（全相等时都是 1）；每步选 λ·rel − (1−λ)·maxSim 最大的、并列取第一个；选完更新 maxSim | `mmr.test.ts` + `parity.test.ts`（300 屏 MMR 结果和 Python 一致） | 中 | 用严格的 `>` 才是「并列取第一个」（np.argmax 的行为） |
| T3 | `ts_merge` | `web/src/demo/engine/recall.ts::mergeWithQuotas` | 和 `core/recall_merge.py` 一样：按优先级每路取配额、跳过重复（但记到 `allSources`）、轮流补满、最后把各路剩下的列表也扫一遍记来源 | `merge.test.ts` + `parity.test.ts`（合并 300/300 一致，含每个候选「被哪些路提出过」） | 中 | 最后那一步最容易漏：不记的话 `in_*` 特征不一样，排序分数就对不上 |

## 写完之后

- 单测全绿后跑一次 `make ci`：集成测试（分片合并、事件管道等）会间接用到这些函数。
- 想看自己的实现对真实数字的影响：例如重写 `ndcg` 后重跑 `uv run python scripts/stage1_baselines.py`，
  `results/stage1_baselines.json` 应该和 main 分支的一模一样。
