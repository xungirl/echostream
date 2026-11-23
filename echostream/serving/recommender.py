"""The /recommend pipeline with a latency budget and graceful degradation.

    features (Redis, sharded) ──► user vector (numpy) ──► recall ──► merge ──► rank ──► (MMR) ──► top-n
                                                           ├─ two-tower ANN: scatter-gather over shards
                                                           └─ ItemCF / popularity / content: in-process

Every stage gets min(its own cap, time left in the request budget). Failure
of a stage degrades the answer instead of failing the request:
  feature store down  -> no real-time features (batch history only)
  ANN slow/failed     -> partial ANN (some shards) or skip the ANN path
  ranker timeout      -> keep recall (merge) order
  whole budget spent  -> popularity list
Each fallback is reported in `degraded` and counted in Prometheus.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from prometheus_client import Counter, Histogram

from echostream.core.hashing import ab_bucket
from echostream.core.mmr import mmr_rerank
from echostream.core.recall_merge import Candidate
from echostream.rank.features import POS_THR, build_features
from echostream.serving.ann_client import ShardClient
from echostream.serving.bundle import ServingBundle
from echostream.serving.config import TIMEOUTS, Timeouts
from echostream.serving.feature_store import FeatureStore

STAGE_SECONDS = Histogram("echostream_stage_seconds", "time per /recommend stage", ["stage"],
                          buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
FALLBACKS = Counter("echostream_fallback_total", "degraded responses by kind", ["kind"])
SHADOW_OVERLAP = Histogram("echostream_shadow_overlap_at10", "overlap@10 between served and shadow model",
                           buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


@dataclass
class Rollout:
    """Which model version serves which user (Stage 8)."""
    default: str
    canary: str | None = None
    canary_percent: int = 0
    shadow: str | None = None
    salt: str = "rollout"

    def version_for(self, user_id: int) -> str:
        if self.canary and ab_bucket(user_id, self.salt, 100) < self.canary_percent:
            return self.canary
        return self.default


@dataclass
class Result:
    user_id: int
    version: str
    items: list[dict]
    timings_ms: dict[str, float]
    shards: list[dict]
    degraded: list[str] = field(default_factory=list)


class Recommender:
    def __init__(self, bundles: dict[str, ServingBundle], rollout: Rollout, shards: ShardClient,
                 features: FeatureStore | None, timeouts: Timeouts = TIMEOUTS, mmr_lambda: float | None = None):
        self.bundles = bundles
        self.rollout = rollout
        self.shards = shards
        self.features = features
        self.t = timeouts
        self.mmr_lambda = mmr_lambda
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ranker")

    # ------------------------------------------------------------------ helpers
    def _popularity(self, b: ServingBundle, user_id: int, n: int, exclude: np.ndarray) -> list[Candidate]:
        res = b.engine.popularity("pop_recent", n, exclude)
        return [Candidate(i, "pop_recent", s, {"pop_recent": s}) for i, s in res]

    def _items_json(self, b: ServingBundle, cands: list[Candidate], rank_scores: np.ndarray | None) -> list[dict]:
        out = []
        for pos, c in enumerate(cands):
            row = b.item_meta.iloc[c.item]
            out.append({"item_id": c.item, "title": row.title, "category": row.category_name, "source": c.source,
                        "sources": {k: round(v, 5) for k, v in c.all_sources.items()},
                        "recall_score": round(c.score, 5),
                        "rank_score": None if rank_scores is None else round(float(rank_scores[pos]), 5)})
        return out

    # ----------------------------------------------------------------- pipeline
    async def recommend(self, user_id: int, n: int = 10, version: str | None = None,
                        mmr_lambda: float | None = None) -> Result:
        version = version or self.rollout.version_for(user_id)
        b = self.bundles[version]
        t_start = time.monotonic()
        deadline = t_start + self.t.total_ms / 1000
        try:
            res = await asyncio.wait_for(self._pipeline(b, user_id, n, deadline, mmr_lambda), self.t.total_ms / 1000)
        except asyncio.TimeoutError:
            FALLBACKS.labels("budget_exceeded").inc()
            cands = self._popularity(b, user_id, n, b.seen(user_id))
            res = Result(user_id, version, self._items_json(b, cands, None), {}, [], ["budget_exceeded_popularity"])
        res.timings_ms["total"] = round((time.monotonic() - t_start) * 1000, 3)
        STAGE_SECONDS.labels("total").observe(res.timings_ms["total"] / 1000)
        if self.rollout.shadow and self.rollout.shadow != version and self.rollout.shadow in self.bundles:
            asyncio.get_running_loop().create_task(self._shadow(user_id, n, res))
        return res

    async def _shadow(self, user_id: int, n: int, served: Result) -> None:
        """Score with the shadow model too; compare, never return it."""
        try:
            sb = self.bundles[self.rollout.shadow]
            deadline = time.monotonic() + self.t.total_ms / 1000
            shadow = await self._pipeline(sb, user_id, n, deadline, None)
            a = {x["item_id"] for x in served.items[:10]}
            s = {x["item_id"] for x in shadow.items[:10]}
            SHADOW_OVERLAP.observe(len(a & s) / max(len(a), 1))
        except Exception:
            FALLBACKS.labels("shadow_error").inc()

    async def _pipeline(self, b: ServingBundle, user_id: int, n: int, deadline: float,
                        mmr_lambda: float | None) -> Result:
        timings: dict[str, float] = {}
        degraded: list[str] = []

        def lap(name: str, t0: float) -> float:
            dt = time.monotonic() - t0
            timings[name] = round(dt * 1000, 3)
            STAGE_SECONDS.labels(name).observe(dt)
            return time.monotonic()

        # 1) real-time features from the user's feature-store node
        t = time.monotonic()
        rt_hist: list[tuple[int, float, float]] = []
        if self.features is not None:
            try:
                budget = min(self.t.features_ms / 1000, deadline - time.monotonic())
                rt = await asyncio.wait_for(self.features.get(user_id), max(budget, 0.001))
                rt_hist = rt["hist"]
            except Exception:
                degraded.append("feature_store_unavailable")
                FALLBACKS.labels("feature_store_unavailable").inc()
        t = lap("features", t)

        # 2) user state = batch snapshot + real-time events
        hist_pos = b.pos_history(user_id)
        rt_pos = [i for i, wr, _ in rt_hist if wr > POS_THR]
        if rt_pos:
            hist_pos = np.concatenate([hist_pos, np.array(rt_pos, dtype=hist_pos.dtype)])
        rt_items = np.array([i for i, _, _ in rt_hist], dtype=np.int64)
        exclude = np.union1d(b.seen(user_id), rt_items).astype(np.int64)
        recent = b.recent(user_id) + [(i, wr) for i, wr, _ in rt_hist]
        uvec = b.engine.user_vector(user_id, hist_pos)
        t = lap("user_vector", t)

        # 3) recall: ANN scatter-gather (network) runs while local paths compute
        t_recall = t
        ann_task = asyncio.ensure_future(self.shards.search(uvec, 200, deadline, b.version, exclude.tolist()))
        await asyncio.sleep(0)  # let the shard requests go out before the CPU-bound local paths
        paths = {"itemcf": b.engine.itemcf(hist_pos, 200, exclude)}
        for name in b.engine.a.pop_scores:
            paths[name] = b.engine.popularity(name, 200, exclude)
        paths["content"] = b.engine.content(hist_pos, 200, exclude)
        t_local_done = time.monotonic()
        timings["recall_local"] = round((t_local_done - t) * 1000, 3)
        merged_ann, calls = await ann_task
        t = lap("recall_ann_wait", t_local_done)
        timings["recall"] = round((t - t_recall) * 1000, 3)
        shard_info = [{"shard": c.shard, "ok": c.ok, "replica": c.replica, "ms": round(c.ms, 2),
                       "attempts": c.attempts} for c in calls]
        if len(merged_ann.missing_shards) == len(calls):
            degraded.append("ann_unavailable")
            FALLBACKS.labels("ann_unavailable").inc()
        elif merged_ann.partial:
            degraded.append(f"ann_partial:missing_shards={merged_ann.missing_shards}")
            FALLBACKS.labels("ann_partial").inc()
        paths = {"two_tower": list(zip(merged_ann.ids, merged_ann.scores)), **paths}
        cands = b.engine.merge(paths)
        t = lap("merge", t)
        if not cands:
            cands = self._popularity(b, user_id, n, exclude)
            degraded.append("empty_recall_popularity")

        # 4) rank (CPU, in a thread so the timeout can fire)
        rank_scores = None
        remaining = deadline - time.monotonic()
        if b.ranker is not None and remaining > 0.002:
            def score() -> np.ndarray:
                X = build_features(b.rank_assets, b.engine, user_id, uvec, hist_pos, recent, cands)
                return b.ranker.predict(X, num_threads=1)
            try:
                fut = asyncio.get_running_loop().run_in_executor(self.pool, score)
                rank_scores = await asyncio.wait_for(fut, min(self.t.ranker_ms / 1000, remaining))
            except asyncio.TimeoutError:
                degraded.append("ranker_timeout_recall_order")
                FALLBACKS.labels("ranker_timeout").inc()
        elif b.ranker is not None:
            degraded.append("ranker_skipped_no_budget")
            FALLBACKS.labels("ranker_skipped").inc()
        if rank_scores is not None:
            order = np.lexsort((np.array([c.item for c in cands]), -rank_scores))
            cands = [cands[j] for j in order]
            rank_scores = rank_scores[order]
        t = lap("rank", t)

        # 5) optional diversity re-rank over the top of the ranked list
        lam = self.mmr_lambda if mmr_lambda is None else mmr_lambda
        if lam is not None and lam < 1.0 and rank_scores is not None:
            top = min(len(cands), max(5 * n, 50))
            sim = item_similarity(b, [c.item for c in cands[:top]])
            pick = mmr_rerank(rank_scores[:top], sim, lam, n)
            cands = [cands[j] for j in pick]
            rank_scores = rank_scores[pick]
            lap("mmr", t)
        return Result(user_id, b.version, self._items_json(b, cands[:n], None if rank_scores is None else rank_scores[:n]),
                      timings, shard_info, degraded)


def item_similarity(b: ServingBundle, items: list[int]) -> np.ndarray:
    """Similarity for MMR: mean of category Jaccard and caption-embedding cosine (clipped to [0, 1])."""
    cats = b.rank_assets.item_cats[items]
    inter = cats @ cats.T
    sizes = cats.sum(1)
    union = sizes[:, None] + sizes[None, :] - inter
    jac = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    emb = b.engine.a.caption_emb[items]
    cos = np.clip(emb @ emb.T, 0.0, 1.0)
    return 0.5 * jac + 0.5 * cos
