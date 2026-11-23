"""Degradation paths of the /recommend pipeline, driven in-process with fake
dependencies (no cluster needed, deterministic)."""
import asyncio
import dataclasses

import numpy as np
import pytest

from echostream.core.scatter_gather import MergeResult
from echostream.serving.ann_client import ShardCall
from echostream.serving.config import Timeouts
from echostream.serving.recommender import Recommender, Rollout
from tests.integration.conftest import BUNDLE, needs_bundle

pytestmark = needs_bundle


@pytest.fixture(scope="module")
def bundle():
    from echostream.serving.bundle import load_bundle
    return load_bundle(BUNDLE)


class FakeShards:
    """Answers like the real shards (exact search over the bundle's vectors)."""

    def __init__(self, bundle, delay=0.0, fail=False, ignore_deadline=False):
        self.b, self.delay, self.fail, self.ignore_deadline = bundle, delay, fail, ignore_deadline

    async def search(self, uvec, k, deadline, version=None, exclude=None):
        await asyncio.sleep(self.delay)
        if self.fail:
            return MergeResult([], [], [0, 1]), [ShardCall(0, False), ShardCall(1, False)]
        res = self.b.engine.two_tower(uvec, k, np.array(exclude or [], dtype=np.int64))
        return MergeResult([i for i, _ in res], [s for _, s in res]), [ShardCall(0, True, 0)]


class BrokenFeatures:
    async def get(self, uid, hist_n=50):
        raise ConnectionError("node down")


class SlowRanker:
    def predict(self, X, num_threads=1):
        import time
        time.sleep(0.3)
        return np.zeros(len(X))


def run(rec, user=42, n=10):
    return asyncio.run(rec.recommend(user, n))


def make(bundle, shards=None, features=None, timeouts=Timeouts(), b=None):
    b = b or bundle
    return Recommender({b.version: b}, Rollout(b.version), shards or FakeShards(bundle), features, timeouts)


def test_happy_path_no_degradation(bundle):
    res = run(make(bundle))
    assert res.degraded == [] and len(res.items) == 10


def test_ranker_timeout_falls_back_to_recall_order(bundle):
    slow = dataclasses.replace(bundle, ranker=SlowRanker())
    res = run(make(bundle, b=slow, timeouts=Timeouts(ranker_ms=20)))
    assert "ranker_timeout_recall_order" in res.degraded
    assert len(res.items) == 10 and res.items[0]["rank_score"] is None


def test_all_shards_failed_skips_ann_path(bundle):
    res = run(make(bundle, shards=FakeShards(bundle, fail=True)))
    assert "ann_unavailable" in res.degraded
    assert len(res.items) == 10
    assert all("two_tower" not in x["sources"] for x in res.items)


def test_feature_store_down_uses_batch_features_only(bundle):
    res = run(make(bundle, features=BrokenFeatures()))
    assert "feature_store_unavailable" in res.degraded and len(res.items) == 10


def test_budget_exceeded_returns_popularity(bundle):
    res = run(make(bundle, shards=FakeShards(bundle, delay=0.5), timeouts=Timeouts(total_ms=100)))
    assert res.degraded == ["budget_exceeded_popularity"]
    assert len(res.items) == 10 and all(x["source"] == "pop_recent" for x in res.items)


def test_canary_routing_is_stable_and_sized():
    r = Rollout(default="v1", canary="v2", canary_percent=5)
    users = range(20000)
    share = sum(r.version_for(u) == "v2" for u in users) / 20000
    assert 0.04 < share < 0.06
    assert [r.version_for(u) for u in range(100)] == [r.version_for(u) for u in range(100)]
