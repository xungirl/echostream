from echostream.core.scatter_gather import merge_topk


def test_merge_is_exact_global_topk():
    shards = [([1, 4], [0.9, 0.2]), ([2, 5], [0.8, 0.7]), ([3], [0.95])]
    r = merge_topk(shards, 3)
    assert r.ids == [3, 1, 2] and r.scores == [0.95, 0.9, 0.8]
    assert not r.partial


def test_failed_shard_gives_partial_result():
    r = merge_topk([([1], [0.5]), None, ([2], [0.6]), None], 5)
    assert r.ids == [2, 1] and r.missing_shards == [1, 3] and r.partial


def test_ties_and_duplicates_deterministic():
    r = merge_topk([([7, 3], [0.5, 0.5]), ([3], [0.9])], 2)
    assert r.ids == [3, 7] and r.scores == [0.9, 0.5]
    r2 = merge_topk([([9, 2], [0.5, 0.5])], 2)
    assert r2.ids == [2, 9]  # tie broken by id


def test_all_failed_and_k_zero():
    assert merge_topk([None, None], 5).ids == []
    assert merge_topk([([1], [1.0])], 0).ids == []
