from echostream.core.recall_merge import merge_with_quotas


def test_quotas_dedup_and_exclude():
    paths = {
        "ann": [(1, 0.9), (2, 0.8), (3, 0.7)],
        "cf": [(2, 5.0), (4, 4.0), (5, 3.0)],
        "pop": [(6, 100.0)],
    }
    out = merge_with_quotas(paths, {"ann": 2, "cf": 2, "pop": 1}, total=5, exclude={1})
    assert [c.item for c in out] == [2, 3, 4, 5, 6]
    assert [c.source for c in out] == ["ann", "ann", "cf", "cf", "pop"]
    # item 2 was proposed by both ann and cf
    assert out[0].all_sources == {"ann": 0.8, "cf": 5.0}


def test_leftover_slots_filled_round_robin():
    paths = {"a": [(1, 1.0), (2, 0.9), (3, 0.8)], "b": [(10, 1.0)]}
    out = merge_with_quotas(paths, {"a": 1, "b": 3}, total=3)
    assert [c.item for c in out] == [1, 10, 2]


def test_total_cap_and_empty_paths():
    paths = {"a": [(i, 1.0) for i in range(10)], "b": []}
    assert len(merge_with_quotas(paths, {"a": 10, "b": 5}, total=4)) == 4
    assert merge_with_quotas({"a": []}, {"a": 3}, total=3) == []
