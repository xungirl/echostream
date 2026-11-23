from echostream.bench.loadgen import Sample, per_second, summarize


def test_summary_and_per_second_windows():
    s = [Sample(0.1, 10.0, 200), Sample(0.5, 30.0, 200, ["feature_store_unavailable"]), Sample(1.2, 50.0, 0),
         Sample(1.7, 20.0, 200, ["ann_partial:missing_shards=[1]"])]
    out = summarize(s, 2.0)
    assert out["requests"] == 4 and out["achieved_rps"] == 2.0
    assert out["error_rate"] == 0.25
    assert out["degraded_rate"] == round(2 / 3, 5)          # among successful responses
    assert out["degraded_labels"] == {"feature_store_unavailable": 1, "ann_partial": 1}
    secs = per_second(s, 2.0)
    assert [x["n"] for x in secs] == [2, 2]
    assert secs[1]["availability"] == 0.5
