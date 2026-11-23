import numpy as np
import pytest

from echostream.core.split import assert_no_future_leak, temporal_split, time_cutoffs


def test_time_cutoffs_and_split_order():
    ts = np.arange(100, dtype=float)
    val_start, test_start = time_cutoffs(ts, val_frac=0.1, test_frac=0.2)
    assert (val_start, test_start) == (70.0, 80.0)
    part = temporal_split(ts, val_start, test_start)
    assert (part == 0).sum() == 70 and (part == 1).sum() == 10 and (part == 2).sum() == 20
    # every train event is strictly earlier than every val event, and val earlier than test
    assert ts[part == 0].max() < ts[part == 1].min() <= ts[part == 1].max() < ts[part == 2].min()


def test_split_unsorted_input():
    ts = np.array([5.0, 1.0, 9.0, 3.0])
    assert temporal_split(ts, 3.0, 5.0).tolist() == [2, 0, 2, 1]


def test_leak_guard():
    assert_no_future_leak(np.array([1.0, 2.0]), cutoff=3.0)
    with pytest.raises(ValueError):
        assert_no_future_leak(np.array([1.0, 3.0]), cutoff=3.0)
