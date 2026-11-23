"""parse_fields / latency buckets: the two poison-pill crashes found while writing the Go port."""
import math

import pytest

from echostream.serving.events import NOT_UTF8, PoisonEvent, as_text, parse_fields
from echostream.serving.worker import LAT_BUCKETS_MS, lat_bucket

GOOD = {"event_id": "e1", "user_id": "42", "item_id": "7", "action": "play", "watch_ratio": "0.5",
        "ts": "1.5", "cats": "1,2", "produced_ms": "1000.0"}


def test_good_event_parses():
    ev = parse_fields({k.encode(): v.encode() for k, v in GOOD.items()})
    assert ev["user_id"] == 42 and ev["watch_ratio"] == 0.5 and ev["cats"] == "1,2"


def test_invalid_utf8_is_poison_not_a_crash():
    # used to raise UnicodeDecodeError, which escaped handle() and killed the worker on every redelivery
    with pytest.raises(PoisonEvent, match=NOT_UTF8):
        parse_fields({**{k.encode(): v.encode() for k, v in GOOD.items()}, b"event_id": b"\xff\xfe"})
    assert as_text(b"\xff") == b"\xff" and as_text(b"ok") == "ok"


@pytest.mark.parametrize("bad, reason", [
    ({"event_id": "x"}, "missing fields"), ({**GOOD, "user_id": "x"}, "invalid literal for int()"),
    ({**GOOD, "watch_ratio": "-1"}, "invalid values"), ({**GOOD, "event_id": "a:b"}, "invalid values"),
    ({**GOOD, "action": "like"}, "invalid values"), ({**GOOD, "ts": "abc"}, "could not convert string to float")])
def test_poison_reasons(bad, reason):
    with pytest.raises(PoisonEvent, match=reason):
        parse_fields(bad)


def test_latency_bucket_handles_nan():
    # a NaN produced_ms used to raise StopIteration inside the coroutine -> RuntimeError -> worker crash
    assert lat_bucket(math.nan) == len(LAT_BUCKETS_MS) - 1
    assert lat_bucket(0.0) == 0 and lat_bucket(0.7) == 1 and lat_bucket(1e9) == len(LAT_BUCKETS_MS) - 1
    assert LAT_BUCKETS_MS == sorted(LAT_BUCKETS_MS) and LAT_BUCKETS_MS[-1] == math.inf
