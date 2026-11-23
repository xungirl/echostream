"""Event schema shared by the API (producer) and the workers (consumers)."""
from __future__ import annotations

import time

from echostream.core.hashing import partition_for
from echostream.serving.config import EVENTS

ACTIONS = ("play", "complete", "skip")
REQUIRED = ("event_id", "user_id", "item_id", "action", "watch_ratio", "ts", "produced_ms")


def stream_name(partition: int) -> str:
    return f"{EVENTS.stream_prefix}:{partition}"


def stream_for_user(user_id: int, n_partitions: int) -> str:
    return stream_name(partition_for(user_id, n_partitions))


def to_fields(ev: dict, cats: str) -> dict[str, str]:
    return {"event_id": str(ev["event_id"]), "user_id": str(ev["user_id"]), "item_id": str(ev["item_id"]),
            "action": ev["action"], "watch_ratio": repr(float(ev["watch_ratio"])), "ts": repr(float(ev["ts"])),
            "cats": cats, "produced_ms": repr(time.time() * 1000.0)}


class PoisonEvent(ValueError):
    pass


NOT_UTF8 = "fields are not valid UTF-8"


def as_text(x):
    """bytes -> str if it is valid UTF-8, else keep the raw bytes (used for the DLQ copy)."""
    if isinstance(x, bytes):
        try:
            return x.decode()
        except UnicodeDecodeError:
            return x
    return x


def parse_fields(fields: dict) -> dict:
    """Validate a raw stream entry. Raises PoisonEvent for anything malformed."""
    try:
        f = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
             for k, v in fields.items()}
    except UnicodeDecodeError as e:   # not PoisonEvent -> would crash the worker on every redelivery
        raise PoisonEvent(NOT_UTF8) from e
    missing = [k for k in REQUIRED if k not in f]
    if missing:
        raise PoisonEvent(f"missing fields {missing}")
    try:
        ev = {"event_id": f["event_id"], "user_id": int(f["user_id"]), "item_id": int(f["item_id"]),
              "action": f["action"], "watch_ratio": float(f["watch_ratio"]), "ts": float(f["ts"]),
              "cats": f.get("cats", ""), "produced_ms": float(f["produced_ms"])}
    except ValueError as e:
        raise PoisonEvent(str(e)) from e
    if ev["action"] not in ACTIONS or ev["watch_ratio"] < 0 or ev["user_id"] < 0 or ev["item_id"] < 0 \
            or not ev["event_id"] or ":" in ev["event_id"]:
        raise PoisonEvent(f"invalid values {f}")
    return ev
