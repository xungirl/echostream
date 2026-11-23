"""Scripted viewer sessions: the same swipes are replayed by the Python golden generator, the
quantization sweep and (from the golden file) the TypeScript parity test.

A session = the first screen, then for each step one event on a video of the current screen
followed by a fresh screen. The demo's three buttons map to KuaiRec watch ratios (KuaiRec has
no like button; "liked" means watch_ratio > 2.0, i.e. watched more than twice):
    skip  -> 0.1  (action "skip":     left within the first 30%)
    watch -> 1.0  (action "complete": watched to the end once; not a positive)
    like  -> 3.0  (action "complete": watched three times;     a positive)
"""
from __future__ import annotations

from echostream.domain.kuairec import KuaiRecAdapter

WATCH_RATIO = {"skip": 0.1, "watch": 1.0, "like": 3.0}
SCRIPT = ["like", "skip", "watch", "like", "skip"]
TS0 = 1_700_000_000.0          # fixed event timestamps (they do not affect recommendations)
_ADAPTER = KuaiRecAdapter()


def action_for(kind: str) -> tuple[float, str]:
    wr = WATCH_RATIO[kind]
    return wr, _ADAPTER.event_type(wr)


def screen_ids(res) -> list[int]:
    return [x["item_id"] for x in res.items]


async def run_session(lp, uid: int, script: list[str] = SCRIPT, events: list[dict] | None = None) -> list[dict]:
    """Returns one record per screen: {"event": the event applied before it (None for the first),
    "result": Result, "trace": Trace}. With `events`, replay those events instead of choosing
    targets from the current screen (used to compare two models on identical inputs)."""
    out = []
    res, tr = await lp.recommend(uid)
    out.append({"event": None, "result": res, "trace": tr})
    for k, kind in enumerate(script):
        if events is not None:
            ev = events[k]
        else:
            wr, action = action_for(kind)
            ids = screen_ids(res)
            ev = {"kind": kind, "item_id": ids[k % len(ids)], "watch_ratio": wr, "action": action, "ts": TS0 + k}
        await lp.post_event(uid, ev["item_id"], ev["watch_ratio"], ev["action"], ev["ts"])
        res, tr = await lp.recommend(uid)
        out.append({"event": ev, "result": res, "trace": tr})
    return out
