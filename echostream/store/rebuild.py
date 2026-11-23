"""Rebuild the Redis feature store from the Postgres event archive.

Redis holds derived data only; if a feature node loses its memory (restart
without persistence, or a bad deploy) this re-applies every archived event,
per partition in original stream order (so per-user order is preserved), with
the same idempotent Lua script the workers use.

    python -m echostream.store.rebuild [--no-flush]
"""
from __future__ import annotations

import argparse
import asyncio
import time
from collections import defaultdict

from echostream.serving.config import CLUSTER, EVENTS
from echostream.serving.feature_store import FeatureStore
from echostream.store import db

ORDERED = """SELECT event_id, user_id, item_id, action, watch_ratio, ts, cats FROM event_archive
             ORDER BY partition, split_part(stream_id, '-', 1)::bigint, split_part(stream_id, '-', 2)::bigint"""


async def rebuild(feature_nodes: dict[str, str] | None = None, flush: bool = True, batch: int = 2000) -> dict:
    t0 = time.time()
    fs = FeatureStore(feature_nodes or CLUSTER.feature_nodes, socket_timeout=5)
    if flush:
        for c in fs.clients.values():
            await c.flushdb()
    n = 0
    with db.connect(autocommit=False) as conn:          # server-side cursors need a transaction
        cur = conn.cursor(name="archive_scan")   # server-side cursor: constant memory
        cur.execute(ORDERED)
        while rows := cur.fetchmany(batch):
            by_node: dict[str, list] = defaultdict(list)
            for r in rows:
                ev = dict(r)
                by_node[fs.node_for(ev["user_id"])].append(
                    FeatureStore.apply_args(ev, ev["cats"], EVENTS.dedupe_ttl_s, EVENTS.hist_max, EVENTS.positive_threshold))
            await asyncio.gather(*(fs.apply_batch(node, calls) for node, calls in by_node.items()))
            n += len(rows)
    await fs.close()
    return {"events_replayed": n, "seconds": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-flush", action="store_true")
    args = ap.parse_args()
    print(asyncio.run(rebuild(flush=not args.no_flush)))


if __name__ == "__main__":
    main()
