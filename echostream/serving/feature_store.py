"""Sharded real-time feature store on several Redis nodes.

A user's keys all live on ONE node chosen by consistent hashing on the user id,
so applying an event (dedupe check + history push + counters) is a single
atomic Lua script on that node. Keys per user:

    rt:{uid}:hist   list of "item:watch_ratio:ts", newest first (capped)
    rt:{uid}:cnt    hash: n, play, complete, skip, pos, sum_wr, last_ts,
                    pos_cat:<c>, skip_cat:<c>
    ev:{event_id}   dedupe marker with TTL (exactly-once *effect* on top of
                    at-least-once delivery)
"""
from __future__ import annotations

import asyncio

import redis.asyncio as aioredis
from redis.exceptions import NoScriptError

from echostream.core.hashing import ConsistentHashRing

APPLY_EVENT_LUA = """
-- KEYS: dedupe, hist, cnt, rec-cache   ARGV: ttl, item, wr, ts, action, cats(csv), hist_max, pos_threshold
if not redis.call('SET', KEYS[1], '1', 'NX', 'EX', tonumber(ARGV[1])) then
  return 0
end
redis.call('LPUSH', KEYS[2], ARGV[2] .. ':' .. ARGV[3] .. ':' .. ARGV[4])
redis.call('LTRIM', KEYS[2], 0, tonumber(ARGV[7]) - 1)
redis.call('HINCRBY', KEYS[3], 'n', 1)
redis.call('HINCRBY', KEYS[3], ARGV[5], 1)
redis.call('HINCRBYFLOAT', KEYS[3], 'sum_wr', ARGV[3])
redis.call('HSET', KEYS[3], 'last_ts', ARGV[4])
redis.call('DEL', KEYS[4])   -- cached recommendations are now stale
local positive = tonumber(ARGV[3]) > tonumber(ARGV[8])
if positive then redis.call('HINCRBY', KEYS[3], 'pos', 1) end
for c in string.gmatch(ARGV[6], '[^,]+') do
  if positive then redis.call('HINCRBY', KEYS[3], 'pos_cat:' .. c, 1) end
  if ARGV[5] == 'skip' then redis.call('HINCRBY', KEYS[3], 'skip_cat:' .. c, 1) end
end
return 1
"""


def hist_key(uid: int) -> str:
    return f"rt:{uid}:hist"


def cnt_key(uid: int) -> str:
    return f"rt:{uid}:cnt"


def parse_hist(raw: list[bytes | str]) -> list[tuple[int, float, float]]:
    """Newest-first list of 'item:wr:ts' -> [(item, wr, ts)] oldest-first."""
    out = []
    for x in reversed(raw):
        s = x.decode() if isinstance(x, bytes) else x
        item, wr, ts = s.split(":")
        out.append((int(item), float(wr), float(ts)))
    return out


class FeatureStore:
    def __init__(self, nodes: dict[str, str], vnodes: int = 100, socket_timeout: float = 0.5):
        self.ring = ConsistentHashRing(list(nodes), vnodes)
        self.clients = {name: aioredis.from_url(url, socket_timeout=socket_timeout,
                                                socket_connect_timeout=socket_timeout)
                        for name, url in nodes.items()}
        self._sha: dict[str, str] = {}

    def node_for(self, uid: int) -> str:
        return self.ring.get_node(f"user:{uid}")

    def client_for(self, uid: int) -> aioredis.Redis:
        return self.clients[self.node_for(uid)]

    async def get(self, uid: int, hist_n: int = 50) -> dict:
        c = self.client_for(uid)
        async with c.pipeline(transaction=False) as p:
            p.lrange(hist_key(uid), 0, hist_n - 1)
            p.hgetall(cnt_key(uid))
            raw_hist, raw_cnt = await p.execute()
        counts = {k.decode(): float(v) for k, v in raw_cnt.items()}
        return {"hist": parse_hist(raw_hist), "counts": counts, "node": self.node_for(uid)}

    async def script_sha(self, node: str) -> str:
        if node not in self._sha:
            self._sha[node] = await self.clients[node].script_load(APPLY_EVENT_LUA)
        return self._sha[node]

    @staticmethod
    def apply_args(ev: dict, cats: str, ttl: int, hist_max: int, thr: float) -> tuple[list[str], list]:
        keys = [f"ev:{ev['event_id']}", hist_key(ev["user_id"]), cnt_key(ev["user_id"]), f"rec:{ev['user_id']}"]
        args = [ttl, ev["item_id"], repr(float(ev["watch_ratio"])), repr(float(ev["ts"])), ev["action"],
                cats, hist_max, thr]
        return keys, args

    async def apply_batch(self, node: str, calls: list[tuple[list[str], list]]) -> list:
        """Run the apply script for several events on one node, in order, in one
        round trip. Returns per event: 1 applied, 0 duplicate, or an Exception
        (a deterministic script error -> caller dead-letters it). Connection
        errors propagate (transient -> caller retries later)."""
        c = self.clients[node]
        sha = await self.script_sha(node)
        async with c.pipeline(transaction=False) as p:
            for keys, args in calls:
                p.evalsha(sha, len(keys), *keys, *args)
            results = await p.execute(raise_on_error=False)
        if any(isinstance(r, NoScriptError) for r in results):
            # node restarted and lost its script cache: reload and re-run the whole
            # batch -- safe because the script is idempotent (dedupe key)
            self._sha.pop(node, None)
            return await self.apply_batch(node, calls)
        return results

    async def ping_all(self) -> dict[str, bool]:
        async def ping(name, c):
            try:
                return name, bool(await asyncio.wait_for(c.ping(), 0.3))
            except Exception:
                return name, False
        return dict(await asyncio.gather(*(ping(n, c) for n, c in self.clients.items())))

    async def close(self) -> None:
        for c in self.clients.values():
            await c.aclose()
