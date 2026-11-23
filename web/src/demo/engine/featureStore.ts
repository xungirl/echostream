/**
 * The real-time feature store, in the browser. Same effect as the event worker's Lua script
 * (echostream/serving/feature_store.py APPLY_EVENT_LUA), which the Redis node runs atomically:
 *   dedupe on event_id -> LPUSH "item:wr:ts" to the user's history (newest first), LTRIM to 100
 *   -> counters n, <action>, sum_wr, last_ts, pos, pos_cat:<c>, skip_cat:<c>
 * and the same read as FeatureStore.get: the newest 50 history entries, returned oldest first.
 */

export type Action = "play" | "complete" | "skip";

export interface RtEvent {
  eventId: string;
  item: number;
  watchRatio: number;
  action: Action;
  ts: number;
  cats: number[]; // the item's tag categories (the cats CSV the API attaches)
}

/** KuaiRecAdapter.event_type: how POST /events derives the action from the watch ratio. */
export function eventType(watchRatio: number): Action {
  if (watchRatio < 0.3) return "skip";
  if (watchRatio >= 1.0) return "complete";
  return "play";
}

export class RealtimeStore {
  /** newest first, like the Redis list */
  hist: { item: number; wr: number; ts: number }[] = [];
  counts = new Map<string, number>();
  private applied = new Set<string>();

  constructor(
    private histMax: number,
    private posThreshold: number,
  ) {}

  private incr(key: string, by = 1) {
    this.counts.set(key, (this.counts.get(key) ?? 0) + by);
  }

  /** 1 = applied, 0 = duplicate (already applied: at-least-once delivery, exactly-once effect). */
  apply(ev: RtEvent): 0 | 1 {
    if (this.applied.has(ev.eventId)) return 0;
    this.applied.add(ev.eventId);
    this.hist.unshift({ item: ev.item, wr: ev.watchRatio, ts: ev.ts });
    this.hist.length = Math.min(this.hist.length, this.histMax);
    this.incr("n");
    this.incr(ev.action);
    this.incr("sum_wr", ev.watchRatio);
    this.counts.set("last_ts", ev.ts);
    const positive = ev.watchRatio > this.posThreshold;
    if (positive) this.incr("pos");
    for (const c of ev.cats) {
      if (positive) this.incr(`pos_cat:${c}`);
      if (ev.action === "skip") this.incr(`skip_cat:${c}`);
    }
    return 1;
  }

  /** FeatureStore.get(hist_n): newest n entries, oldest first. */
  recent(n: number): { item: number; wr: number; ts: number }[] {
    return this.hist.slice(0, n).reverse();
  }
}
