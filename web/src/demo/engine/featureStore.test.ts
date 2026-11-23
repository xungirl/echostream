// @vitest-environment node
import { describe, expect, it } from "vitest";
import { eventType, RealtimeStore, type RtEvent } from "./featureStore";

const ev = (k: number, wr: number, cats = [3]): RtEvent => ({ eventId: `e${k}`, item: 100 + k, watchRatio: wr, action: eventType(wr), ts: k, cats });

describe("in-browser feature store (same effect as the worker's Lua script)", () => {
  it("maps watch ratios to actions like POST /events", () => {
    expect([eventType(0.1), eventType(0.5), eventType(1.0), eventType(3)]).toEqual(["skip", "play", "complete", "complete"]);
  });

  it("dedupes on event id: a redelivered event changes nothing", () => {
    const s = new RealtimeStore(100, 2.0);
    expect(s.apply(ev(1, 3))).toBe(1);
    expect(s.apply(ev(1, 3))).toBe(0);
    expect(s.hist).toHaveLength(1);
    expect(s.counts.get("n")).toBe(1);
  });

  it("keeps the newest 100 and reads the newest 50, oldest first", () => {
    const s = new RealtimeStore(100, 2.0);
    for (let k = 0; k < 120; k++) s.apply(ev(k, 1));
    expect(s.hist).toHaveLength(100);
    const r = s.recent(50);
    expect(r).toHaveLength(50);
    expect(r[0].item).toBe(100 + 70);
    expect(r[49].item).toBe(100 + 119);
  });

  it("counts positives (watch ratio > 2) and skips per category", () => {
    const s = new RealtimeStore(100, 2.0);
    s.apply(ev(1, 3, [1, 2]));
    s.apply(ev(2, 0.1, [2]));
    s.apply(ev(3, 2.0, [1])); // exactly 2.0 is not a positive
    expect(s.counts.get("pos")).toBe(1);
    expect(s.counts.get("pos_cat:1")).toBe(1);
    expect(s.counts.get("pos_cat:2")).toBe(1);
    expect(s.counts.get("skip_cat:2")).toBe(1);
    expect(s.counts.get("skip")).toBe(1);
    expect(s.counts.get("complete")).toBe(2);
    expect(s.counts.get("sum_wr")).toBeCloseTo(5.1, 12);
  });
});
