import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import Waterfall, { computeBars } from "../components/Waterfall";
import type { ShardCall } from "../types";

const timings = { features: 1, user_vector: 1, recall: 6, recall_local: 2, recall_ann_wait: 4, merge: 1, rank: 1, total: 10 };
const shards: ShardCall[] = [
  { shard: 0, ok: true, replica: 0, ms: 3, attempts: [{ replica: 0, status: "ok", ms: 3 }] },
  { shard: 1, ok: true, replica: 1, ms: 5, attempts: [{ replica: 0, status: "timeout", ms: 4.1 }, { replica: 1, status: "ok", ms: 0.9 }] },
];

describe("waterfall", () => {
  it("lays bars out proportionally on one ms axis", () => {
    const { scale, bars } = computeBars(timings, shards);
    const by = Object.fromEntries(bars.map((b) => [b.key, b]));
    expect(scale).toBe(10);
    expect(by.total.width).toBeCloseTo(100);
    expect(by.features.width).toBeCloseTo(10);
    expect(by.user_vector.left).toBeCloseTo(10);
    expect(by.recall.left).toBeCloseTo(20);
    expect(by.recall.width).toBeCloseTo(60);
    // sub-bars: local starts with recall, ann wait follows it
    expect(by.recall_local.left).toBeCloseTo(20);
    expect(by.recall_local.width).toBeCloseTo(20);
    expect(by.recall_ann_wait.left).toBeCloseTo(40);
    expect(by.recall_ann_wait.width).toBeCloseTo(40);
    expect(by.merge.left).toBeCloseTo(80);
    expect(by.rank.left).toBeCloseTo(90);
    expect(by.mmr).toBeUndefined(); // absent timing -> no bar
    // shard calls start when recall starts
    expect(by.shard1.left).toBeCloseTo(20);
    expect(by.shard1.width).toBeCloseTo(50);
  });

  it("marks failed shard attempts in red and shows which replica answered", () => {
    render(<Waterfall timings={timings} shards={shards} />);
    const failed = document.querySelectorAll(".attempt.failed");
    expect(failed).toHaveLength(1);
    expect(failed[0]).toHaveTextContent("r0 timeout 4.1ms");
    expect(screen.getByTestId("wf-shard1")).toHaveTextContent("→ r1");
    expect(screen.getByTestId("wf-shard0").querySelector(".attempt")).toBeNull();
  });

  it("marks a shard with no answer", () => {
    const dead: ShardCall[] = [{ shard: 2, ok: false, replica: null, ms: 40, attempts: [{ replica: 0, status: "error", ms: 1 }, { replica: 1, status: "timeout", ms: 39 }] }];
    render(<Waterfall timings={timings} shards={dead} />);
    expect(screen.getByTestId("wf-shard2").querySelector(".wf-bar.bad")).not.toBeNull();
    expect(screen.getByText("no replica answered")).toBeInTheDocument();
    // the shard call (40 ms) is longer than total, so the axis grows to fit it
    expect(computeBars(timings, dead).scale).toBe(42);
  });
});
