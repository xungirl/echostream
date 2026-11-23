// @vitest-environment node
// Mirrors tests/unit/test_recall_merge.py (paths renamed to the pipeline's path names)
import { describe, expect, it } from "vitest";
import type { PathName } from "./bundle";
import { mergeWithQuotas, type Paths } from "./recall";

const paths = (p: Partial<Paths>): Paths => ({ two_tower: [], itemcf: [], pop_recent: [], pop_platform: [], content: [], ...p });
const q = (x: Partial<Record<PathName, number>>) => ({ two_tower: 0, itemcf: 0, pop_recent: 0, pop_platform: 0, content: 0, ...x });
const ORDER: PathName[] = ["two_tower", "itemcf", "pop_recent", "pop_platform", "content"];

describe("quota merge", () => {
  it("takes quotas in priority order, dedups, remembers every proposing path", () => {
    const out = mergeWithQuotas(
      paths({ two_tower: [[1, 0.9], [2, 0.8], [3, 0.7]], itemcf: [[2, 5], [4, 4], [5, 3]], pop_recent: [[6, 100]] }),
      ORDER, q({ two_tower: 2, itemcf: 2, pop_recent: 1 }), 5);
    expect(out.map((c) => c.item)).toEqual([1, 2, 4, 5, 6]);
    expect(out.map((c) => c.source)).toEqual(["two_tower", "two_tower", "itemcf", "itemcf", "pop_recent"]);
    expect(out[1].allSources).toEqual({ two_tower: 0.8, itemcf: 5 });
  });

  it("fills leftover slots round-robin", () => {
    const out = mergeWithQuotas(paths({ two_tower: [[1, 1], [2, 0.9], [3, 0.8]], itemcf: [[10, 1]] }), ORDER, q({ two_tower: 1, itemcf: 3 }), 3);
    expect(out.map((c) => c.item)).toEqual([1, 10, 2]);
  });

  it("caps at total and handles empty paths", () => {
    const many = Array.from({ length: 10 }, (_, i): [number, number] => [i, 1]);
    expect(mergeWithQuotas(paths({ two_tower: many }), ORDER, q({ two_tower: 10, itemcf: 5 }), 4)).toHaveLength(4);
    expect(mergeWithQuotas(paths({}), ORDER, q({ two_tower: 3 }), 3)).toEqual([]);
  });
});
