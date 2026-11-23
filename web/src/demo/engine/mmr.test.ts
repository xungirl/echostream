// @vitest-environment node
// Mirrors tests/unit/test_mmr.py
import { describe, expect, it } from "vitest";
import { mmrRerank } from "./mmr";

const zeros = (n: number) => new Float32Array(n * n);

describe("MMR", () => {
  it("lambda = 1 is relevance order", () => {
    expect(mmrRerank([0.1, 0.9, 0.5], zeros(3), 1, 3)).toEqual([1, 2, 0]);
  });

  it("diversity pushes a duplicate-category item down", () => {
    // categories {1}, {1}, {2} -> Jaccard similarity
    const sim = new Float32Array([1, 1, 0, 1, 1, 0, 0, 0, 1]);
    // after picking 0: item1 = .5*.9 - .5*1 = -0.05 ; item2 = .5*0 - 0 = 0 -> item2 wins
    expect(mmrRerank([1, 0.95, 0.5], sim, 0.5, 3)).toEqual([0, 2, 1]);
    expect(mmrRerank([1, 0.95, 0.5], sim, 1, 3)).toEqual([0, 1, 2]);
  });

  it("edge cases: empty, ties keep the first, k larger than n", () => {
    expect(mmrRerank([], zeros(0), 0.5, 3)).toEqual([]);
    expect(mmrRerank([0.3, 0.3], zeros(2), 0.5, 5)).toEqual([0, 1]);
  });
});
