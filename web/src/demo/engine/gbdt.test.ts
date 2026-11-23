// @vitest-environment node
import { describe, expect, it } from "vitest";
import type { RankerJson } from "./bundle";
import { leafIndex, predict, predictRows, rawScore } from "./gbdt";
import { loadTestModel } from "./testBundle";

// tree:            x0 <= 0.5 ?
//                 /           \
//          leaf0 (-1)      x1 <= 2 ?
//                         /        \
//                   leaf1 (0.5)   leaf2 (2)
const tree = { split_feature: [0, 1], threshold: [0.5, 2], left_child: [-1, -2], right_child: [1, -3], leaf_value: [-1, 0.5, 2] };
const model: RankerJson = { objective: "binary", sigmoid: 1, num_features: 2, feature_names: ["a", "b"], trees: [tree, tree] };

describe("LightGBM tree evaluation", () => {
  it("follows x <= threshold to the left, negative children are leaves (~child)", () => {
    expect(leafIndex(tree, [0.5, 9])).toBe(0); // equal goes left
    expect(leafIndex(tree, [0.6, 2])).toBe(1);
    expect(leafIndex(tree, [0.6, 2.0000001])).toBe(2);
  });

  it("treats NaN as 0 (missing_type None)", () => {
    expect(leafIndex(tree, [NaN, 3])).toBe(0);
    expect(leafIndex(tree, [1, NaN])).toBe(1);
  });

  it("sums one leaf per tree, then a sigmoid", () => {
    expect(rawScore(model, [1, 3])).toBe(4);
    expect(predict(model, [1, 3])).toBeCloseTo(1 / (1 + Math.exp(-4)), 15);
    expect(Array.from(predictRows(model, new Float32Array([0, 0, 1, 1]), 2))).toEqual([predict(model, [0, 0]), predict(model, [1, 1])]);
  });

  it("handles single-leaf trees", () => {
    expect(rawScore({ ...model, trees: [{ split_feature: [], threshold: [], left_child: [], right_child: [], leaf_value: [0.25] }] }, [5, 5])).toBe(0.25);
  });

  it("the shipped ranker: 210 trees x 63 leaves over 40 features", () => {
    const r = loadTestModel().ranker;
    expect(r.trees).toHaveLength(210);
    expect(r.num_features).toBe(40);
    expect(r.trees.every((t) => t.leaf_value.length === 63 && t.split_feature.length === 62)).toBe(true);
  });
});
