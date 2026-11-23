/**
 * LightGBM inference for the ranker, from the trees in ranker.json (no library).
 *
 * Every split in this model is numerical with missing_type "None" (LightGBM decision_type 2):
 * a NaN feature is treated as 0, then x <= threshold goes left. Child indices < 0 are leaves
 * (leaf = ~child). The raw score is the sum of one leaf per tree, added in tree order in
 * float64 like LightGBM; the binary objective turns it into a probability with a sigmoid.
 */
import type { RankerJson } from "./bundle";

export function leafIndex(tree: RankerJson["trees"][number], x: ArrayLike<number>): number {
  if (tree.split_feature.length === 0) return 0; // single-leaf tree
  let node = 0;
  for (;;) {
    let v = x[tree.split_feature[node]];
    if (Number.isNaN(v)) v = 0;
    const next = v <= tree.threshold[node] ? tree.left_child[node] : tree.right_child[node];
    if (next < 0) return ~next;
    node = next;
  }
}

export function rawScore(model: RankerJson, x: ArrayLike<number>): number {
  let s = 0;
  for (const t of model.trees) s += t.leaf_value[leafIndex(t, x)];
  return s;
}

export function predict(model: RankerJson, x: ArrayLike<number>): number {
  return 1 / (1 + Math.exp(-model.sigmoid * rawScore(model, x)));
}

/** Predict every row of a row-major feature matrix [rows, num_features]. */
export function predictRows(model: RankerJson, X: Float32Array, rows: number): Float64Array {
  const d = model.num_features;
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) out[r] = predict(model, X.subarray(r * d, (r + 1) * d));
  return out;
}
