/**
 * Diversity re-rank (echostream/core/mmr.py + rank/offline.blend_similarity).
 * Greedy: pick argmax of  lambda * relevance - (1 - lambda) * max similarity to anything picked,
 * relevance min-max normalized. Similarity = mean(category Jaccard, caption cosine clipped to [0, 1]).
 */
import type { Model } from "./bundle";

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

/** [n, n] float32 similarity between the given items. */
export function blendSimilarity(m: Model, items: number[]): Float32Array {
  const n = items.length;
  const d = m.capDim;
  const sim = new Float32Array(n * n);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      const ba = m.itemCatBits[items[a]];
      const bb = m.itemCatBits[items[b]];
      const inter = popcount(ba & bb);
      const union = popcount(ba) + popcount(bb) - inter;
      const jac = union > 0 ? Math.fround(inter / union) : 1;
      let s = 0;
      const oa = items[a] * d;
      const ob = items[b] * d;
      for (let j = 0; j < d; j++) s += m.caption[oa + j] * m.caption[ob + j];
      const cos = Math.min(Math.max(Math.fround(s), 0), 1);
      sim[a * n + b] = 0.5 * jac + 0.5 * cos; // float32 store = numpy's float32 sum
    }
  }
  return sim;
}

/** Returns k positions into the candidate list. */
export function mmrRerank(relevance: ArrayLike<number>, sim: Float32Array, lam: number, k: number): number[] {
  const n = relevance.length;
  k = Math.min(k, n);
  if (n === 0) return [];
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    lo = Math.min(lo, relevance[i]);
    hi = Math.max(hi, relevance[i]);
  }
  const span = hi - lo;
  const rel = Array.from(relevance, (r) => (span > 0 ? (r - lo) / span : 1));
  const maxSim = new Float64Array(n);
  const available = new Uint8Array(n).fill(1);
  const chosen: number[] = [];
  for (let step = 0; step < k; step++) {
    let best = -1;
    let bestVal = -Infinity;
    for (let i = 0; i < n; i++) {
      if (!available[i]) continue;
      const v = lam * rel[i] - (1 - lam) * maxSim[i];
      if (v > bestVal) {
        bestVal = v;
        best = i; // strict '>' keeps the first maximum, like np.argmax
      }
    }
    chosen.push(best);
    available[best] = 0;
    for (let i = 0; i < n; i++) maxSim[i] = Math.max(maxSim[i], sim[best * n + i]);
  }
  return chosen;
}
