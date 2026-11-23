/**
 * The read path's first half: user tower -> five recall paths -> quota merge.
 * Mirrors echostream/core/two_tower.py (user_tower_numpy), echostream/recall/engine.py,
 * echostream/core/itemcf.py and echostream/core/recall_merge.py line by line.
 *
 * Float rules (why the browser can match numpy): Float32Array stores round every value to
 * float32, and one +, -, *, / or sqrt computed in float64 and rounded once to float32 is
 * exactly numpy's float32 result. Only long dot products (BLAS) differ: numpy sums them in
 * float32 in its own order, we sum in float64 and round once -> differences of ~1 ulp.
 */
import type { Model, PathName } from "./bundle";

export type Scored = [item: number, score: number];
export type Paths = Record<PathName, Scored[]>;

const f32 = Math.fround;

/** Dot product of row `r` of a row-major matrix with `v` (float64 sum, rounded once). */
export function dotRow(m: Float32Array, dim: number, r: number, v: Float32Array): number {
  let s = 0;
  const o = r * dim;
  for (let j = 0; j < dim; j++) s += m[o + j] * v[j];
  return f32(s);
}

/** numpy `x[rows].mean(axis=0)` on float32: sequential float32 sum, then one division. */
export function meanRows(m: Float32Array, dim: number, rows: ArrayLike<number>): Float32Array {
  const out = new Float32Array(dim);
  for (let k = 0; k < rows.length; k++) {
    const o = rows[k] * dim;
    for (let j = 0; j < dim; j++) out[j] = out[j] + m[o + j]; // the store rounds to float32
  }
  for (let j = 0; j < dim; j++) out[j] = out[j] / rows.length;
  return out;
}

/** two_tower.user_tower_numpy: [field embeddings, mean item-id embedding of the last 50 positives] -> MLP -> L2 norm. */
export function userVector(m: Model, userCats: number[], hist: ArrayLike<number>): Float32Array {
  const nF = m.userFields.length;
  const fd = m.fieldDim;
  const inDim = nF * fd + m.idDim;
  const x = new Float32Array(inDim);
  for (let f = 0; f < nF; f++) x.set(m.userFields[f].subarray(userCats[f] * fd, (userCats[f] + 1) * fd), f * fd);
  if (hist.length) x.set(meanRows(m.idEmb, m.idDim, hist), nF * fd);
  const hDim = m.b1.length;
  const h = new Float32Array(hDim);
  for (let i = 0; i < hDim; i++) h[i] = Math.max(f32(dotRow(m.w1, inDim, i, x) + m.b1[i]), 0);
  const oDim = m.b2.length;
  const out = new Float32Array(oDim);
  for (let i = 0; i < oDim; i++) out[i] = dotRow(m.w2, hDim, i, h) + m.b2[i];
  let ss = 0;
  for (let i = 0; i < oDim; i++) ss += out[i] * out[i];
  const norm = f32(Math.sqrt(f32(ss)));
  const d = Math.max(norm, 1e-12);
  for (let i = 0; i < oDim; i++) out[i] = out[i] / d;
  return out;
}

/**
 * engine.topk_excluding: the k best (score desc, then item id asc) among items not excluded
 * (and allowed). `score(i)` is only called for candidate items.
 */
export function topK(
  n: number,
  score: (i: number) => number,
  k: number,
  excluded: Uint8Array,
  allowed?: Uint8Array,
): Scored[] {
  const all: Scored[] = [];
  for (let i = 0; i < n; i++) {
    if (excluded[i] || (allowed && !allowed[i])) continue;
    all.push([i, score(i)]);
  }
  all.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  return all.slice(0, k);
}

/** Two-tower path: exact inner-product search over every item vector (the shards' Flat index). */
export function twoTower(m: Model, uvec: Float32Array, k: number, excluded: Uint8Array): Scored[] {
  return topK(m.nItems, (i) => dotRow(m.itemEmb, m.embDim, i, uvec), k, excluded);
}

/**
 * itemcf.itemcf_scores with recency decay: score[j] = sum_h w(age of h) * sim(h, j), over the
 * 20 most recent positives. numpy does this in float32 with np.add.at, item by item in history
 * order, so every step is one float32 multiply and one float32 add - reproduced exactly here.
 */
export function itemcfScores(m: Model, hist: ArrayLike<number>): Float32Array {
  const scores = new Float32Array(m.nItems);
  const n = hist.length;
  for (let pos = 0; pos < n; pos++) {
    const w = f32(m.c.cf_weights[n - 1 - pos]);
    const h = hist[pos];
    for (let k = m.cfStart[h]; k < m.cfStart[h + 1]; k++) {
      const j = m.cfIds[k];
      scores[j] = scores[j] + f32(w * m.cfSims[k]);
    }
  }
  return scores;
}

export function itemcf(m: Model, hist: ArrayLike<number>, k: number, excluded: Uint8Array): Scored[] {
  if (hist.length === 0) return [];
  const s = itemcfScores(m, lastN(hist, m.c.cf_hist));
  return topK(m.nItems, (i) => s[i], k, excluded).filter(([, sc]) => sc > 0);
}

export function popularity(m: Model, name: "pop_recent" | "pop_platform", k: number, excluded: Uint8Array): Scored[] {
  const p = m.pop[name];
  return topK(m.nItems, (i) => p[i], k, excluded);
}

/** Content path: caption similarity to the mean of the 10 most recent positives, cold videos only. */
export function contentQuery(m: Model, hist: ArrayLike<number>): Float32Array {
  return meanRows(m.caption, m.capDim, lastN(hist, m.c.content_hist));
}

export function content(m: Model, hist: ArrayLike<number>, k: number, excluded: Uint8Array): Scored[] {
  if (hist.length === 0) return [];
  const q = contentQuery(m, hist);
  return topK(m.nItems, (i) => dotRow(m.caption, m.capDim, i, q), k, excluded, m.cold);
}

export function lastN(a: ArrayLike<number>, n: number): number[] {
  return Array.from(a).slice(Math.max(0, a.length - n));
}

// ---------------------------------------------------------------- merge
export interface Candidate {
  item: number;
  source: PathName; // the path that claimed it first
  score: number; // that path's score
  allSources: Partial<Record<PathName, number>>; // every path that proposed it
}

/**
 * recall_merge.merge_with_quotas: take up to quota[path] items from each path in priority order,
 * skipping duplicates; fill leftover slots round-robin; remember every path that proposed an item.
 */
export function mergeWithQuotas(paths: Paths, order: PathName[], quotas: Record<PathName, number>, total: number): Candidate[] {
  const byItem = new Map<number, Candidate>();
  const merged: Candidate[] = [];
  const cursor: Record<string, number> = {};
  for (const p of order) cursor[p] = 0;
  const note = (p: PathName, item: number, score: number) => {
    const c = byItem.get(item);
    if (c && !(p in c.allSources)) c.allSources[p] = score;
  };
  const takeFrom = (p: PathName, limit: number): number => {
    let taken = 0;
    const lst = paths[p];
    while (cursor[p] < lst.length && taken < limit && merged.length < total) {
      const [item, score] = lst[cursor[p]++];
      if (byItem.has(item)) {
        note(p, item, score);
        continue;
      }
      const c: Candidate = { item, source: p, score, allSources: { [p]: score } };
      byItem.set(item, c);
      merged.push(c);
      taken++;
    }
    return taken;
  };
  for (const p of order) takeFrom(p, quotas[p] ?? 0);
  let progress = true;
  while (merged.length < total && progress) {
    progress = false;
    for (const p of order) if (takeFrom(p, 1)) progress = true;
  }
  for (const p of order) for (const [item, score] of paths[p].slice(cursor[p])) note(p, item, score);
  return merged;
}
