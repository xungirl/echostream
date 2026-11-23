/**
 * Ranking features: echostream/rank/features.py build_features, the ONE feature builder the
 * offline trainer and the online API share. 40 features per candidate, in FEATURE_NAMES order:
 *   user (6)  item (14)  cross: cat_affinity, dur_gap  recall: 5 path scores + 5 "proposed by" flags
 *   + n_paths  realtime (7): window size, positive / skip rate, category hits, caption similarity
 *   to the last 5 positives.
 */
import { PATHS, type Model, type UserRecord } from "./bundle";
import { contentQuery, dotRow, itemcfScores, lastN, type Candidate } from "./recall";

export const N_FEATURES = 40;
const f32 = Math.fround;

/** Dot product of item i's tag-category row (0/1) with a dense per-category vector. */
function catDot(m: Model, item: number, v: Float32Array): number {
  const bits = m.itemCatBits[item];
  let s = 0;
  for (let c = 0; c < m.c.n_tag_categories; c++) if ((bits >>> c) & 1) s += v[c];
  return f32(s);
}

/**
 * user: the user's record (batch features); uvec: user vector; histPos: batch + real-time
 * positives; recent: the last events (item, watch_ratio), oldest first (batch window + real-time).
 */
export function buildFeatures(
  m: Model,
  user: UserRecord,
  uvec: Float32Array,
  histPos: number[],
  recent: [number, number][],
  cands: Candidate[],
): Float32Array {
  const n = cands.length;
  const X = new Float32Array(n * N_FEATURES);
  const c = m.c;
  const userAff = Float32Array.from(user.cat_aff);

  const cf = histPos.length ? itemcfScores(m, lastN(histPos, c.cf_hist)) : null;
  const q = histPos.length ? contentQuery(m, histPos) : null;

  // real-time window
  const win = recent.slice(Math.max(0, recent.length - c.recent_n));
  const wr = win.map(([, w]) => f32(w));
  const rPos = win.filter((_, k) => wr[k] > c.pos_thr).map(([i]) => i);
  const rSkip = win.filter((_, k) => wr[k] < c.skip_thr).map(([i]) => i);
  const catCount = (items: number[]) => {
    const v = new Float32Array(c.n_tag_categories);
    for (const i of items) for (let k = 0; k < c.n_tag_categories; k++) if ((m.itemCatBits[i] >>> k) & 1) v[k] += 1;
    return v;
  };
  const posCats = catCount(rPos);
  const skipCats = catCount(rSkip);
  const last = rPos.slice(Math.max(0, rPos.length - 5));
  const rtN = win.length;
  const rtPosRate = rtN ? rPos.length / rtN : 0;
  const rtSkipRate = rtN ? rSkip.length / rtN : 0;

  for (let r = 0; r < n; r++) {
    const item = cands[r].item;
    const x = X.subarray(r * N_FEATURES, (r + 1) * N_FEATURES);
    let k = 0;
    for (const v of user.user_stats) x[k++] = v;
    for (let j = 0; j < m.nItemStats; j++) x[k++] = m.itemStats[item * m.nItemStats + j];
    x[k++] = catDot(m, item, userAff);
    x[k++] = Math.abs(f32(m.itemLogDur[item] - user.log_dur));
    // every path's score for every candidate (not only the path that proposed it)
    x[k++] = dotRow(m.itemEmb, m.embDim, item, uvec);
    x[k++] = cf ? cf[item] : 0;
    x[k++] = m.pop.pop_recent[item];
    x[k++] = m.pop.pop_platform[item];
    x[k++] = q ? dotRow(m.caption, m.capDim, item, q) : 0;
    let nPaths = 0;
    for (const p of PATHS) {
      const has = p in cands[r].allSources ? 1 : 0;
      x[k++] = has;
      nPaths += has;
    }
    x[k++] = nPaths;
    x[k++] = rtN;
    x[k++] = rtPosRate;
    x[k++] = rtSkipRate;
    x[k++] = rtN && rPos.length ? catDot(m, item, posCats) : 0;
    x[k++] = rtN && rSkip.length ? catDot(m, item, skipCats) : 0;
    if (last.length) {
      // numpy's float32 add.reduce over a short row: out = first + (sequential sum of the rest)
      const s0 = dotOf(m, item, last[0]);
      let mx = s0;
      let tail = 0;
      for (let j = 1; j < last.length; j++) {
        const s = dotOf(m, item, last[j]);
        mx = Math.max(mx, s);
        tail = f32(tail + s);
      }
      x[k++] = mx;
      x[k++] = f32(s0 + tail) / last.length;
    } else {
      x[k++] = 0;
      x[k++] = 0;
    }
  }
  return X;
}

/** Caption cosine of two items (float64 sum, rounded once). */
export function dotOf(m: Model, a: number, b: number): number {
  const d = m.capDim;
  let s = 0;
  for (let j = 0; j < d; j++) s += m.caption[a * d + j] * m.caption[b * d + j];
  return f32(s);
}
