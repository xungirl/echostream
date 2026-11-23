/**
 * One screen, end to end (echostream/serving/recommender.py _pipeline, minus the network):
 *   real-time features -> user state -> user vector -> 5 recall paths -> quota merge (200)
 *   -> LightGBM rank -> MMR (lambda 0.7, top 50) -> 10 videos
 * Returns every intermediate result so the page can explain how the screen was made.
 */
import type { Model, PathName, UserRecord } from "./bundle";
import { buildFeatures } from "./features";
import { predictRows } from "./gbdt";
import { blendSimilarity, mmrRerank } from "./mmr";
import { content, itemcf, mergeWithQuotas, popularity, twoTower, userVector, type Candidate, type Paths } from "./recall";
import type { RealtimeStore } from "./featureStore";

export interface Ranked extends Candidate {
  rankScore: number;
}

export interface Screen {
  uid: number;
  userVec: Float32Array;
  histPos: number[]; // batch positives + real-time positives (the model's view of "liked")
  nExcluded: number; // videos skipped as already seen (batch history + real-time events)
  paths: Paths; // each path's top 200
  merged: Candidate[]; // after the quota merge, in merge order
  rankScores: Float64Array; // ranker probability per merged candidate (merge order)
  ranked: Ranked[]; // merged, sorted by rank score (desc), ties by item id
  shown: Ranked[]; // after MMR: the screen
  timingsMs: Record<string, number>;
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export function recommend(m: Model, user: UserRecord, store: RealtimeStore, n = m.c.screen_n): Screen {
  const c = m.c;
  const t0 = now();
  const times: Record<string, number> = {};
  let t = t0;
  const lap = (name: string) => {
    const x = now();
    times[name] = x - t;
    t = x;
  };

  // 1-2) user state = batch snapshot + real-time events (newest 50)
  const rt = store.recent(c.hist_n);
  const histPos = [...user.pos, ...rt.filter((e) => e.wr > c.pos_thr).map((e) => e.item)];
  const excluded = new Uint8Array(m.nItems);
  for (const i of user.seen) excluded[i] = 1;
  for (const e of rt) excluded[e.item] = 1;
  const nExcluded = excluded.reduce((a, b) => a + b, 0);
  const recent: [number, number][] = [
    ...user.recent_items.map((i, k): [number, number] => [i, user.recent_wr[k]]),
    ...rt.map((e): [number, number] => [e.item, e.wr]),
  ];
  const uvec = userVector(m, user.user_cats, histPos.slice(Math.max(0, histPos.length - c.hist_len)));
  lap("user_vector");

  // 3) recall
  const paths: Paths = {
    two_tower: twoTower(m, uvec, c.total, excluded),
    itemcf: itemcf(m, histPos, c.total, excluded),
    pop_recent: popularity(m, "pop_recent", c.total, excluded),
    pop_platform: popularity(m, "pop_platform", c.total, excluded),
    content: content(m, histPos, c.total, excluded),
  };
  lap("recall");
  const order = Object.keys(c.quotas) as PathName[];
  const merged = mergeWithQuotas(paths, order, c.quotas, c.total);
  lap("merge");

  // 4) rank
  const X = buildFeatures(m, user, uvec, histPos, recent, merged);
  const rankScores = predictRows(m.ranker, X, merged.length);
  const ranked: Ranked[] = merged
    .map((cand, j) => ({ ...cand, rankScore: rankScores[j] }))
    .sort((a, b) => b.rankScore - a.rankScore || a.item - b.item);
  lap("rank");

  // 5) diversity re-rank over the top of the ranked list
  const top = ranked.slice(0, Math.min(ranked.length, Math.max(5 * n, 50)));
  const sim = blendSimilarity(m, top.map((r) => r.item));
  const pick = mmrRerank(top.map((r) => r.rankScore), sim, c.mmr_lambda, n);
  const shown = pick.map((j) => top[j]);
  lap("mmr");
  times.total = now() - t0;
  return { uid: user.uid, userVec: uvec, histPos, nExcluded, paths, merged, rankScores, ranked, shown, timingsMs: times };
}

