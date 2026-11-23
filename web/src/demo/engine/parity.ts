/**
 * Replays the golden sessions (scripts/gen_browser_golden.py) through the TypeScript engine and
 * measures agreement with the Python read path. Two views:
 *
 *  stage by stage - every stage gets Python's own inputs (Python's path lists -> our merge,
 *    Python's merged candidates -> our features + trees, Python's ranked list -> our MMR), so a
 *    difference in one stage cannot hide or cause a difference in another;
 *  end to end - our engine runs alone from the user record + events, like in the browser.
 *
 * Used by parity.test.ts (asserts) and scripts/parity-report.ts (numbers for the docs).
 */
import { PATHS, type Model, type PathName, type UserRecord } from "./bundle";
import { buildFeatures } from "./features";
import { RealtimeStore } from "./featureStore";
import { predictRows } from "./gbdt";
import { blendSimilarity, mmrRerank } from "./mmr";
import { mergeWithQuotas, type Candidate, type Paths, type Scored } from "./recall";
import { recommend } from "./recommender";
import { f32, f64, u16, u8, type Golden, type GoldenScreen } from "./testBundle";

export interface PathAgreement {
  lists: number;
  identical: number; // same ids in the same order
  equivalent: number; // identical, or different only among (near-)tied items (see compareLists)
  maxAbsScoreDiff: number; // over items both lists contain
}

export interface ParityReport {
  sessions: number;
  screens: number;
  stage: {
    userVecMaxAbsDiff: number;
    paths: Record<PathName, PathAgreement>;
    mergeIdentical: number; // our merge of Python's path lists == Python's merged 200
    featureRows: number;
    featureValues: number;
    featureExact: number;
    featureMaxAbsDiff: number;
    featureMaxAbsDiffByName: Record<string, number>;
    treeRows: number; // our trees on Python's features vs LightGBM's probabilities
    treeMaxAbsDiff: number;
    mmrIdentical: number; // our MMR on Python's ranked list == Python's screen
  };
  endToEnd: {
    mergedIdentical: number;
    rankMaxAbsDiff: number; // over screens whose merged 200 is identical
    shownIdentical: number;
    shownMismatches: { uid: number; step: number }[];
  };
  computeMs: number[];
}

const same = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  a.length === b.length && Array.from(a).every((x, i) => x === b[i]);

export function catsOf(m: Model, item: number): number[] {
  const out: number[] = [];
  for (let c = 0; c < m.c.n_tag_categories; c++) if ((m.itemCatBits[item] >>> c) & 1) out.push(c);
  return out;
}

export function pythonPaths(g: GoldenScreen): Paths {
  const out = {} as Paths;
  for (const p of PATHS) {
    const ids = u16(g.paths[p].ids);
    const sc = f32(g.paths[p].scores);
    out[p] = Array.from(ids, (id, k): Scored => [id, sc[k]]);
  }
  return out;
}

export function pythonMerged(g: GoldenScreen): Candidate[] {
  const ids = u16(g.merged_ids);
  const src = u8(g.merged_source);
  const mem = u8(g.merged_members);
  // the per-path scores of a candidate are not used downstream, only WHICH paths proposed it
  return Array.from(ids, (item, k) => ({
    item,
    source: PATHS[src[k]],
    score: NaN,
    allSources: Object.fromEntries(PATHS.filter((_, j) => (mem[k] >> j) & 1).map((p) => [p, 1])),
  }));
}

export const SCORE_TOL = 1e-6;

/**
 * Python list vs ours:
 *  "identical"  - same ids in the same order;
 *  "equivalent" - position by position the scores agree within SCORE_TOL, and every item scoring
 *                 clearly above the last slot (by more than SCORE_TOL) is in both lists. So the
 *                 lists differ only among (near-)tied items: which of several items tied with the
 *                 200th score are kept (numpy's argpartition keeps an arbitrary subset), or the
 *                 order of two items whose float32 scores differ by one rounding step.
 */
export function compareLists(py: Scored[], ts: Scored[]): "identical" | "equivalent" | "different" {
  if (same(py.map(([i]) => i), ts.map(([i]) => i))) return "identical";
  if (py.length !== ts.length || py.length === 0) return "different";
  if (py.some(([, s], k) => Math.abs(s - ts[k][1]) > SCORE_TOL)) return "different";
  const cut = py[py.length - 1][1] + SCORE_TOL;
  const above = (l: Scored[]) => l.filter(([, s]) => s > cut).map(([i]) => i).sort((a, b) => a - b);
  return same(above(py), above(ts)) ? "equivalent" : "different";
}

function recentOf(user: UserRecord, store: RealtimeStore, histN: number): [number, number][] {
  return [
    ...user.recent_items.map((i, k): [number, number] => [i, user.recent_wr[k]]),
    ...store.recent(histN).map((e): [number, number] => [e.item, e.wr]),
  ];
}

const signature = (cs: Candidate[]) =>
  cs.map((c) => `${c.item}/${c.source}/${PATHS.filter((p) => p in c.allSources).join("+")}`).join();

export function runParity(m: Model, golden: Golden): ParityReport {
  const users = new Map<number, UserRecord>([...m.users, ...golden.users].map((u) => [u.uid, u]));
  const order = Object.keys(m.c.quotas) as PathName[];
  const st: ParityReport["stage"] = {
    userVecMaxAbsDiff: 0,
    paths: Object.fromEntries(
      PATHS.map((p) => [p, { lists: 0, identical: 0, equivalent: 0, maxAbsScoreDiff: 0 }]),
    ) as Record<PathName, PathAgreement>,
    mergeIdentical: 0,
    featureRows: 0,
    featureValues: 0,
    featureExact: 0,
    featureMaxAbsDiff: 0,
    featureMaxAbsDiffByName: {},
    treeRows: 0,
    treeMaxAbsDiff: 0,
    mmrIdentical: 0,
  };
  const e2e: ParityReport["endToEnd"] = { mergedIdentical: 0, rankMaxAbsDiff: 0, shownIdentical: 0, shownMismatches: [] };
  const rep: ParityReport = { sessions: golden.sessions.length, screens: 0, stage: st, endToEnd: e2e, computeMs: [] };

  for (const s of golden.sessions) {
    const user = users.get(s.uid)!;
    const store = new RealtimeStore(m.c.hist_max, m.c.pos_thr);
    for (const g of s.screens) {
      if (g.event) {
        store.apply({
          eventId: `golden-${s.uid}-${g.step}`,
          item: g.event.item_id,
          watchRatio: g.event.watch_ratio,
          action: g.event.action,
          ts: g.event.ts,
          cats: catsOf(m, g.event.item_id),
        });
      }
      const scr = recommend(m, user, store);
      rep.screens++;
      rep.computeMs.push(scr.timingsMs.total);

      // ---- stage by stage, each stage fed with Python's inputs
      const uv = f32(g.user_vec);
      for (let i = 0; i < uv.length; i++) st.userVecMaxAbsDiff = Math.max(st.userVecMaxAbsDiff, Math.abs(uv[i] - scr.userVec[i]));

      const pyPaths = pythonPaths(g);
      for (const p of PATHS) {
        const r = st.paths[p];
        r.lists++;
        const verdict = compareLists(pyPaths[p], scr.paths[p]);
        if (verdict === "identical") r.identical++;
        if (verdict !== "different") r.equivalent++;
        const mine = new Map(scr.paths[p]);
        for (const [i, sc] of pyPaths[p]) if (mine.has(i)) r.maxAbsScoreDiff = Math.max(r.maxAbsScoreDiff, Math.abs(sc - mine.get(i)!));
      }

      const pyMerged = pythonMerged(g);
      if (signature(mergeWithQuotas(pyPaths, order, m.c.quotas, m.c.total)) === signature(pyMerged)) st.mergeIdentical++;

      const pyRank = f64(g.rank_scores);
      if (g.features) {
        const X = f32(g.features);
        const mine = buildFeatures(m, user, scr.userVec, scr.histPos, recentOf(user, store, m.c.hist_n), pyMerged);
        const names = m.c.feature_names;
        st.featureRows += pyMerged.length;
        for (let k = 0; k < X.length; k++) {
          const d = Math.abs(X[k] - mine[k]);
          st.featureValues++;
          if (d === 0) st.featureExact++;
          st.featureMaxAbsDiff = Math.max(st.featureMaxAbsDiff, d);
          const name = names[k % names.length];
          st.featureMaxAbsDiffByName[name] = Math.max(st.featureMaxAbsDiffByName[name] ?? 0, d);
        }
        const p = predictRows(m.ranker, X, pyMerged.length);
        st.treeRows += pyMerged.length;
        for (let k = 0; k < p.length; k++) st.treeMaxAbsDiff = Math.max(st.treeMaxAbsDiff, Math.abs(p[k] - pyRank[k]));
      }

      const ranked = pyMerged
        .map((c, k) => ({ item: c.item, score: pyRank[k] }))
        .sort((a, b) => b.score - a.score || a.item - b.item);
      const top = ranked.slice(0, Math.min(ranked.length, m.c.mmr_top));
      const sim = blendSimilarity(m, top.map((r) => r.item));
      const pick = mmrRerank(top.map((r) => r.score), sim, m.c.mmr_lambda, m.c.screen_n);
      if (same(pick.map((j) => top[j].item), g.shown_ids)) st.mmrIdentical++;

      // ---- end to end: our engine alone
      if (signature(scr.merged) === signature(pyMerged)) {
        e2e.mergedIdentical++;
        for (let k = 0; k < pyRank.length; k++) e2e.rankMaxAbsDiff = Math.max(e2e.rankMaxAbsDiff, Math.abs(pyRank[k] - scr.rankScores[k]));
      }
      if (same(g.shown_ids, scr.shown.map((r) => r.item))) e2e.shownIdentical++;
      else e2e.shownMismatches.push({ uid: s.uid, step: g.step });
    }
  }
  return rep;
}
