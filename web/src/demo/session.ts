/**
 * A viewer's session in the demo: the user's batch record + the in-browser real-time store.
 * Every reaction becomes an event (like POST /events), is applied to the store (like the worker),
 * and the next screen is recomputed from scratch - so each step is "one swipe, one echo".
 */
import type { Model, UserRecord } from "./engine/bundle";
import { eventType, RealtimeStore } from "./engine/featureStore";
import { recommend, type Screen } from "./engine/recommender";
import { REACTIONS, type Reaction } from "./labels";

export interface ProfileBar {
  category: number;
  en: string;
  zh: string;
  count: number;
  share: number;
}

export interface Step {
  n: number; // 0 = first screen
  screen: Screen;
  reaction: { kind: Reaction; item: number } | null;
  fresh: number[]; // shown now, not on the previous screen
  dropped: { item: number; reacted: boolean }[]; // on the previous screen, gone now
  profile: ProfileBar[];
  likedCount: number; // positives the taste vector can see (last hist_len of them)
}

export function catsOf(m: Model, item: number): number[] {
  const out: number[] = [];
  for (let c = 0; c < m.c.n_tag_categories; c++) if ((m.itemCatBits[item] >>> c) & 1) out.push(c);
  return out;
}

/** Category shares of the last `window` liked videos - exactly the videos the user tower reads. */
export function profileOf(m: Model, histPos: number[], top = 6): ProfileBar[] {
  const last = histPos.slice(Math.max(0, histPos.length - m.c.hist_len));
  if (!last.length) return [];
  const counts = new Map<number, number>();
  for (const i of last) counts.set(m.catalog.category[i], (counts.get(m.catalog.category[i]) ?? 0) + 1);
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, top)
    .map(([c, n]) => ({ category: c, en: m.catalog.categories[c].en, zh: m.catalog.categories[c].zh, count: n, share: n / last.length }));
}

export class DemoSession {
  readonly store: RealtimeStore;
  steps: Step[] = [];
  private seq = 0;

  constructor(
    readonly model: Model,
    readonly user: UserRecord,
  ) {
    this.store = new RealtimeStore(model.c.hist_max, model.c.pos_thr);
    this.steps.push(this.makeStep(null, null));
  }

  get current(): Step {
    return this.steps[this.steps.length - 1];
  }

  react(item: number, kind: Reaction): Step {
    const wr = REACTIONS.find((r) => r.kind === kind)!.watchRatio;
    this.seq += 1;
    this.store.apply({
      eventId: `demo-${this.user.uid}-${this.seq}`,
      item,
      watchRatio: wr,
      action: eventType(wr),
      ts: Date.now() / 1000,
      cats: catsOf(this.model, item),
    });
    const step = this.makeStep(this.current, { kind, item });
    this.steps.push(step);
    return step;
  }

  /** Recompute the current screen without a new event (used to measure compute time). */
  recompute(): Screen {
    return recommend(this.model, this.user, this.store);
  }

  private makeStep(prev: Step | null, reaction: Step["reaction"]): Step {
    const screen = recommend(this.model, this.user, this.store);
    const now = new Set(screen.shown.map((r) => r.item));
    const before = prev ? prev.screen.shown.map((r) => r.item) : [];
    const beforeSet = new Set(before);
    return {
      n: prev ? prev.n + 1 : 0,
      screen,
      reaction,
      fresh: prev ? screen.shown.map((r) => r.item).filter((i) => !beforeSet.has(i)) : [],
      dropped: before.filter((i) => !now.has(i)).map((i) => ({ item: i, reacted: reaction?.item === i })),
      profile: profileOf(this.model, screen.histPos),
      likedCount: Math.min(screen.histPos.length, this.model.c.hist_len),
    };
  }
}
