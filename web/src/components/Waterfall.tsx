import type { Attempt, ShardCall, Timings } from "../types";

// Pipeline stages run one after another; inside "recall" the ANN scatter-gather
// to the shards starts at the same moment as the local recall paths, and
// "recall_ann_wait" is the extra time spent waiting for the shards afterwards.
const STAGES = ["features", "user_vector", "recall", "merge", "rank", "mmr"] as const;

export interface Bar {
  key: string;
  label: string;
  kind: "total" | "stage" | "sub" | "shard";
  start: number; // ms from request start
  ms: number;
  left: number; // % of the track
  width: number; // % of the track
  shard?: ShardCall;
}

/** Lay out every bar on one shared ms axis; widths are proportional to time. */
export function computeBars(t: Timings, shards: ShardCall[] = []): { scale: number; bars: Bar[] } {
  const raw: Omit<Bar, "left" | "width">[] = [];
  let cursor = 0;
  let recallStart = 0;

  for (const s of STAGES) {
    const ms = t[s];
    if (ms === undefined) continue; // e.g. mmr is absent when MMR is off
    raw.push({ key: s, label: s, kind: "stage", start: cursor, ms });
    if (s === "recall") {
      recallStart = cursor;
      const local = t.recall_local ?? 0;
      if (t.recall_local !== undefined) raw.push({ key: "recall_local", label: "↳ local", kind: "sub", start: cursor, ms: local });
      if (t.recall_ann_wait !== undefined)
        raw.push({ key: "recall_ann_wait", label: "↳ ann wait", kind: "sub", start: cursor + local, ms: t.recall_ann_wait });
    }
    cursor += ms;
  }

  const total = t.total ?? cursor;
  for (const sh of shards) {
    raw.push({ key: `shard${sh.shard}`, label: `shard ${sh.shard}`, kind: "shard", start: recallStart, ms: sh.ms, shard: sh });
  }

  const scale = Math.max(total, cursor, ...raw.map((b) => b.start + b.ms), 1e-9);
  const pct = (x: number) => (x / scale) * 100;
  raw.unshift({ key: "total", label: "total", kind: "total", start: 0, ms: total });
  const bars: Bar[] = raw.map((b) => ({
    ...b,
    left: pct(b.start),
    width: pct(b.ms),
  }));
  return { scale, bars };
}

export const failedAttempts = (s: ShardCall): Attempt[] => s.attempts.filter((a) => a.status !== "ok");

export default function Waterfall({ timings, shards }: { timings: Timings; shards: ShardCall[] }) {
  const { bars } = computeBars(timings, shards);
  return (
    <div className="waterfall">
      {bars.map((b) => {
        const failed = b.shard ? failedAttempts(b.shard) : [];
        const bad = b.shard !== undefined && !b.shard.ok;
        return (
          <div key={b.key} className={`wf-row ${b.kind}`} data-testid={`wf-${b.key}`}>
            <div className="wf-label">
              {b.label}
              {b.shard && (
                <span className="wf-replica">{b.shard.ok ? ` → r${b.shard.replica}` : " → none"}</span>
              )}
            </div>
            <div className="wf-track">
              <div
                className={`wf-bar ${b.kind}${bad ? " bad" : ""}`}
                style={{ left: `${b.left}%`, width: `${Math.max(b.width, 0.4)}%` }}
                data-width={b.width.toFixed(2)}
              />
            </div>
            <div className="wf-ms mono">{b.ms.toFixed(2)}</div>
            {(failed.length > 0 || bad) && (
              <div className="wf-attempts">
                {failed.map((a, i) => (
                  <span key={i} className="attempt failed mono">
                    r{a.replica} {a.status} {a.ms.toFixed(1)}ms
                  </span>
                ))}
                {bad && <span className="attempt failed mono">no replica answered</span>}
              </div>
            )}
          </div>
        );
      })}
      <div className="wf-axis mono">ms · one shared axis · shards start with recall</div>
    </div>
  );
}
