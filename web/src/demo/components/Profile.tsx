/** The interest profile: categories of the liked videos the taste vector is built from. */
import type { CSSProperties } from "react";
import type { Model } from "../engine/bundle";
import type { RealtimeStore } from "../engine/featureStore";
import type { Step } from "../session";

const plural = (k: number, word: string) => (
  <>
    <b>{k}</b> {k === 1 ? word : `${word}s`}
  </>
);

export default function Profile({ model, step, prev, store }: { model: Model; step: Step; prev: Step | null; store: RealtimeStore }) {
  const before = new Map((prev?.profile ?? []).map((b) => [b.category, b.share]));
  const n = (k: string) => store.counts.get(k) ?? 0;
  const max = Math.max(0.0001, ...step.profile.map((b) => b.share));
  return (
    <section className="panel profile" aria-labelledby="profile-h">
      <h2 id="profile-h">What the model thinks you like</h2>
      {step.profile.length === 0 ? (
        <p className="muted">No likes yet. A brand-new viewer gets a taste vector from an empty history, so the first screens lean on what is popular. Like a few videos and watch this fill in.</p>
      ) : (
        <>
          <p className="muted small">Categories of the last {step.likedCount} liked videos: the input of the taste vector.</p>
          <ul className="bars">
            {step.profile.map((b) => {
              const d = prev ? b.share - (before.get(b.category) ?? 0) : 0;
              return (
                <li key={b.category} className="bar-row">
                  <span className="bar-label" title={`${b.zh} (category name hand-translated)`}>
                    {b.en}
                  </span>
                  <span className="bar-track">
                    <span className="bar-fill" style={{ "--w": `${(100 * b.share) / max}%` } as CSSProperties} />
                  </span>
                  <span className="bar-val">
                    {Math.round(100 * b.share)}%
                    {Math.abs(d) >= 0.005 && <span className={d > 0 ? "delta up" : "delta down"}>{d > 0 ? "▲" : "▼"}</span>}
                  </span>
                </li>
              );
            })}
          </ul>
        </>
      )}
      <p className="session-line" data-testid="session-line">
        This session: {plural(n("n"), "reaction")} · {plural(n("pos"), "like")} · {plural(n("skip"), "skip")}
        <span className="muted"> — skips and watches do not change the taste vector; they reach the ranker as real-time signals.</span>
      </p>
    </section>
  );
}
