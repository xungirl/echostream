/** "How this screen was made": catalog -> 5 recall paths -> merge -> rank -> MMR -> screen. */
import type { CSSProperties } from "react";
import { PATHS, type Model, type PathName } from "../engine/bundle";
import { PATH_INFO } from "../labels";
import type { Step } from "../session";
import { CountUp } from "./anim";

export default function Funnel({ model, step }: { model: Model; step: Step }) {
  const s = step.screen;
  const c = model.c;
  const kept = (p: PathName) => s.merged.filter((x) => x.source === p).length;
  const toMmr = Math.min(s.ranked.length, c.mmr_top);
  // key = step: re-mounting replays the staggered "data flows down" animation on every screen
  return (
    <section className="panel funnel" aria-labelledby="funnel-h" key={step.n}>
      <h2 id="funnel-h">How this screen was made</h2>
      <ol className="stages">
        <li className="stage" style={{ "--i": 0, "--w": "100%" } as CSSProperties}>
          <div className="stage-bar catalog" />
          <div className="stage-text">
            <b className="num">
              <CountUp value={model.nItems} />
            </b>{" "}
            videos in the catalog
            <span className="sub">
              minus <CountUp value={s.nExcluded} /> this viewer already watched
            </span>
          </div>
        </li>
        <li className="stage paths" style={{ "--i": 1 } as CSSProperties}>
          <div className="stage-label">
            <b>5 recall paths</b> each nominate up to {c.total}
          </div>
          <ul className="path-rows">
            {PATHS.map((p) => (
              <li key={p} className="path-row" style={{ "--c": PATH_INFO[p].color } as CSSProperties} data-testid={`path-${p}`}>
                <span className="path-name">
                  <i className="dot" />
                  {PATH_INFO[p].name} <span className="tech">{PATH_INFO[p].tech}</span>
                </span>
                <span className="path-bar" aria-hidden="true">
                  <span className="found" style={{ width: `${(100 * s.paths[p].length) / c.total}%` }} />
                  <span className="kept" style={{ width: `${(100 * kept(p)) / c.total}%` }} />
                </span>
                <span className="path-count">
                  <b>
                    <CountUp value={s.paths[p].length} />
                  </b>{" "}
                  found · <b>
                    <CountUp value={kept(p)} />
                  </b>{" "}
                  kept
                </span>
                <span className="path-what">{PATH_INFO[p].what}</span>
              </li>
            ))}
          </ul>
          {c.quotas.content === 0 && (
            <p className="note">
              “New &amp; similar” has a merge quota of 0 in this model version (quotas were tuned on validation data in Stage 3), so
              its videos get in only when the other paths leave slots empty; otherwise they count as a signal for the ranker.
            </p>
          )}
        </li>
        <li className="stage" style={{ "--i": 2, "--w": "62%" } as CSSProperties}>
          <div className="stage-bar merged" />
          <div className="stage-text">
            <b className="num">
              <CountUp value={s.merged.length} />
            </b>{" "}
            candidates after the merge
            <span className="sub">each path fills its quota, duplicates removed</span>
          </div>
        </li>
        <li className="stage" style={{ "--i": 3, "--w": "42%" } as CSSProperties}>
          <div className="stage-bar ranked" />
          <div className="stage-text">
            <b className="num">
              <CountUp value={toMmr} />
            </b>{" "}
            best by the ranker
            <span className="sub">a LightGBM model scores all {s.merged.length} on 40 signals: taste, video stats, what you just did</span>
          </div>
        </li>
        <li className="stage" style={{ "--i": 4, "--w": "24%" } as CSSProperties}>
          <div className="stage-bar shown" />
          <div className="stage-text">
            <b className="num">
              <CountUp value={s.shown.length} />
            </b>{" "}
            on your screen
            <span className="sub">a diversity re-rank (MMR) mixes topics so the screen is not 10 of the same</span>
          </div>
        </li>
      </ol>
    </section>
  );
}
