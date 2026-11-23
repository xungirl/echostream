/** What the last reaction changed: new videos in, videos out, taste shift. */
import type { Model } from "../engine/bundle";
import { REACTIONS } from "../labels";
import type { Step } from "../session";
import { titleOf, useTitleLang } from "../titles";

export default function Echo({ model, step, prev }: { model: Model; step: Step; prev: Step | null }) {
  const want = useTitleLang();
  const t = (i: number) => titleOf(model, i, want);
  if (!step.reaction || !prev) {
    return (
      <section className="panel echo" aria-labelledby="echo-h">
        <h2 id="echo-h">The echo</h2>
        <p className="muted">React to any video. The next screen is recomputed from scratch with your reaction included, and this box shows what changed.</p>
      </section>
    );
  }
  const r = REACTIONS.find((x) => x.kind === step.reaction!.kind)!;
  const pushed = step.dropped.filter((d) => !d.reacted);
  return (
    <section className="panel echo" aria-labelledby="echo-h" data-testid="echo">
      <h2 id="echo-h">The echo</h2>
      <p className="echo-head">
        <span className={`verb ${r.kind}`}>{r.label}</span> <span lang={t(step.reaction.item).lang}>“{t(step.reaction.item).text}”</span>
        <span className="muted"> ({r.hint})</span>
      </p>
      <p className="echo-sum">
        <b>{step.fresh.length}</b> new {step.fresh.length === 1 ? "video" : "videos"} entered the screen, <b>{step.dropped.length}</b> left
        {pushed.length > 0 ? ` (${pushed.length} pushed out, 1 you reacted to)` : ""}.
      </p>
      <div className="echo-cols">
        <div>
          <h3>New</h3>
          <ul className="chips">
            {step.fresh.map((i) => (
              <li key={i} className="chip in" lang={t(i).lang}>
                {t(i).text}
              </li>
            ))}
            {step.fresh.length === 0 && <li className="muted">none</li>}
          </ul>
        </div>
        <div>
          <h3>Dropped out</h3>
          <ul className="chips" data-testid="dropped">
            {step.dropped.map((d) => (
              <li key={d.item} className={d.reacted ? "chip out reacted" : "chip out"} lang={t(d.item).lang} title={d.reacted ? "you reacted to it, so it counts as seen" : "pushed out by the new ranking"}>
                {t(d.item).text}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
