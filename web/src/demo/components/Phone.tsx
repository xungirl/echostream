/** The phone-shaped feed: one screen of 10 videos, each with Skip / Watched / Like. */
import { useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import { PATHS, type Model } from "../engine/bundle";
import type { Ranked } from "../engine/recommender";
import { PATH_INFO, REACTIONS, type Reaction } from "../labels";
import type { Step } from "../session";
import { useFlip } from "./anim";

interface Props {
  model: Model;
  step: Step;
  onReact: (item: number, kind: Reaction) => void;
}

export default function Phone({ model, step, onReact }: Props) {
  const list = useRef<HTMLOListElement>(null);
  const [leaving, setLeaving] = useState<{ item: number; kind: Reaction } | null>(null);
  useFlip(list, [step.n]);
  const fresh = new Set(step.fresh);
  const rankOf = new Map(step.screen.ranked.map((r, k) => [r.item, k + 1]));

  const react = (item: number, kind: Reaction) => {
    if (leaving) return;
    setLeaving({ item, kind });
    // let the swipe animation play, then recompute
    window.setTimeout(() => {
      setLeaving(null);
      onReact(item, kind);
    }, 260);
  };

  const onKey = (e: KeyboardEvent, item: number) => {
    const r = REACTIONS.find((x) => x.key === e.key.toLowerCase());
    if (r && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      react(item, r.kind);
    }
  };

  return (
    <section className="phone" aria-label="Recommended videos">
      <div className="phone-bar">
        <span className="phone-title">For You</span>
        <span className="phone-meta" data-testid="compute-time">
          screen {step.n + 1} · computed in your browser in <b>{step.screen.timingsMs.total.toFixed(1)} ms</b>
        </span>
        {step.reaction && (
          <span className="phone-echo" data-testid="phone-echo">
            {REACTIONS.find((x) => x.kind === step.reaction!.kind)!.label} → <b>{step.fresh.length}</b> new, <b>{step.dropped.length}</b> left
          </span>
        )}
      </div>
      <ol className="feed" ref={list} data-testid="feed">
        {step.screen.shown.map((r, k) => (
          <Card
            key={r.item}
            model={model}
            r={r}
            pos={k + 1}
            rankPos={rankOf.get(r.item)!}
            fresh={fresh.has(r.item)}
            leaving={leaving?.item === r.item ? leaving.kind : null}
            disabled={!!leaving}
            onReact={react}
            onKey={onKey}
          />
        ))}
      </ol>
    </section>
  );
}

function Card(props: {
  model: Model;
  r: Ranked;
  pos: number;
  rankPos: number;
  fresh: boolean;
  leaving: Reaction | null;
  disabled: boolean;
  onReact: (item: number, kind: Reaction) => void;
  onKey: (e: KeyboardEvent, item: number) => void;
}) {
  const { model, r, pos, rankPos, fresh, leaving, disabled, onReact, onKey } = props;
  const cat = model.catalog.categories[model.catalog.category[r.item]];
  const title = model.catalog.titles[r.item] || "（无标题）";
  const cls = ["card", fresh ? "fresh" : "", leaving ? `leaving ${leaving}` : ""].join(" ").trim();
  return (
    <li className={cls} data-flip={r.item} data-testid={`card-${r.item}`} onKeyDown={(e) => onKey(e, r.item)}>
      <div className="card-head">
        <span className="pos">{pos}</span>
        <span className="cat" title={`${cat.zh} (category name hand-translated)`}>
          {cat.en}
        </span>
        {fresh && <span className="new-tag">new</span>}
        <span className="why" title="position in the ranker's order, before the diversity re-rank">
          ranker #{rankPos}
        </span>
      </div>
      <p className="title" lang="zh">
        {title}
      </p>
      <div className="badges" aria-label="found by">
        {PATHS.filter((p) => p in r.allSources).map((p) => (
          <span key={p} className={p === r.source ? "badge first" : "badge"} style={{ "--c": PATH_INFO[p].color } as CSSProperties} title={PATH_INFO[p].what}>
            {PATH_INFO[p].name}
          </span>
        ))}
      </div>
      <div className="reactions">
        {REACTIONS.map((x) => (
          <button
            key={x.kind}
            className={`react ${x.kind}`}
            disabled={disabled}
            onClick={() => onReact(r.item, x.kind)}
            aria-label={`${x.label} video ${pos}: ${title} (${x.hint})`}
            title={`${x.label}: ${x.hint} · key ${x.key.toUpperCase()}`}
          >
            {x.label}
          </button>
        ))}
      </div>
    </li>
  );
}
