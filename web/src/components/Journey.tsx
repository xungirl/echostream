// The project's signature timeline: vertical rail, gradient dots, bright = done,
// newest done = "current" (pulsing), gray dashed = todo. The bright fill grows
// with scroll and stops at the centre of the last done dot (measured from layout).
import { useEffect, useRef } from "react";
import type { JourneyItem } from "../types";

export function statusOf(items: JourneyItem[]): string[] {
  const lastDone = items.map((i) => i.done).lastIndexOf(true);
  return items.map((it, i) => (i === lastDone ? "done current" : it.done ? "done" : "todo"));
}

export default function Journey({ items }: { items: JourneyItem[] }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const fillRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const rm = rootRef.current, fill = fillRef.current;
    if (!rm || !fill) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

    const target = () => {
      const dots = rm.querySelectorAll<HTMLElement>(".rm-item.done .rm-dot");
      if (!dots.length) return 0;
      const d = dots[dots.length - 1];
      const item = d.closest<HTMLElement>(".rm-item")!;
      return item.offsetTop + d.offsetTop + d.offsetHeight / 2;
    };
    const update = () => {
      if (reduce) {
        fill.style.height = `${target()}px`; // no growth animation
        return;
      }
      const rect = rm.getBoundingClientRect();
      const viewH = window.innerHeight || document.documentElement.clientHeight;
      const progress = Math.max(0, Math.min(1, (viewH * 0.8 - rect.top) / (rect.height * 0.9 || 1)));
      fill.style.height = `${target() * progress}px`;
    };
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, [items]);

  const status = statusOf(items);
  return (
    <div className="roadmap" ref={rootRef}>
      <div className="rm-line" />
      <div className="rm-fill" ref={fillRef} />
      {items.map((it, i) => (
        <div key={it.stage} className={`rm-item ${status[i]}`} data-testid={`stage-${it.stage}`}>
          <div className="rm-dot" />
          <div className="rm-card">
            <div className="rm-date">
              Stage {it.stage}
              {it.done ? "" : " · planned"}
              {status[i].includes("current") && <span className="rm-tag">latest</span>}
            </div>
            <h3>
              {it.title} {it.zh && <span className="zh">{it.zh}</span>}
            </h3>
            {it.result && <p>{it.result}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}
