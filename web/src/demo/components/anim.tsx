/** Small animation helpers (no libraries). Both respect prefers-reduced-motion. */
import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";

export const reducedMotion = () =>
  typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** A number that counts from its previous value to the new one. */
export function CountUp({ value, ms = 600, format = (v: number) => Math.round(v).toLocaleString("en-US") }: { value: number; ms?: number; format?: (v: number) => string }) {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  useEffect(() => {
    const start = from.current;
    from.current = value;
    if (start === value || reducedMotion() || typeof requestAnimationFrame !== "function") {
      setShown(value);
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      const ease = 1 - (1 - k) ** 3;
      setShown(start + (value - start) * ease);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, ms]);
  return <>{format(shown)}</>;
}

/**
 * FLIP: children with a data-flip key that moved between renders slide from their old position
 * to the new one (First, Last, Invert, Play).
 */
export function useFlip(container: RefObject<HTMLElement>, deps: unknown[]) {
  const last = useRef(new Map<string, DOMRect>());
  useLayoutEffect(() => {
    const el = container.current;
    if (!el) return;
    const now = new Map<string, DOMRect>();
    el.querySelectorAll<HTMLElement>("[data-flip]").forEach((c) => now.set(c.dataset.flip!, c.getBoundingClientRect()));
    if (!reducedMotion()) {
      el.querySelectorAll<HTMLElement>("[data-flip]").forEach((c) => {
        const before = last.current.get(c.dataset.flip!);
        const after = now.get(c.dataset.flip!);
        if (!before || !after || typeof c.animate !== "function") return;
        const dy = before.top - after.top;
        if (Math.abs(dy) > 1) c.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 450, easing: "cubic-bezier(.2,.8,.2,1)" });
      });
    }
    last.current = now;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
