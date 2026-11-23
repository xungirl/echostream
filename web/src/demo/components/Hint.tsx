/** First-visit hint: three steps, dismissible, remembered per browser. */
import { useEffect, useRef } from "react";

const STEPS = [
  ["Pick a viewer", "Six real KuaiRec users with different tastes, or a brand-new user."],
  ["React to a video", "Skip, Watched or Like. (Keys S / W / L work on the focused video.)"],
  ["Watch the echo", "The next screen is recomputed right away. New videos glow, and the right side shows how the screen was made."],
];

export default function Hint({ onClose }: { onClose: () => void }) {
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    btn.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="hint-backdrop" role="dialog" aria-modal="true" aria-labelledby="hint-h" data-testid="hint">
      <div className="hint">
        <h2 id="hint-h">See a recommender react to you</h2>
        <ol>
          {STEPS.map(([h, t], k) => (
            <li key={h}>
              <span className="step-n">{k + 1}</span>
              <div>
                <b>{h}</b>
                <p>{t}</p>
              </div>
            </li>
          ))}
        </ol>
        <button ref={btn} className="primary" onClick={onClose}>
          Got it
        </button>
      </div>
    </div>
  );
}
