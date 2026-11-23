/**
 * echostream browser demo: the real trained models, running in this tab.
 * Loading -> pick a viewer -> react to videos -> every reaction recomputes the next screen.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { loadModel, type LoadTimings, type Model, type UserRecord } from "./engine/bundle";
import { REACTIONS } from "./labels";
import { DemoSession, type Step } from "./session";
import Phone from "./components/Phone";
import Funnel from "./components/Funnel";
import Echo from "./components/Echo";
import Profile from "./components/Profile";
import Hint from "./components/Hint";
import { TitleLangContext, hasEnglishTitles, type TitleLang } from "./titles";

const HINT_KEY = "echostream-demo-hint-seen";
const THEME_KEY = "echostream-demo-theme";
const LANG_KEY = "echostream-demo-title-lang";
const REPORT = "https://xungirl.com/echostream/";
const REPO = "https://github.com/xungirl/echostream";

const store = {
  get: (k: string) => {
    try {
      return window.localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set: (k: string, v: string) => {
    try {
      window.localStorage.setItem(k, v);
    } catch {
      /* private mode: fine, nothing to remember */
    }
  },
};

/** Hooks for the Playwright smoke test and benchmark (read-only except bench). */
declare global {
  interface Window {
    __echostream?: {
      ttiMs?: number;
      load?: LoadTimings;
      screens: number;
      bench?: (n: number, uid?: number) => number[];
    };
  }
}

export default function DemoApp({ modelBase = "./model/" }: { modelBase?: string }) {
  const [model, setModel] = useState<Model | null>(null);
  const [progress, setProgress] = useState<[number, number]>([0, 0]);
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<DemoSession | null>(null);
  const [step, setStep] = useState<Step | null>(null);
  const [hint, setHint] = useState(() => store.get(HINT_KEY) !== "1");
  const [theme, setTheme] = useState<string | null>(() => store.get(THEME_KEY));
  const [titleLang, setTitleLang] = useState<TitleLang>(() => (store.get(LANG_KEY) === "zh" ? "zh" : "en"));
  const [announce, setAnnounce] = useState("");
  const ttiDone = useRef(false);

  useEffect(() => {
    window.__echostream = window.__echostream ?? { screens: 0 };
    loadModel(modelBase, (a, b) => setProgress([a, b]))
      .then(({ model, timings }) => {
        window.__echostream!.load = timings;
        window.__echostream!.bench = (n: number, uid?: number) => {
          const user = model.users.find((u) => u.uid === uid) ?? model.users[0];
          const s = new DemoSession(model, user);
          const out: number[] = [];
          for (let k = 0; k < n; k++) {
            const top = s.current.screen.shown[k % s.current.screen.shown.length].item;
            out.push(s.react(top, REACTIONS[k % 3].kind).screen.timingsMs.total);
          }
          return out;
        };
        setModel(model);
        const s = new DemoSession(model, model.users[0]);
        setSession(s);
        setStep(s.current);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [modelBase]);

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (!step) return;
    window.__echostream!.screens += 1;
    if (!ttiDone.current) {
      ttiDone.current = true;
      performance.mark("echostream-interactive");
      window.__echostream!.ttiMs = performance.now();
    }
  }, [step]);

  const pickUser = useCallback(
    (u: UserRecord) => {
      if (!model) return;
      const s = new DemoSession(model, u);
      setSession(s);
      setStep(s.current);
      setAnnounce(`Now viewing as ${u.kind === "new" ? "a brand-new user" : `KuaiRec user ${u.uid}`}.`);
    },
    [model],
  );

  const react = useCallback(
    (item: number, kind: (typeof REACTIONS)[number]["kind"]) => {
      if (!session) return;
      const next = session.react(item, kind); // synchronous: one event + one full recompute (~10 ms)
      setStep(next);
      setAnnounce(`${REACTIONS.find((r) => r.kind === kind)!.label}. New screen: ${next.fresh.length} new videos, ${next.dropped.length} left.`);
    },
    [session],
  );

  const closeHint = useCallback(() => {
    setHint(false);
    store.set(HINT_KEY, "1");
  }, []);

  const toggleTheme = () => {
    const dark = theme ? theme === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    setTheme(next);
    store.set(THEME_KEY, next);
  };

  const prev = session && step && step.n > 0 ? session.steps[step.n - 1] : null;
  const pct = progress[1] ? Math.round((100 * progress[0]) / progress[1]) : 0;

  return (
    <TitleLangContext.Provider value={titleLang}>
    <div className="demo">
      <header className="top">
        <a className="brand" href={REPO} aria-label="echostream on GitHub">
          <img src="./brand/icon.svg" alt="" width={40} height={40} />
          <span>
            <b className="word">
              <span className="echo">echo</span>stream
            </b>{" "}
            <span className="zh" lang="zh">
              流光回响
            </span>
          </span>
        </a>
        <p className="tagline">A short-video recommender, running live in your browser. Every swipe changes the next screen.</p>
        <div className="top-actions">
          <button className="ghost" onClick={() => setHint(true)}>
            How it works
          </button>
          {model && hasEnglishTitles(model) && (
            <button
              className="ghost"
              aria-pressed={titleLang === "zh"}
              aria-label={titleLang === "en" ? "Show the original Chinese titles" : "Show English titles"}
              onClick={() => {
                const next: TitleLang = titleLang === "en" ? "zh" : "en";
                setTitleLang(next);
                store.set(LANG_KEY, next);
              }}
            >
              {titleLang === "en" ? "中文原文" : "English"}
            </button>
          )}
          <button className="ghost" onClick={toggleTheme} aria-label="Toggle light and dark theme">
            Theme
          </button>
        </div>
      </header>

      {error && (
        <p className="error" role="alert">
          Could not load the model: {error}
        </p>
      )}

      {!model && !error && (
        <div className="loading" role="status" aria-live="polite" data-testid="loading">
          <img src="./brand/icon.svg" alt="" width={72} height={72} className="pulse" />
          <p>Loading the trained models into your browser…</p>
          <div className="progress" aria-hidden="true">
            <span style={{ width: `${pct}%` }} />
          </div>
          <p className="muted small">
            {(progress[0] / 1e6).toFixed(1)} / {(progress[1] / 1e6 || 0).toFixed(1)} MB · two-tower model, ItemCF table, caption embeddings, LightGBM ranker
          </p>
        </div>
      )}

      {model && session && step && (
        <>
          <section className="picker" aria-labelledby="picker-h">
            <h2 id="picker-h">Watch as…</h2>
            <p className="muted small">Real viewers from the KuaiRec dataset (anonymous ids) with their real watch histories, or a viewer with no history.</p>
            <div className="users" role="radiogroup" aria-labelledby="picker-h">
              {model.users.map((u) => (
                <button
                  key={u.uid}
                  role="radio"
                  aria-checked={u.uid === session.user.uid}
                  className={u.uid === session.user.uid ? "user on" : "user"}
                  onClick={() => pickUser(u)}
                  data-testid={`user-${u.uid}`}
                >
                  <b>{u.label}</b>
                  <span className="muted small">
                    {u.kind === "new" ? "cold start · no history" : `KuaiRec user #${u.uid} · ${u.n_positives} likes`}
                  </span>
                </button>
              ))}
            </div>
          </section>

          <main className="layout">
            <Phone model={model} step={step} onReact={react} />
            <Funnel model={model} step={step} />
            <div className="side">
              <Echo model={model} step={step} prev={prev} />
              <Profile model={model} step={step} prev={prev} store={session.store} />
            </div>
          </main>
        </>
      )}

      <p className="sr-only" aria-live="polite">
        {announce}
      </p>

      <footer className="foot">
        <p>
          Real models trained on KuaiRec, running in your browser. The distributed serving layer (sharded index, Redis Streams workers, Go worker) is
          not part of this page; see the <a href={REPORT}>report</a>. Code: <a href={REPO}>github.com/xungirl/echostream</a>.
        </p>
        <p className="muted small">
          Data: KuaiRec (Gao et al., CIKM 2022), © its authors, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>; the
          captions, statistics and models in this page are derived from it and shared under the same license. Video titles are the original
          Chinese captions, shown in English as machine translations (Qwen, translated once and cached; hover a title or press “中文原文”
          for the original); English category names are hand-translated. Embeddings are stored as int8 to keep the download small; the report
          lists how often screens match the full-precision model. Times shown are computation in this browser tab, not server latency.
        </p>
      </footer>

      {hint && model && <Hint onClose={closeHint} />}
    </div>
    </TitleLangContext.Provider>
  );
}
