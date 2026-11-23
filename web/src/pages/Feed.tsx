import { useCallback, useEffect, useRef, useState } from "react";
import { getJSON, postJSON, sleep } from "../api";
import { num } from "../format";
import type { Item, Recommendation } from "../types";
import Waterfall from "../components/Waterfall";

const PRESETS = [42, 7, 1024, 5000];
const MAX_USER = 7175;
const WATCHED = 2.5; // watch_ratio > 2.0 is a positive label
const SKIPPED = 0.1;

export const SOURCE_COLORS: Record<string, string> = {
  two_tower: "#a78bfa",
  itemcf: "#22d3ee",
  pop_recent: "#fbbf24",
  pop_platform: "#fb923c",
  content: "#34d399",
};

/** Unique id for the dedup check in the event workers (no ':' or spaces). */
export function newEventId(userId: number, itemId: number): string {
  return `web-${userId}-${itemId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function Feed({ initialUser = 42, refreshDelayMs = 400 }: { initialUser?: number; refreshDelayMs?: number }) {
  const [uidText, setUidText] = useState(String(initialUser));
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [fresh, setFresh] = useState<Set<number>>(new Set()); // item ids new since the previous refresh
  const [gone, setGone] = useState<Item[]>([]); // items that dropped out
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const prevRef = useRef<Recommendation | null>(null);

  const load = useCallback(async (uid: number) => {
    setBusy(true);
    setError(null);
    try {
      const r = await getJSON<Recommendation>(`/recommend?user_id=${uid}&n=10`);
      const prev = prevRef.current;
      if (prev && prev.user_id === r.user_id) {
        const before = new Set(prev.items.map((i) => i.item_id));
        const now = new Set(r.items.map((i) => i.item_id));
        setFresh(new Set(r.items.filter((i) => !before.has(i.item_id)).map((i) => i.item_id)));
        setGone(prev.items.filter((i) => !now.has(i.item_id)));
      } else {
        setFresh(new Set());
        setGone([]);
      }
      prevRef.current = r;
      setRec(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    load(initialUser);
  }, [load, initialUser]);

  const pick = (uid: number) => {
    setUidText(String(uid));
    setNote("");
    load(uid);
  };

  const refresh = () => {
    const uid = Number(uidText);
    if (!Number.isInteger(uid) || uid < 0 || uid > MAX_USER) {
      setError(`user_id must be an integer in 0..${MAX_USER}`);
      return;
    }
    load(uid);
  };

  const sendEvent = async (item: Item, watched: boolean) => {
    if (!rec) return;
    const uid = rec.user_id;
    const event = { event_id: newEventId(uid, item.item_id), user_id: uid, item_id: item.item_id, watch_ratio: watched ? WATCHED : SKIPPED };
    try {
      const res = await postJSON<{ accepted: number; streams: string[] }>("/events", { events: [event] });
      setNote(`${watched ? "watched" : "skipped"} #${item.item_id} → ${res.streams.join(", ")} · refreshing…`);
      await sleep(refreshDelayMs); // events are applied asynchronously by the workers
      await load(uid);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <section className="feed-page">
      <div className="toolbar panel">
        <label>
          user_id
          <input type="number" min={0} max={MAX_USER} value={uidText} onChange={(e) => setUidText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && refresh()} aria-label="user id" />
        </label>
        <button className="primary" onClick={refresh} disabled={busy}>
          Refresh 刷新
        </button>
        <button onClick={() => pick(Math.floor(Math.random() * (MAX_USER + 1)))}>random user</button>
        <span className="presets">
          {PRESETS.map((u) => (
            <button key={u} className="ghost" onClick={() => pick(u)}>
              #{u}
            </button>
          ))}
        </span>
      </div>

      {error && <div className="banner error">{error}</div>}

      {rec && (
        <div className="meta">
          <span className="chip neutral">user <b className="mono">{rec.user_id}</b></span>
          <span className="chip neutral">model <b className="mono">{rec.model_version}</b></span>
          <span className="chip neutral">served_by <b className="mono">{rec.served_by}</b></span>
          {rec.degraded.length === 0 && <span className="chip ok">not degraded</span>}
          {rec.degraded.map((d) => (
            <span key={d} className={d === "cache_hit" ? "chip neutral" : "chip bad"}>{d}</span>
          ))}
          {note && <span className="note">{note}</span>}
        </div>
      )}

      <div className="feed-grid">
        <div>
          <div className="cards">
            {rec?.items.map((it, i) => (
              <article key={it.item_id} className={fresh.has(it.item_id) ? "card new" : "card"} data-testid={`card-${it.item_id}`}>
                <div className="card-top">
                  <span className="rank mono">#{i + 1}</span>
                  {fresh.has(it.item_id) && <span className="new-tag">NEW 新</span>}
                  <span className="cat">{it.category}</span>
                </div>
                <h3 className="title">{it.title || `item ${it.item_id}`}</h3>
                <div className="badges">
                  {Object.entries(it.sources ?? {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([s, v]) => (
                      <span key={s} className={s === it.source ? "badge primary" : "badge"} style={{ "--c": SOURCE_COLORS[s] ?? "#94a3b8" } as React.CSSProperties}>
                        {s} <b className="mono">{num(v, 2)}</b>
                      </span>
                    ))}
                </div>
                <div className="scores mono">
                  rank {num(it.rank_score)} · recall {num(it.recall_score)} · id {it.item_id}
                </div>
                <div className="actions">
                  <button className="watch" onClick={() => sendEvent(it, true)}>Watched ▶</button>
                  <button className="skip" onClick={() => sendEvent(it, false)}>Skip ⏭</button>
                </div>
              </article>
            ))}
          </div>
          {gone.length > 0 && (
            <div className="gone panel" data-testid="gone">
              <h4>Dropped since last refresh · 被挤出的</h4>
              <ul>
                {gone.map((g) => (
                  <li key={g.item_id}>
                    <span className="mono">#{g.item_id}</span> {g.title}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <aside className="panel">
          <h3>Latency waterfall · 延迟瀑布</h3>
          {rec ? <Waterfall timings={rec.timings_ms} shards={rec.shards ?? []} /> : <p className="muted">no request yet</p>}
          <div className="legend">
            {Object.entries(SOURCE_COLORS).map(([s, c]) => (
              <span key={s} className="badge" style={{ "--c": c } as React.CSSProperties}>{s}</span>
            ))}
          </div>
        </aside>
      </div>
    </section>
  );
}
