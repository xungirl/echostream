import { useCallback, useEffect, useState } from "react";
import { getJSON, postJSON } from "../api";
import { num } from "../format";
import type { ClusterState, Proc, ShardReplica, SupervisorStatus } from "../types";

interface Action {
  label: string;
  action: string;
  ms?: number;
}

/** Which chaos buttons a process gets (names come from /supervisor/status). */
export function actionsFor(name: string): Action[] {
  const a: Action[] = [
    { label: "kill", action: "kill" },
    { label: "stop", action: "stop" },
    { label: "start", action: "start" },
  ];
  if (/^redis-feat\d+$/.test(name)) a.push({ label: "freeze", action: "freeze" }, { label: "thaw", action: "thaw" });
  if (/^shard\d+-r\d+$/.test(name)) a.push({ label: "slow 500ms", action: "delay", ms: 500 }, { label: "unslow", action: "delay", ms: 0 });
  return a;
}

function replicaState(r: ShardReplica): [string, string] {
  if (!r.up) return ["bad", "down"];
  if (r.ejected) return ["warn", "ejected"];
  if (r.delay_ms) return ["warn", `slow +${r.delay_ms}ms`];
  return ["ok", "up"];
}

function ProcFlags({ p }: { p?: Proc }) {
  if (!p) return null;
  return (
    <>
      {p.held_down && <span className="chip bad">held down</span>}
      {p.frozen && <span className="chip warn">frozen</span>}
      {!!p.restarts && <span className="chip neutral">restarts {p.restarts}</span>}
    </>
  );
}

const GROUP_ORDER = ["shards", "features", "events", "workers", "api", "lb", "db"];
const groupRank = (g?: string) => {
  const i = GROUP_ORDER.indexOf(g ?? "");
  return i < 0 ? GROUP_ORDER.length : i;
};

export default function Cluster({ pollMs = 2000 }: { pollMs?: number }) {
  const [cluster, setCluster] = useState<ClusterState | null>(null);
  const [sup, setSup] = useState<SupervisorStatus>({});
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState("");

  const poll = useCallback(async () => {
    const [c, s] = await Promise.allSettled([getJSON<ClusterState>("/cluster"), getJSON<SupervisorStatus>("/supervisor/status")]);
    if (c.status === "fulfilled") {
      setCluster(c.value);
      setError(null);
    } else setError(c.reason instanceof Error ? c.reason.message : String(c.reason));
    if (s.status === "fulfilled") setSup(s.value ?? {});
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, pollMs);
    return () => clearInterval(id);
  }, [poll, pollMs]);

  const chaos = async (a: Action, target: string) => {
    const q = `target=${encodeURIComponent(target)}` + (a.ms !== undefined ? `&ms=${a.ms}` : "");
    try {
      const r = await postJSON(`/chaos/${a.action}?${q}`);
      setMsg(`${a.label} ${target} → ${JSON.stringify(r)}`);
      poll();
    } catch (e) {
      setMsg(`${a.label} ${target} failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  const procs = Object.entries(sup).sort(([an, a], [bn, b]) => groupRank(a.group) - groupRank(b.group) || an.localeCompare(bn));
  const nReplicas = Math.max(0, ...(cluster?.shards ?? []).map((s) => s.length));
  const ev = cluster?.events;
  const workerIds = Array.from(new Set([...(ev?.workers_alive ?? []), ...Object.keys(ev?.workers ?? {})])).sort();

  return (
    <section className="cluster-page">
      <div className="banner info">
        chaos actions need the cluster started with <code>ECHO_DEMO_MODE=1</code> (<code>make demo</code> does this)
      </div>
      {error && <div className="banner error">{error}</div>}
      {msg && <div className="banner neutral mono">{msg}</div>}

      <div className="cluster-grid">
        <div className="panel">
          <h3>ANN shards × replicas · 分片 × 副本</h3>
          {cluster ? (
            <div className="table-wrap">
              <table className="shard-grid">
                <thead>
                  <tr>
                    <th>shard</th>
                    {Array.from({ length: nReplicas }, (_, r) => <th key={r}>replica {r}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {cluster.shards.map((reps, s) => (
                    <tr key={s}>
                      <th>{s}</th>
                      {reps.map((r) => {
                        const [cls, text] = replicaState(r);
                        const p = sup[`shard${r.shard}-r${r.replica}`];
                        return (
                          <td key={r.replica} className={`replica ${cls}`} data-testid={`replica-${r.shard}-${r.replica}`}>
                            <span className={`chip ${cls}`}>{text}</span>
                            {r.up && (
                              <div className="mono small">
                                {(r.versions ?? []).map((v) => `${v}:${r.n_items?.[v] ?? "?"}`).join(" ")} · pid {r.pid ?? "?"} · {num(r.uptime_s, 1)}s
                              </div>
                            )}
                            <ProcFlags p={p} />
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">waiting for /cluster…</p>
          )}
        </div>

        <div className="panel">
          <h3>Feature store · 特征节点</h3>
          <div className="chips">
            {Object.entries(cluster?.feature_nodes ?? {}).map(([n, up]) => (
              <span key={n} className="node" data-testid={`feat-${n}`}>
                <span className={up ? "chip ok" : "chip bad"}>{n} {up ? "up" : "down"}</span>
                <ProcFlags p={sup[`redis-${n}`]} />
              </span>
            ))}
          </div>
          <h3>API replicas · 接口副本</h3>
          <div className="chips">
            {procs
              .filter(([n]) => n.startsWith("api"))
              .map(([n, p]) => (
                <span key={n} className={p.running ? "chip ok" : "chip bad"}>{n}{cluster?.api_replica === n ? " ← answered" : ""}</span>
              ))}
            {cluster && <span className="muted small">this poll served by {cluster.api_replica}</span>}
          </div>
        </div>

        <div className="panel">
          <h3>
            Event streams · 事件流{" "}
            {ev && <span className={ev.up ? "chip ok" : "chip bad"}>{ev.up ? "up" : "down"}</span>}{" "}
            {ev?.up && <span className={ev.dlq_length ? "chip bad" : "chip neutral"}>DLQ {ev.dlq_length ?? 0}</span>}
          </h3>
          {ev?.up && (
            <>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>worker</th><th>alive</th><th>partitions</th><th>applied</th><th>duplicates</th><th>dlq</th></tr>
                  </thead>
                  <tbody>
                    {workerIds.map((w) => {
                      const st = ev.workers?.[w] ?? {};
                      const alive = ev.workers_alive?.includes(w);
                      return (
                        <tr key={w}>
                          <td>{w}</td>
                          <td><span className={alive ? "chip ok" : "chip bad"}>{alive ? "alive" : "dead"}</span></td>
                          <td className="mono">{st.partitions ?? "—"}</td>
                          <td className="mono">{st.applied ?? "—"}</td>
                          <td className="mono">{st.duplicates ?? "—"}</td>
                          <td className="mono">{st.dlq ?? "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>partition</th><th>length</th><th>pending</th><th>lag</th><th>owner</th></tr>
                  </thead>
                  <tbody>
                    {(ev.partitions ?? []).map((p) => (
                      <tr key={p.partition}>
                        <td>{p.partition}</td>
                        <td className="mono">{p.length ?? "—"}</td>
                        <td className={p.pending ? "mono warn-text" : "mono"}>{p.pending ?? "—"}</td>
                        <td className={p.lag ? "mono warn-text" : "mono"}>{p.lag ?? "—"}</td>
                        <td className="mono">{p.owner ?? <span className="bad-text">unowned</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="panel">
        <h3>Processes · 进程 (supervisor)</h3>
        {procs.length === 0 ? (
          <p className="muted">supervisor status unavailable</p>
        ) : (
          <div className="table-wrap">
            <table className="procs">
              <thead>
                <tr><th>process</th><th>group</th><th>state</th><th>pid</th><th>port</th><th>restarts</th><th>uptime</th><th>chaos</th></tr>
              </thead>
              <tbody>
                {procs.map(([name, p]) => (
                  <tr key={name} data-testid={`proc-${name}`}>
                    <td className="mono">{name}</td>
                    <td>{p.group ?? "—"}</td>
                    <td>
                      <span className={p.running ? "chip ok" : "chip bad"}>{p.running ? "running" : "stopped"}</span>
                      {p.held_down && <span className="chip bad">held down</span>}
                      {p.frozen && <span className="chip warn">frozen</span>}
                    </td>
                    <td className="mono">{p.pid ?? "—"}</td>
                    <td className="mono">{p.port ?? "—"}</td>
                    <td className={p.restarts ? "mono warn-text" : "mono"}>{p.restarts ?? 0}</td>
                    <td className="mono">{p.uptime_s != null ? `${num(p.uptime_s, 1)}s` : "—"}</td>
                    <td className="chaos-btns">
                      {actionsFor(name).map((a) => (
                        <button key={a.label} className={`mini ${a.action}`} aria-label={`${a.label} ${name}`} onClick={() => chaos(a, name)}>
                          {a.label}
                        </button>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
