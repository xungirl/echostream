import { useEffect, useState } from "react";
import { loadData } from "../api";
import { NA, num, pval } from "../format";
import { CIPlot } from "../components/charts";

interface Arm { n_users?: number; pos_rate: number; ci95?: [number, number]; mean_wr?: number; diversity?: number }
interface Comparison { compare?: string; diff?: number; ci95?: [number, number]; welch_p?: number; holm_p?: number; "significant_at_0.05"?: boolean }
interface Within extends Partial<Arm> { coverage?: number; small_matrix_ndcg10_slate1?: number; compare?: string; diff?: number }
interface OffOn {
  offline?: Record<string, number>;
  online_pos_rate?: Record<string, number>;
  kendall_tau?: number;
  spearman_rho?: number;
  disagreeing_pairs?: string[];
}
export interface ABData {
  population?: number;
  catalog?: number;
  mmr_lambda?: number;
  arms?: Record<string, Arm>;
  primary?: Record<string, Comparison>;
  within_subjects?: Record<string, Within>;
  offline_vs_online?: Record<string, OffOn>;
  deviations?: unknown[];
}

const Missing = () => <p className="na">{NA}</p>;
const ci = (c?: [number, number], d = 4) => (c ? `[${num(c[0], d)}, ${num(c[1], d)}]` : "—");

/** 1 = best. Ties share a rank. */
function ranks(vals: Record<string, number>): Record<string, number> {
  const sorted = Object.values(vals).sort((a, b) => b - a);
  return Object.fromEntries(Object.entries(vals).map(([k, v]) => [k, sorted.indexOf(v) + 1]));
}

function OfflineOnline({ label, d }: { label: string; d: OffOn }) {
  const off = d.offline ?? {};
  const on = d.online_pos_rate ?? {};
  const ro = ranks(off), rn = ranks(on);
  return (
    <div className="subpanel">
      <h4 className="mono">{label}</h4>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>variant</th><th>offline</th><th>offline rank</th><th>online pos_rate</th><th>online rank</th></tr>
          </thead>
          <tbody>
            {Object.keys(off).map((v) => (
              <tr key={v} className={ro[v] !== rn[v] ? "disagree" : ""}>
                <td>{v}</td>
                <td className="mono">{num(off[v], 5)}</td>
                <td className="mono">{ro[v]}</td>
                <td className="mono">{num(on[v], 5)}</td>
                <td className="mono">{rn[v] ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mono small">
        Kendall τ {num(d.kendall_tau)} · Spearman ρ {num(d.spearman_rho)}
      </p>
      <p className="small">
        <b>Offline and online disagree on · 离线与线上排序不一致:</b>{" "}
        {d.disagreeing_pairs?.length ? d.disagreeing_pairs.join("; ") : "no pairs (orders agree)"}
      </p>
    </div>
  );
}

export default function AB() {
  const [data, setData] = useState<ABData | null | undefined>(undefined);
  useEffect(() => {
    loadData<ABData>("ab.json").then(setData);
  }, []);

  if (data === undefined) return <p className="muted">loading…</p>;
  if (data === null)
    return (
      <section className="panel">
        <h2>Simulated online A/B · 模拟线上实验</h2>
        <Missing />
        <p className="muted small">Run the Stage-7 experiment, then <code>make web-data</code>.</p>
      </section>
    );

  const arms = Object.entries(data.arms ?? {});
  const primary = Object.entries(data.primary ?? {});
  const within = Object.entries(data.within_subjects ?? {});

  return (
    <section className="ab-page">
      <div className="panel">
        <h2>Simulated online A/B · 模拟线上实验</h2>
        <p className="muted small">
          Pre-registered (PREREG.md). Users bucketed by consistent hash; outcome = positive-watch rate per user.
          {data.population !== undefined && <> Population <b className="mono">{data.population}</b> users.</>}
          {data.mmr_lambda !== undefined && <> MMR λ <b className="mono">{data.mmr_lambda}</b>.</>}
        </p>
        <h3>Arms · 各组 positive rate (95% CI)</h3>
        {arms.length ? (
          <>
            <CIPlot rows={arms.map(([k, a]) => ({ label: k, mean: a.pos_rate, lo: a.ci95?.[0], hi: a.ci95?.[1] }))} />
            <div className="table-wrap">
              <table>
                <thead><tr><th>arm</th><th>n_users</th><th>pos_rate</th><th>95% CI</th><th>mean watch ratio</th><th>diversity</th></tr></thead>
                <tbody>
                  {arms.map(([k, a]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td className="mono">{a.n_users ?? "—"}</td>
                      <td className="mono">{num(a.pos_rate, 5)}</td>
                      <td className="mono">{ci(a.ci95, 5)}</td>
                      <td className="mono">{num(a.mean_wr)}</td>
                      <td className="mono">{num(a.diversity)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <Missing />
        )}
      </div>

      <div className="panel">
        <h3>Primary comparisons · 主要对比 (Welch t, Holm-corrected)</h3>
        {primary.length ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>id</th><th>compare</th><th>diff</th><th>95% CI</th><th>Welch p</th><th>Holm p</th><th>significant @0.05</th></tr></thead>
              <tbody>
                {primary.map(([k, c]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{c.compare ?? "—"}</td>
                    <td className="mono">{num(c.diff, 5)}</td>
                    <td className="mono">{ci(c.ci95, 5)}</td>
                    <td className="mono">{pval(c.welch_p)}</td>
                    <td className="mono">{pval(c.holm_p)}</td>
                    <td>
                      {c["significant_at_0.05"] === undefined ? "—" : (
                        <span className={c["significant_at_0.05"] ? "chip ok" : "chip neutral"}>{c["significant_at_0.05"] ? "yes" : "no"}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Missing />
        )}
      </div>

      <div className="panel">
        <h3>Within-subjects · 组内（每个用户看所有方案）</h3>
        {within.length ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>variant / pair</th><th>pos_rate or diff</th><th>95% CI</th><th>coverage</th><th>small-matrix NDCG@10 (slate 1)</th></tr></thead>
              <tbody>
                {within.map(([k, w]) => (
                  <tr key={k}>
                    <td>{w.compare ? `${k}: ${w.compare}` : k}</td>
                    <td className="mono">{num(w.pos_rate ?? w.diff, 5)}</td>
                    <td className="mono">{ci(w.ci95, 5)}</td>
                    <td className="mono">{num(w.coverage)}</td>
                    <td className="mono">{num(w.small_matrix_ndcg10_slate1, 5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Missing />
        )}
      </div>

      <div className="panel">
        <h3>Offline metric vs online outcome · 离线指标 vs 线上结果</h3>
        <p className="muted small">Rows where the two ranks differ are marked. We show the disagreements instead of hiding them.</p>
        {data.offline_vs_online && Object.keys(data.offline_vs_online).length ? (
          <div className="two-col">
            {Object.entries(data.offline_vs_online).map(([k, d]) => (
              <OfflineOnline key={k} label={k} d={d} />
            ))}
          </div>
        ) : (
          <Missing />
        )}
      </div>

      <div className="panel">
        <h3>Deviations from the pre-registration · 与预注册的偏差</h3>
        {data.deviations === undefined ? (
          <Missing />
        ) : data.deviations.length === 0 ? (
          <p className="small">none recorded</p>
        ) : (
          <ul className="small">
            {data.deviations.map((d, i) => (
              <li key={i}>{typeof d === "string" ? d : JSON.stringify(d)}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
