import { useEffect, useState } from "react";
import { loadData } from "../api";
import { NA, cell, ciText, isCI, isObj, num } from "../format";
import type { CI, JourneyItem } from "../types";
import { CIPlot, DataTable, XYChart } from "../components/charts";
import type { CIRow } from "../components/charts";
import Journey from "../components/Journey";

interface MMR {
  lambdas?: Record<string, Record<string, CI>>;
  selected_lambda?: number;
}
interface HistoryRun {
  run_id: string;
  ts?: string;
  candidate?: { name?: string };
  metrics?: Record<string, CI>;
}
export interface QualityData {
  baselines_test_ndcg10?: Record<string, CI>;
  two_tower_test_ndcg10?: Record<string, CI>;
  recall_rank_test_ndcg10?: CI | null;
  recall_only_test_ndcg10?: CI | null;
  ranker_auc?: number | CI | null;
  ranker_gauc?: number | CI | null;
  mmr?: MMR | null;
  history?: HistoryRun[];
}

const Missing = () => <p className="na">{NA}</p>;
const toRow = (label: string, c: CI): CIRow => ({ label, mean: c.mean, lo: c.ci95?.[0], hi: c.ci95?.[1] });

function modelRows(q: QualityData): CIRow[] {
  const rows: CIRow[] = [];
  for (const [k, v] of Object.entries(q.baselines_test_ndcg10 ?? {})) if (isCI(v)) rows.push(toRow(`baseline · ${k}`, v));
  for (const [k, v] of Object.entries(q.two_tower_test_ndcg10 ?? {})) if (isCI(v)) rows.push(toRow(`two-tower · ${k}`, v));
  if (isCI(q.recall_only_test_ndcg10)) rows.push(toRow("recall only (merge order)", q.recall_only_test_ndcg10));
  if (isCI(q.recall_rank_test_ndcg10)) rows.push({ ...toRow("recall + rank", q.recall_rank_test_ndcg10), highlight: true });
  return rows;
}

/** Renders any JSON summary as a compact table (systems.json has no fixed schema). */
function Generic({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <Missing />;
  if (Array.isArray(value) && value.length && value.every(isObj)) {
    const cols = Array.from(new Set(value.flatMap((r) => Object.keys(r))));
    return <DataTable cols={cols} rows={value.map((r) => cols.map((c) => cell(r[c])))} />;
  }
  if (isObj(value)) {
    const entries = Object.entries(value);
    if (entries.length && entries.every(([, v]) => isObj(v))) {
      const cols = Array.from(new Set(entries.flatMap(([, v]) => Object.keys(v as object))));
      return <DataTable cols={["", ...cols]} rows={entries.map(([k, v]) => [k, ...cols.map((c) => cell((v as Record<string, unknown>)[c]))])} />;
    }
    return <DataTable cols={["key", "value"]} rows={entries.map(([k, v]) => [k, cell(v)])} />;
  }
  return <p className="mono">{cell(value)}</p>;
}

export default function Quality() {
  const [q, setQ] = useState<QualityData | null | undefined>(undefined);
  const [systems, setSystems] = useState<Record<string, unknown> | null>(null);
  const [journey, setJourney] = useState<JourneyItem[] | null>(null);

  useEffect(() => {
    loadData<QualityData>("quality.json").then(setQ);
    loadData<Record<string, unknown>>("systems.json").then(setSystems);
    loadData<JourneyItem[]>("journey.json").then(setJourney);
  }, []);

  const rows = q ? modelRows(q) : [];
  const lambdas = Object.entries(q?.mmr?.lambdas ?? {}).sort((a, b) => Number(a[0]) - Number(b[0]));
  const sel = q?.mmr?.selected_lambda;
  const history = (q?.history ?? []).filter((h) => isCI(h.metrics?.["ndcg@10"]));

  return (
    <section className="quality-page">
      <div className="panel">
        <h2>Offline quality · 离线质量 <span className="muted small">test phase, NDCG@10, 95% bootstrap CI</span></h2>
        {q === undefined ? <p className="muted">loading…</p> : rows.length ? <CIPlot rows={rows} bars /> : <Missing />}
        {q && (
          <p className="mono small">
            ranker AUC {q.ranker_auc == null ? NA : ciText(q.ranker_auc)} · GAUC {q.ranker_gauc == null ? NA : ciText(q.ranker_gauc)}
          </p>
        )}
      </div>

      <div className="two-col">
        <div className="panel">
          <h3>MMR trade-off · 相关性 vs 多样性</h3>
          {lambdas.length ? (
            <>
              <XYChart
                xLabel="diversity@10"
                yLabel="NDCG@10"
                points={lambdas
                  .filter(([, m]) => isCI(m["ndcg@10"]) && isCI(m["diversity@10"]))
                  .map(([k, m]) => ({
                    x: m["diversity@10"].mean,
                    y: m["ndcg@10"].mean,
                    lo: m["ndcg@10"].ci95?.[0],
                    hi: m["ndcg@10"].ci95?.[1],
                    label: `λ=${k}`,
                    highlight: sel !== undefined && Number(k) === sel,
                  }))}
              />
              <DataTable
                cols={["λ", "NDCG@10", "diversity@10", "coverage@10"]}
                rows={lambdas.map(([k, m]) => [`${k}${Number(k) === sel ? " ✓ selected" : ""}`, ciText(m["ndcg@10"]), ciText(m["diversity@10"]), ciText(m["coverage@10"])])}
              />
            </>
          ) : (
            <Missing />
          )}
        </div>

        <div className="panel">
          <h3>Eval history · 评测记录 <span className="muted small">make eval</span></h3>
          {history.length ? (
            <>
              <XYChart
                xLabel="run #"
                yLabel="NDCG@10"
                xDigits={0}
                points={history.map((h, i) => ({
                  x: i + 1,
                  y: h.metrics!["ndcg@10"].mean,
                  lo: h.metrics!["ndcg@10"].ci95?.[0],
                  hi: h.metrics!["ndcg@10"].ci95?.[1],
                }))}
              />
              <DataTable
                cols={["#", "run", "candidate", "NDCG@10", "recall@200"]}
                rows={history.map((h, i) => [i + 1, h.run_id, h.candidate?.name ?? "—", ciText(h.metrics?.["ndcg@10"]), ciText(h.metrics?.["recall@200"])])}
              />
            </>
          ) : (
            <Missing />
          )}
        </div>
      </div>

      <div className="panel">
        <h3>Systems · 系统实验 <span className="muted small">load, chaos, replay, synthetic scale, rollout, release guard</span></h3>
        {systems === null ? (
          <Missing />
        ) : (
          Object.entries(systems).map(([k, v]) => (
            <details key={k} className="sys" open={k === "load" || k === "replay"}>
              <summary>{k}</summary>
              <Generic value={v} />
            </details>
          ))
        )}
      </div>

      <div className="panel">
        <h2>Project journey · 项目历程</h2>
        {journey?.length ? <Journey items={journey} /> : <Missing />}
      </div>

      <p className="muted small">Numbers on this page are read from <code>/data/*.json</code>, produced by <code>scripts/build_web_data.py</code> from <code>results/</code>; nothing is typed in by hand.</p>
    </section>
  );
}
