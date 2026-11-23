// Two tiny SVG charts (no chart library): a CI dot/bar plot and an x–y chart.
import { num } from "../format";

export interface CIRow {
  label: string;
  mean: number;
  lo?: number;
  hi?: number;
  highlight?: boolean;
}

function ticks(d0: number, d1: number, n = 4): number[] {
  return Array.from({ length: n + 1 }, (_, i) => d0 + ((d1 - d0) * i) / n);
}

/** Horizontal rows: dot = mean, whisker = 95% CI. `bars` draws bars from 0. */
export function CIPlot({ rows, bars = false, digits = 4 }: { rows: CIRow[]; bars?: boolean; digits?: number }) {
  const W = 760, rowH = 28, padL = 190, padR = 200, top = 8, bottom = 24;
  const H = top + rows.length * rowH + bottom;
  let d0 = Math.min(...rows.map((r) => r.lo ?? r.mean));
  let d1 = Math.max(...rows.map((r) => r.hi ?? r.mean));
  if (bars) d0 = Math.min(0, d0);
  const span = d1 - d0 || Math.abs(d1) || 1;
  if (!bars) d0 -= span * 0.08;
  d1 += span * 0.08;
  const x = (v: number) => padL + ((v - d0) / (d1 - d0)) * (W - padL - padR);

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img">
      {ticks(d0, d1).map((t, i) => (
        <g key={i}>
          <line className="grid" x1={x(t)} x2={x(t)} y1={top} y2={H - bottom} />
          <text className="tick" x={x(t)} y={H - 8} textAnchor="middle">{num(t, digits - 1)}</text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2;
        const cls = r.highlight ? "hl" : "";
        return (
          <g key={r.label} className={`ci-row ${cls}`}>
            <text className="row-label" x={padL - 10} y={y + 4} textAnchor="end">{r.label}</text>
            {bars && <rect className="bar" x={x(Math.min(0, r.mean))} y={y - 8} width={Math.abs(x(r.mean) - x(0))} height={16} rx={3} />}
            {r.lo !== undefined && r.hi !== undefined && (
              <g className="whisker">
                <line x1={x(r.lo)} x2={x(r.hi)} y1={y} y2={y} />
                <line x1={x(r.lo)} x2={x(r.lo)} y1={y - 5} y2={y + 5} />
                <line x1={x(r.hi)} x2={x(r.hi)} y1={y - 5} y2={y + 5} />
              </g>
            )}
            <circle className="dot" cx={x(r.mean)} cy={y} r={4.5} />
            <text className="val" x={W - padR + 10} y={y + 4}>
              {num(r.mean, digits)}
              {r.lo !== undefined && r.hi !== undefined ? ` [${num(r.lo, digits)}, ${num(r.hi, digits)}]` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export interface Pt {
  x: number;
  y: number;
  label?: string;
  lo?: number;
  hi?: number;
  highlight?: boolean;
}

/** Points (optionally joined by a line), with optional vertical CI whiskers. */
export function XYChart({ points, xLabel, yLabel, line = true, xDigits = 3, yDigits = 4 }: { points: Pt[]; xLabel: string; yLabel: string; line?: boolean; xDigits?: number; yDigits?: number }) {
  const W = 680, H = 300, padL = 64, padR = 24, padT = 16, padB = 44;
  const ext = (vals: number[]) => {
    let a = Math.min(...vals), b = Math.max(...vals);
    const s = b - a || Math.abs(b) || 1;
    a -= s * 0.1;
    b += s * 0.1;
    return [a, b];
  };
  const [x0, x1] = ext(points.map((p) => p.x));
  const [y0, y1] = ext(points.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]));
  const X = (v: number) => padL + ((v - x0) / (x1 - x0)) * (W - padL - padR);
  const Y = (v: number) => H - padB - ((v - y0) / (y1 - y0)) * (H - padT - padB);

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img">
      {ticks(y0, y1).map((t, i) => (
        <g key={`y${i}`}>
          <line className="grid" x1={padL} x2={W - padR} y1={Y(t)} y2={Y(t)} />
          <text className="tick" x={padL - 6} y={Y(t) + 4} textAnchor="end">{num(t, yDigits)}</text>
        </g>
      ))}
      {ticks(x0, x1).map((t, i) => (
        <text key={`x${i}`} className="tick" x={X(t)} y={H - padB + 16} textAnchor="middle">{num(t, xDigits)}</text>
      ))}
      <text className="axis-label" x={(W + padL) / 2} y={H - 6} textAnchor="middle">{xLabel}</text>
      <text className="axis-label" x={14} y={H / 2} textAnchor="middle" transform={`rotate(-90 14 ${H / 2})`}>{yLabel}</text>
      {line && points.length > 1 && <polyline className="line" points={points.map((p) => `${X(p.x)},${Y(p.y)}`).join(" ")} />}
      {points.map((p, i) => (
        <g key={i} className={p.highlight ? "pt hl" : "pt"}>
          {p.lo !== undefined && p.hi !== undefined && <line className="whisker" x1={X(p.x)} x2={X(p.x)} y1={Y(p.lo)} y2={Y(p.hi)} />}
          <circle className="dot" cx={X(p.x)} cy={Y(p.y)} r={p.highlight ? 6 : 4.5} />
          {p.label && <text className="pt-label" x={X(p.x) + 8} y={Y(p.y) - 8}>{p.label}</text>}
        </g>
      ))}
    </svg>
  );
}

/** Generic table for "whatever keys exist" JSON. */
export function DataTable({ cols, rows }: { cols: string[]; rows: (string | number)[][] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j} className={j ? "mono" : ""}>{v}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
