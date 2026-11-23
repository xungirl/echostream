import type { CI } from "./types";

export const NA = "not measured yet";

export function isObj(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

export function isCI(x: unknown): x is CI {
  return isObj(x) && typeof x.mean === "number";
}

/** Number -> fixed decimals; anything else -> a dash. */
export function num(x: unknown, digits = 4): string {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  return Number.isInteger(x) ? String(x) : x.toFixed(digits);
}

/** p-values: tiny ones in scientific notation. */
export function pval(x: unknown): string {
  if (typeof x !== "number") return "—";
  return x !== 0 && x < 1e-4 ? x.toExponential(1) : x.toFixed(4);
}

export function ciText(x: unknown, digits = 4): string {
  if (typeof x === "number") return num(x, digits);
  if (!isCI(x)) return "—";
  const ci = x.ci95 ? ` [${num(x.ci95[0], digits)}, ${num(x.ci95[1], digits)}]` : "";
  return num(x.mean, digits) + ci;
}

/** Compact text for one table cell of arbitrary JSON. */
export function cell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return num(v, 3);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v;
  if (isCI(v)) return ciText(v, 3);
  if (Array.isArray(v) && v.every((x) => !isObj(x) && !Array.isArray(x))) return v.map(cell).join(" / ");
  const s = JSON.stringify(v);
  return s.length > 90 ? s.slice(0, 87) + "…" : s;
}
