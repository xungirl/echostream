/**
 * Writes the parity numbers (TypeScript engine vs Python golden) for the docs:
 *   cd web && npx vite-node scripts/parity-report.ts   -> ../results/stage13_parity.json
 */
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { runParity } from "../src/demo/engine/parity";
import { loadGolden, loadTestModel } from "../src/demo/engine/testBundle";

const rep = runParity(loadTestModel(), loadGolden());
const { computeMs, ...rest } = rep;
const sorted = [...computeMs].sort((a, b) => a - b);
const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
const out = {
  what: "TypeScript engine vs the Python read path (scripts/gen_browser_golden.py), same bundle, users and events",
  node: process.version,
  date: new Date().toISOString().slice(0, 10),
  ...rest,
  node_compute_ms: { n: sorted.length, median: q(0.5), p95: q(0.95), note: "Node.js (vitest-free), not a browser" },
};
const file = resolve(__dirname, "../../results/stage13_parity.json");
writeFileSync(file, JSON.stringify(out, null, 1) + "\n");
console.log(JSON.stringify(out, null, 1));
