// @vitest-environment node
/**
 * Golden parity: the TypeScript engine vs the real Python read path (scripts/gen_browser_golden.py)
 * on the same bundle, the same 50 users and the same events (300 screens).
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { decodeArray, PATHS } from "./bundle";
import { runParity } from "./parity";
import { BUNDLE_DIR, loadGolden, loadTestModel } from "./testBundle";

const golden = loadGolden();
const model = loadTestModel();
const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");

describe("browser bundle", () => {
  it("is the bundle the golden file was generated from", () => {
    for (const [f, h] of Object.entries(golden.bundle_files)) expect(sha(readFileSync(resolve(BUNDLE_DIR, f))), f).toBe(h);
  });

  it("decodes every array bit-identically to the Python reader", () => {
    const bin = gunzipSync(readFileSync(resolve(BUNDLE_DIR, model.manifest.model_file)));
    const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
    for (const [name, h] of Object.entries(golden.decoded_sha256)) {
      const a = decodeArray(buf, model.manifest.arrays[name]);
      const v = a instanceof Float32Array ? a : Int32Array.from(a);
      expect(sha(new Uint8Array(v.buffer, v.byteOffset, v.byteLength)), name).toBe(h);
    }
  });
});

describe("parity with the Python read path (50 users, 300 screens)", () => {
  const rep = runParity(model, golden);
  const st = rep.stage;

  it("replays every golden screen", () => {
    expect(rep.sessions).toBe(50);
    expect(rep.screens).toBe(300);
  });

  it("user vector within 1e-6", () => {
    expect(st.userVecMaxAbsDiff).toBeLessThan(1e-6);
  });

  it("every recall path returns the same candidates (up to the order / selection of tied items)", () => {
    for (const p of PATHS) {
      expect(st.paths[p].equivalent, p).toBe(rep.screens);
      expect(st.paths[p].maxAbsScoreDiff, p).toBeLessThan(1e-6);
    }
    // the two-tower path (exact float32 dot products, no ties) is identical on every screen
    expect(st.paths.two_tower.identical).toBe(rep.screens);
  });

  it("merging Python's path lists gives Python's merged 200 exactly", () => {
    expect(st.mergeIdentical).toBe(rep.screens);
  });

  it("features within 1e-6 and tree evaluation equal to LightGBM", () => {
    expect(st.featureRows).toBeGreaterThan(3000);
    expect(st.featureMaxAbsDiff).toBeLessThan(1e-6);
    expect(st.treeMaxAbsDiff).toBeLessThan(1e-12);
  });

  it("MMR on Python's ranked list picks Python's screen", () => {
    expect(st.mmrIdentical).toBe(rep.screens);
  });

  it("end to end: the same 10 videos in the same order on every screen", () => {
    expect(rep.endToEnd.shownMismatches).toEqual([]);
    expect(rep.endToEnd.rankMaxAbsDiff).toBeLessThan(1e-6);
  });
});
