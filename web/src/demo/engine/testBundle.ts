/** Test helper (Node only): load the committed browser bundle and the golden file from disk. */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { resolve } from "node:path";
import { decodeModel, type Model, type UserRecord } from "./bundle";

const WEB = resolve(__dirname, "../../..");
export const BUNDLE_DIR = resolve(WEB, "demo-public/model");
export const GOLDEN_FILE = resolve(WEB, "demo-golden/golden.json.gz");

const json = (name: string) => JSON.parse(readFileSync(resolve(BUNDLE_DIR, name), "utf8"));

let cached: Model | null = null;
export function loadTestModel(): Model {
  if (cached) return cached;
  const manifest = json("manifest.json");
  const bin = gunzipSync(readFileSync(resolve(BUNDLE_DIR, manifest.model_file)));
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  cached = decodeModel(manifest, buf, json("ranker.json"), json("catalog.json"), json("users.json").users);
  return cached;
}

export interface GoldenScreen {
  step: number;
  event: null | { kind: string; item_id: number; watch_ratio: number; action: "play" | "complete" | "skip"; ts: number };
  user_vec: string;
  paths: Record<string, { ids: string; scores: string }>;
  merged_ids: string;
  merged_source: string;
  merged_members: string;
  rank_scores: string;
  shown_ids: number[];
  shown_scores: string;
  features?: string;
}

export interface Golden {
  bundle_files: Record<string, string>;
  decoded_sha256: Record<string, string>;
  script: string[];
  mmr_lambda: number;
  paths: string[];
  users: UserRecord[];
  sessions: { uid: number; screens: GoldenScreen[] }[];
}

export function loadGolden(): Golden {
  return JSON.parse(gunzipSync(readFileSync(GOLDEN_FILE)).toString("utf8"));
}

/** base64 little-endian bytes -> typed arrays */
function bytes(s: string): ArrayBuffer {
  const b = Buffer.from(s, "base64");
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
export const u8 = (s: string) => new Uint8Array(bytes(s));
export const u16 = (s: string) => new Uint16Array(bytes(s));
export const f32 = (s: string) => new Float32Array(bytes(s));
export const f64 = (s: string) => new Float64Array(bytes(s));
