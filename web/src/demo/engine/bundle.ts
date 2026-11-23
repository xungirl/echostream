/**
 * Load and decode the browser bundle written by scripts/export_browser_model.py
 * (format: echostream/browser/bundle_format.py). Decoding is bit-exact with the Python reader:
 * every dequantized value is one float32 multiply (or a table lookup / half-float widening).
 */

export const PATHS = ["two_tower", "itemcf", "pop_recent", "pop_platform", "content"] as const;
export type PathName = (typeof PATHS)[number];

export interface ArrayEntry {
  offset: number;
  bytes: number;
  codec: "f32" | "f16" | "i8row" | "u8code" | "u8" | "u16" | "i32" | "u32" | "varint";
  shape: number[];
  scale?: { offset: number; bytes: number };
}

export interface Constants {
  quotas: Record<PathName, number>;
  total: number;
  hist_len: number;
  cf_hist: number;
  cf_weights: number[];
  content_hist: number;
  recent_n: number;
  pos_thr: number;
  skip_thr: number;
  hist_max: number;
  hist_n: number;
  mmr_lambda: number;
  screen_n: number;
  mmr_top: number;
  user_field_vocab: number[];
  n_tag_categories: number;
  feature_names: string[];
}

export interface Manifest {
  format: string;
  format_version: number;
  created: string;
  source: { version: string; sha256: Record<string, string> };
  quant: Record<string, string | number>;
  n_items: number;
  n_users: number;
  constants: Constants;
  model_file: string;
  arrays: Record<string, ArrayEntry>;
  files: Record<string, { bytes: number; sha256: string }>;
}

export interface Tree {
  split_feature: number[];
  threshold: number[];
  left_child: number[];
  right_child: number[];
  leaf_value: number[];
}

export interface RankerJson {
  objective: "binary";
  sigmoid: number;
  num_features: number;
  feature_names: string[];
  trees: Tree[];
}

export interface Category {
  zh: string;
  en: string;
  n_videos: number;
}

export interface Catalog {
  titles: string[];
  category: number[];
  categories: Category[];
  category_names_note: string;
}

export interface UserRecord {
  uid: number;
  kind: "real" | "new";
  label: string;
  top_categories: { category: number; en: string; share: number }[];
  n_positives: number;
  n_seen: number;
  user_cats: number[];
  user_stats: number[];
  cat_aff: number[];
  log_dur: number;
  pos: number[];
  seen: number[];
  recent_items: number[];
  recent_wr: number[];
}

/** Everything the engine needs, decoded into typed arrays (row-major). */
export interface Model {
  manifest: Manifest;
  c: Constants;
  nItems: number;
  itemEmb: Float32Array; // [n, 64] two-tower item vectors
  embDim: number;
  idEmb: Float32Array; // [n, 32] user-tower item-id embeddings
  idDim: number;
  caption: Float32Array; // [n, 512] caption embeddings (L2-normalized)
  capDim: number;
  userFields: Float32Array[]; // per field [vocab, 8]
  fieldDim: number;
  w1: Float32Array; // [128, in]
  b1: Float32Array;
  w2: Float32Array; // [64, 128]
  b2: Float32Array;
  cfStart: Int32Array; // [n + 1] row offsets into cfIds / cfSims
  cfIds: Int32Array; // neighbour ids, sorted by id inside a row
  cfSims: Float32Array;
  pop: Record<"pop_recent" | "pop_platform", Float32Array>;
  cold: Uint8Array; // 1 = no interactions before the cutoff (the content path's only candidates)
  itemStats: Float32Array; // [n, 14]
  nItemStats: number;
  itemLogDur: Float32Array;
  itemCatBits: Uint32Array; // tag categories as a bit set
  ranker: RankerJson;
  catalog: Catalog;
  users: UserRecord[];
}

// ---------------------------------------------------------------- decoding
const HALF = (() => {
  // half -> float32 lookup (65,536 entries): exact, and the same as numpy's astype(float32)
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) t[h] = s * m * 2 ** -24;
    else if (e === 31) t[h] = m ? NaN : s * Infinity;
    else t[h] = s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

function slice(buf: ArrayBuffer, offset: number, bytes: number): ArrayBuffer {
  return buf.slice(offset, offset + bytes); // copy -> aligned for typed-array views
}

export function decodeArray(buf: ArrayBuffer, e: ArrayEntry): Float32Array | Int32Array | Uint8Array | Uint32Array {
  const raw = slice(buf, e.offset, e.bytes);
  const scale = e.scale ? new Float32Array(slice(buf, e.scale.offset, e.scale.bytes)) : null;
  switch (e.codec) {
    case "f32":
      return new Float32Array(raw);
    case "f16": {
      const h = new Uint16Array(raw);
      const out = new Float32Array(h.length);
      for (let i = 0; i < h.length; i++) out[i] = HALF[h[i]];
      return out;
    }
    case "i8row": {
      const q = new Int8Array(raw);
      const cols = e.shape[1];
      const out = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) out[i] = q[i] * scale![(i / cols) | 0]; // one f32 multiply
      return out;
    }
    case "u8code": {
      const q = new Uint8Array(raw);
      const out = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) out[i] = scale![q[i]];
      return out;
    }
    case "u8":
      return new Uint8Array(raw);
    case "u16":
      return Int32Array.from(new Uint16Array(raw));
    case "i32":
      return new Int32Array(raw);
    case "u32":
      return new Uint32Array(raw);
    case "varint": {
      const b = new Uint8Array(raw);
      const out = new Int32Array(e.shape[0]);
      let pos = 0;
      for (let k = 0; k < out.length; k++) {
        let v = 0;
        let shift = 0;
        for (;;) {
          const x = b[pos++];
          v |= (x & 0x7f) << shift;
          if (x < 0x80) break;
          shift += 7;
        }
        out[k] = v;
      }
      return out;
    }
  }
}

export function decodeModel(
  manifest: Manifest,
  modelBin: ArrayBuffer,
  ranker: RankerJson,
  catalog: Catalog,
  users: UserRecord[],
): Model {
  if (manifest.format !== "echostream-browser-bundle" || manifest.format_version !== 1) {
    throw new Error(`unsupported bundle format ${manifest.format} v${manifest.format_version}`);
  }
  const A = manifest.arrays;
  const get = <T>(name: string) => decodeArray(modelBin, A[name]) as unknown as T;
  const n = manifest.n_items;
  const c = manifest.constants;

  // ItemCF: per-row neighbour ids are gap-coded (the first gap of a row is the absolute id)
  const counts = get<Uint8Array>("cf_counts");
  const gaps = get<Int32Array>("cf_id_gaps");
  const cfStart = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) cfStart[i + 1] = cfStart[i] + counts[i];
  const cfIds = new Int32Array(gaps.length);
  for (let i = 0; i < n; i++) {
    let id = 0;
    for (let k = cfStart[i]; k < cfStart[i + 1]; k++) {
      id += gaps[k];
      cfIds[k] = id;
    }
  }

  return {
    manifest,
    c,
    nItems: n,
    itemEmb: get<Float32Array>("item_emb"),
    embDim: A.item_emb.shape[1],
    idEmb: get<Float32Array>("id_emb"),
    idDim: A.id_emb.shape[1],
    caption: get<Float32Array>("caption"),
    capDim: A.caption.shape[1],
    userFields: c.user_field_vocab.map((_, f) => get<Float32Array>(`user_field_${f}`)),
    fieldDim: A.user_field_0.shape[1],
    w1: get<Float32Array>("w1"),
    b1: get<Float32Array>("b1"),
    w2: get<Float32Array>("w2"),
    b2: get<Float32Array>("b2"),
    cfStart,
    cfIds,
    cfSims: get<Float32Array>("cf_sims"),
    pop: { pop_recent: get<Float32Array>("pop_recent"), pop_platform: get<Float32Array>("pop_platform") },
    cold: get<Uint8Array>("cold"),
    itemStats: get<Float32Array>("item_stats"),
    nItemStats: A.item_stats.shape[1],
    itemLogDur: get<Float32Array>("item_log_dur"),
    itemCatBits: get<Uint32Array>("item_cat_bits"),
    ranker,
    catalog,
    users,
  };
}

export interface LoadTimings {
  downloadMs: number;
  decodeMs: number;
}

/** Fetch the bundle (relative URL, so the page works from any sub-path). Reports download progress. */
export async function loadModel(
  base: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<{ model: Model; timings: LoadTimings }> {
  const t0 = performance.now();
  const manifest: Manifest = await (await fetch(`${base}manifest.json`)).json();
  const total = Object.values(manifest.files).reduce((s, f) => s + f.bytes, 0);
  let loaded = 0;
  const fetchBytes = async (name: string): Promise<ArrayBuffer> => {
    const res = await fetch(`${base}${name}`);
    if (!res.ok || !res.body) throw new Error(`${name}: HTTP ${res.status}`);
    const reader = res.body.getReader();
    const parts: Uint8Array[] = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      size += value.length;
      loaded += value.length; // bytes after any transport decompression (= file size on disk)
      onProgress?.(Math.min(loaded, total), total);
    }
    const out = new Uint8Array(size);
    let o = 0;
    for (const p of parts) {
      out.set(p, o);
      o += p.length;
    }
    return out.buffer;
  };
  const text = async (name: string) => JSON.parse(new TextDecoder().decode(await fetchBytes(name)));
  const gunzip = async (buf: ArrayBuffer) =>
    new Response(new Response(buf).body!.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  const [binGz, ranker, catalog, users] = await Promise.all([
    fetchBytes(manifest.model_file),
    text("ranker.json"),
    text("catalog.json"),
    text("users.json"),
  ]);
  const t1 = performance.now();
  const bin = manifest.model_file.endsWith(".gz") ? await gunzip(binGz) : binGz;
  const model = decodeModel(manifest, bin, ranker, catalog, users.users);
  return { model, timings: { downloadMs: t1 - t0, decodeMs: performance.now() - t1 } };
}
