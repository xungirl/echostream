"""The browser bundle: everything the in-browser engine (web/src/demo/engine/) loads.

    manifest.json   format version, source serving bundle + checksums, pipeline constants,
                    quantization, and a table of every array inside model.bin
    model.bin.gz    all numeric arrays, concatenated, little-endian, 4-byte aligned, gzipped
    ranker.json     the LightGBM trees as flat arrays (one entry per tree)
    catalog.json    per item: title (original Chinese), first-level category; category names zh + en
    users.json      demo users: anonymous KuaiRec id, label, user features, batch history

Array codecs (chosen per array by QuantConfig; the default keeps float32):
    f32     raw float32
    f16     IEEE half precision, widened to float32
    i8row   int8 + one float32 scale per row:  x = float32(q * scale), scale = max|row| / 127
    u8code  uint8 index into a shared table of 256 float32 values (equal-count buckets of the
            sorted data, each value = the bucket mean): x = table[q]
    u16     uint16 integers (item ids), i32 / u8 / u32 integers

Every decode is specified so TypeScript reproduces it bit for bit (a float32 product of two
float32 numbers is correctly rounded in both languages). The ItemCF table is stored as
per-item neighbour lists sorted by item id (delta + LEB128 varint) - the order of neighbours
inside one row does not change ItemCF scores (each neighbour is added once per history item).

This module is shared by scripts/export_browser_model.py (write), scripts/gen_browser_golden.py
(read back -> the real Python Recommender) and scripts/stage13_quant_sweep.py (experiments).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from echostream.rank.features import FEATURE_NAMES, POS_THR, RECENT_N, SKIP_THR, RankAssets
from echostream.recall.engine import CF_DECAY, CF_HIST, CONTENT_HIST, HIST_LEN, RecallAssets, RecallEngine
from echostream.serving.bundle import ServingBundle, pack_lists
from echostream.serving.config import EVENTS

FORMAT = "echostream-browser-bundle"
FORMAT_VERSION = 1
HIST_N = 50            # FeatureStore.get(hist_n=50): the recommender reads the newest 50 real-time events
MMR_LAMBDA = 0.7       # Stage 6: chosen by the pre-written rule (results/stage6_mmr.json)
SCREEN_N = 10          # videos per screen (GET /recommend?n=10)
CATEGORY_EN = Path(__file__).with_name("category_en.json")


@dataclass(frozen=True)
class QuantConfig:
    """How each big array is stored. 'f32' everywhere = exactly the serving bundle."""
    item_emb: str = "f32"        # f32 | f16 | i8row       two-tower item vectors [n_items, 64]
    id_emb: str = "f32"          # f32 | f16 | i8row       user-tower item-id embeddings [n_items, 32]
    caption: str = "f32"         # f32 | f16 | i8row       caption embeddings [n_items, 512]
    cf_sims: str = "f32"         # f32 | f16 | u8code      ItemCF similarities
    cf_topk: int = 200           # neighbours kept per item (the serving table has 200)

    @property
    def name(self) -> str:
        return f"item{self.item_emb}-id{self.id_emb}-cap{self.caption}-cf{self.cf_sims}x{self.cf_topk}"


# ----------------------------------------------------------------------------- codecs
def encode(x: np.ndarray, codec: str) -> tuple[bytes, np.ndarray | None]:
    """-> (payload, per-row scales / code table, or None)."""
    x = np.asarray(x)
    if codec == "f32":
        return x.astype("<f4").tobytes(), None
    if codec == "f16":
        return x.astype("<f2").tobytes(), None
    if codec == "i8row":
        scale = (np.abs(x).max(axis=1) / 127.0).astype(np.float32)
        safe = np.where(scale > 0, scale, 1.0).astype(np.float32)
        return np.rint(x / safe[:, None]).astype(np.int8).tobytes(), scale
    if codec == "u8code":
        flat = x.astype(np.float32).ravel()
        order = np.argsort(flat, kind="stable")
        q = np.empty(len(flat), np.uint8)
        q[order] = (np.arange(len(flat)) * 256 // max(len(flat), 1)).astype(np.uint8)
        table = np.array([flat[q == k].mean() if (q == k).any() else 0.0 for k in range(256)], np.float32)
        return q.tobytes(), table
    if codec in ("u8", "u16", "i32", "u32"):
        return x.astype({"u8": "<u1", "u16": "<u2", "i32": "<i4", "u32": "<u4"}[codec]).tobytes(), None
    raise ValueError(codec)


def decode(buf: bytes, codec: str, shape: tuple[int, ...], scale: np.ndarray | None = None) -> np.ndarray:
    if codec == "f32":
        return np.frombuffer(buf, "<f4").reshape(shape).astype(np.float32)
    if codec == "f16":
        return np.frombuffer(buf, "<f2").reshape(shape).astype(np.float32)
    if codec == "i8row":
        q = np.frombuffer(buf, "<i1").reshape(shape).astype(np.float32)
        return (q * scale.astype(np.float32)[:, None]).astype(np.float32)
    if codec == "u8code":
        return scale.astype(np.float32)[np.frombuffer(buf, "<u1")].reshape(shape)
    dt = {"u8": "<u1", "u16": "<u2", "i32": "<i4", "u32": "<u4"}[codec]
    return np.frombuffer(buf, dt).reshape(shape).copy()


def varint_encode(values: np.ndarray) -> bytes:
    out = bytearray()
    for v in values.tolist():
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def varint_decode(buf: bytes, count: int) -> np.ndarray:
    out = np.empty(count, np.int64)
    pos = 0
    for k in range(count):
        v, shift = 0, 0
        while True:
            b = buf[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        out[k] = v
    return out


def cf_to_rows(neighbors: np.ndarray, sims: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense [n, 200] (sorted by sim, -1 padded) -> per-row neighbour lists sorted by id.
    Returns (counts [n], ids_flat, sims_flat)."""
    counts, ids, ss = [], [], []
    for nb, s in zip(neighbors, sims):
        valid = nb >= 0
        nb, s = nb[valid][:topk], s[valid][:topk]
        order = np.argsort(nb, kind="stable")
        counts.append(len(nb))
        ids.append(nb[order])
        ss.append(s[order])
    return np.array(counts), np.concatenate(ids), np.concatenate(ss).astype(np.float32)


def rows_to_cf(counts: np.ndarray, ids: np.ndarray, sims: np.ndarray, width: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of cf_to_rows, padded back to the serving layout (-1 / 0.0)."""
    n = len(counts)
    nb = np.full((n, width), -1, np.int64)
    sm = np.zeros((n, width), np.float32)
    off = 0
    for r, c in enumerate(counts.tolist()):
        nb[r, :c] = ids[off:off + c]
        sm[r, :c] = sims[off:off + c]
        off += c
    return nb, sm


# ----------------------------------------------------------------------------- writer
class BinWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.table: dict[str, dict] = {}

    def _put(self, payload: bytes) -> tuple[int, int]:
        while len(self.buf) % 4:
            self.buf.append(0)
        off = len(self.buf)
        self.buf += payload
        return off, len(payload)

    def add(self, name: str, x: np.ndarray, codec: str) -> None:
        payload, scale = encode(x, codec)
        off, n = self._put(payload)
        entry = {"offset": off, "bytes": n, "codec": codec, "shape": list(np.shape(x))}
        if scale is not None:
            s_off, s_n = self._put(scale.astype("<f4").tobytes())
            entry["scale"] = {"offset": s_off, "bytes": s_n}
        self.table[name] = entry

    def add_raw(self, name: str, payload: bytes, codec: str, count: int) -> None:
        off, n = self._put(payload)
        self.table[name] = {"offset": off, "bytes": n, "codec": codec, "shape": [count]}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ranker_to_json(model_text: str) -> dict:
    """Parse LightGBM's text model into flat per-tree arrays (no lightgbm needed in the browser).
    Every split in this model is numerical with missing_type None (decision_type 2), so a node
    sends x to the left child iff x <= threshold; children < 0 are leaves (~child = leaf index)."""
    header, _, rest = model_text.partition("\nTree=")
    meta = dict(line.split("=", 1) for line in header.splitlines() if "=" in line)
    trees = []
    for block in ("Tree=" + rest).split("\nTree=")[0:]:
        if not block.strip() or block.startswith("end of trees"):
            continue
        kv = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "num_leaves" not in kv:
            continue
        n_leaves = int(kv["num_leaves"])
        if int(kv.get("num_cat", "0")) != 0:
            raise ValueError("categorical splits are not supported by the browser evaluator")
        leaf_value = [float(v) for v in kv["leaf_value"].split()]
        if n_leaves == 1:
            trees.append({"split_feature": [], "threshold": [], "left_child": [], "right_child": [],
                          "leaf_value": leaf_value})
            continue
        dts = {int(v) for v in kv["decision_type"].split()}
        if dts - {0, 2}:   # bit 0 categorical, bits 2-3 missing type: only "numerical, missing None" here
            raise ValueError(f"unsupported decision_type {sorted(dts)}")
        trees.append({"split_feature": [int(v) for v in kv["split_feature"].split()],
                      "threshold": [float(v) for v in kv["threshold"].split()],
                      "left_child": [int(v) for v in kv["left_child"].split()],
                      "right_child": [int(v) for v in kv["right_child"].split()],
                      "leaf_value": leaf_value,
                      "shrinkage": float(kv.get("shrinkage", "1"))})
    objective = meta.get("objective", "")
    if not objective.startswith("binary"):
        raise ValueError(f"unsupported objective {objective!r}")
    sigmoid = float(objective.split("sigmoid:")[1]) if "sigmoid:" in objective else 1.0
    return {"objective": "binary", "sigmoid": sigmoid, "num_features": int(meta["max_feature_idx"]) + 1,
            "feature_names": FEATURE_NAMES, "trees": trees}


# ----------------------------------------------------------------------------- reader
@dataclass
class BrowserBundle:
    """A decoded browser bundle (the same numbers the TypeScript engine sees)."""
    path: Path
    manifest: dict
    arrays: dict[str, np.ndarray]
    catalog: dict
    users: list[dict]
    ranker_json: dict

    @property
    def constants(self) -> dict:
        return self.manifest["constants"]


def read_browser_bundle(path: Path) -> BrowserBundle:
    path = Path(path)
    m = json.loads((path / "manifest.json").read_text())
    if m.get("format") != FORMAT or m.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION}: {path}")
    blob = gzip.decompress((path / m["model_file"]).read_bytes())
    arrays: dict[str, np.ndarray] = {}
    for name, e in m["arrays"].items():
        raw = blob[e["offset"]:e["offset"] + e["bytes"]]
        if e["codec"] == "varint":
            arrays[name] = varint_decode(raw, e["shape"][0])
            continue
        scale = None
        if "scale" in e:
            s = e["scale"]
            scale = np.frombuffer(blob[s["offset"]:s["offset"] + s["bytes"]], "<f4")
        arrays[name] = decode(raw, e["codec"], tuple(e["shape"]), scale)
    return BrowserBundle(path, m, arrays, json.loads((path / "catalog.json").read_text()),
                         json.loads((path / "users.json").read_text())["users"],
                         json.loads((path / "ranker.json").read_text()))


def item_cats_matrix(bits: np.ndarray, n_cats: int) -> np.ndarray:
    return ((bits[:, None].astype(np.int64) >> np.arange(n_cats)) & 1).astype(np.float32)


def cf_tables(bb: BrowserBundle) -> tuple[np.ndarray, np.ndarray]:
    a = bb.arrays
    gaps = a["cf_id_gaps"]
    counts = a["cf_counts"].astype(np.int64)
    ids = np.empty_like(gaps)
    off = 0
    for c in counts.tolist():   # gaps restart at every row: first id is absolute
        ids[off:off + c] = np.cumsum(gaps[off:off + c])
        off += c
    return rows_to_cf(counts, ids, a["cf_sims"])


def to_serving_bundle(bb: BrowserBundle, users: list[dict], ranker, n_users: int) -> ServingBundle:
    """Build the production ServingBundle object from the browser bundle + user records, so the
    real Recommender runs on exactly the numbers the browser loads. Users not in `users` get
    empty rows (the recommender only ever touches the requested user)."""
    a, c = bb.arrays, bb.constants
    n_cats = c["n_tag_categories"]
    nb, sims = cf_tables(bb)
    n_fields = len(c["user_field_vocab"])
    ut = {f"field_{f}": a[f"user_field_{f}"] for f in range(n_fields)}
    ut.update(item_id_emb=a["id_emb"], w1=a["w1"], b1=a["b1"], w2=a["w2"], b2=a["b2"])
    user_cats = np.zeros((n_users, n_fields), np.int64)
    user_stats = np.zeros((n_users, 6), np.float32)
    user_aff = np.zeros((n_users, n_cats), np.float32)
    user_log_dur = np.zeros(n_users, np.float32)
    empty = np.empty(0, np.int32)
    pos, seen, r_items = [empty] * n_users, [empty] * n_users, [empty] * n_users
    r_wr = [np.empty(0, np.float32)] * n_users
    for u in users:
        i = u["uid"]
        user_cats[i] = u["user_cats"]
        user_stats[i] = np.array(u["user_stats"], np.float32)
        user_aff[i] = np.array(u["cat_aff"], np.float32)
        user_log_dur[i] = np.float32(u["log_dur"])
        pos[i] = np.array(u["pos"], np.int32)
        seen[i] = np.array(u["seen"], np.int32)
        r_items[i] = np.array(u["recent_items"], np.int32)
        r_wr[i] = np.array(u["recent_wr"], np.float32)
    assets = RecallAssets(
        item_vecs=a["item_emb"], user_tower=ut, user_cats=user_cats, cf_neighbors=nb, cf_sims=sims,
        pop_scores={"pop_recent": a["pop_recent"], "pop_platform": a["pop_platform"]},
        caption_emb=a["caption"], item_hist_count=(~a["cold"].astype(bool)).astype(np.float32),
        quotas=dict(c["quotas"]), total=c["total"])
    ra = RankAssets(item_stats=a["item_stats"], user_stats=user_stats, user_cat_aff=user_aff,
                    item_cats=item_cats_matrix(a["item_cat_bits"], n_cats), item_log_dur=a["item_log_dur"],
                    user_log_dur=user_log_dur)
    pf, po = pack_lists(pos, np.int32)
    sf, so = pack_lists(seen, np.int32)
    rif, rio = pack_lists(r_items, np.int32)
    rwf, _ = pack_lists(r_wr, np.float32)
    cat_names = [x["zh"] for x in bb.catalog["categories"]]
    meta = pd.DataFrame({"item_id": np.arange(len(bb.catalog["titles"])), "title": bb.catalog["titles"],
                         "category_name": [cat_names[k] for k in bb.catalog["category"]],
                         "cats": [",".join(str(k) for k in range(n_cats) if (b >> k) & 1)
                                  for b in a["item_cat_bits"].tolist()]})
    return ServingBundle(bb.manifest["source"]["version"], bb.manifest, RecallEngine(assets), ra, ranker,
                         pf, po, sf, so, rif, rwf, rio, meta, meta.cats.tolist(), bb.path)


def constants_block(quotas: dict, total: int, user_field_vocab: list[int], n_tag_categories: int) -> dict:
    return {"quotas": quotas, "total": total, "hist_len": HIST_LEN, "cf_hist": CF_HIST, "cf_decay": CF_DECAY,
            # ItemCF weights exactly as Python computes them (CF_DECAY ** age as a float64, then the
            # float32 multiply in itemcf_scores): shipped so the browser does not depend on Math.pow
            "cf_weights": [CF_DECAY ** k for k in range(CF_HIST)],
            "content_hist": CONTENT_HIST, "recent_n": RECENT_N, "pos_thr": POS_THR, "skip_thr": SKIP_THR,
            "hist_max": EVENTS.hist_max, "hist_n": HIST_N, "mmr_lambda": MMR_LAMBDA, "screen_n": SCREEN_N,
            "mmr_top": max(5 * SCREEN_N, 50), "user_field_vocab": user_field_vocab,
            "n_tag_categories": n_tag_categories, "feature_names": FEATURE_NAMES}


def dump_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

