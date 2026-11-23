"""Stage 13 browser bundle: codecs round-trip, and the committed bundle + golden file are current."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from echostream.browser.bundle_format import (cf_to_rows, decode, encode, rows_to_cf, sha256_file,
                                              varint_decode, varint_encode)
from echostream.core.itemcf import itemcf_scores

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "web" / "demo-public" / "model"
GOLDEN = ROOT / "web" / "demo-golden" / "golden.json.gz"
SERVING = ROOT / "artifacts" / "serving" / "current"


def test_codecs_round_trip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 16)).astype(np.float32)
    assert np.array_equal(decode(*encode(x, "f32")[:1], "f32", x.shape), x)
    for codec, tol in (("f16", 1e-2), ("i8row", 0.02)):
        payload, scale = encode(x, codec)
        y = decode(payload, codec, x.shape, scale)
        assert y.dtype == np.float32 and np.abs(y - x).max() < tol
    s = rng.random(1000).astype(np.float32)
    payload, table = encode(s, "u8code")
    assert len(table) == 256 and np.abs(decode(payload, "u8code", s.shape, table) - s).max() < 0.01


def test_varint_and_cf_rows_preserve_itemcf_scores():
    vals = np.array([0, 1, 127, 128, 300, 16383, 16384, 10727])
    assert varint_decode(varint_encode(vals), len(vals)).tolist() == vals.tolist()
    rng = np.random.default_rng(1)
    n, k = 40, 8
    nb = np.full((n, k), -1, np.int64)
    sims = np.zeros((n, k), np.float32)
    for i in range(n):
        c = rng.integers(0, k + 1)
        nb[i, :c] = rng.choice(np.setdiff1d(np.arange(n), [i]), c, replace=False)
        sims[i, :c] = np.sort(rng.random(c).astype(np.float32))[::-1]
    counts, ids, ss = cf_to_rows(nb, sims, k)
    nb2, sims2 = rows_to_cf(counts, ids, ss, k)
    hist = rng.integers(0, n, 12).tolist()
    # neighbours are re-ordered by id inside each row: the float32 scores must not change at all
    assert np.array_equal(itemcf_scores(hist, nb, sims, n, 0.98), itemcf_scores(hist, nb2, sims2, n, 0.98))


def test_golden_file_matches_committed_bundle():
    golden = json.loads(gzip.decompress(GOLDEN.read_bytes()))
    for f, h in golden["bundle_files"].items():
        assert sha256_file(BUNDLE / f) == h, f"{f} changed: rerun scripts/gen_browser_golden.py"


@pytest.mark.skipif(not (SERVING / "manifest.json").exists(), reason="needs the serving bundle (make bundle)")
def test_browser_bundle_was_exported_from_current_serving_bundle():
    m = json.loads((BUNDLE / "manifest.json").read_text())
    stale = [f for f, h in m["source"]["sha256"].items() if sha256_file(SERVING / f) != h]
    assert not stale, f"re-export: uv run python scripts/export_browser_model.py ({stale})"
