"""Write the cross-language golden files the Go worker's tests check against.

Python is the reference: every expected value below is computed by the Python
code the Go worker must agree with (hashing.py, events.py, feature_store.py,
worker.py, and CPython's own int()/float()/repr()).

    uv run python scripts/gen_go_golden.py            # (re)write go/worker/testdata/*.json
    uv run python scripts/gen_go_golden.py --check    # exit 1 if the files are stale

tests/unit/test_go_golden.py runs --check, so changing the Python side without
regenerating (and re-running `make go-test`) fails CI.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

from echostream.core.hashing import ConsistentHashRing, partition_for, stable_hash
from echostream.serving.events import PoisonEvent, parse_fields
from echostream.serving.feature_store import APPLY_EVENT_LUA
from echostream.serving.worker import LAT_BUCKETS_MS

OUT = Path(__file__).resolve().parents[1] / "go" / "worker" / "testdata"
SEED = 20261122


def hashing_golden() -> dict:
    rng = random.Random(SEED)
    users = list(range(10000)) + [2**31 - 1, 2**31, 2**40 + 7, 2**62, 2**63 - 1] + \
        [rng.randrange(0, 2**63) for _ in range(1000)]
    keys = ["", "a", "user:42", "echostream", "feat0#0", "ünïcödé", "x" * 300] + [f"k{rng.random()}" for _ in range(200)]
    rings = []
    for n in (1, 2, 3, 4, 5, 8):
        nodes = [f"feat{i}" for i in range(n)]
        ring = ConsistentHashRing(nodes, 100)
        rings.append({"nodes": nodes, "vnodes": 100,
                      "node_index": [nodes.index(ring.get_node(f"user:{u}")) for u in users]})
    parts = [{"n": n, "partition": [partition_for(u, n) for u in users]} for n in (1, 2, 3, 4, 6, 8, 16)]
    return {"user_ids": users, "stable_hash": [{"key": k, "hash": str(stable_hash(k))} for k in keys],
            "rings": rings, "partitions": parts}


def fuzz_number_strings(rng: random.Random, n: int) -> list[str]:
    atoms = ["0", "1", "7", "9", "00", "_", ".", "e", "E", "+", "-", " ", "\t", "inf", "Infinity", "nan", "NaN",
             "x", "0x", "١", "٣", "１", " ", " ", "\x1c", "j", ",", "e+", "e-", "1e308", "9" * 20]
    return ["".join(rng.choice(atoms) for _ in range(rng.randint(0, 6))) for _ in range(n)]


def pycompat_golden() -> dict:
    rng = random.Random(SEED)
    fixed = ["abc", "", " 7 ", "1_0", "_1", "1__0", "1_", "٣", "１２", "0x10", "+5", "-0", "-", "+", "7.0",
             "\xa012 ", "\x1c5\x1f", "1e5", "007", "0_7", "9223372036854775807", "-9223372036854775808",
             "1_000.000_1e1_0", "1._5", "1_e5", ".5", "5.", "1e400", "-1e-400", "inf", "-Infinity", "+nan",
             "-nAn", "0x1p3", "١.٥", "1e", "e5", " 2.5 ", "infinit", "nan(1)", "1.5e+3", "1.e3", ".e3", "1E-3",
             ".", "..5", "1.5.5", "--1", "+-1", "1 2", "١٢٣", "๑๒", "\U0001d7ce\U0001d7d9", "2.5\x00"]
    strs = fixed + fuzz_number_strings(rng, 2000)
    ints, floats = [], []
    for s in strs:
        try:
            v = int(s)
            # Go uses int64; Python ints beyond that are reported (and treated as poison) in Go
            ints.append({"in": s, "ok": str(v), "fits_int64": -2**63 <= v < 2**63})
        except ValueError as e:
            ints.append({"in": s, "err": str(e)})
        try:
            floats.append({"in": s, "repr": repr(float(s))})
        except ValueError as e:
            floats.append({"in": s, "err": str(e)})
    values = [0.0, -0.0, 1.0, 0.1, 2.5, 12.0, 1e16, 1e15, 123456789012345678.0, 1e-4, 1e-5, 1593878903.438,
              5e-324, 1.7976931348623157e308, 2.5e-7, 12345.678e10, 9999999999999998.0, 0.30000000000000004,
              float("inf"), float("-inf")]
    values += [struct.unpack("<d", rng.randbytes(8))[0] for _ in range(2000)]
    values += [rng.uniform(0, 5) for _ in range(500)] + [rng.uniform(1.5e9, 1.7e9) for _ in range(500)]
    values += [round(rng.uniform(0, 100), rng.randint(0, 6)) for _ in range(500)]
    values = [v for v in values if v == v]          # NaN has no hex form; covered by FormatFloat's own test
    reprs = [{"hex": v.hex(), "repr": repr(v)} for v in values]
    text = ["", "abc", "it's", 'say "hi"', "both ' \"", "\\", "tab\t\n\r", "\x00\x01\x1f\x7f", "é", "​",
            " ", "\U0001F600", "퟿", "\x85", " ", "͸", "日本語", "a'b\"c", "", "\U000e0001"]
    text += ["".join(chr(rng.choice([rng.randint(0x20, 0x7e), rng.randint(0, 0x2ff), rng.randint(0x300, 0xd7ff),
                                     rng.randint(0xe000, 0x10ffff)])) for _ in range(rng.randint(0, 8)))
             for _ in range(1500)]
    return {"ints": ints, "floats": floats, "float_repr": reprs,
            "str_repr": [{"in": t, "repr": repr(t)} for t in text]}


def events_golden() -> dict:
    good = {"event_id": "e1", "user_id": "42", "item_id": "7", "action": "play", "watch_ratio": "0.5",
            "ts": "1593878903.438", "cats": "1,2", "produced_ms": "1763851234567.123"}
    cases = [good, {**good, "cats": ""}, {k: v for k, v in good.items() if k != "cats"},
             {"event_id": "bad-1", "user_id": "not-a-number"}, {"garbage": "x"}, {},
             {**good, "user_id": "-1"}, {**good, "item_id": "x"}, {**good, "watch_ratio": "abc"},
             {**good, "watch_ratio": "-0.1"}, {**good, "watch_ratio": "nan"}, {**good, "watch_ratio": "1e400"},
             {**good, "ts": "１２.5"}, {**good, "produced_ms": "nan"}, {**good, "action": "like"},
             {**good, "event_id": ""}, {**good, "event_id": "a:b"}, {**good, "user_id": " 0_42 "},
             {**good, "user_id": "٤٢"}, {**good, "extra": "it's \"quoted\"\n", "action": "PLAY"},
             {"ts": "1", "event_id": "e", "action": "skip", "user_id": "1", "item_id": "2", "watch_ratio": "0",
              "produced_ms": "0"},
             {**good, "event_id": "日本​", "action": "nope"}]
    out = []
    for f in cases:
        fields = [[k, v] for k, v in f.items()]
        try:
            ev = parse_fields(f)
            out.append({"fields": fields, "event": {
                "event_id": ev["event_id"], "user_id": ev["user_id"], "item_id": ev["item_id"],
                "action": ev["action"], "watch_ratio": repr(ev["watch_ratio"]), "ts": repr(ev["ts"]),
                "cats": ev["cats"], "produced_ms": repr(ev["produced_ms"])}})
        except PoisonEvent as e:
            out.append({"fields": fields, "poison": str(e)})
    return {"cases": out}


def _runs(pred) -> list[tuple[int, int]]:
    out, start = [], None
    for c in range(0x80, 0x110000):
        if pred(chr(c)):
            start = c if start is None else start
        elif start is not None:
            out.append((start, c - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def unicode_tables_go() -> str:
    """CPython's own printable and decimal-digit tables as Go source. Go's
    unicode package is a newer Unicode version, so using it would quote ~5k newly assigned
    characters differently from CPython 3.12."""
    import unicodedata

    def table(name, runs):
        rows = "\n".join(f"\t{{0x{a:x}, 0x{b:x}}}," for a, b in runs)
        return f"var {name} = []runeRange{{\n{rows}\n}}\n"
    return ("// Code generated by scripts/gen_go_golden.py from CPython "
            f"{sys.version.split()[0]} (Unicode {unicodedata.unidata_version}). DO NOT EDIT.\n\n"
            "package pycompat\n\n"
            "// pyPrintable: non-ASCII code points where str.isprintable() is true.\n"
            + table("pyPrintable", _runs(str.isprintable))
            + "\n// pyDecimal: non-ASCII code points where str.isdecimal() is true; each run starts at digit 0.\n"
            + table("pyDecimal", _runs(str.isdecimal)))


def worker_golden() -> dict:
    return {"apply_event_lua": APPLY_EVENT_LUA,
            "lat_buckets_ms": [b if b != float("inf") else "inf" for b in LAT_BUCKETS_MS]}


FILES = {"hashing_golden.json": hashing_golden, "pycompat_golden.json": pycompat_golden,
         "events_golden.json": events_golden, "worker_golden.json": worker_golden}


def render(fn) -> str:
    return json.dumps(fn(), ensure_ascii=True, separators=(",", ":")) + "\n"


GO_SOURCES = {OUT.parent / "internal" / "pycompat" / "unicode_py312.go": unicode_tables_go}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FILES.items():
        text = render(fn)
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(OUT.parents[2])} ({len(text) // 1024} KiB)")
    for path, fn in GO_SOURCES.items():
        text = fn()
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(OUT.parents[2])}")
    if stale:
        print(f"stale golden files (run scripts/gen_go_golden.py): {stale}")
        sys.exit(1)


if __name__ == "__main__":
    main()
