"""Export the browser bundle (Stage 13) from the current serving bundle.

    uv run python scripts/export_browser_model.py                 # int8 preset -> web/demo-public/model/
    uv run python scripts/export_browser_model.py --quant f32     # exactly the serving bundle (~33 MB gzipped)

Presets (scripts/stage13_quant_sweep.py measured each one against float32 with the real Python read path):
    f32   float32 everywhere                                   = the serving bundle
    f16   half precision embeddings + ItemCF similarities
    int8  int8 embeddings (one scale per row), ItemCF similarities as 256 shared values   <- shipped
The output is deterministic (no timestamps; gzip mtime 0), so a re-export of the same artifacts
gives the same bytes. The bundle is committed: building the demo needs no data download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from echostream.browser.export import PRESETS, write_browser_bundle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "web" / "demo-public" / "model"
SERVING = ROOT / "artifacts" / "serving" / "current"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=sorted(PRESETS), default="int8")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--serving", type=Path, default=SERVING)
    args = ap.parse_args()
    m = write_browser_bundle(args.out, args.serving, PRESETS[args.quant])
    total = sum(f["bytes"] for f in m["files"].values()) + (args.out / "manifest.json").stat().st_size
    users = json.loads((args.out / "users.json").read_text())["users"]
    print(f"wrote {args.out} ({args.quant}): {total / 1e6:.2f} MB on disk; model.bin {m['model_bytes'] / 1e6:.2f} MB "
          f"before gzip; demo users {[(u['uid'], u['label']) for u in users]}")


if __name__ == "__main__":
    main()
