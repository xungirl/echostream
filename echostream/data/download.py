"""`make data`: download KuaiRec, verify checksums, unzip, convert CSV -> parquet.

The data is CC BY-SA 4.0 (Gao et al., CIKM 2022). We never commit or
redistribute it; this script re-creates data/ from the official Zenodo record.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
CHECKSUMS = ROOT / "data_checksums.json"
CSV_DIR = RAW / "KuaiRec 2.0" / "data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "KuaiRec.zip"
    if not zpath.exists():
        print(f"downloading {URL} (~432 MB)")
        subprocess.run(["curl", "-L", "--fail", "-o", str(zpath), URL], check=True)
    return zpath


def verify(zpath: Path, write: bool = False) -> None:
    actual = {"KuaiRec.zip": sha256(zpath)}
    for csv in sorted(CSV_DIR.glob("*.csv")):
        actual[csv.name] = sha256(csv)
    if write or not CHECKSUMS.exists():
        CHECKSUMS.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"wrote {CHECKSUMS}")
        return
    expected = json.loads(CHECKSUMS.read_text())
    bad = [k for k, v in expected.items() if actual.get(k) != v]
    if bad:
        sys.exit(f"checksum mismatch: {bad}")
    print(f"checksums OK ({len(expected)} files)")


def to_parquet() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in ("big_matrix", "small_matrix"):
        df = pd.read_csv(CSV_DIR / f"{name}.csv")
        out = pd.DataFrame({
            "user_id": df.user_id.astype("int32"),
            "item_id": df.video_id.astype("int32"),
            "ts": df.timestamp.astype("float64"),
            "date": df.date.fillna(-1).astype("int32"),
            "play_duration": df.play_duration.astype("int32"),
            "video_duration": df.video_duration.astype("int32"),
            "watch_ratio": df.watch_ratio.astype("float32"),
        })
        out.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name}: {len(out):,} rows")
    for name in ("item_categories", "kuairec_caption_category", "user_features",
                 "item_daily_features", "social_network"):
        df = pd.read_csv(CSV_DIR / f"{name}.csv", low_memory=False)
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} rows")


def main() -> None:
    zpath = download()
    if not CSV_DIR.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW)
    verify(zpath, write="--write-checksums" in sys.argv)
    to_parquet()


if __name__ == "__main__":
    main()
