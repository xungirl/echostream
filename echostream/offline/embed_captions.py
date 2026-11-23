"""Embed every item's caption text with a small local Chinese sentence model
(BAAI/bge-small-zh-v1.5, 512-d). Model weights are cached in .cache/hf (gitignored).

    uv run python -m echostream.offline.embed_captions  -> artifacts/caption_emb.npy
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

import numpy as np  # noqa: E402

from echostream.domain.kuairec import KuaiRecAdapter  # noqa: E402

MODEL = "BAAI/bge-small-zh-v1.5"
OUT = ROOT / "artifacts" / "caption_emb.npy"


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    a = KuaiRecAdapter()
    texts = a.item_text()
    corpus = [texts[i] for i in range(a.n_items)]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    t = time.time()
    emb = model.encode(corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, emb.astype(np.float32))
    print(f"embedded {len(corpus)} captions -> {emb.shape} on {device} in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
