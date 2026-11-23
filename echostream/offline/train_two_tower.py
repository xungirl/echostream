"""Train the two-tower retrieval model (PyTorch, Apple MPS when available).

One training example = (user features, user's previous <=L positive items,
next positive item). Loss = in-batch sampled softmax, optionally with logQ
correction and B extra uniformly-sampled random negatives ("mixed negatives").

This module imports torch and must not import faiss (see echostream/__init__.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from echostream.core.losses import in_batch_softmax_loss
from echostream.core.two_tower import TwoTower, export_user_tower
from echostream.domain.base import DomainAdapter
from echostream.offline.phase import ARTIFACTS, Phase, history_frame

HIST_LEN = 50
MAX_CATS = 4


@dataclass
class TTConfig:
    temperature: float = 0.1
    logq: bool = True
    mixed_negatives: int = 1024     # number of extra uniform negatives per batch (0 = off)
    epochs: int = 4
    batch_size: int = 2048
    lr: float = 2e-3
    dim: int = 64
    id_dropout: float = 0.3
    seed: int = 0
    # Stage-9 fault injection hooks (default off): used to create *real* regressions
    caption_noise: float = 0.0


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def item_feature_tensors(adapter: DomainAdapter, caption_emb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    cats = np.zeros((adapter.n_items, MAX_CATS), dtype=np.int64)
    for item, cs in adapter.item_categories().items():
        for j, c in enumerate(sorted(cs)[:MAX_CATS]):
            cats[item, j] = c + 1  # 0 is padding
    return torch.from_numpy(cats), torch.from_numpy(caption_emb.astype(np.float32))


def build_examples(adapter: DomainAdapter, phase: Phase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(users [N], targets [N], hist [N, HIST_LEN] padded with -1), in time order per user."""
    pos = history_frame(adapter, phase.name)
    users, targets, hists = [], [], []
    for u, seq in pos.groupby("user_id").item_id:
        seq = seq.to_numpy(np.int64)
        for j in range(1, len(seq)):
            h = seq[max(0, j - HIST_LEN):j]
            users.append(u)
            targets.append(seq[j])
            hists.append(np.pad(h, (HIST_LEN - len(h), 0), constant_values=-1))
    return np.array(users), np.array(targets), np.stack(hists)


def pad_history(hist: np.ndarray) -> np.ndarray:
    h = hist[-HIST_LEN:]
    return np.pad(h, (HIST_LEN - len(h), 0), constant_values=-1)


def train(adapter: DomainAdapter, phase: Phase, cfg: TTConfig, caption_emb: np.ndarray, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = device()
    if cfg.caption_noise > 0:
        caption_emb = caption_emb + rng.normal(0, cfg.caption_noise, caption_emb.shape).astype(np.float32)

    users, targets, hists = build_examples(adapter, phase)
    n_items = adapter.n_items
    user_cats_np, vocab = adapter.user_categorical_features()
    user_cats = torch.from_numpy(user_cats_np).to(dev)
    item_cats, item_caps = (t.to(dev) for t in item_feature_tensors(adapter, caption_emb))

    counts = np.bincount(targets, minlength=n_items).astype(np.float64)
    q = counts / counts.sum()
    B, M = cfg.batch_size, cfg.mixed_negatives
    # expected number of times item j appears among a batch's negatives:
    # B*q_j from the in-batch positives + M/N from the uniform sampler
    logq_all = torch.tensor(np.log(B * q + M / n_items + 1e-12), dtype=torch.float32, device=dev)

    model = TwoTower(n_items, adapter.n_categories, caption_emb.shape[1], vocab, cfg.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    hists_t = torch.from_numpy(hists)
    t0 = time.time()
    losses = []
    for epoch in range(cfg.epochs):
        perm = rng.permutation(len(users))
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(perm) - B + 1, B):
            idx = perm[start:start + B]
            u = torch.from_numpy(users[idx]).to(dev)
            tgt = torch.from_numpy(targets[idx]).to(dev)
            h = hists_t[idx].to(dev)
            h_mask = (h >= 0).float()
            uvec = model.user_tower(user_cats[u], h.clamp(min=0), h_mask)
            id_mask = (torch.rand(B, device=dev) > cfg.id_dropout).float()
            ivec = model.item_tower(tgt, item_cats[tgt], item_caps[tgt], id_mask)
            kw = {}
            if M > 0:
                neg = torch.from_numpy(rng.integers(0, n_items, M)).to(dev)
                nmask = (torch.rand(M, device=dev) > cfg.id_dropout).float()
                kw = dict(extra_neg_emb=model.item_tower(neg, item_cats[neg], item_caps[neg], nmask),
                          extra_neg_ids=neg, extra_neg_logq=logq_all[neg] if cfg.logq else None)
            loss = in_batch_softmax_loss(uvec, ivec, tgt, cfg.temperature,
                                         item_logq=logq_all[tgt] if cfg.logq else None, **kw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
        log(f"  epoch {epoch} loss {losses[-1]:.4f} ({time.time() - t0:.0f}s)")

    model.eval()
    seen_in_training = counts > 0
    with torch.no_grad():
        all_items = torch.arange(n_items, device=dev)
        id_mask = torch.from_numpy(seen_in_training.astype(np.float32)).to(dev)
        item_vecs = model.item_tower(all_items, item_cats, item_caps, id_mask).cpu().numpy()
    return {"model": model.cpu(), "item_vecs": item_vecs, "losses": losses, "train_sec": time.time() - t0,
            "n_examples": int(len(users)), "seen_in_training": seen_in_training,
            "user_cats": user_cats_np}


def user_vectors(result: dict, phase: Phase, users: np.ndarray, batch: int = 4096) -> np.ndarray:
    """User vectors from each user's latest HIST_LEN history positives (as of the phase cutoff)."""
    model = result["model"]
    cats = torch.from_numpy(result["user_cats"])
    out = []
    with torch.no_grad():
        for s in range(0, len(users), batch):
            ub = users[s:s + batch]
            h = torch.from_numpy(np.stack([pad_history(phase.pos_history[u]) for u in ub]))
            out.append(model.user_tower(cats[ub], h.clamp(min=0), (h >= 0).float()).numpy())
    return np.concatenate(out)


def save(result: dict, cfg: TTConfig, tag: str) -> Path:
    d = ARTIFACTS / "two_tower" / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), d / "model.pt")
    np.save(d / "item_emb.npy", result["item_vecs"])
    np.savez(d / "user_tower.npz", **export_user_tower(result["model"]))
    np.save(d / "user_cats.npy", result["user_cats"])
    (d / "config.json").write_text(json.dumps({**asdict(cfg), "losses": result["losses"],
                                               "train_sec": round(result["train_sec"], 1),
                                               "n_examples": result["n_examples"]}, indent=2))
    return d


def main() -> None:
    """Refit the selected config (artifacts/two_tower/main/config.json) on a phase:
    python -m echostream.offline.train_two_tower --phase rank --tag rank_main"""
    import argparse

    from echostream.domain.kuairec import KuaiRecAdapter
    from echostream.offline.phase import load_phase

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (fault injection)")
    args = ap.parse_args()
    cfg_d = json.loads((ARTIFACTS / "two_tower" / "main" / "config.json").read_text())
    cfg_d = {k: v for k, v in cfg_d.items() if k in TTConfig.__dataclass_fields__}
    for kv in args.set:
        k, v = kv.split("=")
        cfg_d[k] = type(cfg_d[k])(v) if not isinstance(cfg_d[k], bool) else v.lower() == "true"
    cfg = TTConfig(**cfg_d)
    a = KuaiRecAdapter()
    phase = load_phase(a, args.phase)
    r = train(a, phase, cfg, np.load(ARTIFACTS / "caption_emb.npy"))
    d = save(r, cfg, args.tag)
    if args.phase == "test":
        np.save(d / "user_vecs_test.npy", user_vectors(r, phase, np.arange(a.n_users)))
    print("saved", d)


if __name__ == "__main__":
    main()
