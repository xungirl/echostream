"""Two-tower retrieval model.

user tower:  [user categorical features] + mean(item-id embeddings of recent
             positive history)  -> MLP -> L2-normalized vector u
item tower:  item-id embedding + mean(category embeddings) + projected caption
             embedding            -> MLP -> L2-normalized vector v
score(u, v) = <u, v>  (so retrieval = maximum inner product search in faiss)

The towers never see each other's inputs, which is exactly what lets us
precompute every item vector offline and only run the user tower per request.
`user_tower_numpy` re-implements the user tower forward pass in numpy so the
serving layer does not need PyTorch; a unit test checks both agree.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, field_vocab_sizes: list[int], item_id_emb: nn.Embedding, dim: int = 64, field_dim: int = 8):
        super().__init__()
        self.fields = nn.ModuleList([nn.Embedding(v, field_dim) for v in field_vocab_sizes])
        self.item_id_emb = item_id_emb  # shared with the item tower
        in_dim = field_dim * len(field_vocab_sizes) + item_id_emb.embedding_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, user_cats: torch.Tensor, hist: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        """user_cats [B, F] int; hist [B, L] item ids (padded); hist_mask [B, L] 1 = real item."""
        field_vecs = [emb(user_cats[:, f]) for f, emb in enumerate(self.fields)]
        h = self.item_id_emb(hist) * hist_mask.unsqueeze(-1)
        pooled = h.sum(1) / hist_mask.sum(1, keepdim=True).clamp(min=1.0)
        x = torch.cat(field_vecs + [pooled], dim=1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, n_items: int, n_cats: int, caption_dim: int, dim: int = 64, id_dim: int = 32):
        super().__init__()
        self.item_id_emb = nn.Embedding(n_items, id_dim)
        self.cat_emb = nn.Embedding(n_cats + 1, 16, padding_idx=0)  # 0 = padding
        self.caption_proj = nn.Linear(caption_dim, 32)
        self.mlp = nn.Sequential(nn.Linear(id_dim + 16 + 32, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, item_ids: torch.Tensor, cats: torch.Tensor, caption: torch.Tensor) -> torch.Tensor:
        """item_ids [B]; cats [B, C] category ids + 1 (0 = pad); caption [B, caption_dim]."""
        c = self.cat_emb(cats)
        c_mask = (cats > 0).float().unsqueeze(-1)
        c_pooled = (c * c_mask).sum(1) / c_mask.sum(1).clamp(min=1.0)
        x = torch.cat([self.item_id_emb(item_ids), c_pooled, self.caption_proj(caption)], dim=1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_items: int, n_cats: int, caption_dim: int, user_field_vocab: list[int], dim: int = 64):
        super().__init__()
        self.item_tower = ItemTower(n_items, n_cats, caption_dim, dim)
        self.user_tower = UserTower(user_field_vocab, self.item_tower.item_id_emb, dim)


def export_user_tower(model: TwoTower) -> dict[str, np.ndarray]:
    """Flatten the user tower weights into plain numpy arrays for serving."""
    ut = model.user_tower
    params = {f"field_{f}": emb.weight.detach().cpu().numpy() for f, emb in enumerate(ut.fields)}
    params["item_id_emb"] = ut.item_id_emb.weight.detach().cpu().numpy()
    params["w1"] = ut.mlp[0].weight.detach().cpu().numpy()
    params["b1"] = ut.mlp[0].bias.detach().cpu().numpy()
    params["w2"] = ut.mlp[2].weight.detach().cpu().numpy()
    params["b2"] = ut.mlp[2].bias.detach().cpu().numpy()
    return params


def user_tower_numpy(params: dict[str, np.ndarray], user_cats: np.ndarray, hist: list[int]) -> np.ndarray:
    """Single-user forward pass in numpy. user_cats [F] ints, hist = item ids."""
    n_fields = sum(1 for k in params if k.startswith("field_"))
    field_vecs = [params[f"field_{f}"][user_cats[f]] for f in range(n_fields)]
    if hist:
        pooled = params["item_id_emb"][np.asarray(hist)].mean(axis=0)
    else:
        pooled = np.zeros(params["item_id_emb"].shape[1], dtype=np.float32)
    x = np.concatenate(field_vecs + [pooled]).astype(np.float32)
    h = np.maximum(params["w1"] @ x + params["b1"], 0.0)
    out = params["w2"] @ h + params["b2"]
    return (out / max(np.linalg.norm(out), 1e-12)).astype(np.float32)
