"""In-batch sampled softmax with logQ correction (Yi et al., RecSys 2019).

For a batch of B (user, positive item) pairs, each user's positive is the
"correct class" and the other B-1 items in the batch serve as negatives.
Because popular items appear in batches more often, they are punished as
negatives more often than they should be. logQ correction subtracts log(q_j),
q_j = probability item j shows up in a batch, from every logit so the softmax
estimates the full-catalog softmax without that popularity bias.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def in_batch_softmax_loss(
    user_emb: torch.Tensor,           # [B, d], L2-normalized
    item_emb: torch.Tensor,           # [B, d], L2-normalized positives
    item_ids: torch.Tensor,           # [B] ids of the positives (to mask duplicates)
    temperature: float = 0.1,
    item_logq: torch.Tensor | None = None,   # [B] log sampling prob of each positive
    extra_neg_emb: torch.Tensor | None = None,  # [M, d] uniformly sampled random negatives
    extra_neg_ids: torch.Tensor | None = None,  # [M]
    extra_neg_logq: torch.Tensor | None = None, # [M] log prob under the uniform sampler
) -> torch.Tensor:
    logits = user_emb @ item_emb.T / temperature  # [B, B]
    if item_logq is not None:
        logits = logits - item_logq[None, :]
    # The same item can be the positive of two users in one batch; it must not
    # count as a negative for the other user.
    same_item = item_ids[None, :] == item_ids[:, None]
    eye = torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_item & ~eye, float("-inf"))

    if extra_neg_emb is not None:
        neg_logits = user_emb @ extra_neg_emb.T / temperature  # [B, M]
        if extra_neg_logq is not None:
            neg_logits = neg_logits - extra_neg_logq[None, :]
        # a random negative that equals the user's positive is not a negative
        neg_logits = neg_logits.masked_fill(extra_neg_ids[None, :] == item_ids[:, None], float("-inf"))
        logits = torch.cat([logits, neg_logits], dim=1)

    labels = torch.arange(len(item_ids), device=logits.device)
    return F.cross_entropy(logits, labels)
