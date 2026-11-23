import math

import torch

from echostream.core.losses import in_batch_softmax_loss


def test_matches_hand_computed_cross_entropy():
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ids = torch.tensor([10, 11])
    loss = in_batch_softmax_loss(u, v, ids, temperature=1.0)
    # each row: logits [1, 0] with label 0 -> -log(e / (e + 1))
    expected = -math.log(math.e / (math.e + 1))
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_logq_correction_subtracts_log_prob():
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ids = torch.tensor([10, 11])
    logq = torch.log(torch.tensor([0.5, 0.25]))
    loss = in_batch_softmax_loss(u, v, ids, temperature=1.0, item_logq=logq)
    # row 0 logits: [1 - log .5, 0 - log .25]; row 1: [0 - log .5, 1 - log .25]
    def ce(logits, label):
        return -torch.log_softmax(torch.tensor(logits), 0)[label]
    expected = (ce([1 - math.log(.5), -math.log(.25)], 0) + ce([-math.log(.5), 1 - math.log(.25)], 1)) / 2
    assert torch.isclose(loss, expected, atol=1e-6)


def test_duplicate_positive_is_not_a_negative():
    u = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    ids_dup = torch.tensor([7, 7, 8])
    loss = in_batch_softmax_loss(u, v, ids_dup, temperature=0.5)
    # manual: row0 cannot see col1 and row1 cannot see col0
    logits = u @ v.T / 0.5
    logits[0, 1] = float("-inf")
    logits[1, 0] = float("-inf")
    expected = torch.nn.functional.cross_entropy(logits, torch.arange(3))
    assert torch.isclose(loss, expected, atol=1e-6)


def test_extra_random_negatives_increase_loss_and_mask_collisions():
    u = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[1.0, 0.0]])
    ids = torch.tensor([1])
    base = in_batch_softmax_loss(u, v, ids, temperature=1.0)
    assert base.item() == 0.0  # only one class
    neg = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss = in_batch_softmax_loss(u, v, ids, temperature=1.0, extra_neg_emb=neg, extra_neg_ids=torch.tensor([2, 1]))
    # neg id 1 == positive id -> masked; remaining logits [1, 0]
    assert torch.isclose(loss, torch.tensor(-math.log(math.e / (math.e + 1))), atol=1e-6)
    assert loss > base
