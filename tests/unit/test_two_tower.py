import numpy as np
import torch

from echostream.core.two_tower import TwoTower, export_user_tower, user_tower_numpy


def make_model():
    torch.manual_seed(0)
    return TwoTower(n_items=20, n_cats=5, caption_dim=6, user_field_vocab=[3, 4], dim=8)


def test_output_shapes_and_unit_norm():
    m = make_model()
    u = m.user_tower(torch.tensor([[1, 2], [0, 3]]), torch.tensor([[1, 2, 0], [3, 0, 0]]),
                     torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    v = m.item_tower(torch.tensor([1, 4]), torch.tensor([[1, 2], [3, 0]]), torch.randn(2, 6))
    assert u.shape == (2, 8) and v.shape == (2, 8)
    assert torch.allclose(u.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(v.norm(dim=1), torch.ones(2), atol=1e-5)


def test_padding_does_not_change_user_vector():
    m = make_model()
    cats = torch.tensor([[1, 2]])
    a = m.user_tower(cats, torch.tensor([[5, 6]]), torch.tensor([[1.0, 1.0]]))
    b = m.user_tower(cats, torch.tensor([[5, 6, 9, 9]]), torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    assert torch.allclose(a, b, atol=1e-6)


def test_numpy_export_matches_torch():
    m = make_model().eval()
    params = export_user_tower(m)
    with torch.no_grad():
        t = m.user_tower(torch.tensor([[2, 1]]), torch.tensor([[3, 7, 11]]), torch.ones(1, 3))[0].numpy()
    n = user_tower_numpy(params, np.array([2, 1]), [3, 7, 11])
    assert np.allclose(t, n, atol=1e-5)
    # empty history works too
    with torch.no_grad():
        t0 = m.user_tower(torch.tensor([[0, 0]]), torch.tensor([[0]]), torch.zeros(1, 1))[0].numpy()
    assert np.allclose(t0, user_tower_numpy(params, np.array([0, 0]), []), atol=1e-5)


def test_id_mask_zero_means_content_only():
    m = make_model().eval()
    cats, cap = torch.tensor([[1, 2], [1, 2]]), torch.ones(2, 6)
    with torch.no_grad():
        v = m.item_tower(torch.tensor([3, 9]), cats, cap, id_mask=torch.zeros(2))
    # different ids, same content, id masked -> identical vectors
    assert torch.allclose(v[0], v[1], atol=1e-6)
