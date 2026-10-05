import math

import numpy as np
import pytest
import torch

from src.neuron_model.spine_generation.models.spine_mogen.flow import (
    cosine_schedule,
    flow_matching_loss,
    sample_midpoint,
    t_schedule,
)
from src.neuron_model.spine_generation.models.spine_mogen.pointinfinity import (
    PointInfinity,
    PointInfinityConfig,
    knn_features,
    knn_indices,
    timestep_embedding,
)
from src.neuron_model.spine_generation.models.spine_mogen.weights import (
    flax_to_state_dict,
    infer_config,
    load_pointinfinity_from_flax,
    state_dict_to_flax,
)

SMALL = PointInfinityConfig(point_dim=16, latent_dim=32, n_latents=8, n_blocks=2, n_subblocks=2, n_heads=4, k_nn=4, cond_dim=6)


def randomize(model, seed=0):
    """Replace the zero-inits so every parameter matters in the tests."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return model


def test_forward_shapes_and_permutation_equivariance():
    torch.manual_seed(0)
    model = randomize(PointInfinity(SMALL)).eval()
    x = torch.randn(2, 64, 3)
    t = torch.tensor([0.1, 0.9])
    out = model(x, t)
    assert out.shape == (2, 64, 3)
    perm = torch.randperm(64)
    # per-point output must follow a permutation of the input points (set model)
    assert torch.allclose(model(x[:, perm], t), out[:, perm], atol=1e-5)


def test_zero_cond_equals_no_cond_and_cond_changes_output():
    model = randomize(PointInfinity(SMALL)).eval()
    x, t = torch.randn(1, 32, 3), torch.tensor([0.5])
    assert torch.equal(model(x, t), model(x, t, torch.zeros(1, 6)))
    assert not torch.allclose(model(x, t), model(x, t, torch.ones(1, 6)))


def test_fresh_model_outputs_bias_only():
    # zero-initialised attention out-projections and last MLP layers -> residual stream = tokenizer output
    model = PointInfinity(SMALL).eval()
    x = torch.randn(1, 32, 3)
    with torch.no_grad():
        expected = model.Dense_2(model.Dense_1(torch.cat([x, knn_features(x, SMALL.k_nn)], dim=-1)))
    assert torch.allclose(model(x, torch.tensor([0.3])), expected, atol=1e-6)


def test_knn_includes_self_up_to_8192_and_excludes_above():
    x = torch.randn(1, 50, 3)
    idx = knn_indices(x, 4)
    assert torch.equal(idx[0, :, 0], torch.arange(50))  # nearest = itself (distance 0)
    feats = knn_features(x, 4)
    assert torch.allclose(feats[0, :, :3], torch.zeros(50, 3), atol=1e-5)  # first offset = 0
    big = torch.randn(1, 8193, 3)
    assert not (knn_indices(big, 2)[0, :, 0] == torch.arange(8193)).any()


def test_knn_matches_direct_distance_for_near_degenerate_points():
    """Regression for the expand-of-squares distance formula (sq_a+sq_b-2ab): it cancels two
    large, nearly-equal terms down to a tiny true distance and picked the wrong neighbour for
    points that are almost coincident - caught by check_mogen_parity.py against the official
    JAX model (same formula as connectomics.jax.spatial.squared_distances)."""
    torch.manual_seed(0)
    base = torch.randn(1, 30, 3)
    coord = torch.cat([base, base[:, :5] + 1e-4 * torch.randn(1, 5, 3)], dim=1)  # near-duplicate points
    direct = ((coord[:, :, None, :] - coord[:, None, :, :]) ** 2).sum(-1)
    expand = coord.pow(2).sum(-1)[:, :, None] + coord.pow(2).sum(-1)[:, None, :] - 2.0 * coord @ coord.transpose(1, 2)
    assert not torch.equal(torch.topk(-expand, 5, dim=-1).indices, torch.topk(-direct, 5, dim=-1).indices)
    assert torch.equal(knn_indices(coord, 4), torch.topk(-direct, 4, dim=-1).indices)  # coord has <=8192 points -> self included, no exclusion shift


def test_timestep_embedding_matches_formula():
    t = torch.tensor([0.0, 0.25, 1.0])
    emb = timestep_embedding(t, 8)
    half = 4
    freq = np.exp(np.arange(half) * -(math.log(10000.0) / (half - 1)))
    args = np.asarray(t)[:, None] * 1000.0 * freq[None]
    assert np.allclose(emb.numpy(), np.concatenate([np.sin(args), np.cos(args)], axis=1), atol=1e-5)


def test_cosine_schedule_matches_reference_formula():
    t = torch.linspace(0, 1, 11)
    ref = ((np.abs(np.abs(np.cos(t.numpy() * np.pi)) - 1) ** 2.0) - 1) * ((t.numpy() < 0.5) - 0.5) + 0.5
    assert np.allclose(t_schedule(t, "cosine_2.0").numpy(), ref, atol=1e-6)
    assert torch.equal(t_schedule(t, "mogen_cosine"), t_schedule(t, "cosine_2.0"))
    s = cosine_schedule(t, 2.0)
    assert s[0] == 0 and s[-1] == 1 and torch.all(s[1:] >= s[:-1])  # monotone, endpoints fixed
    with pytest.raises(ValueError):
        t_schedule(t, "bogus")


def test_midpoint_solver_is_exact_for_constant_velocity_and_second_order():
    shift = torch.tensor([1.0, -2.0, 0.5])
    x0 = torch.randn(2, 10, 3)
    x1 = sample_midpoint(lambda x, t: shift.expand_as(x), x0, n_steps=7, schedule="cosine_2.0")
    assert torch.allclose(x1, x0 + shift, atol=1e-5)
    # dx/dt = x -> x(1) = e * x0; midpoint error should shrink ~4x when steps double
    err = lambda n: (sample_midpoint(lambda x, t: x, x0, n_steps=n, schedule="linear") - math.e * x0).abs().max().item()
    assert err(20) < err(10) / 3.5


def test_flow_matching_loss_is_zero_for_the_true_velocity():
    torch.manual_seed(0)
    x1 = torch.randn(3, 16, 3)
    captured = {}

    def oracle(x_t, t):
        # recover x0 from x_t and x1: x_t = (1-t) x0 + t x1  ->  v = x1 - x0 = (x1 - x_t) / (1 - t)
        te = t.view(-1, 1, 1)
        return (x1 - x_t) / (1 - te)

    g = torch.Generator().manual_seed(1)
    loss = flow_matching_loss(oracle, x1, schedule="linear", generator=g)
    assert loss.item() < 1e-6


def test_weight_conversion_roundtrip_and_config_inference():
    model = randomize(PointInfinity(SMALL)).eval()
    flax = state_dict_to_flax(model)
    # Flax-side layouts
    assert flax["PointInfinityBlock_0/MultiHeadDotProductAttention_0/query/kernel"].shape == (32, 4, 8)
    assert flax["PointInfinityBlock_0/MultiHeadDotProductAttention_0/key/kernel"].shape == (16, 4, 8)  # latents read points
    assert flax["PointInfinityBlock_0/MultiHeadDotProductAttention_3/out/kernel"].shape == (4, 4, 16)  # write: points query
    assert flax["Dense_1/kernel"].shape == (3 * (1 + 4), 16)
    assert infer_config(flax) == SMALL
    restored = load_pointinfinity_from_flax(flax).eval()
    x, t = torch.randn(2, 40, 3), torch.tensor([0.2, 0.7])
    assert torch.equal(restored(x, t), model(x, t))
    state = flax_to_state_dict(flax)
    for name, tensor in model.state_dict().items():
        assert torch.equal(state[name], tensor)


def test_conversion_rejects_mismatched_tree():
    flax = state_dict_to_flax(PointInfinity(SMALL))
    del flax["PointInfinityBlock_1/RMSNorm_5/scale"]
    with pytest.raises(KeyError):
        load_pointinfinity_from_flax(flax)


def test_mouse_mixed_sized_model_builds():
    model = PointInfinity(PointInfinityConfig(cond_dim=10))
    n_params = sum(p.numel() for p in model.parameters())
    assert 5e6 < n_params < 3e7  # the checkpoint is ~224 MB incl. EMA + optimizer state
