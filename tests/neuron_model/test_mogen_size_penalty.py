"""Distributional size penalty of the MoGen fine-tuning (flow.size_penalty_loss, train.make_loss_fn with flow.size_penalty)."""

import torch

from src.neuron_model.spine_generation.models.spine_mogen.flow import log_size_stats, rollout_to_one, size_penalty_loss


class _Scale(torch.nn.Module):
    """dx/dt = k x: the end point of a rollout from t0 is x_t exp(k (1 - t0)), so k sets the size of the end points."""

    def __init__(self):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(0.8))

    def forward(self, x, t, cond=None):
        return self.k * x


def test_log_size_stats_and_rollout():
    x = torch.randn(3, 500, 3) * torch.tensor([2.0, 5.0, 2.0])
    s = log_size_stats(x)
    assert s.shape == (3, 2) and (s[:, 1] > s[:, 0]).all()  # axial spread larger than radial here
    m = _Scale()
    end = rollout_to_one(m, x, torch.zeros(3), n_steps=20)
    assert torch.allclose(log_size_stats(end) - s, torch.full((3, 2), 0.8), atol=0.02)


def test_penalty_drives_the_size_gap_to_zero():
    torch.manual_seed(0)
    x1 = torch.randn(8, 400, 3) * 0.3
    m = _Scale()
    opt = torch.optim.SGD(m.parameters(), lr=0.2)
    gen = torch.Generator().manual_seed(1)
    first = size_penalty_loss(m, x1, n_sub=8, t_min=0.3, t_max=0.9, n_steps=4, generator=gen)[1]["size_gap_radial"]
    for _ in range(200):
        loss, info = size_penalty_loss(m, x1, n_sub=8, t_min=0.3, t_max=0.9, n_steps=4)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = size_penalty_loss(m, x1, n_sub=8, t_min=0.3, t_max=0.9, n_steps=4, generator=torch.Generator().manual_seed(1))[1]["size_gap_radial"]
    assert first > 0.2 and abs(last) < 0.15 * first  # stochastic start points: a small residual


def test_loss_fn_with_the_penalty_on_a_tiny_pointinfinity(fake_dataset):
    from src.neuron_model.spine_generation.data.datasets import make_loader
    from src.neuron_model.spine_generation.data.index import load_spine_index
    from src.neuron_model.spine_generation.data.splits import build_group_split
    from src.neuron_model.spine_generation.models.spine_mogen.train import build_model, make_loss_fn, make_validate_fn, point_cloud_datasets

    tiny = {"point_dim": 16, "latent_dim": 32, "n_latents": 8, "n_blocks": 1, "n_subblocks": 1, "n_heads": 4, "k_nn": 4, "coord_dim": 3, "cond_dim": 6}
    data = {"n_points": 2048, "coord_scale": 1e-2, "jitter_std": 1.0, "batch_size": 4}
    flow = {"time_schedule": "mogen_cosine", "size_penalty": {"weight": 1.0, "every": 1, "n_sub": 2, "n_steps": 2},
            "size_monitor": {"n_batches": 1, "n_steps": 2}}
    index = load_spine_index([fake_dataset / "manifest.parquet"]).with_split(build_group_split(load_spine_index([fake_dataset / "manifest.parquet"]).frame, seed=1))
    train_ds, val_ds = point_cloud_datasets(index, data)
    model = build_model({"init": "random", "architecture": tiny})
    batch = next(iter(make_loader(train_ds, batch_size=4, shuffle=True, seed=0)))
    loss, metrics = make_loss_fn(flow, torch.device("cpu"))(model, batch, 0)
    loss.backward()
    assert {"fm_loss", "size_gap_radial", "size_gap_axial", "size_penalty", "loss"} <= set(metrics)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    off = make_loss_fn({"time_schedule": "mogen_cosine"}, torch.device("cpu"))(model, batch, 0)[1]
    assert "size_penalty" not in off and off["loss"] == off["fm_loss"]
    val = make_validate_fn(val_ds, flow, torch.device("cpu"), batch_size=2, seed=0, max_batches=1)(model, 0)
    assert "val_size_gap_radial" in val
