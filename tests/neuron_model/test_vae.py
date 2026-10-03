import json

import numpy as np
import pytest
import torch

from src.neuron_model.spine_generation.data.bbox import compute_frozen_bbox
from src.neuron_model.spine_generation.data.datasets import make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import build_group_split
from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
from src.neuron_model.spine_generation.experiment.run import create_run
from src.neuron_model.spine_generation.experiment.seed import seed_everything
from src.neuron_model.spine_generation.models.spine_vae.generate import latent_to_mesh
from src.neuron_model.spine_generation.models.spine_vae.latent import GaussianLatent, kl_loss, kl_weight, latent_statistics
from src.neuron_model.spine_generation.models.spine_vae.losses import LossWeights, eikonal_loss, normal_loss, sdf_gradient
from src.neuron_model.spine_generation.models.spine_vae.model import SpineVAE, SpineVAEConfig
from src.neuron_model.spine_generation.models.spine_vae.pointnext_encoder import PointNeXtConfig, PointNeXtEncoder, farthest_point_sample
from src.neuron_model.spine_generation.models.spine_vae.siren_sdf_decoder import SirenConfig, SirenSDFDecoder
from src.neuron_model.spine_generation.models.spine_vae.train import make_loss_fn, make_sample_fn, make_validate_fn, sdf_datasets

TINY_ENC = PointNeXtConfig(input_points=256, base_width=8, stage_widths=(8, 16, 32), strides=(1, 4, 4), blocks=(0, 1, 1), k_neighbors=8, global_feature_dim=32)
TINY = SpineVAEConfig(latent_dim=8, encoder=TINY_ENC, decoder=SirenConfig(hidden_dim=32, hidden_layers=3))


def test_encoder_shape_and_permutation_invariance():
    torch.manual_seed(0)
    cfg = PointNeXtConfig(input_points=128, base_width=8, stage_widths=(8, 16), strides=(1, 1), blocks=(1, 1), k_neighbors=8, global_feature_dim=16, pooling="max_avg")
    encoder = PointNeXtEncoder(cfg).eval()
    xyz, normals = torch.randn(2, 128, 3), torch.randn(2, 128, 3)
    h = encoder(xyz, normals)
    perm = torch.randperm(128)
    assert h.shape == (2, 16)
    assert torch.allclose(encoder(xyz[:, perm], normals[:, perm]), h, atol=1e-5)
    with pytest.raises(ValueError):
        encoder(xyz)  # use_normals=True needs normals
    no_normals = PointNeXtEncoder(PointNeXtConfig(**{**cfg.__dict__, "use_normals": False}))
    assert no_normals(xyz).shape == (2, 16)


def test_farthest_point_sampling_spreads_points():
    xyz = torch.cat([torch.zeros(1, 50, 3), torch.ones(1, 50, 3) * 10], dim=1)  # two far clusters
    idx = farthest_point_sample(xyz, 2, generator=torch.Generator().manual_seed(0))
    picked = xyz[0, idx[0]]
    assert torch.linalg.norm(picked[0] - picked[1]) > 10  # one point per cluster


def test_siren_initialisation_keeps_activation_scale():
    torch.manual_seed(0)
    decoder = SirenSDFDecoder(latent_dim=0, cfg=SirenConfig(hidden_dim=256, hidden_layers=5))
    x = torch.rand(1, 4096, 3) * 2 - 1
    h = x
    stds = []
    with torch.no_grad():
        h = torch.cat([h, torch.zeros(1, 4096, 0)], dim=-1)
        for layer in decoder.net:
            h = layer(h)
            stds.append(float(h.std()))
    # sine of a well-initialised pre-activation ~ arcsine distribution, std ~ 0.707 at every depth
    assert all(0.5 < s < 0.85 for s in stds[1:]), stds


def test_decoder_conditional_input_reserved():
    decoder = SirenSDFDecoder(latent_dim=4, cfg=SirenConfig(hidden_dim=16, hidden_layers=2, cond_dim=3))
    q, z = torch.randn(2, 10, 3), torch.randn(2, 4)
    assert torch.equal(decoder(q, z), decoder(q, z, torch.zeros(2, 3)))
    with pytest.raises(ValueError):
        SirenSDFDecoder(latent_dim=4, cfg=SirenConfig(hidden_dim=16, hidden_layers=2))(q, z, torch.zeros(2, 3))


def test_kl_free_bits_warmup_and_latent_stats():
    mu, logvar = torch.zeros(4, 6), torch.zeros(4, 6)
    assert float(kl_loss(mu, logvar)) == 0.0
    assert float(kl_loss(mu, logvar, free_bits=0.1)) == pytest.approx(0.6)  # every dim clamped to 0.1 nats
    assert float(kl_loss(torch.ones(4, 6), logvar)) == pytest.approx(3.0)  # 0.5 * mu^2 per dim
    assert kl_weight(0, 1e-3, 100) == 0.0 and kl_weight(50, 1e-3, 100) == pytest.approx(5e-4) and kl_weight(500, 1e-3, 100) == 1e-3
    stats = latent_statistics(torch.tensor([[0.0, 1.0], [0.0, -1.0]]), torch.zeros(2, 2))
    assert stats["active_latent_dimensions"] == 1 and stats["mean_std_z"] == pytest.approx(1.0)
    z, m, lv = GaussianLatent(5, 3)(torch.randn(2, 5), sample=False)
    assert torch.equal(z, m)


def test_eikonal_and_normal_losses_vanish_for_exact_sphere_sdf():
    q = torch.randn(1, 500, 3).requires_grad_(True)
    f = q.norm(dim=-1) - 1.0
    grad = sdf_gradient(f, q)
    assert float(eikonal_loss(grad).detach()) < 1e-10
    normals = (q / q.norm(dim=-1, keepdim=True)).detach()
    sample_type = torch.zeros(1, 500, dtype=torch.long)
    assert float(normal_loss(grad, normals, torch.ones(1, 500, dtype=torch.bool), sample_type).detach()) < 1e-6


def _sphere_batch(n_points=256, n_query=512, radius=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    points = torch.randn(1, n_points, 3, generator=g)
    points = points / points.norm(dim=-1, keepdim=True) * radius
    query = (torch.rand(1, n_query, 3, generator=g) * 2 - 1) * 0.9
    surface = torch.randn(1, 128, 3, generator=g)
    surface = surface / surface.norm(dim=-1, keepdim=True) * radius
    query = torch.cat([surface, query], dim=1)
    return {
        "points": points,
        "normals": points / radius,
        "query_points": query,
        "sdf": query.norm(dim=-1) - radius,
        "sample_type": torch.cat([torch.zeros(1, 128), torch.full((1, n_query), 2.0)], dim=1).long(),
        "query_normals": torch.cat([surface / radius, torch.zeros(1, n_query, 3)], dim=1),
        "query_has_normal": torch.cat([torch.ones(1, 128), torch.zeros(1, n_query)], dim=1).bool(),
    }


def test_overfit_one_sphere_and_extract_mesh():
    seed_everything(0)
    model = SpineVAE(TINY)
    batch = _sphere_batch()
    weights = LossWeights(sdf=1.0, surface=1.0, eikonal=0.1, normal=0.1, beta=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = None
    for step in range(150):
        terms = model.compute_losses(batch, weights, step, sample=False)
        optimizer.zero_grad()
        terms["total_loss"].backward()
        optimizer.step()
        first = first if first is not None else float(terms["sdf_loss"])
    assert float(terms["sdf_loss"]) < 0.3 * first
    with torch.no_grad():
        _, mu, _ = model.encode(batch["points"], batch["normals"], sample=False)
    result = latent_to_mesh(model, mu[0].numpy(), bbox=((-1, -1, -1), (1, 1, 1)), resolution=32, coord_scale=1.0, device=torch.device("cpu"), check_self_intersections=False)
    assert result["mesh_raw"] is not None and not result["sdf_evaluation"]["surface_touches_grid_boundary"]
    volume = abs(float(result["mesh_raw"].volume))
    assert volume == pytest.approx(4 / 3 * np.pi * 0.5**3, rel=0.35)


def test_vae_training_loop_on_fake_dataset(fake_dataset, tmp_path):
    seed_everything(0)
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    index = index.with_split(build_group_split(index.frame, seed=1))
    data_cfg = {"n_points": 2048, "coord_scale": 1e-3, "use_normals": True, "query_counts": {"surface": 8, "near": 8, "uniform": 8}}
    train_ds, val_ds = sdf_datasets(index, data_cfg)
    bbox = compute_frozen_bbox(index.with_split(build_group_split(index.frame, seed=1), "train"), margin=50.0, n_points=2048)
    cfg = SpineVAEConfig(latent_dim=4, encoder=PointNeXtConfig(input_points=2048, base_width=8, stage_widths=(8, 16), strides=(4, 4), blocks=(0, 1), k_neighbors=4, global_feature_dim=8, sampler="random"), decoder=SirenConfig(hidden_dim=16, hidden_layers=2))
    model = SpineVAE(cfg)
    run = create_run({"seed": 0}, experiment_id="vae_tiny", runs_root=tmp_path / "runs")
    weights = LossWeights(eikonal=0.1, kl_warmup_steps=2)
    device = torch.device("cpu")
    result = run_training(
        run,
        LoopConfig(max_steps=2, log_every=1, validate_every=2, sample_every=2, checkpoint_every=2),
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        make_loader(train_ds, batch_size=2, shuffle=True, seed=0, drop_last=True),
        make_loss_fn(weights, device),
        device=device,
        validate_fn=make_validate_fn(val_ds, weights, device, batch_size=2, seed=0, max_batches=1),
        sample_fn=make_sample_fn(run.meshes, val_ds, bbox=(bbox.low, bbox.high), coord_scale=1e-3, resolution=12, n_reconstructions=1, n_prior=1, seed=0, device=device),
    )
    metrics = result["metrics"]
    for key in ("val_loss", "val_sdf_loss", "val_eikonal_loss", "val_mean_abs_mu", "val_active_latent_dimensions", "mesh_validity_on_validation_samples"):
        assert key in metrics, key
    sample_dir = run.meshes / "step_0000002" / "prior" / "sample_00000"
    assert (sample_dir / "latent.npy").exists() and (sample_dir / "sdf_evaluation_config.json").exists()
    meta = json.loads((sample_dir / "generation_metadata.json").read_text())
    assert meta["generation_config"]["mode"] == "prior" and meta["checkpoint_id"] == "step2"
    train_log = [json.loads(line) for line in (run.logs / "metrics_train.jsonl").read_text().splitlines()]
    assert train_log[0]["kl_weight"] == 0.0 and train_log[1]["kl_weight"] == pytest.approx(5e-4)  # warm-up by step
