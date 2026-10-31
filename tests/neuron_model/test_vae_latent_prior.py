"""Latent diagnostics and learned priors of the VAE (models/spine_vae/latent_prior.py, analyze_vae_latent.py,
fit_latent_prior.py, generate_vae.py --prior learned)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import build_group_split, save_splits
from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.checkpoint import CheckpointManager, state_dicts
from src.neuron_model.spine_generation.experiment.run import create_run
from src.neuron_model.spine_generation.models.spine_vae.latent_prior import (
    GaussianPrior, StandardNormalPrior, encode_spines, fit_flow, hole_ratio, latent_report, load_prior, load_vae_run,
    nll_per_dim, save_prior,
)
from src.neuron_model.spine_generation.models.spine_vae.model import SpineVAE, SpineVAEConfig
from src.neuron_model.spine_generation.models.spine_vae.pointnext_encoder import PointNeXtConfig
from src.neuron_model.spine_generation.models.spine_vae.siren_sdf_decoder import SirenConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _banana(n, d, rng):
    a = rng.standard_normal((n, d))
    a[:, 1] += a[:, 0] ** 2 - 1.0
    a[: n // 2, 2] += 4.0
    return a[rng.permutation(n)]


def test_flow_is_invertible_and_beats_the_gaussian_on_non_gaussian_codes():
    rng = np.random.default_rng(0)
    a = _banana(4000, 6, rng)
    train, val = a[:3000], a[3000:]
    gauss = GaussianPrior.fit(train)
    flow, info = fit_flow(train, val, epochs=120, hidden=64, n_layers=6, log=None)
    assert nll_per_dim(flow, val) < nll_per_dim(gauss, val) - 0.02
    assert nll_per_dim(gauss, val) < nll_per_dim(StandardNormalPrior(6), val)
    z = torch.as_tensor(val[:16], dtype=torch.float32)
    eps, _ = flow.flow(z)
    assert torch.allclose(flow.flow.inverse(eps), z, atol=1e-4)
    # log-density integrates the change of variables: 1-D check of the whitening part via a Gaussian-only flow
    g_flow, _ = fit_flow(train[:, :2], None, epochs=1, lr=0.0, log=None)  # lr 0 = pure whitening = the fitted Gaussian
    assert np.allclose(g_flow.log_prob(val[:50, :2]), GaussianPrior.fit(train[:, :2]).log_prob(val[:50, :2]), atol=1e-3)


def test_priors_save_and_load(tmp_path):
    rng = np.random.default_rng(1)
    train = rng.standard_normal((500, 4)) @ np.diag([1.0, 2.0, 0.5, 1.0])
    flow, _ = fit_flow(train, None, epochs=3, hidden=16, n_layers=2, log=None)
    gauss = GaussianPrior.fit(train)
    for prior in (flow, gauss):
        d = tmp_path / prior.kind
        save_prior(prior, d)
        again = load_prior(d)
        assert np.allclose(again.log_prob(train[:5]), prior.log_prob(train[:5]), atol=1e-5)
        assert again.sample(7, seed=3).shape == (7, 4)


def test_latent_report_and_hole_ratio():
    rng = np.random.default_rng(2)
    centres = rng.standard_normal((300, 8)) * 0.1 + 3.0  # codes far from N(0, I)
    mu = np.stack([centres + 0.003 * rng.standard_normal(centres.shape) for _ in range(2)], axis=1)
    logvar = np.full_like(mu, np.log(0.01 ** 2))
    report = latent_report(mu, logvar, seed=0)
    assert report["retrieval_resampling"] > 0.95 and report["retrieval_posterior"] > 0.9
    assert report["posterior_to_between_median"] < 0.2
    assert report["hole_ratio_standard_normal"]["median_ratio"] > 10   # N(0, I) samples miss the codes
    assert report["hole_ratio_fitted_gaussian"]["median_ratio"] < 2.0
    assert report["kl_fitted_gaussian_to_standard_normal"] > 10
    assert hole_ratio(centres[:150], centres[150:])["median_ratio"] < 2.0


def _tiny_vae_run(fake_dataset, tmp_path):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    splits_path = tmp_path / "splits.parquet"
    save_splits(build_group_split(index.frame, seed=1), splits_path, version="test", info={})
    cfg = SpineVAEConfig(latent_dim=4, encoder=PointNeXtConfig(input_points=2048, base_width=8, stage_widths=(8, 16), strides=(4, 4), blocks=(0, 1),
                                                               k_neighbors=4, global_feature_dim=8, sampler="random"),
                         decoder=SirenConfig(hidden_dim=16, hidden_layers=2))
    torch.manual_seed(0)
    model = SpineVAE(cfg)
    run_cfg = {"seed": 0, "device": "cpu",
               "data": {"manifests": [str(fake_dataset / "manifest.parquet")], "splits": str(splits_path), "n_points": 2048,
                        "coord_scale": 1e-3, "use_normals": True, "variants": [1, 2]},
               "reconstruction": {"marching_cubes_config": str(REPO_ROOT / "configs/neuron-model/reconstruction/marching_cubes.yaml")}}
    run = create_run(run_cfg, experiment_id="vae_latent", runs_root=tmp_path / "runs")
    write_json(run.root / "model_config.json", cfg.to_dict())
    CheckpointManager(run.checkpoints).save(1, state_dicts(model=model), {})
    return run


def test_encode_and_scripts_on_a_tiny_run(fake_dataset, tmp_path):
    run = _tiny_vae_run(fake_dataset, tmp_path)
    model, _, cfg, ckpt, device = load_vae_run(run.root)
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    codes = encode_spines(model, index, cfg["data"], variants=(1, 2), device=device, batch_size=4)
    assert codes["mu"].shape == (len(index), 2, 4) and np.isfinite(codes["mu"]).all()

    import trimesh

    for path in index.frame["local_sealed_mesh_path"]:  # the fake dataset has no meshes: spheres resting on y = 0
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=100.0)
        sphere.apply_translation([0.0, 100.0, 0.0])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sphere.export(path)

    def run_script(name, *extra):
        out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/neuron_model" / name), "--run", str(run.root), *extra],
                             capture_output=True, text=True, cwd=REPO_ROOT)
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
        return out.stdout

    stdout = run_script("analyze_vae_latent.py", "--split", "train", "--n-spines", "20", "--decode-spines", "2", "--n-posterior", "2", "--resolution", "24")
    assert "retrieval_posterior" in stdout and "[mesh]" in stdout
    out_dir = run.root / "latent_analysis" / "train_seed1"
    mesh_report = json.loads((out_dir / "mesh_report.json").read_text())
    assert "Length" in mesh_report["metrics"]

    run_script("fit_latent_prior.py", "--n-train", "20", "--n-val", "5", "--gmm", "2", "--flow-epochs", "3", "--flow-hidden", "16", "--flow-layers", "2")
    report = json.loads((run.root / "latent_prior" / "report.json").read_text())
    assert {"standard_normal", "gaussian", "gmm2", "flow"} <= set(report["priors"])
    assert (run.root / "latent_prior" / "flow_prior.pt").exists()
    prior = load_prior(run.root / "latent_prior" / "flow_prior.pt")
    assert prior.sample(3, seed=0).shape == (3, 4)


def test_repeat_statistics_separates_within_and_between_spread():
    import pandas as pd

    from src.neuron_model.spine_generation.models.spine_vae.latent_prior import repeat_statistics

    rng = np.random.default_rng(3)
    rows = []
    for spine in range(200):
        true = rng.normal(1000.0, 100.0)                       # between-spine sd 100
        rows.append({"spine": spine, "kind": "real", "Length": true})
        rows.append({"spine": spine, "kind": "mu", "Length": true * 0.9})  # reconstructions 10 % shorter
        for _ in range(6):
            rows.append({"spine": spine, "kind": "posterior", "Length": true * 0.9 + rng.normal(0.0, 30.0)})  # within sd 30
    stats = repeat_statistics(pd.DataFrame(rows), ["Length"])["Length"]
    assert stats["within_to_between_posterior"] == pytest.approx(30.0 / 90.0, rel=0.15)
    assert stats["icc_posterior"] > 0.85
    assert stats["recon_bias_sd"] == pytest.approx(-1.0, abs=0.15)      # -0.1 * 1000 / 100
    assert stats["recon_median_rel_err"] == pytest.approx(0.1, abs=0.01)
    assert "within_resampling_sd" not in stats


def test_hole_and_coverage_ratio_flag_a_too_narrow_prior():
    rng = np.random.default_rng(4)
    codes = rng.standard_normal((400, 6))
    good = hole_ratio(rng.standard_normal((400, 6)), codes)
    narrow = hole_ratio(0.2 * rng.standard_normal((400, 6)), codes)
    wide = hole_ratio(4.0 * rng.standard_normal((400, 6)), codes)
    assert narrow["coverage_ratio"] > good["coverage_ratio"] * 1.3      # narrow misses real codes
    assert wide["median_ratio"] > good["median_ratio"] * 2               # wide falls into holes
