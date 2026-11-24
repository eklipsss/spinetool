"""VN + FoldingNet: equivariance of the VN layers, invariance of the code, decoding, a tiny fit, training and generation."""

import math

import numpy as np
import pytest
import torch

from src.neuron_model.spine_generation.models.spine_foldnet import VNFoldConfig, VNFoldingAE, chamfer, make_grid
from src.neuron_model.spine_generation.models.spine_foldnet.vn import VNBlock, gram_schmidt, rotation_from_6d


def _rotation(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q, dtype=torch.float32)


def _small(mode="frame", grid="plane"):
    torch.manual_seed(0)
    return VNFoldingAE(VNFoldConfig(mode=mode, latent_dim=16, encoder_channels=(4, 8), k_nn=6, grid=grid, grid_points=64, hidden_dim=32)).eval()


def test_vn_block_is_rotation_equivariant():
    torch.manual_seed(0)
    block = VNBlock(3, 5).eval()
    x = torch.randn(2, 3, 3, 10)
    q = _rotation(1)
    rotated_in = torch.einsum("ij,bcjn->bcin", q, x)
    assert torch.allclose(block(rotated_in), torch.einsum("ij,bcjn->bcin", q, block(x)), atol=1e-5)


@pytest.mark.parametrize("mode", ["frame", "invariant"])
def test_code_is_invariant_and_frame_rotates(mode):
    model = _small(mode)
    pts = torch.randn(3, 50, 3)
    q = _rotation(2)
    z, frame = model.encode(pts)
    zr, frame_r = model.encode(pts @ q.T)                      # points as rows: x -> Q x
    assert torch.allclose(z, zr, atol=1e-4)
    assert torch.allclose(frame_r, frame @ q.T, atol=1e-4)     # R(Qx) = R(x) Q^T
    out, out_r = model(pts), model(pts @ q.T)
    if mode == "frame":                                        # fully equivariant autoencoder
        assert torch.allclose(out_r, out @ q.T, atol=1e-4)
    else:                                                      # invariant: the same cloud whatever the input orientation
        assert torch.allclose(out_r, out, atol=1e-4)


def test_latent_vector_round_trip_and_grids():
    model = _small("frame", "sphere")
    z, frame = model.encode(torch.randn(2, 40, 3))
    z2, frame2 = model.split_latent(model.latent_vector(z, frame))
    assert torch.allclose(z, z2) and torch.allclose(frame, frame2, atol=1e-3)
    assert torch.allclose(rotation_from_6d(frame[:, :2].reshape(-1, 6)) @ rotation_from_6d(frame[:, :2].reshape(-1, 6)).transpose(1, 2),
                          torch.eye(3).expand(2, 3, 3), atol=1e-5)
    assert make_grid(8192, "plane").shape == (8192, 2) and make_grid(100, "sphere").shape == (100, 3)
    assert model.decode(z, frame, make_grid(300, "sphere")).shape == (2, 300, 3)  # any number of decoded points
    r = gram_schmidt(torch.randn(4, 3, 3))
    assert torch.allclose(torch.det(r), torch.ones(4), atol=1e-5)


def test_tiny_autoencoder_fits_two_shapes():
    torch.manual_seed(0)
    model = VNFoldingAE(VNFoldConfig(mode="frame", latent_dim=16, encoder_channels=(8, 16), k_nn=8, grid="sphere", grid_points=128, hidden_dim=64))
    u = torch.rand(2, 128, 2)
    sphere = torch.stack([torch.cos(2 * math.pi * u[..., 0]) * torch.sin(math.pi * u[..., 1]), torch.sin(2 * math.pi * u[..., 0]) * torch.sin(math.pi * u[..., 1]),
                          torch.cos(math.pi * u[..., 1])], dim=-1)
    pts = sphere * torch.tensor([[[0.3, 0.3, 0.3]], [[0.2, 0.6, 0.2]]]) + torch.tensor([0.0, 0.5, 0.0])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = chamfer(model(pts), pts).mean()
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.2 * first


def test_train_and_generate_scripts_end_to_end(fake_dataset, tmp_path):
    """train_vn_foldingnet.py (4 steps) -> generate_vn_foldingnet.py: codes, flow, 8192-point clouds in the MoGen layout."""
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    from src.neuron_model.spine_generation.data.index import load_spine_index
    from src.neuron_model.spine_generation.data.splits import build_group_split

    repo = Path(__file__).resolve().parents[2]
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    splits = build_group_split(index.frame, seed=1)
    splits_path = tmp_path / "splits.parquet"
    splits.to_parquet(splits_path)
    cfg = {"defaults": [str(repo / "configs/neuron-model/vn_foldingnet/frame.yaml")], "runs_root": str(tmp_path / "runs"), "device": "cpu",
           "data": {"manifests": [str(fake_dataset / "manifest.parquet")], "splits": str(splits_path), "batch_size": 4, "num_workers": 0,
                    "coord_scale": 1.0e-2},
           "model": {"architecture": {"latent_dim": 8, "encoder_channels": [4, 8], "k_nn": 4, "grid_points": 64, "hidden_dim": 16}},
           "train": {"max_steps": 4, "validate_every": 2, "sample_every": 4, "checkpoint_every": 2, "log_every": 2, "n_samples": 2, "n_val_batches": 2}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = subprocess.run([sys.executable, str(repo / "scripts/neuron_model/train_vn_foldingnet.py"), "--config", str(cfg_path)],
                         capture_output=True, text=True, cwd=repo)
    assert out.returncode == 0, out.stderr[-3000:]
    run_dir = next((tmp_path / "runs").iterdir())
    gen = subprocess.run([sys.executable, str(repo / "scripts/neuron_model/generate_vn_foldingnet.py"), "--run", str(run_dir), "--n-samples", "3",
                          "--seeds", "1", "--no-mesh", "--flow-epochs", "3", "--flow-hidden", "16", "--flow-layers", "2"],
                         capture_output=True, text=True, cwd=repo)
    assert gen.returncode == 0, gen.stderr[-3000:]
    from src.neuron_model.spine_generation.experiment.artifacts import read_pointcloud_ply

    pts, _ = read_pointcloud_ply(run_dir / "meshes_flow_d8_pw8" / "seed_1" / "sample_00000" / "generated_pointcloud.ply")
    assert pts.shape == (8192, 3) and np.isfinite(pts).all()
    codes = np.load(run_dir / "latent_prior_flow" / "codes_train.npz", allow_pickle=True)["codes"]
    assert codes.shape[1] == 8 + 6  # code + 6-D rotation in mode 'frame'
