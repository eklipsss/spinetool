"""Training and generation helpers of the VN + FoldingNet autoencoder (``model.py``).

Data: the same point-cloud datasets as MoGen (``spine_mogen.train.point_cloud_datasets``: random resampling variant and
physical jitter for training, variant 1 for validation; NO random rotations), 2048 points, coord_scale nm -> um.
Generation: codes of the training spines -> RealNVP prior (``spine_vae.latent_prior.fit_flow``, the flow used for our VAE)
-> samples -> decoded on a dense grid of 8192 points -> the Poisson chain d8_pw8 of MoGen -> meshes in the MoGen sample
layout, so ``evaluate_generators.py`` evaluates them unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ...data.datasets import PointCloudDataset, make_loader
from ...experiment.artifacts import write_pointcloud_ply
from .model import VNFoldConfig, VNFoldingAE, chamfer, make_grid


def build_model(model_cfg: Mapping[str, Any]) -> VNFoldingAE:
    return VNFoldingAE(VNFoldConfig.from_dict(dict(model_cfg.get("architecture") or {})))


def make_loss_fn(device, coord_scale: float):
    def loss_fn(model, batch, step: int) -> Tuple[Any, Dict[str, float]]:
        x = batch["points"].to(device, non_blocking=True)
        loss = chamfer(model(x), x).mean()
        rms_nm = float(np.sqrt(max(float(loss.detach()), 0.0) / 2.0) / coord_scale)
        return loss, {"loss": float(loss.detach()), "chamfer_rms_nm": rms_nm}

    return loss_fn


def make_validate_fn(val_dataset: Optional[PointCloudDataset], device, *, batch_size: int, seed: int, coord_scale: float,
                     max_batches: Optional[int] = None):
    import torch

    def validate(model, step: int) -> Dict[str, float]:
        if val_dataset is None:
            return {}
        was_training = model.training
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(make_loader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)):
                if max_batches is not None and i >= max_batches:
                    break
                x = batch["points"].to(device)
                total += float(chamfer(model(x), x).sum())
                n += x.shape[0]
        model.train(was_training)
        loss = total / max(n, 1)
        return {"val_loss": loss, "val_chamfer_rms_nm": float(np.sqrt(loss / 2.0) / coord_scale)}

    return validate


def make_sample_fn(run_samples_dir: Path, val_dataset: Optional[PointCloudDataset], device, *, n_samples: int, coord_scale: float):
    """Reconstructions of the first ``n_samples`` validation clouds (target + reconstruction .ply, physical units)."""
    import torch

    def sample(model, step: int) -> Dict[str, float]:
        if val_dataset is None or n_samples <= 0:
            return {}
        was_training = model.training
        model.eval()
        x = torch.stack([val_dataset[i]["points"] for i in range(min(n_samples, len(val_dataset)))]).to(device)
        with torch.no_grad():
            rec = model(x)
        model.train(was_training)
        out = run_samples_dir / f"step_{step:07d}"
        for i, (a, b) in enumerate(zip(x.cpu().numpy(), rec.cpu().numpy())):
            write_pointcloud_ply(out / f"target_{i:03d}.ply", a / coord_scale)
            write_pointcloud_ply(out / f"reconstruction_{i:03d}.ply", b / coord_scale)
        return {"sample_rms_nm": float(np.sqrt(float(chamfer(rec, x).mean()) / 2.0) / coord_scale)}

    return sample


def load_run_model(run, checkpoint: str = "latest", device: Any = "cpu") -> VNFoldingAE:
    """Model of a training run with its EMA weights (if any), eval mode."""
    from ...experiment.checkpoint import CheckpointManager

    model = VNFoldingAE(VNFoldConfig.from_dict(json.loads((run.root / "model_config.json").read_text(encoding="utf-8"))))
    payload = CheckpointManager(run.checkpoints).load(checkpoint, restore_rng=False)
    state = payload["state"]
    model.load_state_dict(state["ema"]["model"] if "ema" in state else state["model"])
    model.checkpoint_step = int(payload.get("step", -1))
    return model.to(device).eval()


def encode_dataset(model: VNFoldingAE, dataset: PointCloudDataset, device, *, batch_size: int = 64, log=None) -> Tuple[np.ndarray, list]:
    """Latent vectors (``model.latent_vector``) of every item, eval mode; returns (vectors [n, D(+6)], spine keys)."""
    import torch

    out, keys = [], []
    with torch.no_grad():
        for i, batch in enumerate(make_loader(dataset, batch_size=batch_size, shuffle=False, seed=0)):
            z, frame = model.encode(batch["points"].to(device))
            out.append(model.latent_vector(z, frame).cpu().numpy())
            keys.extend(batch["spine_key"])
            if log is not None and (i + 1) % 50 == 0:
                log(f"encoded {len(keys)} / {len(dataset)}")
    return np.concatenate(out).astype(np.float64), keys


def decode_vectors(model: VNFoldingAE, vectors: np.ndarray, device, *, n_points: int, coord_scale: float, batch_size: int = 16,
                   grid_seed: int = 0) -> np.ndarray:
    """Latent vectors -> point clouds in nm [n, n_points, 3], decoded on a grid of ``n_points``."""
    import torch

    grid = make_grid(n_points, model.cfg.grid, model.cfg.grid_extent, seed=grid_seed).to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(vectors), batch_size):
            v = torch.as_tensor(vectors[s: s + batch_size], dtype=torch.float32, device=device)
            z, frame = model.split_latent(v)
            out.append((model.decode(z, frame, grid) / coord_scale).cpu().numpy())
    return np.concatenate(out)
