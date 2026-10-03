"""MoGen training on spine point clouds: pretrained fine-tuning (model A) and scratch (model B).

Both use the same architecture, split, point count, coordinate scale, solver
and evaluation (spec: "Обязательные условия сравнения pretrained и scratch");
only ``model.init`` differs (``pretrained`` vs ``random``). Configs:
``configs/neuron-model/mogen/{pretrained_finetune,scratch_pilot,scratch_final}.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ...data.datasets import PointCloudDataset, make_loader
from ...data.index import SpineIndex
from ...experiment.artifacts import write_pointcloud_ply
from ...experiment.config import resolve_path
from .flow import flow_matching_loss, sample_midpoint
from .pointinfinity import PointInfinity, PointInfinityConfig
from .weights import load_flax_npz, load_pointinfinity_from_flax


def build_model(model_cfg: Mapping[str, Any]) -> PointInfinity:
    """``init: pretrained`` -> exported MoGen weights (``collection``: ema_params/params); ``init: random`` -> fresh."""
    init = model_cfg.get("init", "random")
    if init == "pretrained":
        path = resolve_path(model_cfg["pretrained_weights"])
        flax_params = load_flax_npz(path, model_cfg.get("pretrained_collection", "ema_params"))
        model = load_pointinfinity_from_flax(flax_params)
        expected = {k: v for k, v in (model_cfg.get("architecture") or {}).items() if v is not None}
        if expected:
            mismatch = {k: (v, getattr(model.cfg, k)) for k, v in expected.items() if getattr(model.cfg, k) != v}
            if mismatch:
                raise ValueError(f"Pretrained checkpoint differs from the configured architecture: {mismatch}")
        return model
    if init == "random":
        architecture = dict(model_cfg["architecture"])
        if architecture.get("cond_dim") is None:
            raise ValueError(
                "model.architecture.cond_dim must be set for random init - use the pretrained checkpoint's value "
                "(printed by export_mogen_weights.py) so scratch and fine-tuned models share one architecture"
            )
        return PointInfinity(PointInfinityConfig(**architecture))
    raise ValueError(f"model.init must be 'pretrained' or 'random', got {init!r}")


def make_optimizer(model, opt_cfg: Mapping[str, Any]):
    import torch

    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as exc:  # optional dependency, only for the MoGen reference optimizer
            raise ImportError("optimizer 'prodigy' needs `pip install prodigyopt`") from exc
        return Prodigy(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name!r}")


def point_cloud_datasets(index: SpineIndex, data_cfg: Mapping[str, Any]) -> Tuple[PointCloudDataset, Optional[PointCloudDataset]]:
    """Train (random variant + physical jitter) and val (variant 1, no jitter) datasets."""
    train_index = index.subset(index.frame.loc[index.frame["split"] == "train", "spine_key"])
    val_index = index.subset(index.frame.loc[index.frame["split"] == "val", "spine_key"])
    common = dict(n_points=int(data_cfg["n_points"]), coord_scale=float(data_cfg["coord_scale"]), use_normals=False)
    train = PointCloudDataset(train_index, random_variant=True, jitter_std=float(data_cfg.get("jitter_std", 0.0)), **common)
    val = PointCloudDataset(val_index, random_variant=False, jitter_std=0.0, **common) if len(val_index) else None
    return train, val


def make_loss_fn(flow_cfg: Mapping[str, Any], device):
    schedule = flow_cfg.get("time_schedule", "mogen_cosine")

    def loss_fn(model, batch, step: int) -> Tuple[Any, Dict[str, float]]:
        x1 = batch["points"].to(device, non_blocking=True)
        loss = flow_matching_loss(model, x1, schedule=schedule)
        return loss, {"loss": float(loss.detach())}

    return loss_fn


def make_validate_fn(val_dataset, flow_cfg: Mapping[str, Any], device, *, batch_size: int, seed: int, max_batches: Optional[int] = None):
    """Val loss with fixed noise/time per batch (same draws every evaluation -> comparable across steps)."""
    import torch

    schedule = flow_cfg.get("time_schedule", "mogen_cosine")

    def validate(model, step: int) -> Dict[str, float]:
        if val_dataset is None:
            return {}
        loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)
        total, n = 0.0, 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x1 = batch["points"].to(device)
            generator = torch.Generator().manual_seed(seed + i)
            loss = flow_matching_loss(model, x1, schedule=schedule, generator=generator)
            total += float(loss) * x1.shape[0]
            n += x1.shape[0]
        return {"val_loss": total / max(n, 1)}

    return validate


def make_sample_fn(run_samples_dir, flow_cfg: Mapping[str, Any], data_cfg: Mapping[str, Any], device, *, n_samples: int, seed: int):
    """Write ``n_samples`` generated point clouds (physical units) per call: samples/step_<N>/sample_<i>.ply."""
    import torch

    n_points = int(data_cfg["n_points"])
    coord_scale = float(data_cfg["coord_scale"])

    def sample(model, step: int) -> Dict[str, float]:
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(n_samples, n_points, 3, generator=generator).to(device)
        points = sample_midpoint(model, noise, n_steps=int(flow_cfg["inference_steps"]), schedule=flow_cfg.get("inference_schedule", "mogen_cosine"))
        physical = (points / coord_scale).cpu().numpy()
        out_dir = run_samples_dir / f"step_{step:07d}"
        for i, cloud in enumerate(physical):
            write_pointcloud_ply(out_dir / f"sample_{i:03d}.ply", cloud)
        extent = physical.max(axis=1) - physical.min(axis=1)
        return {"sample_extent_mean": float(extent.mean()), "sample_finite": float(np.isfinite(physical).all())}

    return sample
