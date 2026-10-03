"""VAE training callbacks for ``experiment.loop.run_training``.

Validation runs in the two modes of the spec: reconstruction (real validation
spine -> ``z = mu`` -> SDF -> mesh) and prior generation (``z ~ N(0, I)`` ->
mesh); the latter is what gets compared with Flow Matching.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from ...data.datasets import SDFDataset, make_loader
from ...data.index import SpineIndex
from ...experiment.artifacts import SampleWriter
from ...reconstruction.mesh_validation import aggregate_validity
from .generate import latent_to_mesh, write_vae_sample
from .latent import latent_statistics
from .losses import LossWeights
from .model import SpineVAE, loss_logs


def sdf_datasets(index: SpineIndex, data_cfg: Mapping[str, Any]) -> Tuple[SDFDataset, Optional[SDFDataset]]:
    frame = index.frame
    train_index = index.subset(frame.loc[frame["split"] == "train", "spine_key"])
    val_index = index.subset(frame.loc[frame["split"] == "val", "spine_key"])
    common = dict(
        encoder_points=int(data_cfg["n_points"]),
        query_counts=dict(data_cfg["query_counts"]),
        coord_scale=float(data_cfg["coord_scale"]),
        use_normals=bool(data_cfg.get("use_normals", True)),
    )
    train = SDFDataset(train_index, random_variant=True, **common)
    val = SDFDataset(val_index, random_variant=False, **common) if len(val_index) else None
    return train, val


def to_device(batch: Mapping[str, Any], device) -> Dict[str, Any]:
    import torch

    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def make_loss_fn(weights: LossWeights, device):
    """``step`` (from the loop, correct after a resume too) drives the KL warm-up."""

    def loss_fn(model: SpineVAE, batch, step: int) -> Tuple[Any, Dict[str, float]]:
        terms = model.compute_losses(to_device(batch, device), weights, step)
        return terms["total_loss"], loss_logs(terms)

    return loss_fn


def make_validate_fn(val_dataset, weights: LossWeights, device, *, batch_size: int, seed: int, max_batches: Optional[int] = None):
    """Validation losses with ``z = mu`` (deterministic) + latent statistics over all seen val spines."""
    import torch

    def validate(model: SpineVAE, step: int) -> Dict[str, float]:
        if val_dataset is None:
            return {}
        sums: Dict[str, float] = {}
        mus, logvars = [], []
        n = 0
        loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            with torch.enable_grad():  # eikonal/normal terms need d f / d q even in validation
                terms = model.compute_losses(to_device(batch, device), weights, step, sample=False)
            b = batch["points"].shape[0]
            for key, value in terms.items():
                if not key.startswith("_"):
                    sums[key] = sums.get(key, 0.0) + float(value.detach()) * b
            mus.append(terms["_mu"])
            logvars.append(terms["_logvar"])
            n += b
        metrics = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
        metrics["val_loss"] = metrics.pop("val_total_loss", float("nan"))
        metrics.update({f"val_{k}": v for k, v in latent_statistics(torch.cat(mus), torch.cat(logvars)).items()})
        return metrics

    return validate


def make_sample_fn(
    out_root,
    val_dataset,
    *,
    bbox,
    coord_scale: float,
    resolution: int,
    n_reconstructions: int,
    n_prior: int,
    seed: int,
    device,
    chunk_size: int = 262_144,
    postprocess_cfg: Optional[Mapping[str, Any]] = None,
    check_self_intersections: bool = False,
):
    """Periodic reconstruction + prior samples -> meshes; logs ``mesh_validity_on_validation_samples``."""
    import torch

    writer = SampleWriter(out_root)

    def sample(model: SpineVAE, step: int) -> Dict[str, float]:
        tag = f"step_{step:07d}"
        latents = []
        if val_dataset is not None:
            for i in range(min(n_reconstructions, len(val_dataset))):
                item = val_dataset[i]
                normals = item.get("normals")
                with torch.no_grad():
                    _, mu, _ = model.encode(item["points"][None].to(device), None if normals is None else normals[None].to(device), sample=False)
                latents.append(("reconstruction", i, mu[0].cpu().numpy(), item["spine_key"]))
        prior = model.sample_prior(n_prior, generator=torch.Generator().manual_seed(seed), device=device).cpu().numpy()
        latents += [("prior", i, z, None) for i, z in enumerate(prior)]
        reports: Dict[str, list] = {"reconstruction": [], "prior": []}
        for mode, i, z, key in latents:
            result = latent_to_mesh(model, z, bbox=bbox, resolution=resolution, coord_scale=coord_scale, device=device, chunk_size=chunk_size, postprocess_cfg=postprocess_cfg, check_self_intersections=check_self_intersections)
            config = {"mode": mode, "resolution": resolution, "coord_scale": coord_scale, "spine_key": key, "prior_seed": seed if mode == "prior" else None}
            write_vae_sample(writer, f"{tag}/{mode}", i, z, result, seed=seed, checkpoint_id=f"step{step}", generation_config=config)
            reports[mode].append(result["validation_raw"])
        metrics: Dict[str, float] = {}
        for mode, mode_reports in reports.items():
            if mode_reports:
                rates = aggregate_validity(mode_reports)
                metrics[f"{mode}_valid_mesh_rate"] = rates["valid_mesh_rate"]
                metrics[f"{mode}_mesh_present_rate"] = rates["mesh_present_rate"]
        all_reports = reports["reconstruction"] + reports["prior"]
        metrics["mesh_validity_on_validation_samples"] = aggregate_validity(all_reports)["valid_mesh_rate"] if all_reports else float("nan")
        return metrics

    return sample


def loss_weights(cfg: Mapping[str, Any]) -> LossWeights:
    return LossWeights(**dict(cfg))
