"""MoGen inference: noise -> Flow Matching ODE -> point cloud -> normals -> Screened Poisson -> mesh.

Every stage is saved separately (spec: raw generator output, raw extracted
mesh, postprocessed mesh, configs, seed, checkpoint id) so generator errors can
be told apart from reconstruction/postprocessing errors.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ...experiment.artifacts import SampleWriter, write_json, write_mesh_off, write_pointcloud_ply
from ...reconstruction.estimate_normals import estimate_normals
from ...reconstruction.mesh_validation import validate_mesh
from ...reconstruction.postprocess import postprocess_mesh
from ...reconstruction.screened_poisson import PoissonParams, screened_poisson
from .flow import sample_midpoint


def generate_point_clouds(
    model,
    *,
    n_samples: int,
    n_points: int,
    coord_scale: float,
    n_steps: int,
    schedule: str,
    seed: int,
    device,
    batch_size: int = 8,
    cond=None,
) -> np.ndarray:
    """``[n_samples, n_points, 3]`` in PHYSICAL units (model output / coord_scale).

    Noise for sample ``i`` depends only on ``(seed, i)`` - the same seed gives
    the same samples regardless of ``batch_size``.
    """
    import torch

    out = []
    model.eval()
    for start in range(0, n_samples, batch_size):
        idx = range(start, min(start + batch_size, n_samples))
        noise = torch.stack([torch.randn(n_points, 3, generator=torch.Generator().manual_seed(seed * 1_000_003 + i)) for i in idx]).to(device)
        points = sample_midpoint(model, noise, n_steps=n_steps, schedule=schedule, cond=cond)
        out.append((points / coord_scale).cpu().numpy())
    return np.concatenate(out, axis=0)


def pointcloud_to_mesh(
    points: np.ndarray,
    *,
    k_normal: int,
    poisson: PoissonParams,
    postprocess_cfg: Mapping[str, Any],
    bbox=None,
    check_self_intersections: bool = True,
) -> Dict[str, Any]:
    """Normals -> Poisson -> raw mesh -> light postprocessing, with validity reports for both meshes."""
    started = time.perf_counter()
    normals, normal_diag = estimate_normals(points, k=k_normal)
    raw = screened_poisson(points, normals, poisson)
    post, post_changes = postprocess_mesh(raw, **dict(postprocess_cfg))
    return {
        "normals": normals,
        "normal_diagnostics": normal_diag,
        "mesh_raw": raw,
        "mesh_postprocessed": post,
        "postprocess_changes": post_changes,
        "validation_raw": validate_mesh(raw, bbox=bbox, check_self_intersections=check_self_intersections),
        "validation_postprocessed": validate_mesh(post, bbox=bbox, check_self_intersections=check_self_intersections),
        "reconstruction_seconds": time.perf_counter() - started,
    }


def write_mogen_sample(
    writer: SampleWriter,
    tag: str,
    index: int,
    points: np.ndarray,
    reconstruction: Optional[Dict[str, Any]],
    *,
    seed: int,
    checkpoint_id: str,
    inference_config: Mapping[str, Any],
    poisson: PoissonParams,
    k_normal: int,
) -> None:
    d = writer.sample_dir(tag, index)
    write_pointcloud_ply(d / "generated_pointcloud.ply", points)
    write_json(d / "flow_inference_config.json", inference_config)
    writer.write_metadata(tag, index, seed=seed, checkpoint_id=checkpoint_id, generation_config=inference_config, extra={"model": "mogen"})
    if reconstruction is None:
        return
    write_pointcloud_ply(d / "estimated_normals.ply", points, reconstruction["normals"])
    write_json(d / "poisson_config.json", {**poisson.to_dict(), "k_normal": int(k_normal), "normal_diagnostics": reconstruction["normal_diagnostics"]})
    write_mesh_off(d / "generated_mesh_raw.off", reconstruction["mesh_raw"])
    write_mesh_off(d / "generated_mesh_postprocessed.off", reconstruction["mesh_postprocessed"])
    write_json(d / "mesh_validation_raw.json", reconstruction["validation_raw"])
    write_json(d / "mesh_validation_postprocessed.json", {**reconstruction["validation_postprocessed"], "postprocess_changes": reconstruction["postprocess_changes"]})
