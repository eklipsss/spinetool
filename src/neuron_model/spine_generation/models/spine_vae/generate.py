"""VAE latent -> SDF on the frozen-bbox grid (chunked) -> Marching Cubes -> mesh (+ artifacts).

The grid always spans the frozen train bbox (physical units), never a
per-sample box; queries are converted to model units for the decoder and the
predicted SDF back to physical units, so meshes come out in the local frame's
physical coordinates like the real ``local_sealed`` meshes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from ...experiment.artifacts import SampleWriter, write_json, write_mesh_off
from ...reconstruction.marching_cubes import GridSpec, sdf_to_mesh
from ...reconstruction.mesh_validation import validate_mesh
from ...reconstruction.postprocess import postprocess_mesh


def make_sdf_fn(model, z, *, coord_scale: float, device) -> Callable[[np.ndarray], np.ndarray]:
    """Numpy ``[M,3] physical -> [M] physical`` wrapper around ``model.decode`` for one latent ``z [D]``."""
    import torch

    z = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(1, -1)

    @torch.no_grad()
    def fn(points: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(points, dtype=torch.float32, device=device).unsqueeze(0) * coord_scale
        return (model.decode(q, z)[0] / coord_scale).cpu().numpy()

    return fn


def latent_to_mesh(
    model,
    z,
    *,
    bbox: Sequence[Sequence[float]],
    resolution: int,
    coord_scale: float,
    device,
    chunk_size: int = 262_144,
    iso_level: float = 0.0,
    postprocess_cfg: Optional[Mapping[str, Any]] = None,
    check_self_intersections: bool = True,
) -> Dict[str, Any]:
    grid = GridSpec(tuple(bbox[0]), tuple(bbox[1]), int(resolution))
    mesh, diagnostics, _ = sdf_to_mesh(make_sdf_fn(model, z, coord_scale=coord_scale, device=device), grid, level=iso_level, chunk_size=chunk_size)
    result: Dict[str, Any] = {
        "mesh_raw": mesh,
        "sdf_evaluation": {**diagnostics, "coord_scale": coord_scale},
        "validation_raw": validate_mesh(mesh, bbox=bbox, check_self_intersections=check_self_intersections),
        "mesh_postprocessed": None,
        "validation_postprocessed": {"mesh_present": False, "is_valid": False},
    }
    if mesh is not None:
        post, changes = postprocess_mesh(mesh, **dict(postprocess_cfg or {}))
        result["mesh_postprocessed"] = post
        result["postprocess_changes"] = changes
        result["validation_postprocessed"] = validate_mesh(post, bbox=bbox, check_self_intersections=check_self_intersections)
    return result


def write_vae_sample(
    writer: SampleWriter,
    tag: str,
    index: int,
    z: np.ndarray,
    result: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_id: str,
    generation_config: Mapping[str, Any],
) -> None:
    d = writer.sample_dir(tag, index)
    np.save(d / "latent.npy", np.asarray(z, dtype=np.float32))
    write_json(d / "sdf_evaluation_config.json", result["sdf_evaluation"])
    writer.write_metadata(tag, index, seed=seed, checkpoint_id=checkpoint_id, generation_config=generation_config, extra={"model": "vae"})
    if result["mesh_raw"] is not None:
        write_mesh_off(d / "generated_mesh_raw.off", result["mesh_raw"])
    if result["mesh_postprocessed"] is not None:
        write_mesh_off(d / "generated_mesh_postprocessed.off", result["mesh_postprocessed"])
    write_json(d / "mesh_validation_raw.json", result["validation_raw"])
    write_json(d / "mesh_validation_postprocessed.json", {**result["validation_postprocessed"], "postprocess_changes": result.get("postprocess_changes")})
