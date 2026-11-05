"""Real spines passed through the SAME surface pipeline as the generators: the floor of what the evaluation can tell apart
without any generative error.

A generator is judged on meshes it never produced directly: MoGen writes a point cloud that becomes a mesh by normal estimation
+ Screened Poisson; the VAE writes an SDF that becomes a mesh by Marching Cubes on the frozen-bbox grid; and the attachment
region of every generated mesh is found by a heuristic (``DownFacingCapFinder``), while real spines keep their stage-1 region.
Each step leaves its own trace in the morphometrics (smooth Poisson surface, voxel facets, finder bias), so a classifier or a
distribution distance can separate "real" from "generated" even for a perfect generator. Three controls isolate the steps:

- ``asis``: the real ``local_sealed`` mesh unchanged; evaluated like a generated set it differs from the real reference only by
  the attachment finder (junction metrics) -> the finder floor;
- ``poisson``: real 8192-point surface cloud (preprocessing, variant 1) -> estimated normals -> Screened Poisson -> postprocess,
  with the frozen parameters of ``reconstruction/poisson.yaml`` = exactly the MoGen chain -> MoGen floor;
- ``mc``: exact SDF of the real mesh on the frozen-bbox grid (resolution of the VAE generation) -> Marching Cubes ->
  postprocess = the VAE chain with a perfect decoder -> VAE floor.

The outputs use the generator layout (``<root>/<control>/seed_<s>/sample_<i>/generated_mesh_{raw,postprocessed}.off`` +
``mesh_validation_*.json``), so ``evaluate_generators.py --model real_poisson=<root>/poisson`` evaluates them with the very
same protocol as the generators.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

CONTROLS = ("asis", "poisson", "mc")


def _reports(raw, *, postprocess_cfg: Optional[Mapping[str, Any]], bbox, check_self_intersections: bool) -> Dict[str, Any]:
    from ..reconstruction.mesh_validation import validate_mesh
    from ..reconstruction.postprocess import postprocess_mesh

    out: Dict[str, Any] = {"mesh_raw": raw, "validation_raw": validate_mesh(raw, bbox=bbox, check_self_intersections=check_self_intersections),
                           "mesh_postprocessed": None, "validation_postprocessed": {"mesh_present": False, "is_valid": False}, "postprocess_changes": None}
    if raw is not None and len(getattr(raw, "faces", [])):
        post, changes = postprocess_mesh(raw, **dict(postprocess_cfg or {}))
        out.update(mesh_postprocessed=post, postprocess_changes=changes,
                   validation_postprocessed=validate_mesh(post, bbox=bbox, check_self_intersections=check_self_intersections))
    return out


def control_asis(spine: Mapping[str, Any], *, postprocess_cfg=None, check_self_intersections: bool = False) -> Dict[str, Any]:
    """The real mesh unchanged, also as the 'postprocessed' mesh: postprocessing (merge_vertices) glued coincident vertices of a
    sealed real mesh and made it non-manifold / non-watertight (Windows probe, 1 of 5), which is not what this control is
    about. ``postprocess_cfg`` is accepted and ignored (old job configs)."""
    from ..reconstruction.mesh_validation import validate_mesh
    from ...spine_geometry import load_trimesh

    mesh = load_trimesh(spine["local_sealed_mesh_path"], process=False)
    report = validate_mesh(mesh, bbox=None, check_self_intersections=check_self_intersections)
    return {"mesh_raw": mesh, "validation_raw": report, "mesh_postprocessed": mesh, "validation_postprocessed": dict(report),
            "postprocess_changes": {"skipped": "asis control: the real mesh as it is"}}


def control_poisson(spine: Mapping[str, Any], *, k_normal: int, poisson: Mapping[str, Any], postprocess_cfg=None, n_points: int = 8192,
                    variant: int = 1, check_self_intersections: bool = False) -> Dict[str, Any]:
    """Points only (as a MoGen sample): the true normals of the preprocessing cloud are NOT used."""
    from ...pointcloud_io import load_points
    from ..reconstruction.estimate_normals import estimate_normals
    from ..reconstruction.screened_poisson import PoissonParams, screened_poisson

    points = np.asarray(load_points(spine[f"pointcloud_{n_points}_path"], n_points, variant), dtype=np.float64)
    normals, _ = estimate_normals(points, k=int(k_normal))
    raw = screened_poisson(points, normals, PoissonParams.from_mapping(dict(poisson)))
    return _reports(raw, postprocess_cfg=postprocess_cfg, bbox=None, check_self_intersections=check_self_intersections)


def banded_sdf(reference, grid, margin_voxels: float = 3.0):
    """Exact SDF of ``reference`` inside its own box + ``margin_voxels`` grid steps, distance to that box outside.

    The spine fills ~1 % of the frozen bbox; the exact distance + winding number at all 128^3 nodes took minutes per spine.
    Outside the padded box the true SDF is positive and >= the distance to the box (> margin), so no grid edge with a sign
    change is affected and Marching Cubes returns the same surface."""
    from ...spine_sampling import signed_distance

    step = (np.asarray(grid.high, float) - np.asarray(grid.low, float)) / (int(grid.resolution) - 1)
    pad = margin_voxels * float(step.max())
    lo, hi = np.asarray(reference.bounds[0], float) - pad, np.asarray(reference.bounds[1], float) + pad

    def fn(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        out = np.linalg.norm(np.maximum(np.maximum(lo - q, q - hi), 0.0), axis=1) + pad
        inside = np.all((q >= lo) & (q <= hi), axis=1)
        if inside.any():
            out[inside] = signed_distance(reference, q[inside])[0]
        return out

    return fn


def control_mc(spine: Mapping[str, Any], *, bbox_low: Sequence[float], bbox_high: Sequence[float], resolution: int,
               postprocess_cfg=None, chunk_size: int = 262_144, check_self_intersections: bool = False) -> Dict[str, Any]:
    """Exact signed distance of the real mesh on the same frozen-bbox grid the VAE is decoded on."""
    from ...spine_geometry import load_trimesh
    from ..reconstruction.marching_cubes import GridSpec, sdf_to_mesh

    reference = load_trimesh(spine["local_sealed_mesh_path"], process=False)
    grid = GridSpec(tuple(bbox_low), tuple(bbox_high), int(resolution))
    raw, _, _ = sdf_to_mesh(banded_sdf(reference, grid), grid, chunk_size=chunk_size)
    return _reports(raw, postprocess_cfg=postprocess_cfg, bbox=(tuple(bbox_low), tuple(bbox_high)),
                    check_self_intersections=check_self_intersections)


def run_control(job: Mapping[str, Any]) -> Dict[str, Any]:
    """One (control, spine) job -> files in the generator layout; returns a status row. Never raises (a failure is a row)."""
    from ..experiment.artifacts import write_json, write_mesh_off

    control, spine, out_dir = job["control"], job["spine"], Path(job["sample_dir"])
    row = {"control": control, "seed": job["seed"], "sample": out_dir.name, "spine_key": spine["spine_key"], "error": None}
    started = time.perf_counter()
    try:
        if control == "asis":
            result = control_asis(spine, **job["kwargs"])
        elif control == "poisson":
            result = control_poisson(spine, **job["kwargs"])
        elif control == "mc":
            result = control_mc(spine, **job["kwargs"])
        else:
            raise ValueError(f"unknown control {control!r}")
        out_dir.mkdir(parents=True, exist_ok=True)
        if result["mesh_raw"] is not None:
            write_mesh_off(out_dir / "generated_mesh_raw.off", result["mesh_raw"])
        if result["mesh_postprocessed"] is not None:
            write_mesh_off(out_dir / "generated_mesh_postprocessed.off", result["mesh_postprocessed"])
        write_json(out_dir / "mesh_validation_raw.json", result["validation_raw"])
        write_json(out_dir / "mesh_validation_postprocessed.json", {**result["validation_postprocessed"], "postprocess_changes": result["postprocess_changes"]})
        write_json(out_dir / "generation_metadata.json", {"model": f"real_{control}", "spine_key": spine["spine_key"], "generation_seed": int(job["seed"]),
                                                          "control_config": {k: v for k, v in job["kwargs"].items()}})
        row.update(raw_valid=bool(result["validation_raw"].get("is_valid")), post_valid=bool(result["validation_postprocessed"].get("is_valid")))
    except Exception as exc:  # noqa: BLE001 - one bad spine must not stop the run
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - started
    return row


def assign_samples(keys: Sequence[str], *, n_per_seed: int, seeds: Sequence[int], base_seed: int = 0) -> Dict[int, list]:
    """Disjoint random subsets of ``keys`` (one per generation seed), like independent generated sets."""
    order = np.random.default_rng(base_seed).permutation(len(keys))
    need = n_per_seed * len(seeds)
    if need > len(keys):
        raise ValueError(f"need {need} spines ({n_per_seed} x {len(seeds)} seeds), only {len(keys)} available")
    return {int(s): [keys[i] for i in order[k * n_per_seed:(k + 1) * n_per_seed]] for k, s in enumerate(seeds)}
