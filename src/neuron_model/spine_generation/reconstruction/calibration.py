"""Calibration of the reconstruction backends on REAL spines (before any generator is trained).

Two chains give the lower bound of the error caused by the representation and
the surface-extraction procedure alone (spec, "Валидация", type 1):

- Flow Matching backend::

      local_sealed mesh -> pointcloud_8192 (precomputed) -> estimated normals
                        -> Screened Poisson -> raw mesh -> postprocessed mesh

  Normals are *estimated* (as they will be for MoGen output); a control run
  with the true mesh normals separates the normal-estimation error from the
  Poisson error.

- VAE backend::

      local_sealed mesh -> exact SDF on the frozen-bbox grid -> Marching Cubes -> mesh

Every candidate is compared with the reference mesh
(:func:`evaluation.geometry_metrics.compare_meshes`) and validated
(:func:`mesh_validation.validate_mesh`). Only the validation split is used;
the chosen parameters are then frozen until the final test.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..evaluation.geometry_metrics import compare_meshes
from .estimate_normals import estimate_normals, normal_angle_error
from .marching_cubes import GridSpec, sdf_to_mesh
from .mesh_validation import validate_mesh
from .postprocess import postprocess_mesh
from .screened_poisson import PoissonParams, screened_poisson

VALIDITY_KEYS = ("is_valid", "is_watertight", "n_connected_components", "genus", "has_self_intersections", "surface_touches_bbox")


def poisson_param_grid(grid: Mapping[str, Sequence[Any]], fixed: Optional[Mapping[str, Any]] = None) -> List[PoissonParams]:
    """Cartesian product of ``grid`` values (keys = PoissonParams fields) on top of ``fixed``."""
    keys = list(grid)
    combos = itertools.product(*(grid[k] for k in keys))
    return [PoissonParams.from_mapping({**dict(fixed or {}), **dict(zip(keys, values))}) for values in combos]


def _evaluate_candidate(mesh, reference, *, bbox, compare_kwargs, check_self_intersections, postprocess_kwargs) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    raw_report = validate_mesh(mesh, bbox=bbox, check_self_intersections=check_self_intersections)
    row.update({f"raw_{k}": raw_report.get(k) for k in VALIDITY_KEYS})
    if mesh is None or not raw_report.get("mesh_present"):
        return row
    row.update({f"raw_{k}": v for k, v in compare_meshes(mesh, reference, **compare_kwargs).items()})
    post, post_changes = postprocess_mesh(mesh, **postprocess_kwargs)
    post_report = validate_mesh(post, bbox=bbox, check_self_intersections=check_self_intersections)
    row.update({f"post_{k}": post_report.get(k) for k in VALIDITY_KEYS})
    row.update({f"post_{k}": v for k, v in compare_meshes(post, reference, **compare_kwargs).items()})
    row["post_n_components_removed"] = post_changes["n_components_removed"]
    return row


def calibrate_poisson_spine(
    spine: Mapping[str, Any],
    params_list: Sequence[PoissonParams],
    k_normals: Sequence[int],
    *,
    n_points: int = 8192,
    true_normals_control: bool = True,
    compare_kwargs: Optional[Mapping[str, Any]] = None,
    postprocess_kwargs: Optional[Mapping[str, Any]] = None,
    check_self_intersections: bool = True,
) -> List[Dict[str, Any]]:
    """All (normals source, k_normal, Poisson params) candidates for one spine."""
    from ...spine_geometry import load_trimesh
    from ..data.index import pointcloud_variant_path

    reference = load_trimesh(spine["local_sealed_mesh_path"], process=False)
    with np.load(pointcloud_variant_path(spine[f"pointcloud_{n_points}_path"], 1)) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        true_normals = np.asarray(data["normals"], dtype=np.float64)

    sources: List[Dict[str, Any]] = []
    for k in k_normals:
        normals, diag = estimate_normals(points, k=k)
        sources.append({"normals_source": "estimated", "k_normal": int(k), "normals": normals, **diag, **normal_angle_error(normals, true_normals)})
    if true_normals_control:
        sources.append({"normals_source": "true", "k_normal": None, "normals": true_normals})

    rows: List[Dict[str, Any]] = []
    for source in sources:
        normals = source.pop("normals")
        for params in params_list:
            started = time.perf_counter()
            mesh = screened_poisson(points, normals, params)
            elapsed = time.perf_counter() - started
            row = {"spine_key": spine["spine_key"], "backend": "screened_poisson", **source, **params.to_dict(), "reconstruction_seconds": elapsed}
            row.update(
                _evaluate_candidate(
                    mesh,
                    reference,
                    bbox=None,
                    compare_kwargs=dict(compare_kwargs or {}),
                    check_self_intersections=check_self_intersections,
                    postprocess_kwargs=dict(postprocess_kwargs or {}),
                )
            )
            rows.append(row)
    return rows


def calibrate_marching_cubes_spine(
    spine: Mapping[str, Any],
    bbox_low: Sequence[float],
    bbox_high: Sequence[float],
    resolutions: Sequence[int],
    *,
    chunk_size: int = 262_144,
    compare_kwargs: Optional[Mapping[str, Any]] = None,
    postprocess_kwargs: Optional[Mapping[str, Any]] = None,
    check_self_intersections: bool = True,
) -> List[Dict[str, Any]]:
    """Exact SDF of the real mesh on the frozen-bbox grid -> Marching Cubes, per resolution."""
    from ...spine_geometry import load_trimesh
    from ...spine_sampling import signed_distance

    reference = load_trimesh(spine["local_sealed_mesh_path"], process=False)
    rows: List[Dict[str, Any]] = []
    for resolution in resolutions:
        grid = GridSpec(tuple(bbox_low), tuple(bbox_high), int(resolution))
        started = time.perf_counter()
        mesh, diagnostics, _ = sdf_to_mesh(lambda q: signed_distance(reference, q)[0], grid, chunk_size=chunk_size)
        elapsed = time.perf_counter() - started
        row = {
            "spine_key": spine["spine_key"],
            "backend": "marching_cubes",
            "resolution": int(resolution),
            "voxel_size": float(np.max(grid.spacing)),
            "surface_touches_grid_boundary": diagnostics["surface_touches_grid_boundary"],
            "reconstruction_seconds": elapsed,
        }
        row.update(
            _evaluate_candidate(
                mesh,
                reference,
                bbox=(bbox_low, bbox_high),
                compare_kwargs=dict(compare_kwargs or {}),
                check_self_intersections=check_self_intersections,
                postprocess_kwargs=dict(postprocess_kwargs or {}),
            )
        )
        rows.append(row)
    return rows


def summarize(results: pd.DataFrame, group_columns: Sequence[str], *, prefix: str = "raw_") -> pd.DataFrame:
    """Per-candidate aggregates across spines: median/mean errors and validity rates."""
    metric_columns = [c for c in results.columns if c.startswith(prefix) and pd.api.types.is_numeric_dtype(results[c])]
    grouped = results.groupby(list(group_columns), dropna=False)
    summary = grouped[metric_columns].median().add_suffix("_median")
    summary = summary.join(grouped[metric_columns].mean().add_suffix("_mean"))
    summary["n_spines"] = grouped.size()
    return summary.reset_index()


def select_best(
    summary: pd.DataFrame,
    *,
    error_column: str,
    validity_column: str,
    min_validity: float,
) -> Dict[str, Any]:
    """Lowest median error among candidates whose validity rate reaches ``min_validity``.

    Falls back to the most valid candidate (and says so) if none reaches it.
    """
    eligible = summary[summary[validity_column] >= min_validity]
    if len(eligible):
        best = eligible.sort_values(error_column).iloc[0]
        return {"selected": best.to_dict(), "met_validity_threshold": True}
    best = summary.sort_values([validity_column, error_column], ascending=[False, True]).iloc[0]
    return {"selected": best.to_dict(), "met_validity_threshold": False}


def run_calibration(
    spines: Iterable[Mapping[str, Any]],
    worker,
    *,
    workers: int = 1,
) -> pd.DataFrame:
    """Apply ``worker(spine) -> rows`` to every spine (optionally in a process pool); failures become rows."""
    spines = list(spines)
    rows: List[Dict[str, Any]] = []
    if workers <= 1:
        for spine in spines:
            try:
                rows.extend(worker(spine))
            except Exception as exc:  # one bad spine must not stop a long calibration run
                rows.append({"spine_key": spine["spine_key"], "error": f"{type(exc).__name__}: {exc}"})
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, spine): spine for spine in spines}
            for future, spine in futures.items():
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    rows.append({"spine_key": spine["spine_key"], "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)
