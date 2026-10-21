"""Morphometric tables for evaluation: real spines (from preprocessing) and generated meshes.

Both go through ``src/neuron_model/spine_morphometrics.py`` - the very code preprocessing
stage 8 uses - as the spec requires ("теми же метриками, что и для реальных данных").

Generated meshes have no stage-1 attachment region. ``attachment_finder(mesh)`` (step E,
``attachment.DownFacingCapFinder``) returns an ``AttachmentRegion`` (centre, cap area,
loop area) or ``None``; without it the 8 attachment-dependent metrics are NaN and only
Volume / ConvexHullVolume / ConvexHullRatio / OldChordDistribution are computed.

:func:`finder_bias` runs the same finder on REAL meshes and compares with their stage-1
attachment region - how much of a real-vs-generated gap in junction metrics could be an
artefact of the finder rather than of the generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ...spine_morphometrics import ALL_METRICS, JUNCTION_METRICS, AttachmentRegion, compute_chord_distribution, compute_scalar_metrics
from ..data.index import SpineIndex
from .inputs import threaded_map

AttachmentFinder = Callable[[Any], Optional[AttachmentRegion]]


def _read_stage_table(path: Path) -> pd.DataFrame:
    """A preprocessing stage table: main file + leftover ``<table>.parts/*.parquet`` (latest row wins)."""
    path = Path(path)
    frames = [pd.read_parquet(path)] if path.exists() else []
    parts = path.with_name(path.name + ".parts")
    if parts.is_dir():
        frames.extend(pd.read_parquet(part) for part in sorted(parts.glob("*.parquet")))
    if not frames:
        raise FileNotFoundError(f"morphometrics table {path} not found")
    frame = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "neuron_id", "limb_id", "branch_id", "spine_id"]
    frame["spine_key"] = frame[keys].astype(str).agg("/".join, axis=1)
    return frame.drop_duplicates("spine_key", keep="last")


def _metrics_from_json(path: Any) -> Dict[str, Any]:
    if path is None or (isinstance(path, float) and pd.isna(path)):
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {m: data.get(m) for m in ALL_METRICS}


def real_morphometrics(index: SpineIndex, *, table: Optional[Path] = None, threads: int = 32, label: str = "") -> pd.DataFrame:
    """One row per indexed spine (``spine_key`` + the 12 metrics).

    ``table`` = the stage-8 ``morphometrics.parquet`` (same values as the per-spine JSON, one
    file read instead of one NAS round trip per spine); spines missing from it, or no table,
    -> ``morphometrics.json`` read in threads."""
    keys = list(index.frame["spine_key"])
    by_key: Dict[str, Dict[str, Any]] = {}
    if table is not None:
        frame = _read_stage_table(table)
        frame = frame[frame["spine_key"].isin(set(keys))]
        columns = [m for m in ALL_METRICS if m in frame.columns]
        by_key = {row["spine_key"]: {m: row[m] for m in columns} for row in frame[["spine_key", *columns]].to_dict(orient="records")}
    todo = [(k, p) for k, p in zip(keys, index.frame["morphometrics_path"]) if k not in by_key]
    if todo:
        for (key, _), values in zip(todo, threaded_map(lambda kp: _metrics_from_json(kp[1]), todo, threads=threads,
                                                        label=f"{label} morphometrics json".strip())):
            by_key[key] = values
    frame = pd.DataFrame([{"spine_key": k, **by_key.get(k, {})} for k in keys])
    for metric in ALL_METRICS:
        if metric not in frame.columns:
            frame[metric] = None
    return frame


def generated_morphometrics(
    meshes: Sequence[Optional[Any]],
    *,
    attachment_finder: Optional[AttachmentFinder] = None,
    compute_chords: bool = True,
    sample_ids: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """One row per generated sample; ``mesh=None`` (no surface extracted) -> all metrics NaN.

    ``attachment_found`` records whether the finder returned a region, so a model whose
    meshes often lack a recognisable base is visible in the table (not silently dropped).
    """
    rows = []
    ids = list(sample_ids) if sample_ids is not None else list(range(len(meshes)))
    for sample_id, mesh in zip(ids, meshes):
        row = {"sample_id": sample_id, "mesh_present": mesh is not None, "attachment_found": False, "attachment_method": None}
        row.update({m: np.nan for m in ALL_METRICS})
        if mesh is not None:
            region = attachment_finder(mesh) if attachment_finder is not None else None
            row["attachment_found"] = region is not None
            row["attachment_method"] = None if region is None else region.method
            row.update({k: (np.nan if v is None else v) for k, v in compute_scalar_metrics(mesh, region).items()})
            row["OldChordDistribution"] = compute_chord_distribution(mesh, region) if compute_chords else None
        rows.append(row)
    return pd.DataFrame(rows)


def finder_bias(
    index: SpineIndex,
    attachment_finder: AttachmentFinder,
    *,
    max_spines: Optional[int] = 500,
    seed: int = 0,
) -> Dict[str, Any]:
    """Finder vs stage-1 attachment on real ``local_sealed`` meshes (a random subset of ``index``).

    Per junction metric: median / 5% / 95% of the relative error ``(finder - truth) / |truth|``;
    plus how often the finder found a base and by which method.
    """
    from ...spine_geometry import load_trimesh

    frame = index.frame
    if max_spines is not None and len(frame) > max_spines:
        frame = frame.sample(n=max_spines, random_state=seed)
    errors: Dict[str, list] = {m: [] for m in JUNCTION_METRICS}
    methods: Dict[str, int] = {}
    n_found = 0
    for mesh_path, morph_path in zip(frame["local_sealed_mesh_path"], frame["morphometrics_path"]):
        truth = json.loads(Path(morph_path).read_text(encoding="utf-8"))
        mesh = load_trimesh(Path(mesh_path), process=False)
        region = attachment_finder(mesh)
        if region is None:
            continue
        n_found += 1
        methods[str(region.method)] = methods.get(str(region.method), 0) + 1
        values = compute_scalar_metrics(mesh, region)
        for metric in JUNCTION_METRICS:
            t, v = truth.get(metric), values.get(metric)
            if t is not None and v is not None and t != 0:
                errors[metric].append((v - t) / abs(t))
    per_metric = {}
    for metric, values in errors.items():
        e = np.asarray(values, dtype=float)
        if len(e):
            per_metric[metric] = {
                "median_rel_error": float(np.median(e)),
                "q05_rel_error": float(np.quantile(e, 0.05)),
                "q95_rel_error": float(np.quantile(e, 0.95)),
                "median_abs_rel_error": float(np.median(np.abs(e))),
            }
    return {"n_spines": int(len(frame)), "found_rate": n_found / max(len(frame), 1), "methods": methods, "per_metric": per_metric}


def sample_clouds(meshes: Sequence[Optional[Any]], n_points: int, *, seed: int = 0) -> np.ndarray:
    """``[n_present, n_points, 3]`` surface samples (area-weighted, same sampler as preprocessing);
    meshes that are ``None`` are skipped - keep track of which were present yourself."""
    from .geometry_metrics import sample_surface

    clouds = [sample_surface(mesh, n_points, seed=seed + i) for i, mesh in enumerate(meshes) if mesh is not None]
    return np.stack(clouds).astype(np.float32) if clouds else np.empty((0, n_points, 3), dtype=np.float32)
