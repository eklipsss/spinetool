"""Frozen bounding box for SDF grid evaluation (Marching Cubes).

The spec requires one box in local coordinates, computed on the **train**
split with a fixed margin and frozen until final testing - never fitted per
generated sample, which would hide scale errors. It is computed from the
precomputed surface point clouds (cheap to read; the densest size by default)
and stored as JSON next to the split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .index import SpineIndex


@dataclass(frozen=True)
class FrozenBBox:
    low: tuple   # physical units
    high: tuple
    margin: float
    unit: str
    source: str
    split_version: Optional[str]
    n_spines: int

    def as_arrays(self, coord_scale: float = 1.0):
        return np.asarray(self.low, dtype=float) * coord_scale, np.asarray(self.high, dtype=float) * coord_scale

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists - the bbox is frozen once written")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FrozenBBox":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["low"], data["high"] = tuple(data["low"]), tuple(data["high"])
        return cls(**data)


def _pointcloud_bounds(path, n_points: int):
    from ...pointcloud_io import load_points

    points = np.asarray(load_points(path, n_points, 1), dtype=float)
    return points.min(axis=0), points.max(axis=0)


def _sdf_mesh_bounds(path, sdf_bbox_margin: float):
    # sdf_samples.npz stores the uniform-sampling box = mesh.bounds -/+ sdf_bbox_margin
    # (spine_sampling.sample_sdf_queries); two 3-float members, vs ~100 KB of points.
    with np.load(path) as data:
        low = np.asarray(data["bbox_low"], dtype=float) + sdf_bbox_margin
        high = np.asarray(data["bbox_high"], dtype=float) - sdf_bbox_margin
    return low, high


def compute_frozen_bbox(
    train_index: SpineIndex,
    *,
    margin: float,
    n_points: int = 8192,
    unit: str = "nm",
    split_version: Optional[str] = None,
    source: str = "pointcloud",
    sdf_bbox_margin: float = 100.0,
    io_threads: int = 1,
    progress: bool = False,
) -> FrozenBBox:
    """Axis-aligned min/max over all train spines, expanded by ``margin``.

    ``source="pointcloud"``: min/max of the surface points ``pointcloud_<n_points>`` variant 1.
    ``source="sdf_bounds"``: the exact mesh bounds recovered from the box stored in each
    ``sdf_samples.npz`` (``sdf_bbox_margin`` must equal the preprocessing value). Reads a
    few bytes per spine instead of a whole point cloud - on the slow NAS this is the
    difference between hours and a day for ~10^6 spines. ``io_threads`` > 1 overlaps the
    per-file NAS latency.
    """
    if "split" in train_index.frame.columns and (train_index.frame["split"] != "train").any():
        raise ValueError("compute_frozen_bbox must be given the train split only")
    if source == "pointcloud":
        paths = train_index.frame[f"pointcloud_{n_points}_path"].tolist()
        read = lambda p: _pointcloud_bounds(p, n_points)  # noqa: E731
        source_name = f"pointcloud_{n_points}_1"
    elif source == "sdf_bounds":
        paths = train_index.frame["sdf_samples_path"].tolist()
        read = lambda p: _sdf_mesh_bounds(p, float(sdf_bbox_margin))  # noqa: E731
        source_name = "sdf_samples_mesh_bounds"
    else:
        raise ValueError(f"unknown bbox source {source!r}")

    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=len(paths), desc=f"bbox ({source})", mininterval=5.0)
    if io_threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(io_threads)) as pool:
            results = pool.map(read, paths)
            for lo, hi in results:
                low, high = np.minimum(low, lo), np.maximum(high, hi)
                if bar is not None:
                    bar.update()
    else:
        for path in paths:
            lo, hi = read(path)
            low, high = np.minimum(low, lo), np.maximum(high, hi)
            if bar is not None:
                bar.update()
    if bar is not None:
        bar.close()
    if not np.isfinite(low).all():
        raise ValueError("Empty train index - cannot compute a bounding box")
    return FrozenBBox(
        low=tuple((low - margin).tolist()),
        high=tuple((high + margin).tolist()),
        margin=float(margin),
        unit=unit,
        source=source_name,
        split_version=split_version,
        n_spines=len(train_index),
    )
