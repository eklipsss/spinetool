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


def compute_frozen_bbox(
    train_index: SpineIndex,
    *,
    margin: float,
    n_points: int = 8192,
    unit: str = "nm",
    split_version: Optional[str] = None,
) -> FrozenBBox:
    """Axis-aligned min/max over all train spines' surface points, expanded by ``margin``."""
    if "split" in train_index.frame.columns and (train_index.frame["split"] != "train").any():
        raise ValueError("compute_frozen_bbox must be given the train split only")
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    from ...pointcloud_io import load_points

    for path in train_index.frame[f"pointcloud_{n_points}_path"]:
        points = np.asarray(load_points(path, n_points, 1), dtype=float)
        low = np.minimum(low, points.min(axis=0))
        high = np.maximum(high, points.max(axis=0))
    if not np.isfinite(low).all():
        raise ValueError("Empty train index - cannot compute a bounding box")
    return FrozenBBox(
        low=tuple((low - margin).tolist()),
        high=tuple((high + margin).tolist()),
        margin=float(margin),
        unit=unit,
        source=f"pointcloud_{n_points}_1",
        split_version=split_version,
        n_spines=len(train_index),
    )
