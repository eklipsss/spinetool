"""SDF -> regular grid (chunked) -> Marching Cubes -> mesh.

Used for the VAE decoder output and for the "real SDF -> Marching Cubes"
calibration. The grid always spans the frozen train bounding box (never a
per-sample box). Sign convention: negative inside (same as the SDF samples of
the preprocessing, ``sign_convention="negative_inside"``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    low: Tuple[float, float, float]
    high: Tuple[float, float, float]
    resolution: int  # points per axis (128 for validation, 256 final)

    @property
    def spacing(self) -> np.ndarray:
        return (np.asarray(self.high, float) - np.asarray(self.low, float)) / (self.resolution - 1)

    def axes(self):
        return [np.linspace(lo, hi, self.resolution) for lo, hi in zip(self.low, self.high)]

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "spacing": self.spacing.tolist()}


def grid_points(grid: GridSpec) -> np.ndarray:
    """All grid points, ``[R^3, 3]``, in ``ij`` order (matches the ``[i, j, k]`` volume layout)."""
    xs, ys, zs = grid.axes()
    return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)


def evaluate_sdf_on_grid(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    grid: GridSpec,
    *,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Evaluate ``sdf_fn`` (``[M,3] -> [M]``) chunk by chunk; returns volume ``[R, R, R]``.

    Chunking bounds peak (GPU) memory: 256^3 = 16.8M points would not fit in
    one decoder call.
    """
    xs, ys, zs = grid.axes()
    r = grid.resolution
    volume = np.empty(r * r * r, dtype=np.float32)
    yz = np.stack(np.meshgrid(ys, zs, indexing="ij"), axis=-1).reshape(-1, 2)
    flat_index = 0
    buffer = []
    buffer_len = 0

    def flush() -> None:
        nonlocal flat_index, buffer, buffer_len
        if not buffer:
            return
        points = np.concatenate(buffer, axis=0)
        values = np.asarray(sdf_fn(points), dtype=np.float32).reshape(-1)
        if len(values) != len(points):
            raise ValueError(f"sdf_fn returned {len(values)} values for {len(points)} points")
        volume[flat_index:flat_index + len(values)] = values
        flat_index += len(values)
        buffer, buffer_len = [], 0

    for x in xs:  # one x-slab (R^2 points) at a time keeps the memory bounded too
        slab = np.column_stack([np.full(len(yz), x), yz])
        buffer.append(slab)
        buffer_len += len(slab)
        if buffer_len >= chunk_size:
            flush()
    flush()
    return volume.reshape(r, r, r)


def extract_mesh(volume: np.ndarray, grid: GridSpec, *, level: float = 0.0) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Marching Cubes at ``level``; returns ``(trimesh or None, diagnostics)``.

    ``surface_touches_grid_boundary`` is True when any boundary voxel is inside
    or on the surface (sdf <= level): the object is clipped by the frozen box
    and the mesh is not closed there - a real generator error that must be
    reported, never "fixed" by enlarging the box for that sample.
    """
    import trimesh
    from skimage import measure

    volume = np.asarray(volume, dtype=np.float32)
    boundary = np.concatenate(
        [volume[0].ravel(), volume[-1].ravel(), volume[:, 0].ravel(), volume[:, -1].ravel(), volume[:, :, 0].ravel(), volume[:, :, -1].ravel()]
    )
    diagnostics: Dict[str, Any] = {
        "iso_level": float(level),
        "grid": grid.to_dict(),
        "sdf_min": float(np.nanmin(volume)),
        "sdf_max": float(np.nanmax(volume)),
        "has_nan": bool(np.isnan(volume).any()),
        "surface_touches_grid_boundary": bool((boundary <= level).any()),
        "surface_found": False,
    }
    if diagnostics["has_nan"] or not (diagnostics["sdf_min"] < level < diagnostics["sdf_max"]):
        return None, diagnostics
    vertices, faces, _, _ = measure.marching_cubes(volume, level=level, spacing=tuple(grid.spacing.tolist()))
    vertices = vertices + np.asarray(grid.low, dtype=float)
    # skimage's default gradient_direction="descent" treats lower values as the object, which is
    # exactly a negative-inside SDF -> faces already wind outward (volume > 0); no flip needed.
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    diagnostics["surface_found"] = True
    diagnostics["n_vertices"] = int(len(mesh.vertices))
    diagnostics["n_faces"] = int(len(mesh.faces))
    return mesh, diagnostics


def sdf_to_mesh(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    grid: GridSpec,
    *,
    level: float = 0.0,
    chunk_size: int = 262_144,
) -> Tuple[Optional[Any], Dict[str, Any], np.ndarray]:
    volume = evaluate_sdf_on_grid(sdf_fn, grid, chunk_size=chunk_size)
    mesh, diagnostics = extract_mesh(volume, grid, level=level)
    diagnostics["chunk_size"] = int(chunk_size)
    return mesh, diagnostics, volume
