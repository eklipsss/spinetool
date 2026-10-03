"""Oriented point cloud -> Screened Poisson Surface Reconstruction (Kazhdan & Hoppe 2013).

Backend: ``pymeshlab`` (``generate_surface_reconstruction_screened_poisson``),
the MeshLab build of Kazhdan's PoissonRecon. Chosen over open3d because it
exposes every parameter the spec asks to calibrate:

    depth             octree depth (resolution)
    point_weight      screening weight (``pointweight``); 0 = classic Poisson
    samples_per_node  minimum samples per octree node (``samplespernode``); noise robustness
    scale             reconstruction cube size / bbox size (``scale``)

Parameters are calibrated once on real point clouds (validation split) and
then frozen (``configs/neuron-model/reconstruction/poisson.yaml``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class PoissonParams:
    depth: int = 8
    point_weight: float = 4.0
    samples_per_node: float = 1.5
    scale: float = 1.1
    iters: int = 8
    full_depth: int = 5
    threads: int = 1  # keep light by default; raise on the workstation via config

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PoissonParams":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown Screened Poisson parameters: {sorted(unknown)}")
        return cls(**dict(values))

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "backend": "pymeshlab.generate_surface_reconstruction_screened_poisson"}


def screened_poisson(points: np.ndarray, normals: np.ndarray, params: PoissonParams = PoissonParams()):
    """Reconstruct a surface; returns a ``trimesh.Trimesh`` (raw, no postprocessing)."""
    import pymeshlab
    import trimesh

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points/normals must both be [N,3], got {points.shape} / {normals.shape}")
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=points, v_normals_matrix=normals))
    mesh_set.generate_surface_reconstruction_screened_poisson(
        depth=int(params.depth),
        fulldepth=int(params.full_depth),
        scale=float(params.scale),
        samplespernode=float(params.samples_per_node),
        pointweight=float(params.point_weight),
        iters=int(params.iters),
        threads=int(params.threads),
        preclean=False,
        confidence=False,
    )
    result = mesh_set.current_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(result.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(result.face_matrix(), dtype=np.int64),
        process=False,
    )
