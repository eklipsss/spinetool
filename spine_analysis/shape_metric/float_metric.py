from abc import ABC
from typing import List, Any

import numpy as np
import trimesh
from ipywidgets import widgets
from matplotlib import pyplot as plt

from CGAL.CGAL_Convex_hull_3 import convex_hull_3
from CGAL.CGAL_Polygon_mesh_processing import volume
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.shape_metric.metric_core import SpineMetric


def _safe_trimesh_volume(spine_mesh: trimesh.Trimesh) -> float:
    value = float(spine_mesh.volume)
    if not np.isfinite(value):
        return 0.0
    return abs(value)


def _safe_trimesh_convex_hull_volume(spine_mesh: trimesh.Trimesh) -> float:
    vertices = np.asarray(spine_mesh.vertices, dtype=float)
    finite_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(np.unique(finite_vertices, axis=0)) < 4:
        return 0.0
    try:
        value = float(spine_mesh.convex_hull.volume)
    except Exception:
        return 0.0
    return value if np.isfinite(value) else 0.0


class FloatSpineMetric(SpineMetric, ABC):
    def show(self) -> widgets.Widget:
        return widgets.Label(f"{self._value:.2f}")

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        return np.asarray([metric.value for metric in metrics])

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        plt.boxplot(cls.get_distribution(metrics))


class VolumeSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if isinstance(spine_mesh, trimesh.Trimesh):
            return _safe_trimesh_volume(spine_mesh)
        return abs(volume(spine_mesh))


class ConvexHullVolumeSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if isinstance(spine_mesh, trimesh.Trimesh):
            return _safe_trimesh_convex_hull_volume(spine_mesh)
        hull_mesh = Polyhedron_3()
        convex_hull_3(spine_mesh.points(), hull_mesh)
        return volume(hull_mesh)


class ConvexHullRatioSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if isinstance(spine_mesh, trimesh.Trimesh):
            v = _safe_trimesh_volume(spine_mesh)
            if v <= 0:
                return np.nan
            return (_safe_trimesh_convex_hull_volume(spine_mesh) - v) / v
        hull_mesh = Polyhedron_3()
        convex_hull_3(spine_mesh.points(), hull_mesh)
        v = abs(volume(spine_mesh))
        return (volume(hull_mesh) - v) / v
