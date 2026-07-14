import math
from abc import ABC, abstractmethod
from typing import List, Any, Set, Dict, Optional

import numpy as np
import trimesh

from CGAL.CGAL_Kernel import Vector_3, cross_product
from CGAL.CGAL_Polygon_mesh_processing import area, face_area, volume
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Facet_handle
from spine_analysis.shape_metric.float_metric import FloatSpineMetric
from spine_analysis.shape_metric.utils import _calculate_facet_center, _point_2_vec, _get_junction_triangles, \
    _calculate_junction_center


def _is_trimesh_mesh(spine_mesh: Any) -> bool:
    return isinstance(spine_mesh, trimesh.Trimesh)


def _vector_from_array(point: np.ndarray) -> Vector_3:
    return Vector_3(float(point[0]), float(point[1]), float(point[2]))


def trimesh_boundary_edge_loops(spine_mesh: trimesh.Trimesh) -> List[Dict[str, Any]]:
    """Return connected boundary-edge components for a trimesh spine mesh.

    Each component is reported as a dict with:
    - edges: (n_edges, 2) vertex-index array;
    - vertex_indices: unique vertices participating in the component;
    - is_closed: True when every component vertex has boundary degree 2.
    """
    faces = np.asarray(spine_mesh.faces, dtype=int)
    if len(faces) == 0:
        return []

    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return []

    vertex_to_edge_ids: Dict[int, List[int]] = {}
    for edge_id, (start, end) in enumerate(boundary_edges):
        vertex_to_edge_ids.setdefault(int(start), []).append(edge_id)
        vertex_to_edge_ids.setdefault(int(end), []).append(edge_id)

    unvisited = set(range(len(boundary_edges)))
    loops: List[Dict[str, Any]] = []
    while unvisited:
        seed = unvisited.pop()
        component_edge_ids = {seed}
        stack = [seed]
        while stack:
            edge_id = stack.pop()
            for vertex_id in boundary_edges[edge_id]:
                for next_edge_id in vertex_to_edge_ids.get(int(vertex_id), []):
                    if next_edge_id in unvisited:
                        unvisited.remove(next_edge_id)
                        component_edge_ids.add(next_edge_id)
                        stack.append(next_edge_id)

        component_edges = boundary_edges[np.asarray(sorted(component_edge_ids), dtype=int)]
        component_vertices = np.unique(component_edges.reshape(-1)).astype(int)
        degrees = {int(vertex_id): 0 for vertex_id in component_vertices}
        for start, end in component_edges:
            degrees[int(start)] += 1
            degrees[int(end)] += 1
        is_closed = len(component_vertices) >= 3 and all(degree == 2 for degree in degrees.values())
        loops.append(
            {
                "edges": component_edges.astype(int),
                "vertex_indices": component_vertices,
                "is_closed": bool(is_closed),
                "n_edges": int(len(component_edges)),
            }
        )

    loops.sort(
        key=lambda loop: (
            not bool(loop["is_closed"]),
            -int(loop["n_edges"]),
            int(np.min(loop["vertex_indices"])) if len(loop["vertex_indices"]) else 0,
        )
    )
    return loops


def select_trimesh_junction_boundary_loop(spine_mesh: trimesh.Trimesh) -> Optional[Dict[str, Any]]:
    loops = trimesh_boundary_edge_loops(spine_mesh)
    if not loops:
        return None
    closed_loops = [loop for loop in loops if loop["is_closed"]]
    candidates = closed_loops if closed_loops else loops
    return max(
        candidates,
        key=lambda loop: (
            int(loop["n_edges"]),
            len(loop["vertex_indices"]),
        ),
    )


def _trimesh_junction_vertex_indices(spine_mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(spine_mesh.faces, dtype=int)
    if len(faces) == 0:
        return np.arange(len(spine_mesh.vertices), dtype=int)

    selected_loop = select_trimesh_junction_boundary_loop(spine_mesh)
    if selected_loop is not None:
        return np.asarray(selected_loop["vertex_indices"], dtype=int)

    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)

    degrees = np.bincount(edges.reshape(-1), minlength=len(spine_mesh.vertices))
    if len(degrees) == 0:
        return np.arange(len(spine_mesh.vertices), dtype=int)
    high_degree = np.flatnonzero(degrees > np.quantile(degrees, 0.95))
    if len(high_degree) > 0:
        return high_degree.astype(int)
    return np.array([0], dtype=int)


def _trimesh_junction_center(spine_mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(spine_mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return np.zeros(3, dtype=float)
    junction_indices = _trimesh_junction_vertex_indices(spine_mesh)
    return vertices[junction_indices].mean(axis=0)


def _trimesh_junction_area(spine_mesh: trimesh.Trimesh) -> float:
    faces = np.asarray(spine_mesh.faces, dtype=int)
    if len(faces) == 0:
        return 0.0
    junction_indices = set(int(index) for index in _trimesh_junction_vertex_indices(spine_mesh))
    mask = np.array(
        [bool(junction_indices.intersection(face.tolist())) for face in faces],
        dtype=bool,
    )
    if not np.any(mask):
        return 0.0
    return float(np.asarray(spine_mesh.area_faces)[mask].sum())


class JunctionSpineMetric(FloatSpineMetric, ABC):
    _junction_center: Vector_3
    _surface_vectors: List[Vector_3]
    _junction_triangles: Set[Polyhedron_3_Facet_handle]

    @abstractmethod
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if _is_trimesh_mesh(spine_mesh):
            junction_center = _trimesh_junction_center(spine_mesh)
            self._junction_triangles = set()
            self._junction_center = _vector_from_array(junction_center)
            self._surface_vectors = [
                _vector_from_array(vertex - junction_center)
                for vertex in np.asarray(spine_mesh.vertices, dtype=float)
            ]
            return

        # identify junction triangles
        self._junction_triangles = _get_junction_triangles(spine_mesh)

        # calculate junction center
        self._junction_center = _calculate_junction_center(spine_mesh)

        # calculate vectors to surface
        self._surface_vectors = []
        for point in spine_mesh.points():
            self._surface_vectors.append(_point_2_vec(point) - self._junction_center)


class JunctionCenterSpineMetric(JunctionSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)
        return self._junction_center


class JunctionAreaSpineMetric(JunctionSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if _is_trimesh_mesh(spine_mesh):
            return _trimesh_junction_area(spine_mesh)
        super()._calculate(spine_mesh)

        return sum(face_area(triangle, spine_mesh)
                   for triangle in self._junction_triangles)


class AreaSpineMetric(JunctionAreaSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if _is_trimesh_mesh(spine_mesh):
            return float(spine_mesh.area) - _trimesh_junction_area(spine_mesh)
        return area(spine_mesh) - super()._calculate(spine_mesh)



class JunctionDistanceSpineMetric(JunctionSpineMetric, ABC):
    _distances: List[float]

    @abstractmethod
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)

        self._distances = []
        for v in self._surface_vectors:
            self._distances.append(np.sqrt(v.squared_length()))


class AverageDistanceSpineMetric(JunctionDistanceSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)
        return np.mean(self._distances)


class LengthSpineMetric(JunctionDistanceSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)
        q = np.quantile(self._distances, 0.95)
        return np.mean([d for d in self._distances if d >= q])


class CenterSpineMetric(JunctionDistanceSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Vector_3:
        if _is_trimesh_mesh(spine_mesh):
            vertices = np.asarray(spine_mesh.vertices, dtype=float)
            junction_center = _trimesh_junction_center(spine_mesh)
            if len(vertices) == 0:
                return _vector_from_array(junction_center)
            distances = np.linalg.norm(vertices - junction_center, axis=1)
            q = np.quantile(distances, 0.95)
            far_vertices = vertices[distances > q]
            if len(far_vertices) == 0:
                far_vertices = vertices[[int(np.argmax(distances))]]
            center = (far_vertices.mean(axis=0) + junction_center) / 2.0
            return _vector_from_array(center)

        super()._calculate(spine_mesh)
        q = np.quantile(self._distances, 0.95)

        center = Vector_3(0, 0, 0)
        count = 0
        for vert in spine_mesh.vertices():
            point = vert.point()
            vec = Vector_3(point.x(), point.y(), point.z())
            dist = math.sqrt((vec - self._junction_center).squared_length())
            if dist > q:
                count += 1
                center += vec

        center /= count
        center += self._junction_center
        center /= 2
        return center


class LengthVolumeRatioSpineMetric(LengthSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if _is_trimesh_mesh(spine_mesh):
            volume_value = abs(float(spine_mesh.volume))
            return LengthSpineMetric().calculate(spine_mesh) / volume_value if volume_value else np.nan
        return super()._calculate(spine_mesh) / abs(volume(spine_mesh))


class LengthAreaRatioSpineMetric(LengthSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if _is_trimesh_mesh(spine_mesh):
            area_value = float(spine_mesh.area)
            return LengthSpineMetric().calculate(spine_mesh) / area_value if area_value else np.nan
        return super()._calculate(spine_mesh) / area(spine_mesh)


class CVDSpineMetric(JunctionDistanceSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)
        return np.std(self._distances, ddof=1) / np.mean(self._distances)


class OpenAngleSpineMetric(JunctionSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        if _is_trimesh_mesh(spine_mesh):
            vertices = np.asarray(spine_mesh.vertices, dtype=float)
            junction_center = _trimesh_junction_center(spine_mesh)
            surface_vectors = vertices - junction_center
            if len(surface_vectors) == 0:
                return np.nan
            axis = surface_vectors.mean(axis=0)
            axis_norm = np.linalg.norm(axis)
            if axis_norm <= 1e-12:
                return np.nan
            cross_norms = np.linalg.norm(np.cross(axis, surface_vectors), axis=1)
            dot_values = surface_vectors @ axis
            return float(np.mean(np.arctan2(cross_norms, dot_values)))

        super()._calculate(spine_mesh)

        axis = np.mean(self._surface_vectors)
        angle_sum = 0
        for v in self._surface_vectors:
            angle_sum += math.atan2(np.sqrt(cross_product(axis, v).squared_length()), axis * v)

        return angle_sum / len(self._surface_vectors)
