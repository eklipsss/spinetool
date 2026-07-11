from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import coo_matrix, csr_matrix, diags, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization
from spine_analysis.mesh.utils import _mesh_to_v_f


@dataclass
class DistanceMatrixResult:
    method: str
    distance_matrix: np.ndarray
    elapsed_seconds: float
    mesh: Optional[trimesh.Trimesh] = None
    projected_points: Optional[np.ndarray] = None
    point_vertex_indices: Optional[np.ndarray] = None
    paths: Optional[Dict[Tuple[int, int], np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None


def _as_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must be an array with shape (n, 3)")
    return points_array


def polyhedron_to_trimesh(mesh: Any) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Trimesh):
        return mesh.copy()
    vertices, faces = _mesh_to_v_f(mesh)
    return trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64), process=False)


def _nearest_vertex_indices(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    tree = cKDTree(mesh.vertices)
    _, indices = tree.query(points)
    return indices.astype(int)


def _mesh_edge_graph(mesh: trimesh.Trimesh) -> csr_matrix:
    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces) == 0:
        raise ValueError("mesh has no faces")

    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    vertices = np.asarray(mesh.vertices, dtype=float)
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.concatenate((weights, weights))
    return csr_matrix((data, (row, col)), shape=(len(vertices), len(vertices)))


def _reconstruct_path(predecessors: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return np.array([source], dtype=int)

    path = [target]
    current = target
    guard = 0
    while current != source and guard <= len(predecessors):
        current = int(predecessors[current])
        if current < 0:
            return np.array([], dtype=int)
        path.append(current)
        guard += 1
    return np.array(path[::-1], dtype=int)


def _default_pair(distance_matrix: np.ndarray) -> Tuple[int, int]:
    if distance_matrix.shape[0] < 2:
        return 0, 0
    finite = np.where(np.isfinite(distance_matrix), distance_matrix, -np.inf)
    np.fill_diagonal(finite, -np.inf)
    pair = np.unravel_index(np.argmax(finite), finite.shape)
    return int(pair[0]), int(pair[1])


def _shortest_paths_on_mesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    method: str,
    pair_for_path: Optional[Tuple[int, int]] = None,
) -> DistanceMatrixResult:
    start = perf_counter()
    projected_vertex_indices = _nearest_vertex_indices(mesh, points)
    graph = _mesh_edge_graph(mesh)

    distances, predecessors = dijkstra(
        graph,
        directed=False,
        indices=projected_vertex_indices,
        return_predecessors=True,
    )
    distance_matrix = distances[:, projected_vertex_indices]

    if pair_for_path is None:
        pair_for_path = _default_pair(distance_matrix)

    paths: Dict[Tuple[int, int], np.ndarray] = {}
    if len(points) >= 2:
        i, j = pair_for_path
        path_vertices = _reconstruct_path(
            predecessors[i], projected_vertex_indices[i], projected_vertex_indices[j]
        )
        if len(path_vertices) > 0:
            paths[(i, j)] = np.asarray(mesh.vertices[path_vertices], dtype=float)

    return DistanceMatrixResult(
        method=method,
        distance_matrix=distance_matrix,
        elapsed_seconds=perf_counter() - start,
        mesh=mesh,
        projected_points=np.asarray(mesh.vertices[projected_vertex_indices], dtype=float),
        point_vertex_indices=projected_vertex_indices,
        paths=paths,
        metadata={"pair_for_path": pair_for_path},
    )


def calculate_mesh_graph_distance_matrix(
    dendrite_mesh: Any,
    attachment_points: Sequence[Sequence[float]],
    pair_for_path: Optional[Tuple[int, int]] = None,
) -> DistanceMatrixResult:
    """Geodesic approximation as the shortest weighted path over original mesh edges."""
    mesh = polyhedron_to_trimesh(dendrite_mesh)
    points = _as_points(attachment_points)
    return _shortest_paths_on_mesh(mesh, points, "mesh_graph", pair_for_path)


def _point_to_array(point: Any) -> np.ndarray:
    return np.array([point.x(), point.y(), point.z()], dtype=float)


def _skeleton_segments(dendrite_mesh: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    skeleton_polylines = Polylines()
    correspondence_polylines = Polylines()
    surface_mesh_skeletonization(dendrite_mesh, skeleton_polylines, correspondence_polylines)

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for polyline in skeleton_polylines:
        for i in range(len(polyline) - 1):
            segments.append((_point_to_array(polyline[i]), _point_to_array(polyline[i + 1])))
    return segments


def _fallback_centerline(mesh: trimesh.Trimesh, samples: int = 32) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    axis = vh[0]
    positions = (vertices - center) @ axis
    line_values = np.linspace(positions.min(), positions.max(), samples)
    return center + line_values[:, None] * axis[None, :]


def _ordered_centerline_from_segments(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    fallback_mesh: trimesh.Trimesh,
) -> np.ndarray:
    if not segments:
        return _fallback_centerline(fallback_mesh)

    point_ids: Dict[Tuple[float, float, float], int] = {}
    points: List[np.ndarray] = []
    adjacency: Dict[int, List[Tuple[int, float]]] = {}

    def get_id(point: np.ndarray) -> int:
        key = tuple(np.round(point, 8).tolist())
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(point)
            adjacency[point_ids[key]] = []
        return point_ids[key]

    for start, end in segments:
        start_id = get_id(start)
        end_id = get_id(end)
        weight = float(np.linalg.norm(start - end))
        adjacency[start_id].append((end_id, weight))
        adjacency[end_id].append((start_id, weight))

    graph = csr_matrix(
        (
            [w for i, neighbours in adjacency.items() for _, w in neighbours],
            (
                [i for i, neighbours in adjacency.items() for _ in neighbours],
                [j for neighbours in adjacency.values() for j, _ in neighbours],
            ),
        ),
        shape=(len(points), len(points)),
    )

    endpoints = [idx for idx, neighbours in adjacency.items() if len(neighbours) == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(range(len(points)))
    dist, pred = dijkstra(graph, directed=False, indices=candidates, return_predecessors=True)

    best = np.unravel_index(np.nanargmax(np.where(np.isfinite(dist), dist, -np.inf)), dist.shape)
    source = candidates[int(best[0])]
    target = int(best[1])
    path_ids = _reconstruct_path(pred[int(best[0])], source, target)
    if len(path_ids) < 2:
        return _fallback_centerline(fallback_mesh)

    return np.asarray([points[idx] for idx in path_ids], dtype=float)


def _smooth_centerline(centerline: np.ndarray, iterations: int = 2) -> np.ndarray:
    smoothed = np.asarray(centerline, dtype=float).copy()
    if len(smoothed) < 3:
        return smoothed
    for _ in range(iterations):
        next_line = smoothed.copy()
        next_line[1:-1] = (smoothed[:-2] + smoothed[1:-1] + smoothed[2:]) / 3.0
        smoothed = next_line
    return smoothed


def _estimate_radius(mesh: trimesh.Trimesh, centerline: np.ndarray) -> float:
    tree = cKDTree(centerline)
    distances, _ = tree.query(mesh.vertices)
    radius = float(np.percentile(distances, 50))
    if not np.isfinite(radius) or radius <= 0:
        radius = float(np.mean(mesh.extents) / 10.0)
    return radius


def build_stem_surface_mesh(
    centerline: np.ndarray,
    radius: float,
    sections: int = 32,
) -> trimesh.Trimesh:
    centerline = _smooth_centerline(centerline)
    if len(centerline) < 2:
        raise ValueError("centerline must contain at least two points")

    tangents = np.zeros_like(centerline)
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    if len(centerline) > 2:
        tangents[1:-1] = centerline[2:] - centerline[:-2]
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    first_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(first_axis, tangents[0])) > 0.9:
        first_axis = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangents[0], first_axis)
    normal /= np.linalg.norm(normal)

    vertices = []
    previous_normal = normal
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    for center, tangent in zip(centerline, tangents):
        normal = previous_normal - np.dot(previous_normal, tangent) * tangent
        if np.linalg.norm(normal) <= 1e-12:
            normal = np.cross(tangent, first_axis)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal)
        previous_normal = normal

        for angle in angles:
            vertices.append(center + radius * (np.cos(angle) * normal + np.sin(angle) * binormal))

    faces = []
    for ring_idx in range(len(centerline) - 1):
        ring_start = ring_idx * sections
        next_ring_start = (ring_idx + 1) * sections
        for section_idx in range(sections):
            a = ring_start + section_idx
            b = ring_start + (section_idx + 1) % sections
            c = next_ring_start + section_idx
            d = next_ring_start + (section_idx + 1) % sections
            faces.append((a, c, b))
            faces.append((b, c, d))

    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def calculate_stem_graph_distance_matrix(
    dendrite_mesh: Any,
    attachment_points: Sequence[Sequence[float]],
    radius: Optional[float] = None,
    centerline: Optional[np.ndarray] = None,
    sections: int = 32,
    pair_for_path: Optional[Tuple[int, int]] = None,
) -> DistanceMatrixResult:
    """Build a smooth stem-like tube around the skeleton and run graph geodesics on it."""
    start = perf_counter()
    original_mesh = polyhedron_to_trimesh(dendrite_mesh)
    points = _as_points(attachment_points)

    if centerline is None:
        try:
            centerline = _ordered_centerline_from_segments(_skeleton_segments(dendrite_mesh), original_mesh)
        except Exception:
            centerline = _fallback_centerline(original_mesh)
    centerline = _smooth_centerline(np.asarray(centerline, dtype=float))

    if radius is None:
        radius = _estimate_radius(original_mesh, centerline)

    stem_mesh = build_stem_surface_mesh(centerline, radius, sections=sections)
    result = _shortest_paths_on_mesh(stem_mesh, points, "stem_graph", pair_for_path)
    result.elapsed_seconds += perf_counter() - start - result.elapsed_seconds
    result.metadata = {
        **(result.metadata or {}),
        "radius": radius,
        "sections": sections,
        "centerline": centerline,
    }
    return result


def _cotangent(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(np.cross(a, b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _cotangent_laplacian_and_mass(mesh: trimesh.Trimesh) -> Tuple[csr_matrix, csr_matrix]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    n_vertices = len(vertices)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    mass = np.zeros(n_vertices, dtype=float)

    for i, j, k in faces:
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        area = np.linalg.norm(np.cross(vj - vi, vk - vi)) / 2.0
        if area <= 1e-12:
            continue
        mass[[i, j, k]] += area / 3.0

        cot_i = _cotangent(vj - vi, vk - vi)
        cot_j = _cotangent(vi - vj, vk - vj)
        cot_k = _cotangent(vi - vk, vj - vk)
        for a, b, cot in ((j, k, cot_i), (i, k, cot_j), (i, j, cot_k)):
            weight = 0.5 * cot
            rows.extend([a, b, a, b])
            cols.extend([b, a, a, b])
            data.extend([-weight, -weight, weight, weight])

    laplacian = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    mass[mass <= 0] = np.mean(mass[mass > 0]) if np.any(mass > 0) else 1.0
    return laplacian, diags(mass).tocsr()


def _heat_distance_from_source(
    mesh: trimesh.Trimesh,
    source_vertex: int,
    laplacian: csr_matrix,
    mass: csr_matrix,
    t: float,
) -> np.ndarray:
    n_vertices = len(mesh.vertices)
    delta = np.zeros(n_vertices, dtype=float)
    delta[source_vertex] = 1.0

    heat = spsolve((mass + t * laplacian).tocsc(), delta)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    face_vectors = np.zeros((len(faces), 3), dtype=float)
    for face_index, (i, j, k) in enumerate(faces):
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        normal = np.cross(vj - vi, vk - vi)
        double_area = np.linalg.norm(normal)
        if double_area <= 1e-12:
            continue
        grad_i = np.cross(normal, vk - vj) / double_area
        grad_j = np.cross(normal, vi - vk) / double_area
        grad_k = np.cross(normal, vj - vi) / double_area
        grad_heat = heat[i] * grad_i + heat[j] * grad_j + heat[k] * grad_k
        norm = np.linalg.norm(grad_heat)
        if norm > 1e-12:
            face_vectors[face_index] = -grad_heat / norm

    divergence = np.zeros(n_vertices, dtype=float)
    for face_index, (i, j, k) in enumerate(faces):
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        vector = face_vectors[face_index]
        cot_i = _cotangent(vj - vi, vk - vi)
        cot_j = _cotangent(vi - vj, vk - vj)
        cot_k = _cotangent(vi - vk, vj - vk)

        divergence[i] += 0.5 * (cot_k * np.dot(vj - vi, vector) + cot_j * np.dot(vk - vi, vector))
        divergence[j] += 0.5 * (cot_k * np.dot(vi - vj, vector) + cot_i * np.dot(vk - vj, vector))
        divergence[k] += 0.5 * (cot_j * np.dot(vi - vk, vector) + cot_i * np.dot(vj - vk, vector))

    keep = np.ones(n_vertices, dtype=bool)
    keep[source_vertex] = False
    phi = np.zeros(n_vertices, dtype=float)
    phi[keep] = spsolve(laplacian[keep][:, keep].tocsc(), divergence[keep])
    phi -= np.nanmin(phi)
    if phi[source_vertex] != 0:
        phi -= phi[source_vertex]
    phi = np.abs(phi)
    phi[source_vertex] = 0.0
    return phi


def _adjacency_from_mesh(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    graph = _mesh_edge_graph(mesh).tocsr()
    return [graph.indices[graph.indptr[i]:graph.indptr[i + 1]] for i in range(graph.shape[0])]


def _heat_descent_path(
    mesh: trimesh.Trimesh,
    phi: np.ndarray,
    source_vertex: int,
    target_vertex: int,
) -> np.ndarray:
    adjacency = _adjacency_from_mesh(mesh)
    path = [target_vertex]
    current = target_vertex
    visited = {target_vertex}
    for _ in range(len(mesh.vertices)):
        if current == source_vertex:
            break
        neighbours = adjacency[current]
        if len(neighbours) == 0:
            break
        next_vertex = int(neighbours[np.argmin(phi[neighbours])])
        if next_vertex in visited or phi[next_vertex] > phi[current]:
            return np.empty((0, 3), dtype=float)
        path.append(next_vertex)
        visited.add(next_vertex)
        current = next_vertex
    if path[-1] != source_vertex:
        return np.empty((0, 3), dtype=float)
    return np.asarray(mesh.vertices[path[::-1]], dtype=float)


def calculate_heat_distance_matrix(
    dendrite_mesh: Any,
    attachment_points: Sequence[Sequence[float]],
    pair_for_path: Optional[Tuple[int, int]] = None,
) -> DistanceMatrixResult:
    """Heat Method geodesic approximation on the original dendrite mesh."""
    start = perf_counter()
    mesh = polyhedron_to_trimesh(dendrite_mesh)
    points = _as_points(attachment_points)
    projected_vertex_indices = _nearest_vertex_indices(mesh, points)

    laplacian, mass = _cotangent_laplacian_and_mass(mesh)
    edge_lengths = _mesh_edge_graph(mesh).data
    mean_edge = float(np.mean(edge_lengths)) if len(edge_lengths) else 1.0
    t = mean_edge ** 2

    distance_matrix = np.zeros((len(points), len(points)), dtype=float)
    heat_fields: Dict[int, np.ndarray] = {}
    for i, source_vertex in enumerate(projected_vertex_indices):
        phi = _heat_distance_from_source(mesh, int(source_vertex), laplacian, mass, t)
        heat_fields[i] = phi
        distance_matrix[i, :] = phi[projected_vertex_indices]
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    np.fill_diagonal(distance_matrix, 0.0)

    if pair_for_path is None:
        pair_for_path = _default_pair(distance_matrix)

    paths: Dict[Tuple[int, int], np.ndarray] = {}
    if len(points) >= 2:
        i, j = pair_for_path
        path = _heat_descent_path(
            mesh,
            heat_fields[i],
            int(projected_vertex_indices[i]),
            int(projected_vertex_indices[j]),
        )
        if len(path) == 0:
            graph_result = _shortest_paths_on_mesh(mesh, points, "mesh_graph", pair_for_path)
            path = next(iter((graph_result.paths or {}).values()), np.empty((0, 3)))
        if len(path) > 0:
            paths[(i, j)] = path

    return DistanceMatrixResult(
        method="heat",
        distance_matrix=distance_matrix,
        elapsed_seconds=perf_counter() - start,
        mesh=mesh,
        projected_points=np.asarray(mesh.vertices[projected_vertex_indices], dtype=float),
        point_vertex_indices=projected_vertex_indices,
        paths=paths,
        metadata={"pair_for_path": pair_for_path, "t": t},
    )


def _cylindrical_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    r = (np.sqrt(x1 ** 2 + y1 ** 2) + np.sqrt(x2 ** 2 + y2 ** 2)) / 2.0
    phi1 = np.arctan2(y1, x1)
    phi2 = np.arctan2(y2, x2)
    delta_phi = np.abs(phi1 - phi2)
    delta_phi = min(delta_phi, 2 * np.pi - delta_phi)
    return float(np.sqrt((r * delta_phi) ** 2 + (z1 - z2) ** 2))


def _cylinder_mesh_from_points(points: np.ndarray, sections: int = 64) -> trimesh.Trimesh:
    radii = np.linalg.norm(points[:, :2], axis=1)
    radius = float(np.mean(radii[radii > 0])) if np.any(radii > 0) else 1.0
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    height = max(z_max - z_min, radius)
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_translation((0, 0, (z_min + z_max) / 2.0))
    return mesh


def _cylindrical_path(point1: np.ndarray, point2: np.ndarray, steps: int = 80) -> np.ndarray:
    phi1 = np.arctan2(point1[1], point1[0])
    phi2 = np.arctan2(point2[1], point2[0])
    delta = (phi2 - phi1 + np.pi) % (2 * np.pi) - np.pi
    radii = np.linspace(np.linalg.norm(point1[:2]), np.linalg.norm(point2[:2]), steps)
    phis = phi1 + np.linspace(0, delta, steps)
    z = np.linspace(point1[2], point2[2], steps)
    return np.column_stack((radii * np.cos(phis), radii * np.sin(phis), z))


def calculate_cylindrical_distance_matrix(
    cylindrical_points: Sequence[Sequence[float]],
    pair_for_path: Optional[Tuple[int, int]] = None,
) -> DistanceMatrixResult:
    """Existing cylindrical distance model, kept as the baseline method."""
    start = perf_counter()
    points = _as_points(cylindrical_points)
    distance_matrix = squareform(pdist(points, lambda u, v: _cylindrical_distance(u, v)))
    if pair_for_path is None:
        pair_for_path = _default_pair(distance_matrix)

    paths: Dict[Tuple[int, int], np.ndarray] = {}
    if len(points) >= 2:
        i, j = pair_for_path
        paths[(i, j)] = _cylindrical_path(points[i], points[j])

    return DistanceMatrixResult(
        method="cylinder",
        distance_matrix=distance_matrix,
        elapsed_seconds=perf_counter() - start,
        mesh=_cylinder_mesh_from_points(points),
        projected_points=points,
        paths=paths,
        metadata={"pair_for_path": pair_for_path},
    )


def calculate_spine_distance_matrices(
    dendrite_mesh: Any,
    attachment_points: Sequence[Sequence[float]],
    cylindrical_points: Optional[Sequence[Sequence[float]]] = None,
    methods: Iterable[str] = ("cylinder", "stem_graph", "mesh_graph", "heat"),
    radius: Optional[float] = None,
    centerline: Optional[np.ndarray] = None,
    pair_for_path: Optional[Tuple[int, int]] = None,
) -> Dict[str, DistanceMatrixResult]:
    results: Dict[str, DistanceMatrixResult] = {}
    points = _as_points(attachment_points)

    for method in methods:
        if method == "cylinder":
            if cylindrical_points is None:
                continue
            results[method] = calculate_cylindrical_distance_matrix(cylindrical_points, pair_for_path)
        elif method == "stem_graph":
            results[method] = calculate_stem_graph_distance_matrix(
                dendrite_mesh, points, radius=radius, centerline=centerline, pair_for_path=pair_for_path
            )
        elif method == "mesh_graph":
            results[method] = calculate_mesh_graph_distance_matrix(dendrite_mesh, points, pair_for_path)
        elif method in {"heat", "heat_method"}:
            results["heat"] = calculate_heat_distance_matrix(dendrite_mesh, points, pair_for_path)
        else:
            raise ValueError(f"Unknown distance method: {method}")
    return results


def _plot_mesh(ax, mesh: trimesh.Trimesh, color: str = "#9aa3ad", alpha: float = 0.2) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    collection = Poly3DCollection(vertices[faces], alpha=alpha, linewidths=0.15)
    collection.set_facecolor(color)
    collection.set_edgecolor("#6b7280")
    ax.add_collection3d(collection)


def _set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualize_distance_result(
    original_mesh: Any,
    attachment_points: Sequence[Sequence[float]],
    result: DistanceMatrixResult,
    pair: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    original = polyhedron_to_trimesh(original_mesh)
    points = _as_points(attachment_points)
    if pair is None:
        pair = (result.metadata or {}).get("pair_for_path") or _default_pair(result.distance_matrix)

    fig = plt.figure(figsize=(14, 6))
    left = fig.add_subplot(1, 2, 1, projection="3d")
    right = fig.add_subplot(1, 2, 2, projection="3d")

    _plot_mesh(left, original, alpha=0.18)
    left.scatter(points[:, 0], points[:, 1], points[:, 2], c="#1f77b4", s=24)
    left.scatter(points[list(pair), 0], points[list(pair), 1], points[list(pair), 2], c="#d62728", s=55)
    left.set_title("Original dendrite mesh")
    _set_axes_equal(left, np.vstack((original.vertices, points)))

    method_mesh = result.mesh if result.mesh is not None else original
    _plot_mesh(right, method_mesh, alpha=0.18)
    projected = result.projected_points if result.projected_points is not None else points
    right.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c="#1f77b4", s=24)
    right.scatter(projected[list(pair), 0], projected[list(pair), 1], projected[list(pair), 2], c="#d62728", s=55)

    path = (result.paths or {}).get(tuple(pair))
    if path is not None and len(path) > 0:
        right.plot(path[:, 0], path[:, 1], path[:, 2], color="#d62728", linewidth=2.5)

    distance = result.distance_matrix[pair[0], pair[1]] if result.distance_matrix.size else np.nan
    right.set_title(f"{result.method}: d={distance:.3f}, t={result.elapsed_seconds:.3f}s")
    _set_axes_equal(right, np.vstack((method_mesh.vertices, projected)))

    for ax in (left, right):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def summarize_distance_results(results: Dict[str, DistanceMatrixResult]) -> pd.DataFrame:
    rows = []
    for method, result in results.items():
        matrix = result.distance_matrix
        upper = matrix[np.triu_indices_from(matrix, k=1)] if matrix.size else np.array([])
        rows.append(
            {
                "method": method,
                "elapsed_seconds": result.elapsed_seconds,
                "n_points": matrix.shape[0],
                "mean_distance": float(np.mean(upper)) if len(upper) else np.nan,
                "median_distance": float(np.median(upper)) if len(upper) else np.nan,
                "max_distance": float(np.max(upper)) if len(upper) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("elapsed_seconds")


def save_distance_method_comparison(
    dendrite_mesh: Any,
    attachment_points: Sequence[Sequence[float]],
    cylindrical_points: Optional[Sequence[Sequence[float]]] = None,
    output_dir: str = "output_dendrite_distance_comparison",
    methods: Iterable[str] = ("cylinder", "stem_graph", "mesh_graph", "heat"),
    radius: Optional[float] = None,
    centerline: Optional[np.ndarray] = None,
    pair_for_path: Optional[Tuple[int, int]] = None,
) -> Dict[str, DistanceMatrixResult]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = calculate_spine_distance_matrices(
        dendrite_mesh=dendrite_mesh,
        attachment_points=attachment_points,
        cylindrical_points=cylindrical_points,
        methods=methods,
        radius=radius,
        centerline=centerline,
        pair_for_path=pair_for_path,
    )

    summary = summarize_distance_results(results)
    summary.to_csv(output_path / "distance_methods_summary.csv", index=False)

    original = polyhedron_to_trimesh(dendrite_mesh)
    for method, result in results.items():
        np.savetxt(output_path / f"{method}_distance_matrix.csv", result.distance_matrix, delimiter=",")
        visualize_distance_result(
            original,
            attachment_points,
            result,
            save_path=str(output_path / f"{method}_visualization.png"),
        )
        plt.close("all")

    return results
