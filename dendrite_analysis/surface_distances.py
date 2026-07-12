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

from CGAL.CGAL_Polygon_mesh_processing import Polylines, does_self_intersect
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization
from spine_analysis.mesh.utils import _mesh_to_v_f

try:
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


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
    predecessors: Optional[np.ndarray] = None
    heat_fields: Optional[Dict[int, np.ndarray]] = None


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


def _min_pair(distance_matrix: np.ndarray) -> Tuple[int, int]:
    """Return (i, j) with the smallest non-zero finite distance."""
    if distance_matrix.shape[0] < 2:
        return 0, 0
    m = np.where(np.isfinite(distance_matrix) & (distance_matrix > 0), distance_matrix, np.inf)
    np.fill_diagonal(m, np.inf)
    if not np.isfinite(m).any():
        return 0, 1
    pair = np.unravel_index(np.argmin(m), m.shape)
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
        predecessors=predecessors,
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
    # CGAL's mean-curvature-flow skeletonization requires a closed, manifold,
    # non-self-intersecting triangle mesh. Feeding it anything else can hard-crash
    # the interpreter (a native abort/segfault, not a catchable exception), so we
    # check the documented preconditions ourselves and fail in plain Python instead
    # — the caller already falls back to `_fallback_centerline` on any exception here.
    if not bool(dendrite_mesh.is_closed()):
        raise RuntimeError(
            "surface_mesh_skeletonization requires a closed (watertight) mesh; "
            "the input mesh has open boundaries."
        )
    if bool(does_self_intersect(dendrite_mesh)):
        raise RuntimeError(
            "surface_mesh_skeletonization requires a non-self-intersecting mesh; "
            "the input mesh has self-intersecting faces."
        )

    skeleton_polylines = Polylines()
    correspondence_polylines = Polylines()
    surface_mesh_skeletonization(dendrite_mesh, skeleton_polylines, correspondence_polylines)

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for polyline in skeleton_polylines:
        for i in range(len(polyline) - 1):
            segments.append((_point_to_array(polyline[i]), _point_to_array(polyline[i + 1])))
    return segments


def _pca_centerline(mesh: trimesh.Trimesh, samples: int = 32) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=float)
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    axis = vh[0]
    positions = (vertices - center) @ axis
    line_values = np.linspace(positions.min(), positions.max(), samples)
    return center + line_values[:, None] * axis[None, :]


def _graph_diameter_centerline(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 2:
        return vertices.copy()

    graph = _mesh_edge_graph(mesh)
    if graph.shape[0] < 2 or graph.nnz == 0:
        return _pca_centerline(mesh)

    seed = int(np.argmax(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)))
    distances = dijkstra(graph, directed=False, indices=seed)
    finite = np.isfinite(distances)
    if not np.any(finite):
        return _pca_centerline(mesh)
    endpoint_a = int(np.argmax(np.where(finite, distances, -np.inf)))

    distances, predecessors = dijkstra(
        graph,
        directed=False,
        indices=endpoint_a,
        return_predecessors=True,
    )
    finite = np.isfinite(distances)
    if not np.any(finite):
        return _pca_centerline(mesh)
    endpoint_b = int(np.argmax(np.where(finite, distances, -np.inf)))

    path_ids = _reconstruct_path(predecessors, endpoint_a, endpoint_b)
    if len(path_ids) < 2:
        return _pca_centerline(mesh)
    return vertices[np.asarray(path_ids, dtype=int)]


def _fallback_centerline(mesh: trimesh.Trimesh, samples: int = 32) -> np.ndarray:
    try:
        centerline = _graph_diameter_centerline(mesh)
        if len(centerline) >= 2:
            return centerline
    except Exception:
        pass
    return _pca_centerline(mesh, samples=samples)


def centerline_length_from_mesh(mesh: Any) -> float:
    tm = polyhedron_to_trimesh(mesh)
    centerline = _fallback_centerline(tm)
    if len(centerline) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())


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
        heat_fields=heat_fields,
        metadata={"pair_for_path": pair_for_path, "t": t},
    )


def _recompute_path_for_result(
    result: DistanceMatrixResult,
    pair: Tuple[int, int],
) -> Optional[np.ndarray]:
    i, j = pair
    if result.method in ("mesh_graph", "stem_graph"):
        if (
            result.predecessors is not None
            and result.point_vertex_indices is not None
            and result.mesh is not None
        ):
            path_vertices = _reconstruct_path(
                result.predecessors[i],
                int(result.point_vertex_indices[i]),
                int(result.point_vertex_indices[j]),
            )
            if len(path_vertices) > 0:
                return np.asarray(result.mesh.vertices[path_vertices], dtype=float)
    elif result.method == "heat":
        if (
            result.heat_fields is not None
            and result.point_vertex_indices is not None
            and result.mesh is not None
        ):
            phi = result.heat_fields.get(i)
            if phi is not None:
                path = _heat_descent_path(
                    result.mesh,
                    phi,
                    int(result.point_vertex_indices[i]),
                    int(result.point_vertex_indices[j]),
                )
                if len(path) > 0:
                    return path
    elif result.method == "cylinder":
        raw = (result.metadata or {}).get("cylindrical_points")
        if raw is not None:
            raw = np.asarray(raw)
            n = len(raw)
            if i < n and j < n:
                local_path = _cylindrical_path(raw[i], raw[j])
                pca_basis = (result.metadata or {}).get("pca_basis")
                if pca_basis is not None:
                    w, u_ax, v_ax, mean_3d = pca_basis
                    return _cylinder_local_to_3d(local_path, w, u_ax, v_ax, mean_3d)
                return local_path
    return None


def select_shared_pair(
    results: Dict[str, DistanceMatrixResult],
    preferred_methods: Sequence[str] = ("mesh_graph", "stem_graph", "cylinder"),
) -> Tuple[int, int]:
    """Pick a single representative point pair from the best available method."""
    for method in preferred_methods:
        if method in results:
            return _default_pair(results[method].distance_matrix)
    if results:
        return _default_pair(next(iter(results.values())).distance_matrix)
    return (0, 1)


def _cylindrical_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    # Input format: (r, h, theta) — radius, height along dendrite axis, angle
    r1, h1, theta1 = point1
    r2, h2, theta2 = point2
    r_avg = (r1 + r2) / 2.0
    delta_theta = abs(theta1 - theta2)
    delta_theta = min(delta_theta, 2 * np.pi - delta_theta)
    arc = r_avg * delta_theta
    return float(np.sqrt(arc ** 2 + (h1 - h2) ** 2))


def _cylinder_mesh_from_points(points: np.ndarray, sections: int = 64) -> trimesh.Trimesh:
    # Input format: (r, h, theta) — radius col 0, height col 1, angle col 2
    radius = float(np.mean(points[:, 0])) if len(points) else 1.0
    h_min = float(np.min(points[:, 1]))
    h_max = float(np.max(points[:, 1]))
    height = max(h_max - h_min, radius)
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_translation((0, 0, (h_min + h_max) / 2.0))
    return mesh


def _cylindrical_path(point1: np.ndarray, point2: np.ndarray, steps: int = 80) -> np.ndarray:
    # Input format: (r, h, theta) — radius, height along dendrite axis, angle
    r1, h1, theta1 = point1
    r2, h2, theta2 = point2
    delta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    radii = np.linspace(r1, r2, steps)
    thetas = theta1 + np.linspace(0, delta, steps)
    heights = np.linspace(h1, h2, steps)
    # Local Cartesian: x=r·cosθ (along u), y=r·sinθ (along v), z=h (along w)
    return np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas), heights))


def _cylinder_local_to_3d(
    path_local: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Map local cylinder Cartesian (r·cosθ, r·sinθ, h) to 3D world space."""
    return (
        mean[np.newaxis, :]
        + np.outer(path_local[:, 0], u)
        + np.outer(path_local[:, 1], v)
        + np.outer(path_local[:, 2], w)
    )


def _make_oriented_cylinder(
    raw_points: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mean: np.ndarray,
    sections: int = 64,
) -> trimesh.Trimesh:
    """Create a cylinder aligned with dendrite axis *w*, centred at *mean*."""
    radius = float(np.mean(raw_points[:, 0])) if len(raw_points) else 1.0
    h_min = float(np.min(raw_points[:, 1]))
    h_max = float(np.max(raw_points[:, 1]))
    height = max(h_max - h_min, radius)
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    # trimesh creates a Z-aligned cylinder; rotate so Z maps to w (dendrite axis)
    rot = np.column_stack([u, v, w])
    transform = np.eye(4)
    transform[:3, :3] = rot
    mesh.apply_transform(transform)
    mesh.apply_translation(mean + (h_min + h_max) / 2.0 * w)
    return mesh


def calculate_cylindrical_distance_matrix(
    cylindrical_points: Sequence[Sequence[float]],
    pair_for_path: Optional[Tuple[int, int]] = None,
    original_points: Optional[Sequence[Sequence[float]]] = None,
) -> DistanceMatrixResult:
    start = perf_counter()
    raw_points = _as_points(cylindrical_points)  # (r, h, θ)
    distance_matrix = squareform(pdist(raw_points, lambda u, v: _cylindrical_distance(u, v)))
    if pair_for_path is None:
        pair_for_path = _default_pair(distance_matrix)

    # Optionally recompute PCA basis from original 3D points for correct orientation
    pca_basis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if original_points is not None:
        orig = np.asarray(original_points, dtype=float)
        if orig.ndim == 2 and orig.shape[1] == 3 and len(orig) >= 3:
            try:
                from sklearn.decomposition import PCA as _PCA
                _pca = _PCA(n_components=3, random_state=42)
                _pca.fit(orig)
                w = _pca.components_[0]
                u_ax = _pca.components_[1]
                v_ax = _pca.components_[2]
                mean_3d = _pca.mean_
                pca_basis = (w, u_ax, v_ax, mean_3d)
            except Exception:
                pca_basis = None

    # Local Cartesian: x = r·cosθ, y = r·sinθ, z = h
    cart_local = np.column_stack((
        raw_points[:, 0] * np.cos(raw_points[:, 2]),
        raw_points[:, 0] * np.sin(raw_points[:, 2]),
        raw_points[:, 1],
    ))

    if pca_basis is not None:
        w, u_ax, v_ax, mean_3d = pca_basis
        cart_points = _cylinder_local_to_3d(cart_local, w, u_ax, v_ax, mean_3d)
        cyl_mesh = _make_oriented_cylinder(raw_points, w, u_ax, v_ax, mean_3d)
    else:
        cart_points = cart_local
        cyl_mesh = _cylinder_mesh_from_points(raw_points)

    paths: Dict[Tuple[int, int], np.ndarray] = {}
    if len(raw_points) >= 2:
        i, j = pair_for_path
        local_path = _cylindrical_path(raw_points[i], raw_points[j])
        if pca_basis is not None:
            w, u_ax, v_ax, mean_3d = pca_basis
            paths[(i, j)] = _cylinder_local_to_3d(local_path, w, u_ax, v_ax, mean_3d)
        else:
            paths[(i, j)] = local_path

    meta: Dict[str, Any] = {
        "pair_for_path": pair_for_path,
        "cylindrical_points": raw_points,
    }
    if pca_basis is not None:
        meta["pca_basis"] = pca_basis

    return DistanceMatrixResult(
        method="cylinder",
        distance_matrix=distance_matrix,
        elapsed_seconds=perf_counter() - start,
        mesh=cyl_mesh,
        projected_points=cart_points,
        paths=paths,
        metadata=meta,
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

    methods_list = list(methods)
    if _TQDM_AVAILABLE:
        methods_iter: Any = _tqdm(methods_list, desc="distance methods", unit="method", leave=False)
    else:
        methods_iter = methods_list

    for method in methods_iter:
        if method == "cylinder":
            if cylindrical_points is None:
                continue
            results[method] = calculate_cylindrical_distance_matrix(
                cylindrical_points, pair_for_path, original_points=points
            )
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
    min_pair: Optional[Tuple[int, int]] = None,
    max_pair: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Three-panel matplotlib figure: plain mesh | min-distance path (green) | max-distance path (red)."""
    original = polyhedron_to_trimesh(original_mesh)
    points = _as_points(attachment_points)

    if min_pair is None:
        min_pair = _min_pair(result.distance_matrix)
    if max_pair is None:
        max_pair = _default_pair(result.distance_matrix)

    method_mesh = result.mesh if result.mesh is not None else original
    projected = result.projected_points if result.projected_points is not None else points

    min_dist = result.distance_matrix[min_pair[0], min_pair[1]] if result.distance_matrix.size else np.nan
    max_dist = result.distance_matrix[max_pair[0], max_pair[1]] if result.distance_matrix.size else np.nan

    fig = plt.figure(figsize=(21, 6))
    ax_mesh = fig.add_subplot(1, 3, 1, projection="3d")
    ax_min  = fig.add_subplot(1, 3, 2, projection="3d")
    ax_max  = fig.add_subplot(1, 3, 3, projection="3d")

    # Panel 1 — plain dendrite mesh
    _plot_mesh(ax_mesh, original, alpha=0.22)
    ax_mesh.set_title("Dendrite mesh")
    _set_axes_equal(ax_mesh, original.vertices)

    # Panel 2 — minimum distance pair (green)
    _plot_mesh(ax_min, method_mesh, alpha=0.18)
    ax_min.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c="#1f77b4", s=24)
    ax_min.scatter(
        projected[list(min_pair), 0], projected[list(min_pair), 1], projected[list(min_pair), 2],
        c="#2ca02c", s=65,
    )
    min_path = (result.paths or {}).get(tuple(min_pair))
    if min_path is not None and len(min_path) > 0:
        ax_min.plot(min_path[:, 0], min_path[:, 1], min_path[:, 2], color="#2ca02c", linewidth=2.5)
    ax_min.set_title(f"{result.method} — min d={min_dist:.3f}")
    _set_axes_equal(ax_min, np.vstack((method_mesh.vertices, projected)))

    # Panel 3 — maximum distance pair (red)
    _plot_mesh(ax_max, method_mesh, alpha=0.18)
    ax_max.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c="#1f77b4", s=24)
    ax_max.scatter(
        projected[list(max_pair), 0], projected[list(max_pair), 1], projected[list(max_pair), 2],
        c="#d62728", s=65,
    )
    max_path = (result.paths or {}).get(tuple(max_pair))
    if max_path is not None and len(max_path) > 0:
        ax_max.plot(max_path[:, 0], max_path[:, 1], max_path[:, 2], color="#d62728", linewidth=2.5)
    ax_max.set_title(f"{result.method} — max d={max_dist:.3f}, t={result.elapsed_seconds:.3f}s")
    _set_axes_equal(ax_max, np.vstack((method_mesh.vertices, projected)))

    for ax in (ax_mesh, ax_min, ax_max):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def visualize_distance_result_3d(
    original_mesh: Any,
    attachment_points: Sequence[Sequence[float]],
    result: DistanceMatrixResult,
    min_pair: Optional[Tuple[int, int]] = None,
    max_pair: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """Interactive Plotly 3-D figure: plain mesh | min-distance path (green) | max-distance path (red)."""
    if not _PLOTLY_AVAILABLE:
        return None

    original = polyhedron_to_trimesh(original_mesh)
    points = _as_points(attachment_points)

    if min_pair is None:
        min_pair = _min_pair(result.distance_matrix)
    if max_pair is None:
        max_pair = _default_pair(result.distance_matrix)

    method_mesh = result.mesh if result.mesh is not None else original
    projected = result.projected_points if result.projected_points is not None else points
    min_dist = result.distance_matrix[min_pair[0], min_pair[1]] if result.distance_matrix.size else float("nan")
    max_dist = result.distance_matrix[max_pair[0], max_pair[1]] if result.distance_matrix.size else float("nan")

    fig = _make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Dendrite mesh",
            f"{result.method} — min d={min_dist:.3f}",
            f"{result.method} — max d={max_dist:.3f}, t={result.elapsed_seconds:.3f}s",
        ),
    )

    def _add_mesh(mesh: trimesh.Trimesh, col: int, color: str = "lightgray") -> None:
        v = np.asarray(mesh.vertices, dtype=float)
        f = np.asarray(mesh.faces, dtype=int)
        fig.add_trace(
            _go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                opacity=0.25, color=color, showscale=False,
                hoverinfo="skip", name="mesh",
            ),
            row=1, col=col,
        )

    # Col 1 — plain dendrite mesh
    _add_mesh(original, col=1)

    # Col 2 — minimum distance pair (green)
    _add_mesh(method_mesh, col=2, color="lightsteelblue")
    fig.add_trace(
        _go.Scatter3d(
            x=projected[:, 0], y=projected[:, 1], z=projected[:, 2],
            mode="markers", marker=dict(size=4, color="steelblue"),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        _go.Scatter3d(
            x=projected[list(min_pair), 0], y=projected[list(min_pair), 1], z=projected[list(min_pair), 2],
            mode="markers", marker=dict(size=10, color="green"),
            showlegend=False,
        ),
        row=1, col=2,
    )
    min_path = (result.paths or {}).get(tuple(min_pair))
    if min_path is not None and len(min_path) > 0:
        fig.add_trace(
            _go.Scatter3d(
                x=min_path[:, 0], y=min_path[:, 1], z=min_path[:, 2],
                mode="lines", line=dict(color="green", width=6),
                showlegend=False,
            ),
            row=1, col=2,
        )

    # Col 3 — maximum distance pair (red)
    _add_mesh(method_mesh, col=3, color="lightsteelblue")
    fig.add_trace(
        _go.Scatter3d(
            x=projected[:, 0], y=projected[:, 1], z=projected[:, 2],
            mode="markers", marker=dict(size=4, color="steelblue"),
            showlegend=False,
        ),
        row=1, col=3,
    )
    fig.add_trace(
        _go.Scatter3d(
            x=projected[list(max_pair), 0], y=projected[list(max_pair), 1], z=projected[list(max_pair), 2],
            mode="markers", marker=dict(size=10, color="crimson"),
            showlegend=False,
        ),
        row=1, col=3,
    )
    max_path = (result.paths or {}).get(tuple(max_pair))
    if max_path is not None and len(max_path) > 0:
        fig.add_trace(
            _go.Scatter3d(
                x=max_path[:, 0], y=max_path[:, 1], z=max_path[:, 2],
                mode="lines", line=dict(color="crimson", width=6),
                showlegend=False,
            ),
            row=1, col=3,
        )

    fig.update_layout(
        title=f"{result.method} — min d={min_dist:.4f}  |  max d={max_dist:.4f}  |  t={result.elapsed_seconds:.3f}s",
        height=620,
        margin=dict(l=0, r=0, b=0, t=70),
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)

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

    # Compute per-result min and max pairs and their paths
    for result in results.values():
        min_p = _min_pair(result.distance_matrix)
        max_p = _default_pair(result.distance_matrix)
        min_path = _recompute_path_for_result(result, min_p)
        max_path = _recompute_path_for_result(result, max_p)
        result.paths = {}
        if min_path is not None:
            result.paths[min_p] = min_path
        if max_path is not None:
            result.paths[max_p] = max_path
        if result.metadata is None:
            result.metadata = {}
        result.metadata["min_pair"] = min_p
        result.metadata["max_pair"] = max_p

    summary = summarize_distance_results(results)
    summary.to_csv(output_path / "distance_methods_summary.csv", index=False)

    original = polyhedron_to_trimesh(dendrite_mesh)
    for method, result in results.items():
        np.savetxt(output_path / f"{method}_distance_matrix.csv", result.distance_matrix, delimiter=",")
        min_p = (result.metadata or {}).get("min_pair")
        max_p = (result.metadata or {}).get("max_pair")
        visualize_distance_result(
            original,
            attachment_points,
            result,
            min_pair=min_p,
            max_pair=max_p,
            save_path=str(output_path / f"{method}_visualization.png"),
        )
        visualize_distance_result_3d(
            original,
            attachment_points,
            result,
            min_pair=min_p,
            max_pair=max_p,
            save_path=str(output_path / f"{method}_visualization_3d.html"),
        )
        plt.close("all")

    return results
