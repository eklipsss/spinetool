from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

Segment = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class BoundaryLoop:
    edges: np.ndarray
    vertex_indices: np.ndarray
    ordered_vertex_indices: np.ndarray
    is_closed: bool
    n_edges: int
    perimeter: float
    area: float
    center: np.ndarray
    distance_to_skeleton: float


def require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "trimesh is required for spine mesh preprocessing. "
            "Install the project requirements before running mesh stages."
        ) from exc
    return trimesh


def load_trimesh(path: Path, *, process: bool = False):
    trimesh = require_trimesh()
    mesh = trimesh.load_mesh(str(path), process=process)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Mesh scene has no geometry: {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh for {path}, got {type(mesh).__name__}")
    return mesh


def atomic_export_mesh(mesh: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    mesh.export(str(tmp_path), file_type=output_path.suffix.lstrip(".") or None)
    tmp_path.replace(output_path)


def boundary_edge_count(mesh: Any) -> int:
    if len(mesh.faces) == 0:
        return 0
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def count_degenerate_faces(mesh: Any, area_tol: float = 1e-12) -> int:
    if len(mesh.faces) == 0:
        return 0
    return int(np.sum(np.asarray(mesh.area_faces) <= area_tol))


def connected_components_count(mesh: Any) -> int:
    try:
        return int(len(mesh.split(only_watertight=False)))
    except Exception:
        return 0


def mesh_qc_summary(mesh: Any) -> Dict[str, Any]:
    volume: Optional[float] = None
    if bool(getattr(mesh, "is_watertight", False)):
        try:
            volume = float(mesh.volume)
        except Exception:
            volume = None
    return {
        "is_watertight": bool(getattr(mesh, "is_watertight", False)),
        "is_manifold": bool(getattr(mesh, "is_winding_consistent", False)),
        "n_boundary_edges": boundary_edge_count(mesh),
        "n_connected_components": connected_components_count(mesh),
        "n_degenerate_faces": count_degenerate_faces(mesh),
        "has_finite_coordinates": bool(np.isfinite(np.asarray(mesh.vertices, dtype=float)).all()),
        "volume": volume,
        "has_positive_finite_volume": bool(volume is not None and math.isfinite(volume) and volume > 0),
    }


def _boundary_edges(mesh: Any) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces) == 0:
        return np.empty((0, 2), dtype=int)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return unique_edges[counts == 1].astype(int)


def _order_component_vertices(edges: np.ndarray) -> Tuple[np.ndarray, bool]:
    if len(edges) == 0:
        return np.empty((0,), dtype=int), False

    adjacency: Dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    is_closed = len(adjacency) >= 3 and all(len(neighbors) == 2 for neighbors in adjacency.values())
    start = min(adjacency)
    if not is_closed:
        endpoints = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
        if endpoints:
            start = min(endpoints)

    ordered = [start]
    prev: Optional[int] = None
    current = start
    visited_edges = set()
    for _ in range(len(edges) + 1):
        candidates = sorted(adjacency[current])
        next_node = None
        for candidate in candidates:
            edge_key = tuple(sorted((current, candidate)))
            if edge_key in visited_edges:
                continue
            next_node = candidate
            visited_edges.add(edge_key)
            break
        if next_node is None:
            break
        if next_node == start:
            break
        ordered.append(next_node)
        prev, current = current, next_node
        if prev == current:
            break

    return np.asarray(ordered, dtype=int), is_closed


def _loop_area(points: np.ndarray, is_closed: bool) -> float:
    if not is_closed or len(points) < 3:
        return 0.0
    center = points.mean(axis=0)
    normal = np.zeros(3, dtype=float)
    centered = points - center
    for idx in range(len(centered)):
        normal += np.cross(centered[idx], centered[(idx + 1) % len(centered)])
    return float(0.5 * np.linalg.norm(normal))


def _loop_perimeter(points: np.ndarray, is_closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    perimeter = float(np.sum(distances))
    if is_closed:
        perimeter += float(np.linalg.norm(points[0] - points[-1]))
    return perimeter


def point_to_segment_projection(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> Tuple[np.ndarray, float, float]:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vector = end - start
    denom = float(np.dot(vector, vector))
    if denom <= 0:
        return start.copy(), 0.0, float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, vector) / denom, 0.0, 1.0))
    projection = start + t * vector
    return projection, t, float(np.linalg.norm(point - projection))


def closest_point_on_segments(point: np.ndarray, segments: Sequence[Segment]) -> Tuple[np.ndarray, Segment, float, float]:
    if not segments:
        raise ValueError("No skeleton segments were provided.")
    best_projection: Optional[np.ndarray] = None
    best_segment: Optional[Segment] = None
    best_t = 0.0
    best_distance = math.inf
    for start, end in segments:
        projection, t, distance = point_to_segment_projection(point, start, end)
        if distance < best_distance:
            best_projection = projection
            best_segment = (np.asarray(start, dtype=float), np.asarray(end, dtype=float))
            best_t = t
            best_distance = distance
    assert best_projection is not None and best_segment is not None
    return best_projection, best_segment, best_t, best_distance


def distance_to_segments(point: np.ndarray, segments: Sequence[Segment]) -> float:
    return closest_point_on_segments(point, segments)[3]


def boundary_loops(mesh: Any, branch_segments: Optional[Sequence[Segment]] = None) -> List[BoundaryLoop]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = _boundary_edges(mesh)
    if len(edges) == 0:
        return []

    vertex_to_edge_ids: Dict[int, List[int]] = {}
    for edge_id, (start, end) in enumerate(edges):
        vertex_to_edge_ids.setdefault(int(start), []).append(edge_id)
        vertex_to_edge_ids.setdefault(int(end), []).append(edge_id)

    unvisited = set(range(len(edges)))
    loops: List[BoundaryLoop] = []
    while unvisited:
        seed = unvisited.pop()
        component_edge_ids = {seed}
        stack = [seed]
        while stack:
            edge_id = stack.pop()
            for vertex_id in edges[edge_id]:
                for next_edge_id in vertex_to_edge_ids.get(int(vertex_id), []):
                    if next_edge_id in unvisited:
                        unvisited.remove(next_edge_id)
                        component_edge_ids.add(next_edge_id)
                        stack.append(next_edge_id)

        component_edges = edges[np.asarray(sorted(component_edge_ids), dtype=int)]
        vertex_indices = np.unique(component_edges.reshape(-1)).astype(int)
        ordered_indices, is_closed = _order_component_vertices(component_edges)
        if len(ordered_indices) == 0:
            ordered_indices = vertex_indices
        ordered_points = vertices[ordered_indices]
        loop_points = vertices[vertex_indices]
        center = loop_points.mean(axis=0) if len(loop_points) else np.zeros(3, dtype=float)
        perimeter = _loop_perimeter(ordered_points, is_closed)
        area = _loop_area(ordered_points, is_closed)
        skel_distance = (
            distance_to_segments(center, branch_segments)
            if branch_segments
            else math.inf
        )
        loops.append(
            BoundaryLoop(
                edges=component_edges,
                vertex_indices=vertex_indices,
                ordered_vertex_indices=ordered_indices,
                is_closed=bool(is_closed),
                n_edges=int(len(component_edges)),
                perimeter=perimeter,
                area=area,
                center=center,
                distance_to_skeleton=float(skel_distance),
            )
        )

    return loops


def select_attachment_loop(mesh: Any, branch_segments: Sequence[Segment]) -> BoundaryLoop:
    loops = boundary_loops(mesh, branch_segments)
    if not loops:
        raise ValueError("No boundary loops were found in the spine mesh.")

    perimeters = np.asarray([loop.perimeter for loop in loops], dtype=float)
    areas = np.asarray([loop.area for loop in loops], dtype=float)
    distances = np.asarray([loop.distance_to_skeleton for loop in loops], dtype=float)

    def normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
        finite = np.isfinite(values)
        if not finite.any():
            return np.zeros_like(values, dtype=float)
        result = np.zeros_like(values, dtype=float)
        lo = float(np.min(values[finite]))
        hi = float(np.max(values[finite]))
        if hi - lo <= 1e-12:
            result[finite] = 1.0
        else:
            result[finite] = (values[finite] - lo) / (hi - lo)
        if invert:
            result[finite] = 1.0 - result[finite]
        return result

    score = (
        0.4 * normalize(perimeters)
        + 0.35 * normalize(areas)
        + 0.25 * normalize(distances, invert=True)
    )
    closed_bonus = np.asarray([0.05 if loop.is_closed else 0.0 for loop in loops], dtype=float)
    return loops[int(np.argmax(score + closed_bonus))]


def _unit(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if math.isfinite(norm) and norm > 1e-12:
        return vector / norm
    if fallback is None:
        raise ValueError("Cannot normalize a zero-length vector.")
    return _unit(fallback)


def _loop_plane_normal(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    if len(centered) < 3:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return _unit(vh[-1], fallback=np.array([0.0, 1.0, 0.0], dtype=float))


def local_spine_frame(mesh: Any, branch_segments: Sequence[Segment], loop: BoundaryLoop) -> Dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    origin = np.asarray(loop.center, dtype=float)
    projection, nearest_segment, _, _ = closest_point_on_segments(origin, branch_segments)

    tangent = _unit(nearest_segment[1] - nearest_segment[0], fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    radial = origin - projection
    loop_points = vertices[loop.ordered_vertex_indices]
    loop_normal = _loop_plane_normal(loop_points)
    radial = radial - np.dot(radial, tangent) * tangent
    if np.linalg.norm(radial) <= 1e-12:
        radial = loop_normal - np.dot(loop_normal, tangent) * tangent
    radial = _unit(radial, fallback=np.array([0.0, 1.0, 0.0], dtype=float))

    tip_direction = vertices.mean(axis=0) - origin
    if np.dot(radial, tip_direction) < 0:
        radial = -radial

    binormal = _unit(np.cross(tangent, radial), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    radial = _unit(np.cross(binormal, tangent), fallback=radial)

    basis = np.vstack([tangent, radial, binormal])
    local_vertices = (vertices - origin) @ basis.T
    return {
        "origin_global": origin,
        "tangent": tangent,
        "radial": radial,
        "binormal": binormal,
        "basis_rows": basis,
        "vertices_local": local_vertices,
        "branch_projection": projection,
        "attachment_loop": loop,
    }


def graph_from_segments(segments: Iterable[Segment], ndigits: int = 6) -> nx.Graph:
    graph = nx.Graph()
    key_to_node: Dict[Tuple[float, float, float], int] = {}

    def node_for(point: np.ndarray) -> int:
        key = tuple(np.round(np.asarray(point, dtype=float), ndigits).tolist())
        if key not in key_to_node:
            node_id = len(key_to_node)
            key_to_node[key] = node_id
            graph.add_node(node_id, pos=np.asarray(point, dtype=float))
        return key_to_node[key]

    for start, end in segments:
        u = node_for(start)
        v = node_for(end)
        if u == v:
            continue
        length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
        if length <= 0:
            continue
        if graph.has_edge(u, v):
            graph[u][v]["length"] = min(float(graph[u][v]["length"]), length)
        else:
            graph.add_edge(u, v, length=length)
    return graph


def longest_path_polyline(segments: Sequence[Segment]) -> np.ndarray:
    graph = graph_from_segments(segments)
    if graph.number_of_nodes() < 2:
        raise RuntimeError("Skeleton graph has fewer than two nodes.")

    best_length = -math.inf
    best_path: Optional[List[int]] = None
    for component_nodes in nx.connected_components(graph):
        sub = graph.subgraph(component_nodes)
        terminals = [node for node in sub.nodes if sub.degree(node) == 1]
        candidates = terminals if len(terminals) >= 2 else list(sub.nodes)
        for index, source in enumerate(candidates):
            lengths, paths = nx.single_source_dijkstra(sub, source, weight="length")
            for target in candidates[index + 1:]:
                distance = float(lengths.get(target, -math.inf))
                if distance > best_length:
                    best_length = distance
                    best_path = paths[target]

    if best_path is None:
        raise RuntimeError("Cannot identify a longest path in the spine skeleton.")
    return np.vstack([graph.nodes[node]["pos"] for node in best_path])


def sample_polyline_by_arclength(polyline: np.ndarray, n_points: int = 10) -> np.ndarray:
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) < 2:
        raise ValueError("Polyline must contain at least two points.")

    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    keep = np.r_[True, segment_lengths > 0]
    polyline = polyline[keep]
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("Polyline length is zero.")

    samples = []
    for target in np.linspace(0.0, total, n_points):
        seg_idx = min(np.searchsorted(cumulative, target, side="right") - 1, len(segment_lengths) - 1)
        local = 0.0 if segment_lengths[seg_idx] == 0 else (target - cumulative[seg_idx]) / segment_lengths[seg_idx]
        samples.append(polyline[seg_idx] * (1.0 - local) + polyline[seg_idx + 1] * local)
    return np.vstack(samples)


def orient_spine_path_from_dendrite(spine_path: np.ndarray, dendrite_segments: Sequence[Segment]) -> np.ndarray:
    first_distance = distance_to_segments(spine_path[0], dendrite_segments)
    last_distance = distance_to_segments(spine_path[-1], dendrite_segments)
    if last_distance < first_distance:
        return spine_path[::-1].copy()
    return spine_path


def load_skeleton_segments(path: Path) -> List[Segment]:
    skeleton = np.load(path, allow_pickle=True)
    segments = _segments_from_skeleton_object(skeleton)
    result = [(np.asarray(start, dtype=float), np.asarray(end, dtype=float)) for start, end in segments]
    if not result:
        raise RuntimeError(f"No valid 3-D segments found in skeleton: {path}")
    return result


@functools.lru_cache(maxsize=2048)
def _load_skeleton_segments_cached(path: Path) -> Tuple[Segment, ...]:
    return tuple(load_skeleton_segments(path))


def load_skeleton_segments_cached(path: Path) -> List[Segment]:
    """Process-local cached variant of :func:`load_skeleton_segments`.

    Many spines share the same parent branch, so ``branch_skeleton.npy`` gets
    reloaded once per spine (per stage) with the plain function. That is a
    real cost when the raw dataset is read from a slow network share (the
    measured throughput on the Windows workstation NAS is a few MB/s at best,
    see ``docs/neuron-model/windows-workstation-specs.md``): re-reading the
    same tiny file for every spine in a branch adds one round trip per spine
    instead of one per branch.

    Safe to cache: source ``.npy`` files are read-only for the whole run
    (`s-module-preprocessing.md` 2.2), and callers only ever read the
    returned segments, never mutate them. The cache is per worker *process*
    (bounded to 2048 branches), so it only pays off when a worker handles
    more than one spine from the same branch; ``run_stage`` chunks same-batch
    tasks (records are discovered sorted by branch) to make that likely.
    """
    return list(_load_skeleton_segments_cached(Path(path)))


def _polyline_segments(points: Any) -> List[Segment]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
        return []
    pts = pts[:, :3]
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if len(pts) < 2:
        return []
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _segments_from_skeleton_object(skeleton: Any) -> List[Segment]:
    if skeleton is None:
        return []

    if isinstance(skeleton, np.ndarray) and skeleton.dtype == object:
        if skeleton.shape == ():
            return _segments_from_skeleton_object(skeleton.item())
        segments: List[Segment] = []
        for item in skeleton.tolist():
            segments.extend(_segments_from_skeleton_object(item))
        return segments

    if isinstance(skeleton, dict):
        for points_key in ("points", "vertices", "nodes"):
            if points_key in skeleton and "edges" in skeleton:
                points = np.asarray(skeleton[points_key], dtype=float)
                edges = np.asarray(skeleton["edges"], dtype=int)
                if points.ndim == 2 and points.shape[1] >= 3 and edges.ndim == 2 and edges.shape[1] >= 2:
                    valid_edges = edges[:, :2]
                    valid_edges = valid_edges[
                        (valid_edges >= 0).all(axis=1)
                        & (valid_edges < len(points)).all(axis=1)
                    ]
                    return [(points[int(u), :3], points[int(v), :3]) for u, v in valid_edges]
        segments = []
        for key in ("segments", "skeleton", "branches", "polylines", "paths", "lines"):
            if key in skeleton:
                segments.extend(_segments_from_skeleton_object(skeleton[key]))
        if segments:
            return segments
        for value in skeleton.values():
            segments.extend(_segments_from_skeleton_object(value))
        return segments

    if isinstance(skeleton, (list, tuple)):
        try:
            numeric = np.asarray(skeleton, dtype=float)
            if numeric.ndim >= 2:
                return _segments_from_skeleton_object(numeric)
        except Exception:
            pass
        segments = []
        for item in skeleton:
            segments.extend(_segments_from_skeleton_object(item))
        return segments

    try:
        array = np.asarray(skeleton, dtype=float)
    except Exception:
        return []

    if array.ndim == 2 and array.shape[1] >= 3:
        return _polyline_segments(array)

    if array.ndim == 3 and array.shape[-1] >= 3:
        if array.shape[1] == 2:
            return [(array[i, 0, :3], array[i, 1, :3]) for i in range(array.shape[0])]
        segments = []
        for polyline in array:
            segments.extend(_polyline_segments(polyline))
        return segments

    return []


def cgal_skeleton_segments_from_trimesh(mesh: Any) -> List[Segment]:
    from CGAL.CGAL_Polygon_mesh_processing import Polylines
    from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization
    from src.spine_analysis.mesh.utils import v_f_to_mesh_isolated

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    polyhedron = v_f_to_mesh_isolated(vertices, faces)
    if not bool(polyhedron.is_closed()):
        raise RuntimeError("CGAL skeletonization requires a closed/watertight mesh.")

    skeleton_polylines = Polylines()
    correspondence_polylines = Polylines()
    surface_mesh_skeletonization(polyhedron, skeleton_polylines, correspondence_polylines)

    segments: List[Segment] = []
    for polyline in skeleton_polylines:
        points = [np.array([p.x(), p.y(), p.z()], dtype=float) for p in polyline]
        for start, end in zip(points[:-1], points[1:]):
            if np.linalg.norm(end - start) > 0:
                segments.append((start, end))
    if not segments:
        raise RuntimeError("CGAL returned an empty spine skeleton.")
    return segments


# ---------------------------------------------------------------------------
# Attachment loop / attachment cap (s-module-preprocessing.md, 3.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttachmentDetection:
    loop: Optional[BoundaryLoop]
    loops: List[BoundaryLoop]
    scores: np.ndarray
    ambiguous: bool
    reason: Optional[str]

    def candidates(self) -> List[Dict[str, Any]]:
        return [
            {
                "perimeter": float(loop.perimeter),
                "area": float(loop.area),
                "distance_to_branch_skeleton": float(loop.distance_to_skeleton),
                "centroid_global": loop.center.tolist(),
                "n_edges": int(loop.n_edges),
                "is_closed": bool(loop.is_closed),
                "score": float(score),
            }
            for loop, score in zip(self.loops, self.scores)
        ]


def _score_candidates(
    perimeters: np.ndarray,
    areas: np.ndarray,
    distances: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    """Shared scoring formula for attachment-region candidates.

    Used both for the (pre-seal) boundary loops in :func:`detect_attachment_loop`
    and for the (post-seal) face patches in :func:`select_attachment_patch` - same
    inputs (a perimeter/area/distance-to-skeleton triple per candidate), same
    weights, so the two are directly comparable/interchangeable.
    """

    def relative_to_max(values: np.ndarray) -> np.ndarray:
        top = float(np.max(values))
        return values / top if top > 0 else np.ones_like(values)

    finite = np.isfinite(distances)
    proximity = np.zeros_like(distances)
    if finite.any():
        nearest = float(np.min(distances[finite]))
        finite_distances = distances[finite]
        proximity[finite] = np.where(
            finite_distances > 0, nearest / np.where(finite_distances > 0, finite_distances, 1.0), 1.0
        )

    w_perimeter, w_area, w_distance = weights
    return w_perimeter * relative_to_max(perimeters) + w_area * relative_to_max(areas) + w_distance * proximity


def detect_attachment_loop(
    mesh: Any,
    branch_segments: Sequence[Segment],
    *,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ambiguity_ratio: float = 0.85,
) -> AttachmentDetection:
    """Select the boundary loop where the spine is attached to the dendrite.

    Each loop gets a score in (0, 1] from its perimeter and area (relative to the
    largest loop) and its proximity to the branch skeleton (relative to the closest
    loop). If the runner-up scores at least ``ambiguity_ratio`` of the best score,
    the detection is ambiguous and no loop is chosen.
    """
    loops = boundary_loops(mesh, branch_segments)
    if not loops:
        return AttachmentDetection(None, [], np.empty(0), True, "no_boundary_loops")

    perimeters = np.asarray([loop.perimeter for loop in loops], dtype=float)
    areas = np.asarray([loop.area for loop in loops], dtype=float)
    distances = np.asarray([loop.distance_to_skeleton for loop in loops], dtype=float)

    scores = _score_candidates(perimeters, areas, distances, weights)
    order = np.argsort(-scores)
    best = loops[int(order[0])]

    if len(loops) > 1 and scores[order[1]] >= ambiguity_ratio * scores[order[0]]:
        return AttachmentDetection(None, loops, scores, True, "close_candidates")
    if not best.is_closed:
        return AttachmentDetection(None, loops, scores, True, "attachment_loop_not_closed")
    return AttachmentDetection(best, loops, scores, False, None)


def _shared_vertex_ids(meshes_vertices: Sequence[np.ndarray], decimals: int) -> List[np.ndarray]:
    """Assign equal ids to vertices with equal (rounded) coordinates across meshes."""
    stacked = np.vstack([np.round(np.asarray(v, dtype=float), decimals) for v in meshes_vertices])
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    result = []
    offset = 0
    for vertices in meshes_vertices:
        result.append(inverse[offset:offset + len(vertices)])
        offset += len(vertices)
    return result


def find_new_faces(
    original_mesh: Any,
    sealed_mesh: Any,
    *,
    decimals: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faces present in ``sealed_mesh`` but not in ``original_mesh``.

    Repair may drop, split or append vertices, so faces are matched by vertex
    coordinates rather than index. This covers *every* face added while
    sealing, including small side holes that are not part of the attachment
    cap (s-module-preprocessing.md 3.1.2) - useful on its own for visual QA
    of the sealing step, and as the first step of
    :func:`find_attachment_cap_faces`.

    Returns ``(new_face_indices, original_vertex_ids, sealed_vertex_ids)``:
    vertices at the same (rounded) coordinates in both meshes share an id,
    so callers can map vertex indices between the two meshes (e.g. an
    attachment loop's original-mesh indices into the sealed mesh) without
    re-matching coordinates themselves.
    """
    original_ids, sealed_ids = _shared_vertex_ids(
        [np.asarray(original_mesh.vertices), np.asarray(sealed_mesh.vertices)], decimals
    )
    original_faces = np.sort(original_ids[np.asarray(original_mesh.faces, dtype=int)], axis=1)
    sealed_faces = np.sort(sealed_ids[np.asarray(sealed_mesh.faces, dtype=int)], axis=1)
    original_face_keys = set(map(tuple, original_faces.tolist()))
    new_faces = [i for i, face in enumerate(map(tuple, sealed_faces.tolist())) if face not in original_face_keys]
    return np.asarray(new_faces, dtype=int), original_ids, sealed_ids


def group_faces_by_component(sealed_mesh: Any, face_indices: np.ndarray) -> List[np.ndarray]:
    """Group ``face_indices`` (into ``sealed_mesh``) into patches connected through shared edges.

    Each connected component is one filled hole (or, in general, one blob of
    mutually adjacent faces) - used both to pick out the attachment cap
    (:func:`find_attachment_cap_faces`) and, on its own, for visual QA of
    every hole sealing touched (``visualization.py``, one color per patch).

    Returns patches sorted by size, largest first, each as a sorted array of
    face indices.
    """
    sealed_faces = np.asarray(sealed_mesh.faces, dtype=int)
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_index in face_indices.tolist():
        a, b, c = sorted(sealed_faces[face_index])
        for edge in ((a, b), (b, c), (a, c)):
            edge_to_faces.setdefault(edge, []).append(face_index)

    graph = nx.Graph()
    graph.add_nodes_from(face_indices.tolist())
    for faces in edge_to_faces.values():
        for other in faces[1:]:
            graph.add_edge(faces[0], other)

    components = [np.asarray(sorted(component), dtype=int) for component in nx.connected_components(graph)]
    components.sort(key=len, reverse=True)
    return components


def find_attachment_cap_faces(
    original_mesh: Any,
    sealed_mesh: Any,
    loop: BoundaryLoop,
    *,
    decimals: int = 6,
    min_rim_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of the sealed-mesh faces that close ``loop``.

    New faces (see :func:`find_new_faces`) are grouped into patches connected
    through shared edges (:func:`group_faces_by_component`); a patch belongs
    to the attachment cap if most of its rim vertices lie on the attachment
    loop.

    Returns ``(cap_face_indices, loop_vertex_indices_in_sealed_mesh)``.
    """
    new_faces, original_ids, sealed_ids = find_new_faces(original_mesh, sealed_mesh, decimals=decimals)
    sealed_faces = np.sort(sealed_ids[np.asarray(sealed_mesh.faces, dtype=int)], axis=1)

    loop_ids = set(original_ids[loop.vertex_indices].tolist())
    loop_sealed_indices = np.flatnonzero(np.isin(sealed_ids, list(loop_ids))).astype(int)
    if len(new_faces) == 0:
        return np.empty(0, dtype=int), loop_sealed_indices

    cap_faces: List[int] = []
    for component in group_faces_by_component(sealed_mesh, new_faces):
        edge_counts: Dict[Tuple[int, int], int] = {}
        for face_index in component:
            a, b, c = sealed_faces[face_index]
            for edge in ((a, b), (b, c), (a, c)):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        rim_vertices = {v for edge, count in edge_counts.items() if count == 1 for v in edge}
        if not rim_vertices:
            continue
        if len(rim_vertices & loop_ids) / len(rim_vertices) >= min_rim_fraction:
            cap_faces.extend(component.tolist())

    return np.asarray(sorted(cap_faces), dtype=int), loop_sealed_indices


def find_new_face_patches(
    original_mesh: Any,
    sealed_mesh: Any,
    branch_segments: Optional[Sequence[Segment]] = None,
    *,
    decimals: int = 6,
) -> List[Dict[str, Any]]:
    """Every face patch added while sealing, with ``BoundaryLoop``-like features
    computed directly from the sealed mesh's own faces instead of from a
    boundary-edge graph walk on the original mesh.

    ``detect_attachment_loop``'s perimeter/area/``is_closed`` come from
    ordering the *original* mesh's boundary edges into a cycle
    (:func:`_order_component_vertices`); that ordering is only well-defined
    when every boundary vertex in a hole has exactly 2 boundary edges. A
    vertex with more (or fewer) - e.g. two holes touching at a single point -
    makes that walk ambiguous, which is exactly what
    ``attachment_ambiguous:attachment_loop_not_closed`` reports. Sealing
    itself does not have this problem: whatever hole-filling
    (``dendrite_analysis.mesh_repair.repair_mesh``) actually did is a
    concrete, already-triangulated patch, so ``perimeter`` (sum of its rim
    edge lengths - no cycle order needed), ``area`` (sum of its own
    triangles' areas - the real curved-cap area, not a flat-polygon
    approximation) and ``distance_to_skeleton`` (from the rim's centroid) are
    always well-defined, regardless of how the original boundary graph looks.

    Returns one dict per patch (largest first, matching
    :func:`group_faces_by_component`) with keys ``face_indices`` (into
    ``sealed_mesh``), ``perimeter``, ``area``, ``center``,
    ``distance_to_skeleton``, ``vertex_indices``/``edges`` (the patch's rim,
    mapped back to ``original_mesh`` vertex indices - hole-filling does not
    move existing vertices, so a patch's rim vertices already exist there;
    same convention as ``BoundaryLoop.vertex_indices``/``.edges``, so a
    caller can store/plot a patch exactly like a loop), ``sealed_vertex_indices``
    (the same rim, but into ``sealed_mesh`` instead) and ``n_edges``.
    """
    new_faces, original_ids, sealed_ids = find_new_faces(original_mesh, sealed_mesh, decimals=decimals)
    if len(new_faces) == 0:
        return []

    shared_to_original: Dict[int, int] = {}
    for original_index, shared_id in enumerate(original_ids.tolist()):
        shared_to_original.setdefault(shared_id, original_index)

    sealed_vertices = np.asarray(sealed_mesh.vertices, dtype=float)
    sealed_faces = np.asarray(sealed_mesh.faces, dtype=int)

    patches: List[Dict[str, Any]] = []
    for component in group_faces_by_component(sealed_mesh, new_faces):
        edge_counts: Dict[Tuple[int, int], int] = {}
        for face_index in component.tolist():
            a, b, c = sorted(sealed_faces[face_index].tolist())
            for edge in ((a, b), (b, c), (a, c)):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        rim_edges = np.asarray([edge for edge, count in edge_counts.items() if count == 1], dtype=int)
        if len(rim_edges) == 0:
            continue
        rim_vertices = np.unique(rim_edges.reshape(-1))

        perimeter = float(np.sum(np.linalg.norm(sealed_vertices[rim_edges[:, 0]] - sealed_vertices[rim_edges[:, 1]], axis=1)))
        area = float(np.asarray(sealed_mesh.area_faces)[component].sum())
        center = sealed_vertices[rim_vertices].mean(axis=0)
        distance = distance_to_segments(center, branch_segments) if branch_segments else math.inf

        original_vertex_indices = np.asarray(
            [shared_to_original[sealed_ids[v]] for v in rim_vertices.tolist() if sealed_ids[v] in shared_to_original],
            dtype=int,
        )
        original_edges = np.asarray(
            [
                (shared_to_original[sealed_ids[a]], shared_to_original[sealed_ids[b]])
                for a, b in rim_edges.tolist()
                if sealed_ids[a] in shared_to_original and sealed_ids[b] in shared_to_original
            ],
            dtype=int,
        ) if len(rim_edges) else np.empty((0, 2), dtype=int)

        patches.append(
            {
                "face_indices": component,
                "perimeter": perimeter,
                "area": area,
                "center": center,
                "distance_to_skeleton": float(distance),
                "vertex_indices": original_vertex_indices,
                "edges": original_edges,
                "sealed_vertex_indices": rim_vertices,
                "n_edges": int(len(rim_edges)),
            }
        )
    return patches


def select_attachment_patch(
    original_mesh: Any,
    sealed_mesh: Any,
    branch_segments: Optional[Sequence[Segment]] = None,
    *,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ambiguity_ratio: float = 0.85,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Fallback attachment-region selection, scored on sealed face patches rather
    than pre-seal boundary loops (see :func:`find_new_face_patches`).

    Meant to be tried only when :func:`detect_attachment_loop` itself came
    back ambiguous - it uses the same scoring formula
    (:func:`_score_candidates`) and the same ``weights``/``ambiguity_ratio``,
    just on more robust per-candidate features, so it can resolve cases the
    boundary-loop walk can't order into a simple cycle
    (``attachment_loop_not_closed``) and can, in general, also come out
    differently than the loop-based scoring on a ``close_candidates`` tie
    (real triangulated patch area vs. a flat-polygon estimate).

    Returns ``(winner, candidates, reason)``: ``winner`` is one of
    ``candidates`` (see :func:`find_new_face_patches`) or ``None`` if still
    ambiguous or there were no patches at all, in which case ``reason``
    explains why (``"close_candidates"`` or ``"no_new_face_patches"``).
    """
    patches = find_new_face_patches(original_mesh, sealed_mesh, branch_segments)
    if not patches:
        return None, patches, "no_new_face_patches"

    perimeters = np.asarray([patch["perimeter"] for patch in patches], dtype=float)
    areas = np.asarray([patch["area"] for patch in patches], dtype=float)
    distances = np.asarray([patch["distance_to_skeleton"] for patch in patches], dtype=float)
    scores = _score_candidates(perimeters, areas, distances, weights)
    order = np.argsort(-scores)

    if len(patches) > 1 and scores[order[1]] >= ambiguity_ratio * scores[order[0]]:
        return None, patches, "close_candidates"
    return patches[int(order[0])], patches, None


# ---------------------------------------------------------------------------
# Local coordinate frame (s-module-preprocessing.md, 3.3)
# ---------------------------------------------------------------------------


def homogeneous_transform(basis_rows: np.ndarray, origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(global_to_local, local_to_global)`` 4x4 matrices for p_local = R (p - O)."""
    rotation = np.asarray(basis_rows, dtype=float)
    origin = np.asarray(origin, dtype=float)
    global_to_local = np.eye(4)
    global_to_local[:3, :3] = rotation
    global_to_local[:3, 3] = -rotation @ origin
    local_to_global = np.eye(4)
    local_to_global[:3, :3] = rotation.T
    local_to_global[:3, 3] = origin
    return global_to_local, local_to_global


def apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def orient_spines_w_holes(
    sealed_mesh: Any,
    attachment_cap_face_indices: Sequence[int],
    origin_global: np.ndarray,
    branch_segments: Optional[Sequence[Segment]] = None,
    *,
    min_tangent_component: float = 0.1,
) -> Dict[str, Any]:
    """Local frame of a sealed spine built from its attachment cap.

    Follows ``SpineMeshDataset._orient_spine`` from ``notebook_widgets.py``: the
    area-weighted normal of the junction (here: the attachment cap created during
    sealing) is mapped to ``-y``, so the spine grows along ``+y``. The in-plane axis
    is the branch skeleton tangent instead of the PCA axis; PCA is used only as a
    fallback.

    Local axes: ``x = e_t`` (tangent), ``y = e_r`` (radial, into the spine),
    ``z = e_b = e_t x e_r`` (binormal). Origin: attachment loop centroid.
    """
    vertices = np.asarray(sealed_mesh.vertices, dtype=float)
    origin = np.asarray(origin_global, dtype=float)
    cap = np.asarray(attachment_cap_face_indices, dtype=int)
    fallback_reasons: List[str] = []

    cap_normal = np.zeros(3, dtype=float)
    if len(cap):
        face_normals = np.asarray(sealed_mesh.face_normals, dtype=float)[cap]
        face_areas = np.asarray(sealed_mesh.area_faces, dtype=float)[cap]
        cap_normal = np.sum(face_normals * face_areas[:, None], axis=0)
    body_direction = vertices.mean(axis=0) - origin
    if not np.isfinite(cap_normal).all() or np.linalg.norm(cap_normal) <= 1e-12:
        fallback_reasons.append("cap_normal_degenerate")
        cap_normal = -body_direction
    cap_normal = _unit(cap_normal, fallback=np.array([0.0, -1.0, 0.0]))

    radial = -cap_normal
    if np.dot(radial, body_direction) < 0:
        fallback_reasons.append("cap_normal_points_into_spine")
        radial = -radial
        cap_normal = -cap_normal

    projection = None
    tangent = None
    if branch_segments:
        projection, nearest_segment, _, _ = closest_point_on_segments(origin, branch_segments)
        raw_tangent = _unit(nearest_segment[1] - nearest_segment[0], fallback=np.array([1.0, 0.0, 0.0]))
        in_plane = raw_tangent - np.dot(raw_tangent, radial) * radial
        if np.linalg.norm(in_plane) >= min_tangent_component:
            tangent = _unit(in_plane)
        else:
            fallback_reasons.append("tangent_parallel_to_radial")
    else:
        fallback_reasons.append("missing_branch_skeleton")

    if tangent is None:
        centered = vertices - origin
        in_plane_points = centered - np.outer(centered @ radial, radial)
        _, _, vh = np.linalg.svd(in_plane_points - in_plane_points.mean(axis=0), full_matrices=False)
        tangent = vh[0] - np.dot(vh[0], radial) * radial
        tangent = _unit(tangent, fallback=np.cross(radial, [0.0, 0.0, 1.0]))
        fallback_reasons.append("tangent_from_pca")

    binormal = _unit(np.cross(tangent, radial))
    tangent = _unit(np.cross(radial, binormal))
    basis = np.vstack([tangent, radial, binormal])
    global_to_local, local_to_global = homogeneous_transform(basis, origin)

    return {
        "origin_global": origin,
        "tangent": tangent,
        "radial": radial,
        "binormal": binormal,
        "basis_rows": basis,
        "global_to_local": global_to_local,
        "local_to_global": local_to_global,
        "attachment_normal_global": cap_normal,
        "attachment_normal_local": basis @ cap_normal,
        "branch_projection": projection,
        "orientation_fallback": bool(fallback_reasons),
        "orientation_fallback_reasons": fallback_reasons,
    }


# ---------------------------------------------------------------------------
# Geometric QC (s-module-preprocessing.md, 3.1.3 and 3.5)
# ---------------------------------------------------------------------------


def count_nonmanifold_edges(mesh: Any) -> int:
    if len(mesh.faces) == 0:
        return 0
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return int(np.sum(counts > 2))


def count_nonmanifold_vertices(mesh: Any) -> int:
    """Vertices whose incident faces do not form a single edge-connected fan."""
    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces) == 0:
        return 0
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(key: Tuple[int, int]) -> Tuple[int, int]:
        root = key
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    edge_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_index, (a, b, c) in enumerate(faces.tolist()):
        for vertex in (a, b, c):
            find((vertex, face_index))
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(u, v), max(u, v)), []).append(face_index)

    for (u, v), incident in edge_faces.items():
        for other in incident[1:]:
            for vertex in (u, v):
                root_a, root_b = find((vertex, incident[0])), find((vertex, other))
                if root_a != root_b:
                    parent[root_a] = root_b

    roots_per_vertex: Dict[int, set] = {}
    for key in list(parent):
        roots_per_vertex.setdefault(key[0], set()).add(find(key))
    return int(sum(1 for roots in roots_per_vertex.values() if len(roots) > 1))


def face_connected_components(mesh: Any) -> int:
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return 0
    graph = nx.Graph()
    graph.add_nodes_from(range(n_faces))
    graph.add_edges_from(np.asarray(mesh.face_adjacency, dtype=int).tolist())
    return int(nx.number_connected_components(graph))


def _segments_cross_triangles(p0: np.ndarray, p1: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Vectorised Moller-Trumbore test of segments [p0, p1] against triangles.

    Boundaries are inclusive: callers only pass face pairs without shared vertices,
    for which touching along an edge is already an intersection.
    """
    eps = 1e-9
    direction = p1 - p0
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, e2)
    a = np.einsum("ij,ij->i", e1, h)
    scale = np.linalg.norm(direction, axis=1) * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    valid = np.abs(a) > 1e-12 * np.maximum(scale, 1e-300)
    safe_a = np.where(valid, a, 1.0)
    s = p0 - triangles[:, 0]
    u = np.einsum("ij,ij->i", s, h) / safe_a
    q = np.cross(s, e1)
    v = np.einsum("ij,ij->i", direction, q) / safe_a
    t = np.einsum("ij,ij->i", e2, q) / safe_a
    return valid & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)


def count_self_intersecting_pairs(mesh: Any) -> int:
    """Count pairs of faces without shared vertices that intersect or touch.

    Broad phase: KD-tree over triangle centroids (radius from the largest triangle)
    followed by an AABB overlap filter; narrow phase: edge/triangle crossing.
    Coplanar overlaps are not detected.
    """
    from scipy.spatial import cKDTree

    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces) < 2:
        return 0
    triangles = np.asarray(mesh.triangles, dtype=float)
    centroids = triangles.mean(axis=1)
    # Two intersecting triangles have centroids closer than 2/3 of the sum of their diameters.
    diameters = np.max(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2), axis=1)
    pairs = cKDTree(centroids).query_pairs(r=float(4.0 / 3.0 * diameters.max()), output_type="ndarray")
    if len(pairs) == 0:
        return 0
    ia, ib = pairs[:, 0], pairs[:, 1]
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    overlap = np.all((low[ia] <= high[ib]) & (low[ib] <= high[ia]), axis=1)
    shared = (faces[ia][:, :, None] == faces[ib][:, None, :]).any(axis=(1, 2))
    keep = overlap & ~shared
    ia, ib = ia[keep], ib[keep]
    if len(ia) == 0:
        return 0
    hit = np.zeros(len(ia), dtype=bool)
    for first, second in ((ia, ib), (ib, ia)):
        tri_first = triangles[first]
        tri_second = triangles[second]
        for k in range(3):
            hit |= _segments_cross_triangles(tri_first[:, k], tri_first[:, (k + 1) % 3], tri_second)
    return int(np.sum(hit))


def self_intersection_check(mesh: Any) -> Tuple[Optional[bool], Optional[int], str]:
    """Return ``(has_self_intersections, n_intersecting_pairs, method)``.

    Prefers CGAL's ``self_intersections`` (exact geometric predicate, returns
    every intersecting facet pair) over ``does_self_intersect`` (bool only),
    so QC gets a real pair count instead of ``None`` when CGAL is available.
    CGAL catches more cases (e.g. near-coplanar/edge-touching) than the
    ``edge_triangle`` fallback below, so the two methods' counts are not
    directly comparable.
    """
    try:
        from CGAL.CGAL_Polygon_mesh_processing import self_intersections
        from src.spine_analysis.mesh.utils import v_f_to_mesh_isolated

        polyhedron = v_f_to_mesh_isolated(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int))
        for index, facet in enumerate(polyhedron.facets()):
            facet.set_id(index)
        pairs: List[Any] = []
        self_intersections(polyhedron, pairs)
        unique_pairs = {tuple(sorted((item[0].id(), item[1].id()))) for item in pairs}
        return bool(unique_pairs), len(unique_pairs), "cgal"
    except Exception:
        pass
    try:
        n_pairs = count_self_intersecting_pairs(mesh)
        return n_pairs > 0, n_pairs, "edge_triangle"
    except Exception:
        return None, None, "unavailable"


def mesh_geometry_report(
    mesh: Any,
    *,
    degenerate_area_tol: float = 1e-12,
    check_self_intersections: bool = True,
) -> Dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=int)
    n_faces = int(len(faces))
    used_vertices = int(len(np.unique(faces))) if n_faces else 0
    n_edges = int(len(mesh.edges_unique)) if n_faces else 0
    n_boundary = boundary_edge_count(mesh)
    n_nonmanifold_edges = count_nonmanifold_edges(mesh)
    n_nonmanifold_vertices = count_nonmanifold_vertices(mesh)
    is_watertight = bool(mesh.is_watertight) if n_faces else False
    n_components = face_connected_components(mesh)
    euler = used_vertices - n_edges + n_faces

    genus: Optional[float] = None
    if is_watertight and n_components > 0:
        genus = (2 * n_components - euler) / 2.0
        genus = int(genus) if float(genus).is_integer() else genus

    volume: Optional[float] = None
    if is_watertight:
        try:
            volume = float(mesh.volume)
        except Exception:
            volume = None

    has_self_intersections: Optional[bool] = None
    n_self_pairs: Optional[int] = None
    self_method = "skipped"
    if check_self_intersections and n_faces:
        has_self_intersections, n_self_pairs, self_method = self_intersection_check(mesh)

    return {
        "is_watertight": is_watertight,
        "is_manifold": bool(is_watertight and n_nonmanifold_edges == 0 and n_nonmanifold_vertices == 0),
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": n_faces,
        "n_edges": n_edges,
        "n_connected_components": n_components,
        "n_boundary_edges": n_boundary,
        "n_nonmanifold_edges": n_nonmanifold_edges,
        "n_nonmanifold_vertices": n_nonmanifold_vertices,
        "n_degenerate_faces": count_degenerate_faces(mesh, area_tol=degenerate_area_tol),
        "has_self_intersections": has_self_intersections,
        "n_self_intersecting_pairs": n_self_pairs,
        "self_intersection_method": self_method,
        "has_consistent_winding": bool(mesh.is_winding_consistent) if n_faces else False,
        "has_finite_coordinates": bool(np.isfinite(np.asarray(mesh.vertices, dtype=float)).all()),
        "surface_area": float(mesh.area),
        "volume": volume,
        "has_positive_finite_volume": bool(volume is not None and math.isfinite(volume) and volume > 0),
        "euler_characteristic": int(euler),
        "genus": genus,
    }
