from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import matplotlib.pyplot as _plt
    import matplotlib.cm as _cm
    _MPL = True
except ImportError:
    _MPL = False

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

try:
    import yaml as _yaml
    _YAML = True
except ImportError:
    _YAML = False

try:
    from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization as _skeletonize
    from CGAL.CGAL_Polygon_mesh_processing import Polylines as _Polylines
    _CGAL = True
except ImportError:
    _CGAL = False

try:
    import trimesh as _trimesh
    _TRIMESH = True
except ImportError:
    _TRIMESH = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # dataclasses
    "ProjectedSpine",
    "PoissonModelResult",
    "KFunctionResult",
    
    # graph
    "DendriticGraph",
    "build_dendritic_graph",
    "build_graph_from_meshes",

    # spine attachment
    "project_spines_to_graph",
    "projected_spines_dataframe",

    # distances
    "compute_soma_distances",
    "compute_spine_pairwise_distances",
    "compute_spine_pairwise_distances_mesh",
    "compute_distance_to_branching",
    "compute_distance_to_terminals",

    # intensity
    "estimate_binned_intensity",
    "estimate_smooth_intensity",
    "test_intensity_dependence",

    # Poisson model
    "fit_inhomogeneous_poisson",

    # K function
    "ripley_k_network",
    "simulate_poisson_on_graph",
    "compute_simulation_envelopes",

    # group comparison
    "group_k_analysis",
    "permutation_test_groups",

    # visualisation
    "plot_3d_network",
    "plot_intensity_vs_distance",
    "plot_ripley_k",
    "plot_k_deviation",
    "plot_group_comparison",

    # report / pipeline
    "generate_report",
    "load_config",
    "run_analysis",
]


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class ProjectedSpine:
    spine_id: str
    original_point: np.ndarray        
    projected_point: np.ndarray        
    edge_id: str                      
    edge_source: int
    edge_target: int
    edge_position: float                
    distance_to_edge: float
    distance_to_soma: float = 0.0


@dataclass
class PoissonModelResult:
    coefficients: np.ndarray
    covariate_names: List[str]
    standard_errors: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    fitted_intensity: np.ndarray        # λ(xᵢ) at each spine
    residuals: np.ndarray               # Pearson residuals
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KFunctionResult:
    r_values: np.ndarray
    k_observed: np.ndarray
    k_expected: np.ndarray              
    k_lower: Optional[np.ndarray] = None
    k_upper: Optional[np.ndarray] = None
    p_value: Optional[float] = None
    method: str = "homogeneous"
    interpretation: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        data: Dict[str, Any] = {
            "r": self.r_values,
            "k_observed": self.k_observed,
            "k_expected": self.k_expected,
        }
        if self.k_lower is not None:
            data["k_lower"] = self.k_lower
        if self.k_upper is not None:
            data["k_upper"] = self.k_upper
        return pd.DataFrame(data)


# ===========================================================================
# DendriticGraph
# ===========================================================================

class DendriticGraph:

    def __init__(self) -> None:
        self.G: nx.Graph = nx.Graph()
        self.soma_node: Optional[int] = None
        self._soma_distances: Optional[Dict[int, float]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_length(self) -> float:
        return float(sum(d["length"] for _, _, d in self.G.edges(data=True)))

    @property
    def n_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.G.number_of_edges()

    @property
    def nodes_df(self) -> pd.DataFrame:
        rows = []
        for node, data in self.G.nodes(data=True):
            pos = data.get("pos", np.zeros(3))
            rows.append(
                {
                    "node_id": node,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "node_type": data.get("node_type", ""),
                    "dendrite_id": data.get("dendrite_id", ""),
                    "dendrite_type": data.get("dendrite_type", ""),
                }
            )
        return pd.DataFrame(rows)

    @property
    def edges_df(self) -> pd.DataFrame:
        rows = []
        for u, v, data in self.G.edges(data=True):
            rows.append({"source": u, "target": v, "length": data.get("length", 0.0)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Node/position helpers
    # ------------------------------------------------------------------

    def node_position(self, node: int) -> np.ndarray:
        pos = self.G.nodes[node].get("pos", np.zeros(3))
        return np.asarray(pos, dtype=float)

    def all_node_positions(self) -> np.ndarray:
        nodes = list(self.G.nodes())
        if not nodes:
            return np.empty((0, 3))
        return np.stack([self.node_position(n) for n in nodes])

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def soma_distances(self, recompute: bool = False) -> Dict[int, float]:
        if self._soma_distances is not None and not recompute:
            return self._soma_distances

        source = self.soma_node
        if source is None:
            if self.G.number_of_nodes() == 0:
                return {}
            source = next(iter(self.G.nodes()))
            warnings.warn(
                "soma_node is not set; using the first node as source for "
                "soma distance computation.",
                stacklevel=2,
            )

        lengths = nx.single_source_dijkstra_path_length(self.G, source, weight="length")
        self._soma_distances = dict(lengths)
        return self._soma_distances

    def all_pairs_distances(self) -> Dict[Any, Dict[Any, float]]:
        return dict(nx.all_pairs_dijkstra_path_length(self.G, weight="length"))

    def sample_points_on_graph(
        self, step: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.G.number_of_edges() == 0:
            return np.empty((0, 3)), np.empty((0,))

        sd = self.soma_distances()
        all_pts: List[np.ndarray] = []
        all_sd: List[float] = []

        for u, v, edata in self.G.edges(data=True):
            length = float(edata.get("length", 0.0))
            if length <= 0:
                continue
            p_u = self.node_position(u)
            p_v = self.node_position(v)
            d_u = sd.get(u, 0.0)
            d_v = sd.get(v, 0.0)

            n_segs = max(1, int(math.ceil(length / step)))
            ts = (np.arange(n_segs) + 0.5) / n_segs  # midpoints in [0,1]

            for t in ts:
                pt = p_u + t * (p_v - p_u)
                # soma distance: best approach along the edge from either end
                dist = min(d_u + t * length, d_v + (1.0 - t) * length)
                all_pts.append(pt)
                all_sd.append(dist)

        if not all_pts:
            return np.empty((0, 3)), np.empty((0,))

        return np.array(all_pts), np.array(all_sd)


# ===========================================================================
# Graph building
# ===========================================================================

def _graph_from_skeleton_segments(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    dendrite_id: str = "d0",
    dendrite_type: str = "unknown",
) -> DendriticGraph:
    dg = DendriticGraph()

    pos_to_id: Dict[Tuple[float, ...], int] = {}
    next_id = 0

    def _get_node(pt: np.ndarray) -> int:
        nonlocal next_id
        key = tuple(np.round(pt, 6).tolist())
        if key not in pos_to_id:
            pos_to_id[key] = next_id
            dg.G.add_node(
                next_id,
                pos=np.array(pt, dtype=float),
                node_type="intermediate",
                dendrite_id=dendrite_id,
                dendrite_type=dendrite_type,
            )
            next_id += 1
        return pos_to_id[key]

    for seg_start, seg_end in segments:
        u = _get_node(seg_start)
        v = _get_node(seg_end)
        if u == v:
            continue
        length = float(np.linalg.norm(seg_end - seg_start))
        if dg.G.has_edge(u, v):
            if length < dg.G[u][v]["length"]:
                dg.G[u][v]["length"] = length
        else:
            dg.G.add_edge(u, v, length=length)

    for node in dg.G.nodes():
        deg = dg.G.degree(node)
        if deg == 1:
            dg.G.nodes[node]["node_type"] = "terminal"
        elif deg >= 3:
            dg.G.nodes[node]["node_type"] = "branch"
        else:
            dg.G.nodes[node]["node_type"] = "intermediate"

    return dg


def build_dendritic_graph(
    dendrite_mesh: Any,
    soma_point: Optional[np.ndarray] = None,
    dendrite_id: str = "d0",
    dendrite_type: str = "unknown",
    snap_threshold: float = 2.0,
) -> DendriticGraph:
    if not _CGAL:
        raise ImportError("CGAL bindings are required for build_dendritic_graph.")

    from dendrite_analysis.surface_distances import _skeleton_segments  # type: ignore

    segments = _skeleton_segments(dendrite_mesh)
    dg = _graph_from_skeleton_segments(segments, dendrite_id=dendrite_id, dendrite_type=dendrite_type)

    if soma_point is not None and dg.G.number_of_nodes() > 0:
        soma_pt = np.asarray(soma_point, dtype=float)
        positions = np.array([dg.G.nodes[n]["pos"] for n in dg.G.nodes()])
        node_ids = list(dg.G.nodes())
        dists = np.linalg.norm(positions - soma_pt, axis=1)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] <= snap_threshold:
            soma_id = node_ids[nearest_idx]
            dg.soma_node = soma_id
            dg.G.nodes[soma_id]["node_type"] = "soma"

    return dg


def build_graph_from_meshes(
    dendrite_meshes: Dict[str, Any],
    soma_mesh: Optional[Any] = None,
    dendrite_types: Optional[Dict[str, str]] = None,
    snap_threshold: float = 2.0,
) -> DendriticGraph:
    if dendrite_types is None:
        dendrite_types = {}

    soma_point: Optional[np.ndarray] = None
    if soma_mesh is not None and _TRIMESH:
        try:
            from dendrite_analysis.surface_distances import polyhedron_to_trimesh  # type: ignore
            tm = polyhedron_to_trimesh(soma_mesh)
            soma_point = np.asarray(tm.centroid, dtype=float)
        except Exception:
            pass

    merged = DendriticGraph()
    node_offset = 0

    for dendrite_id, mesh in dendrite_meshes.items():
        dtype = dendrite_types.get(dendrite_id, "unknown")
        try:
            sub_graph = build_dendritic_graph(
                mesh,
                soma_point=soma_point,
                dendrite_id=dendrite_id,
                dendrite_type=dtype,
                snap_threshold=snap_threshold,
            )
        except Exception as exc:
            warnings.warn(f"Skeletonisation of {dendrite_id!r} failed: {exc}", stacklevel=2)
            continue

        mapping = {old: old + node_offset for old in sub_graph.G.nodes()}
        relabelled = nx.relabel_nodes(sub_graph.G, mapping)
        merged.G = nx.compose(merged.G, relabelled)

        if sub_graph.soma_node is not None and merged.soma_node is None:
            merged.soma_node = sub_graph.soma_node + node_offset

        node_offset += sub_graph.G.number_of_nodes()

    if merged.G.number_of_nodes() > 1:
        node_list = list(merged.G.nodes())
        positions = np.array([merged.G.nodes[n]["pos"] for n in node_list])
        kd = cKDTree(positions)
        pairs = kd.query_pairs(snap_threshold)
        for i, j in pairs:
            u, v = node_list[i], node_list[j]
            di = merged.G.nodes[u].get("dendrite_id", "")
            dj = merged.G.nodes[v].get("dendrite_id", "")
            if di != dj and not merged.G.has_edge(u, v):
                length = float(np.linalg.norm(positions[i] - positions[j]))
                merged.G.add_edge(u, v, length=length)

    return merged


# ===========================================================================
# Spine attachment
# ===========================================================================

def _project_point_to_segment(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    d = seg_end - seg_start
    seg_len_sq = float(np.dot(d, d))
    if seg_len_sq < 1e-20:
        return seg_start.copy(), 0.0, float(np.linalg.norm(point - seg_start))

    t = float(np.dot(point - seg_start, d) / seg_len_sq)
    t = max(0.0, min(1.0, t))
    proj = seg_start + t * d
    dist = float(np.linalg.norm(point - proj))
    return proj, t, dist


def project_spines_to_graph(
    graph: DendriticGraph,
    spine_points: Dict[str, np.ndarray],
    max_distance_to_edge: float = 2.0,
    use_k_nearest_edges: int = 10,
    allow_unassigned: bool = True,
) -> Tuple[List[ProjectedSpine], List[str]]:
    edges = list(graph.G.edges(data=True))
    if not edges:
        if not allow_unassigned and spine_points:
            raise ValueError("Graph has no edges; cannot project spines.")
        return [], list(spine_points.keys())

    midpoints = []
    edge_list: List[Tuple[int, int, float]] = []  # (u, v, length)
    for u, v, edata in edges:
        pu = graph.node_position(u)
        pv = graph.node_position(v)
        midpoints.append(0.5 * (pu + pv))
        edge_list.append((u, v, float(edata.get("length", np.linalg.norm(pv - pu)))))

    midpoints_arr = np.array(midpoints)
    kd = cKDTree(midpoints_arr)

    soma_dist_map = graph.soma_distances()

    projected: List[ProjectedSpine] = []
    unassigned: List[str] = []

    k = min(use_k_nearest_edges, len(edge_list))

    for spine_id, pt in spine_points.items():
        pt = np.asarray(pt, dtype=float)
        _, idxs = kd.query(pt, k=k)
        if np.isscalar(idxs):
            idxs = [int(idxs)]
        else:
            idxs = list(idxs)

        best_dist = np.inf
        best: Optional[ProjectedSpine] = None

        for idx in idxs:
            u, v, edge_len = edge_list[idx]
            pu = graph.node_position(u)
            pv = graph.node_position(v)
            proj_pt, t, dist = _project_point_to_segment(pt, pu, pv)
            if dist < best_dist:
                best_dist = dist
                d_soma_u = soma_dist_map.get(u, 0.0)
                d_soma_v = soma_dist_map.get(v, 0.0)
                soma_dist = min(d_soma_u + t * edge_len, d_soma_v + (1.0 - t) * edge_len)
                best = ProjectedSpine(
                    spine_id=spine_id,
                    original_point=pt.copy(),
                    projected_point=proj_pt,
                    edge_id=f"{u}_{v}",
                    edge_source=u,
                    edge_target=v,
                    edge_position=t,
                    distance_to_edge=dist,
                    distance_to_soma=soma_dist,
                )

        if best is not None and best_dist <= max_distance_to_edge:
            projected.append(best)
        else:
            if not allow_unassigned:
                raise ValueError(
                    f"Spine {spine_id!r} could not be assigned to any edge "
                    f"(min distance = {best_dist:.4f}, threshold = {max_distance_to_edge})."
                )
            unassigned.append(spine_id)

    return projected, unassigned


def projected_spines_dataframe(spines: List[ProjectedSpine]) -> pd.DataFrame:
    rows = []
    for s in spines:
        rows.append(
            {
                "spine_id": s.spine_id,
                "original_x": s.original_point[0],
                "original_y": s.original_point[1],
                "original_z": s.original_point[2],
                "projected_x": s.projected_point[0],
                "projected_y": s.projected_point[1],
                "projected_z": s.projected_point[2],
                "edge_id": s.edge_id,
                "edge_source": s.edge_source,
                "edge_target": s.edge_target,
                "edge_position": s.edge_position,
                "distance_to_edge": s.distance_to_edge,
                "distance_to_soma": s.distance_to_soma,
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Distance computations
# ===========================================================================

def compute_soma_distances(graph: DendriticGraph) -> Dict[int, float]:
    return graph.soma_distances()


def compute_spine_pairwise_distances(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
) -> np.ndarray:
    n = len(spines)
    dist_matrix = np.zeros((n, n), dtype=float)
    if n == 0:
        return dist_matrix

    all_pairs = dict(nx.all_pairs_dijkstra_path_length(graph.G, weight="length"))

    def _node_dist(a: int, b: int) -> float:
        try:
            return float(all_pairs[a][b])
        except KeyError:
            return np.inf

    edge_lengths = []
    for s in spines:
        u, v = s.edge_source, s.edge_target
        if graph.G.has_edge(u, v):
            edge_lengths.append(float(graph.G[u][v]["length"]))
        else:
            edge_lengths.append(float(np.linalg.norm(s.projected_point - graph.node_position(v))))

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = spines[i], spines[j]
            L_i = edge_lengths[i]
            L_j = edge_lengths[j]
            t_i = si.edge_position
            t_j = sj.edge_position
            u_i, v_i = si.edge_source, si.edge_target
            u_j, v_j = sj.edge_source, sj.edge_target

            if u_i == u_j and v_i == v_j:
                d = abs(t_i - t_j) * L_i
            else:
                d1 = t_i * L_i + _node_dist(u_i, u_j) + t_j * L_j
                d2 = t_i * L_i + _node_dist(u_i, v_j) + (1.0 - t_j) * L_j
                d3 = (1.0 - t_i) * L_i + _node_dist(v_i, u_j) + t_j * L_j
                d4 = (1.0 - t_i) * L_i + _node_dist(v_i, v_j) + (1.0 - t_j) * L_j
                d = min(d1, d2, d3, d4)

            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def compute_spine_pairwise_distances_mesh(
    dendrite_mesh: Any,
    spines: List[ProjectedSpine],
) -> np.ndarray:
    from dendrite_analysis.surface_distances import (  # type: ignore
        calculate_mesh_graph_distance_matrix,
    )

    attachment_points = [s.original_point for s in spines]
    result = calculate_mesh_graph_distance_matrix(dendrite_mesh, attachment_points)
    return result.distance_matrix


def compute_distance_to_branching(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
) -> np.ndarray:
    branch_nodes = [
        n for n, d in graph.G.nodes(data=True)
        if d.get("node_type") in ("branch", "soma") or graph.G.degree(n) >= 3
    ]

    if not branch_nodes:
        return np.full(len(spines), np.inf)

    lengths = nx.multi_source_dijkstra_path_length(graph.G, branch_nodes, weight="length")
    node_to_branch_dist: Dict[int, float] = dict(lengths)

    result = np.empty(len(spines))
    edge_lens = {}

    for i, s in enumerate(spines):
        u, v = s.edge_source, s.edge_target
        key = (u, v)
        if key not in edge_lens:
            if graph.G.has_edge(u, v):
                edge_lens[key] = float(graph.G[u][v]["length"])
            else:
                edge_lens[key] = float(np.linalg.norm(graph.node_position(v) - graph.node_position(u)))
        L = edge_lens[key]
        t = s.edge_position

        d_via_u = node_to_branch_dist.get(u, np.inf) + t * L
        d_via_v = node_to_branch_dist.get(v, np.inf) + (1.0 - t) * L
        result[i] = min(d_via_u, d_via_v)

    return result


def compute_distance_to_terminals(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
) -> np.ndarray:
    terminal_nodes = [n for n in graph.G.nodes() if graph.G.degree(n) == 1]

    if not terminal_nodes:
        return np.full(len(spines), np.inf)

    lengths = nx.multi_source_dijkstra_path_length(graph.G, terminal_nodes, weight="length")
    node_to_terminal_dist: Dict[int, float] = dict(lengths)

    result = np.empty(len(spines))
    edge_lens: Dict[Tuple[int, int], float] = {}

    for i, s in enumerate(spines):
        u, v = s.edge_source, s.edge_target
        key = (u, v)
        if key not in edge_lens:
            if graph.G.has_edge(u, v):
                edge_lens[key] = float(graph.G[u][v]["length"])
            else:
                edge_lens[key] = float(np.linalg.norm(graph.node_position(v) - graph.node_position(u)))
        L = edge_lens[key]
        t = s.edge_position

        d_via_u = node_to_terminal_dist.get(u, np.inf) + t * L
        d_via_v = node_to_terminal_dist.get(v, np.inf) + (1.0 - t) * L
        result[i] = min(d_via_u, d_via_v)

    return result


# ===========================================================================
# Intensity analysis
# ===========================================================================

def estimate_binned_intensity(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    bin_size: float = 25.0,
) -> pd.DataFrame:
    if not spines:
        return pd.DataFrame(
            columns=[
                "distance_bin_start",
                "distance_bin_end",
                "spine_count",
                "network_length",
                "intensity",
            ]
        )

    spine_dists = np.array([s.distance_to_soma for s in spines])
    sample_pts, sample_sd = graph.sample_points_on_graph(step=1.0)

    max_dist = max(
        spine_dists.max() if len(spine_dists) else 0.0,
        sample_sd.max() if len(sample_sd) else 0.0,
    )
    n_bins = max(1, int(math.ceil(max_dist / bin_size)))

    total_len = graph.total_length
    n_samples = len(sample_sd)

    rows = []
    for k in range(n_bins):
        bin_start = k * bin_size
        bin_end = (k + 1) * bin_size

        count = int(np.sum((spine_dists >= bin_start) & (spine_dists < bin_end)))

        if n_samples > 0:
            frac = float(np.sum((sample_sd >= bin_start) & (sample_sd < bin_end))) / n_samples
            net_len = frac * total_len
        else:
            net_len = 0.0

        intensity = count / net_len if net_len > 1e-12 else 0.0
        rows.append(
            {
                "distance_bin_start": bin_start,
                "distance_bin_end": bin_end,
                "spine_count": count,
                "network_length": net_len,
                "intensity": intensity,
            }
        )

    return pd.DataFrame(rows)


def estimate_smooth_intensity(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    bandwidth: Optional[float] = None,
    eval_step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if not spines:
        return np.array([]), np.array([])

    spine_dists = np.array([s.distance_to_soma for s in spines], dtype=float)
    n = len(spine_dists)

    if bandwidth is None:
        sigma = float(np.std(spine_dists))
        if sigma < 1e-10:
            sigma = 1.0
        bandwidth = 1.06 * sigma * (n ** (-0.2))

    h = max(bandwidth, 1e-6)

    d_min = 0.0
    d_max = float(spine_dists.max()) + 3.0 * h
    d_grid = np.arange(d_min, d_max, eval_step)

    if len(d_grid) == 0:
        return d_grid, np.array([])

    diff = d_grid[:, None] - spine_dists[None, :]        
    kde = np.sum(np.exp(-0.5 * (diff / h) ** 2), axis=1) / (n * h * math.sqrt(2 * math.pi))

    sample_pts, sample_sd = graph.sample_points_on_graph(step=eval_step / 2.0)
    total_len = graph.total_length

    rho_net = np.zeros(len(d_grid))
    if len(sample_sd) > 0 and total_len > 0:
        for m, d in enumerate(d_grid):
            window = h
            mask = np.abs(sample_sd - d) <= window
            frac = float(np.sum(mask)) / len(sample_sd)
            rho_net[m] = frac * total_len / (2.0 * window) if window > 0 else 0.0

    lambda_hat = np.where(rho_net > 1e-12, kde / rho_net, 0.0)

    return d_grid, lambda_hat


def test_intensity_dependence(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    n_bins: int = 10,
) -> Dict[str, Any]:
    total_len = graph.total_length
    if total_len <= 0 or not spines:
        return {
            "constant_model": None,
            "distance_model": None,
            "lr_statistic": np.nan,
            "p_value": np.nan,
            "aic_constant": np.nan,
            "aic_distance": np.nan,
            "interpretation": "Insufficient data.",
        }

    spine_dists = np.array([s.distance_to_soma for s in spines])
    max_dist = float(spine_dists.max())
    bin_size = max_dist / n_bins
    bin_df = estimate_binned_intensity(graph, spines, bin_size=bin_size)
    bin_df = bin_df[bin_df["network_length"] > 0].copy()

    if len(bin_df) < 2:
        return {
            "constant_model": None,
            "distance_model": None,
            "lr_statistic": np.nan,
            "p_value": np.nan,
            "aic_constant": np.nan,
            "aic_distance": np.nan,
            "interpretation": "Insufficient bins for test.",
        }

    counts = bin_df["spine_count"].values.astype(float)
    net_len = bin_df["network_length"].values
    d_mid = 0.5 * (bin_df["distance_bin_start"].values + bin_df["distance_bin_end"].values)

    def poisson_neglogL_constant(theta: np.ndarray) -> float:
        mu = net_len * np.exp(theta[0])
        mu = np.maximum(mu, 1e-300)
        return -float(np.sum(counts * np.log(mu) - mu))

    def poisson_neglogL_distance(theta: np.ndarray) -> float:
        log_mu = np.log(net_len) + theta[0] + theta[1] * d_mid
        mu = np.exp(log_mu)
        mu = np.maximum(mu, 1e-300)
        return -float(np.sum(counts * np.log(mu) - mu))

    res0 = minimize(poisson_neglogL_constant, x0=np.array([0.0]), method="L-BFGS-B")
    res1 = minimize(poisson_neglogL_distance, x0=np.array([0.0, 0.0]), method="L-BFGS-B")

    logL0 = -float(res0.fun)
    logL1 = -float(res1.fun)
    lr_stat = max(0.0, 2.0 * (logL1 - logL0))
    p_value = float(1.0 - chi2.cdf(lr_stat, df=1))
    n_spines = int(counts.sum())

    aic0 = -2.0 * logL0 + 2.0 * 1
    aic1 = -2.0 * logL1 + 2.0 * 2

    if p_value < 0.05:
        coef1 = float(res1.x[1])
        direction = "increasing" if coef1 > 0 else "decreasing"
        interp = (
            f"Significant distance dependence (p={p_value:.4f}): spine intensity is "
            f"{direction} with soma distance (β₁={coef1:.4f})."
        )
    else:
        interp = (
            f"No significant distance dependence detected (p={p_value:.4f}). "
            f"Homogeneous intensity is consistent with the data."
        )

    return {
        "constant_model": res0,
        "distance_model": res1,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "aic_constant": aic0,
        "aic_distance": aic1,
        "n_spines": n_spines,
        "interpretation": interp,
    }


# ===========================================================================
# Inhomogeneous Poisson model
# ===========================================================================

def _build_covariates(soma_distances: np.ndarray, covariate_names: List[str]) -> np.ndarray:
    n = len(soma_distances)
    cols = []
    for name in covariate_names:
        if name == "intercept":
            cols.append(np.ones(n))
        elif name == "distance_to_soma":
            cols.append(soma_distances.copy())
        elif name == "distance_to_soma_squared":
            cols.append(soma_distances ** 2)
        elif name == "log_distance_to_soma":
            cols.append(np.log(soma_distances + 1.0))
        else:
            raise ValueError(f"Unknown covariate name: {name!r}")
    if not cols:
        raise ValueError("covariate_names must not be empty.")
    return np.column_stack(cols)


def _network_integral(
    theta: np.ndarray,
    covariate_names: List[str],
    graph: DendriticGraph,
    n_samples_per_unit: float = 2.0,
) -> Tuple[float, np.ndarray]:
    step = max(1.0 / n_samples_per_unit, 1e-6)
    sample_pts, sample_sd = graph.sample_points_on_graph(step=step)

    if len(sample_sd) == 0:
        K = len(theta)
        return 0.0, np.zeros(K)

    X_samp = _build_covariates(sample_sd, covariate_names)
    total_len = graph.total_length
    actual_step = total_len / len(sample_sd)

    log_lambda = X_samp @ theta
    log_lambda = np.clip(log_lambda, -500, 500)
    lam = np.exp(log_lambda)         

    integral = float(np.sum(lam) * actual_step)
    grad_integral = (X_samp.T @ lam) * actual_step  

    return integral, grad_integral


def fit_inhomogeneous_poisson(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    covariates: Sequence[str] = ("intercept", "distance_to_soma", "distance_to_soma_squared"),
) -> PoissonModelResult:
    covariate_names = list(covariates)
    if not spines:
        raise ValueError("Cannot fit model: no spines provided.")

    spine_dists = np.array([s.distance_to_soma for s in spines], dtype=float)
    N = len(spines)
    K = len(covariate_names)

    X_spines = _build_covariates(spine_dists, covariate_names)  # (N, K)

    sum_X = X_spines.sum(axis=0)  # (K,)

    def neg_logL_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        sum_log_lambda = float(X_spines @ theta).real if np.isscalar(X_spines @ theta) \
            else float(np.sum(X_spines @ theta))
        integral, grad_integral = _network_integral(theta, covariate_names, graph)
        neg_ll = -(sum_log_lambda - integral)
        grad = -(sum_X - grad_integral)
        return float(neg_ll), grad

    L = graph.total_length
    theta0 = np.zeros(K)
    if "intercept" in covariate_names:
        idx0 = covariate_names.index("intercept")
        theta0[idx0] = math.log(max(N, 1) / max(L, 1e-6))

    result = minimize(
        neg_logL_and_grad,
        x0=theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )

    theta_hat = result.x

    eps = 1e-5
    hessian = np.zeros((K, K))
    _, g0 = neg_logL_and_grad(theta_hat)
    for k in range(K):
        theta_plus = theta_hat.copy()
        theta_plus[k] += eps
        _, gp = neg_logL_and_grad(theta_plus)
        hessian[:, k] = (gp - g0) / eps

    try:
        cov_matrix = np.linalg.inv(hessian)
        se = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(K, np.nan)

    log_likelihood = -float(result.fun)
    aic = -2.0 * log_likelihood + 2.0 * K
    bic = -2.0 * log_likelihood + K * math.log(max(N, 1))

    log_lambda_spines = X_spines @ theta_hat
    fitted_intensity = np.exp(np.clip(log_lambda_spines, -500, 500))

    n_bins = min(10, max(2, N // 5))
    bin_df = estimate_binned_intensity(graph, spines, bin_size=graph.total_length / n_bins)
    bin_df = bin_df[bin_df["network_length"] > 0]

    if len(bin_df) > 0:
        d_mids = 0.5 * (bin_df["distance_bin_start"].values + bin_df["distance_bin_end"].values)
        X_bins = _build_covariates(d_mids, covariate_names)
        log_lam_bins = X_bins @ theta_hat
        lam_bins = np.exp(np.clip(log_lam_bins, -500, 500))
        expected_counts = lam_bins * bin_df["network_length"].values
        observed_counts = bin_df["spine_count"].values.astype(float)
        residuals = np.where(
            expected_counts > 1e-12,
            (observed_counts - expected_counts) / np.sqrt(expected_counts),
            0.0,
        )
    else:
        residuals = np.array([])

    diagnostics = {
        "converged": bool(result.success),
        "message": result.message,
        "n_iterations": result.get("nit", None),
        "n_spines": N,
        "network_length": L,
    }

    return PoissonModelResult(
        coefficients=theta_hat,
        covariate_names=covariate_names,
        standard_errors=se,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        fitted_intensity=fitted_intensity,
        residuals=residuals,
        diagnostics=diagnostics,
    )


# ===========================================================================
# Ripley K function
# ===========================================================================

def ripley_k_network(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    r_values: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    dist_matrix: Optional[np.ndarray] = None,
    correction: str = "none",
) -> KFunctionResult:
    n = len(spines)
    r_values = np.asarray(r_values, dtype=float)

    if n < 2:
        k_obs = np.zeros_like(r_values)
        k_exp = 2.0 * r_values
        interp = "Too few spines for K function estimation."
        return KFunctionResult(
            r_values=r_values,
            k_observed=k_obs,
            k_expected=k_exp,
            method="homogeneous" if intensity is None else "inhomogeneous",
            interpretation=interp,
        )

    if dist_matrix is None:
        dist_matrix = compute_spine_pairwise_distances(graph, spines)

    L = graph.total_length
    method = "homogeneous" if intensity is None else "inhomogeneous"

    k_obs = np.zeros(len(r_values))

    if intensity is None:
        for ri, r in enumerate(r_values):
            count = float(np.sum(dist_matrix <= r) - n)  # subtract diagonal
            if correction == "geometric" and r > 1e-12:
                # Simple boundary correction: inflate by ratio expected/attainable
                corr_factor = 2.0 * r / max(min(2.0 * r, L), 1e-12)
            else:
                corr_factor = 1.0
            k_obs[ri] = (L / (n * n)) * count * corr_factor
    else:
        lam = np.asarray(intensity, dtype=float)
        lam_outer = np.outer(lam, lam)    
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(lam_outer > 1e-300, 1.0 / lam_outer, 0.0)
        np.fill_diagonal(weights, 0.0) 

        for ri, r in enumerate(r_values):
            mask = (dist_matrix <= r).astype(float)
            np.fill_diagonal(mask, 0.0)
            if correction == "geometric" and r > 1e-12:
                corr_factor = 2.0 * r / max(min(2.0 * r, L), 1e-12)
            else:
                corr_factor = 1.0
            k_obs[ri] = (1.0 / L) * float(np.sum(mask * weights)) * corr_factor

    k_exp = 2.0 * r_values

    dev = k_obs - k_exp
    max_pos_r = float(r_values[np.argmax(dev)]) if len(dev) else 0.0
    max_neg_r = float(r_values[np.argmin(dev)]) if len(dev) else 0.0
    if np.max(np.abs(dev)) < 0.1 * float(np.max(k_exp)) if np.max(k_exp) > 0 else True:
        interp = "Spine distribution is consistent with CSR on the network."
    elif np.max(dev) > 0 and np.max(dev) >= np.abs(np.min(dev)):
        interp = (
            f"Spines show clustering at short distances (K_obs > K_exp, "
            f"largest deviation at r ≈ {max_pos_r:.1f})."
        )
    else:
        interp = (
            f"Spines show regularity / inhibition at short distances "
            f"(K_obs < K_exp, largest deviation at r ≈ {max_neg_r:.1f})."
        )

    return KFunctionResult(
        r_values=r_values,
        k_observed=k_obs,
        k_expected=k_exp,
        method=method,
        interpretation=interp,
    )


def simulate_poisson_on_graph(
    graph: DendriticGraph,
    n_points: int,
    intensity_fn: Optional[Callable[[float], float]] = None,
) -> List[ProjectedSpine]:
    edges = list(graph.G.edges(data=True))
    if not edges:
        return []

    edge_list = [(u, v, float(d.get("length", 0.0))) for u, v, d in edges]
    lengths = np.array([e[2] for e in edge_list])
    total_len = float(lengths.sum())
    if total_len <= 0:
        return []

    soma_dist_map = graph.soma_distances()

    if intensity_fn is None:
        probs = lengths / total_len
        chosen_edges = np.random.choice(len(edge_list), size=n_points, p=probs)
        results = []
        for k, eidx in enumerate(chosen_edges):
            u, v, L = edge_list[eidx]
            t = float(np.random.uniform(0.0, 1.0))
            pu = graph.node_position(u)
            pv = graph.node_position(v)
            pt = pu + t * (pv - pu)
            d_u = soma_dist_map.get(u, 0.0)
            d_v = soma_dist_map.get(v, 0.0)
            soma_dist = min(d_u + t * L, d_v + (1.0 - t) * L)
            results.append(
                ProjectedSpine(
                    spine_id=f"sim_{k}",
                    original_point=pt.copy(),
                    projected_point=pt.copy(),
                    edge_id=f"{u}_{v}",
                    edge_source=u,
                    edge_target=v,
                    edge_position=t,
                    distance_to_edge=0.0,
                    distance_to_soma=soma_dist,
                )
            )
        return results
    else:
        sample_pts, sample_sd = graph.sample_points_on_graph(step=1.0)
        if len(sample_sd) == 0:
            return []

        lambda_vals = np.array([intensity_fn(float(d)) for d in sample_sd])
        lambda_max = float(np.max(lambda_vals))
        if lambda_max <= 0:
            return []

        n_candidates = max(n_points * 5, int(lambda_max * total_len * 2))

        results = []
        candidate_idx = 0

        batch_size = max(n_candidates, 100)
        probs = lengths / total_len

        while len(results) < n_points:
            chosen_edges = np.random.choice(len(edge_list), size=batch_size, p=probs)
            ts = np.random.uniform(0.0, 1.0, size=batch_size)
            uniforms = np.random.uniform(0.0, 1.0, size=batch_size)

            for eidx, t, u_rand in zip(chosen_edges, ts, uniforms):
                u, v, L = edge_list[eidx]
                d_u = soma_dist_map.get(u, 0.0)
                d_v = soma_dist_map.get(v, 0.0)
                soma_dist = min(d_u + t * L, d_v + (1.0 - t) * L)
                lam = float(intensity_fn(soma_dist))
                accept_prob = lam / lambda_max
                if u_rand <= accept_prob:
                    pu = graph.node_position(u)
                    pv = graph.node_position(v)
                    pt = pu + t * (pv - pu)
                    results.append(
                        ProjectedSpine(
                            spine_id=f"sim_{candidate_idx}",
                            original_point=pt.copy(),
                            projected_point=pt.copy(),
                            edge_id=f"{u}_{v}",
                            edge_source=u,
                            edge_target=v,
                            edge_position=float(t),
                            distance_to_edge=0.0,
                            distance_to_soma=soma_dist,
                        )
                    )
                    candidate_idx += 1
                    if len(results) >= n_points:
                        break

        return results[:n_points]


def compute_simulation_envelopes(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    r_values: np.ndarray,
    n_simulations: int = 99,
    alpha: float = 0.05,
    intensity_model: Optional[PoissonModelResult] = None,
) -> KFunctionResult:
    n = len(spines)
    r_values = np.asarray(r_values, dtype=float)

    if n < 2:
        k_exp = 2.0 * r_values
        return KFunctionResult(
            r_values=r_values,
            k_observed=np.zeros_like(r_values),
            k_expected=k_exp,
            k_lower=np.zeros_like(r_values),
            k_upper=k_exp.copy(),
            p_value=1.0,
            method="inhomogeneous",
            interpretation="Too few spines for simulation envelopes.",
        )

    if intensity_model is None:
        try:
            intensity_model = fit_inhomogeneous_poisson(graph, spines)
        except Exception as exc:
            warnings.warn(f"Intensity model fitting failed: {exc}; using homogeneous CSR.", stacklevel=2)
            intensity_model = None

    if intensity_model is not None:
        obs_intensity = intensity_model.fitted_intensity
    else:
        obs_intensity = None

    obs_dist = compute_spine_pairwise_distances(graph, spines)
    k_obs_result = ripley_k_network(
        graph, spines, r_values, intensity=obs_intensity, dist_matrix=obs_dist
    )
    k_observed = k_obs_result.k_observed
    k_expected = k_obs_result.k_expected

    if intensity_model is not None:
        covariate_names = intensity_model.covariate_names
        theta_hat = intensity_model.coefficients

        def intensity_fn(d: float) -> float:
            x = _build_covariates(np.array([d]), covariate_names)[0]
            return float(np.exp(np.clip(np.dot(x, theta_hat), -500, 500)))
    else:
        intensity_fn = None

    k_sim_all = np.zeros((n_simulations, len(r_values)))
    iter_range: Any = range(n_simulations)
    if _TQDM:
        iter_range = _tqdm(iter_range, desc="K envelopes", leave=False)

    for sim in iter_range:
        sim_spines = simulate_poisson_on_graph(graph, n_points=n, intensity_fn=intensity_fn)
        if len(sim_spines) < 2:
            k_sim_all[sim] = k_expected
            continue
        sim_intensity: Optional[np.ndarray] = None
        if intensity_model is not None:
            sim_dists = np.array([s.distance_to_soma for s in sim_spines])
            X_sim = _build_covariates(sim_dists, intensity_model.covariate_names)
            sim_intensity = np.exp(np.clip(X_sim @ intensity_model.coefficients, -500, 500))
        sim_dist_mat = compute_spine_pairwise_distances(graph, sim_spines)
        sim_k = ripley_k_network(
            graph, sim_spines, r_values, intensity=sim_intensity, dist_matrix=sim_dist_mat
        )
        k_sim_all[sim] = sim_k.k_observed

    lo_idx = int(math.floor(alpha / 2.0 * n_simulations))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_simulations))
    lo_idx = max(0, lo_idx)
    hi_idx = min(n_simulations - 1, hi_idx)

    k_sim_sorted = np.sort(k_sim_all, axis=0)
    k_lower = k_sim_sorted[lo_idx]
    k_upper = k_sim_sorted[hi_idx]

    obs_dev = float(np.max(np.abs(k_observed - k_expected)))
    sim_devs = np.max(np.abs(k_sim_all - k_expected[None, :]), axis=1)
    p_value = float(np.mean(sim_devs >= obs_dev))

    if p_value < alpha:
        if np.mean(k_observed - k_expected) > 0:
            interp = (
                f"Significant clustering (p={p_value:.4f}): K_obs > K_exp "
                f"relative to {n_simulations} inhomogeneous CSR simulations."
            )
        else:
            interp = (
                f"Significant regularity (p={p_value:.4f}): K_obs < K_exp "
                f"relative to {n_simulations} inhomogeneous CSR simulations."
            )
    else:
        interp = (
            f"No significant departure from inhomogeneous CSR (p={p_value:.4f}, "
            f"n_sim={n_simulations})."
        )

    return KFunctionResult(
        r_values=r_values,
        k_observed=k_observed,
        k_expected=k_expected,
        k_lower=k_lower,
        k_upper=k_upper,
        p_value=p_value,
        method=k_obs_result.method,
        interpretation=interp,
    )


# ===========================================================================
# Group comparison
# ===========================================================================

def group_k_analysis(
    graphs_and_spines: Dict[str, Tuple[DendriticGraph, List[ProjectedSpine]]],
    r_values: np.ndarray,
    group_by: str = "group",
) -> Dict[str, KFunctionResult]:
    results: Dict[str, KFunctionResult] = {}
    for group_name, (graph, spines) in graphs_and_spines.items():
        try:
            results[group_name] = ripley_k_network(graph, spines, r_values)
        except Exception as exc:
            warnings.warn(f"K function failed for group {group_name!r}: {exc}", stacklevel=2)
    return results


def permutation_test(
    group_a_results: List[KFunctionResult],
    group_b_results: List[ProjectedSpine],
    n_permutations: int = 999,
) -> Dict[str, Any]:
    warnings.warn(
        "permutation_test() has a non-standard signature; use permutation_test_groups() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {"p_value": np.nan, "message": "Use permutation_test_groups."}


def permutation_test_groups(
    group_a: List[Tuple[DendriticGraph, List[ProjectedSpine]]],
    group_b: List[Tuple[DendriticGraph, List[ProjectedSpine]]],
    r_values: np.ndarray,
    n_permutations: int = 999,
) -> Dict[str, Any]:
    r_values = np.asarray(r_values, dtype=float)
    dr = float(r_values[1] - r_values[0]) if len(r_values) > 1 else 1.0

    def _mean_k(group: List[Tuple[DendriticGraph, List[ProjectedSpine]]]) -> np.ndarray:
        ks = []
        for graph, spines in group:
            try:
                res = ripley_k_network(graph, spines, r_values)
                ks.append(res.k_observed)
            except Exception:
                pass
        if not ks:
            return np.zeros_like(r_values)
        return np.mean(np.array(ks), axis=0)

    k_a = _mean_k(group_a)
    k_b = _mean_k(group_b)
    observed_diff = float(np.sum(np.abs(k_a - k_b)) * dr)

    # Pool all data
    all_data = list(group_a) + list(group_b)
    na = len(group_a)
    perm_diffs = []

    iter_range: Any = range(n_permutations)
    if _TQDM:
        iter_range = _tqdm(iter_range, desc="Permutation test", leave=False)

    rng = np.random.default_rng()
    for _ in iter_range:
        perm = rng.permutation(len(all_data))
        perm_a = [all_data[i] for i in perm[:na]]
        perm_b = [all_data[i] for i in perm[na:]]
        k_pa = _mean_k(perm_a)
        k_pb = _mean_k(perm_b)
        perm_diffs.append(float(np.sum(np.abs(k_pa - k_pb)) * dr))

    perm_diffs_arr = np.array(perm_diffs)
    p_value = float(np.mean(perm_diffs_arr >= observed_diff))

    return {
        "p_value": p_value,
        "observed_diff": observed_diff,
        "permutation_distribution": perm_diffs_arr,
        "r_values": r_values,
        "n_permutations": n_permutations,
    }


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_3d_network(
    graph: DendriticGraph,
    spines: Optional[List[ProjectedSpine]] = None,
    color_by: str = "none",
    intensity: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Dendritic Network",
) -> Optional[Any]:
    if not _PLOTLY:
        warnings.warn("plotly is required for plot_3d_network.", stacklevel=2)
        return None

    fig = _go.Figure()

    # Draw skeleton edges
    for u, v, edata in graph.G.edges(data=True):
        pu = graph.node_position(u)
        pv = graph.node_position(v)
        fig.add_trace(
            _go.Scatter3d(
                x=[pu[0], pv[0], None],
                y=[pu[1], pv[1], None],
                z=[pu[2], pv[2], None],
                mode="lines",
                line=dict(color="steelblue", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Draw nodes, coloured by node_type
    node_type_colors = {
        "terminal": "green",
        "branch": "orange",
        "soma": "red",
        "intermediate": "lightblue",
    }
    type_groups: Dict[str, List[int]] = {}
    for node, ndata in graph.G.nodes(data=True):
        nt = ndata.get("node_type", "intermediate")
        type_groups.setdefault(nt, []).append(node)

    for nt, node_ids in type_groups.items():
        pts = np.array([graph.node_position(n) for n in node_ids])
        fig.add_trace(
            _go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=4, color=node_type_colors.get(nt, "gray")),
                name=nt,
                legendgroup=nt,
            )
        )

    # Draw spines
    if spines:
        spine_pts = np.array([s.projected_point for s in spines])
        if color_by == "distance_to_soma":
            colors = [s.distance_to_soma for s in spines]
            colorscale = "Viridis"
            colorbar = dict(title="Soma dist")
        elif color_by == "intensity" and intensity is not None:
            colors = list(intensity)
            colorscale = "Hot"
            colorbar = dict(title="λ(x)")
        elif color_by == "dendrite_type":
            # Map type strings to integer codes
            types = [graph.G.nodes.get(s.edge_source, {}).get("dendrite_type", "") for s in spines]
            unique_types = list(set(types))
            type_to_int = {t: i for i, t in enumerate(unique_types)}
            colors = [type_to_int[t] for t in types]
            colorscale = "Set1"
            colorbar = dict(title="Type")
        else:
            colors = "crimson"
            colorscale = None
            colorbar = None

        marker_kwargs: Dict[str, Any] = dict(size=5, color=colors)
        if colorscale:
            marker_kwargs["colorscale"] = colorscale
            marker_kwargs["showscale"] = True
            if colorbar:
                marker_kwargs["colorbar"] = colorbar

        fig.add_trace(
            _go.Scatter3d(
                x=spine_pts[:, 0], y=spine_pts[:, 1], z=spine_pts[:, 2],
                mode="markers",
                marker=marker_kwargs,
                name="spines",
            )
        )

    fig.update_layout(
        title=title,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(aspectmode="data"),
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)

    return fig


def plot_intensity_vs_distance(
    binned: pd.DataFrame,
    smooth_d: Optional[np.ndarray] = None,
    smooth_lambda: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    if not _MPL:
        warnings.warn("matplotlib is required for plot_intensity_vs_distance.", stacklevel=2)
        return None

    fig, ax = _plt.subplots(figsize=(8, 4))

    if len(binned) > 0:
        d_mid = 0.5 * (binned["distance_bin_start"].values + binned["distance_bin_end"].values)
        width = binned["distance_bin_end"].values[0] - binned["distance_bin_start"].values[0]
        ax.bar(
            d_mid,
            binned["intensity"].values,
            width=width * 0.9,
            color="steelblue",
            alpha=0.6,
            label="Binned intensity",
        )

    if smooth_d is not None and smooth_lambda is not None:
        ax.plot(smooth_d, smooth_lambda, color="crimson", lw=2, label="Kernel estimate")

    ax.set_xlabel("Distance from soma")
    ax.set_ylabel("Intensity (spines / unit length)")
    ax.set_title("Spine intensity vs soma distance")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_ripley_k(
    result: KFunctionResult,
    title: str = "Ripley K (network)",
    save_path: Optional[str] = None,
) -> Optional[Any]:
    if not _MPL:
        warnings.warn("matplotlib is required for plot_ripley_k.", stacklevel=2)
        return None

    fig, ax = _plt.subplots(figsize=(7, 5))
    r = result.r_values
    ax.plot(r, result.k_observed, color="steelblue", lw=2, label="K observed")
    ax.plot(r, result.k_expected, color="black", lw=1.5, ls="--", label="K expected (2r)")

    if result.k_lower is not None and result.k_upper is not None:
        ax.fill_between(r, result.k_lower, result.k_upper, alpha=0.25, color="orange", label="Envelope")

    ax.set_xlabel("r")
    ax.set_ylabel("K(r)")
    ax.set_title(title)
    ax.legend()

    if result.p_value is not None:
        ax.text(
            0.02, 0.97,
            f"p = {result.p_value:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_k_deviation(
    result: KFunctionResult,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    if not _MPL:
        warnings.warn("matplotlib is required for plot_k_deviation.", stacklevel=2)
        return None

    fig, ax = _plt.subplots(figsize=(7, 4))
    r = result.r_values
    deviation = result.k_observed - result.k_expected
    ax.plot(r, deviation, color="steelblue", lw=2, label="K observed − K expected")
    ax.axhline(0, color="black", lw=1, ls="--")

    if result.k_lower is not None and result.k_upper is not None:
        lower_dev = result.k_lower - result.k_expected
        upper_dev = result.k_upper - result.k_expected
        ax.fill_between(r, lower_dev, upper_dev, alpha=0.25, color="orange", label="Envelope")

    ax.set_xlabel("r")
    ax.set_ylabel("K(r) − 2r")
    ax.set_title("K deviation from CSR expectation")
    ax.legend()

    if result.p_value is not None:
        ax.text(0.02, 0.97, f"p = {result.p_value:.4f}", transform=ax.transAxes, va="top", fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_group_comparison(
    results: Dict[str, KFunctionResult],
    save_path: Optional[str] = None,
) -> Optional[Any]:
    if not _MPL:
        warnings.warn("matplotlib is required for plot_group_comparison.", stacklevel=2)
        return None

    fig, ax = _plt.subplots(figsize=(8, 5))
    colors = _plt.rcParams["axes.prop_cycle"].by_key()["color"] if _MPL else []

    for idx, (group_name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)] if colors else "steelblue"
        ax.plot(result.r_values, result.k_observed, lw=2, color=color, label=group_name)
        if result.k_lower is not None and result.k_upper is not None:
            ax.fill_between(result.r_values, result.k_lower, result.k_upper, alpha=0.15, color=color)

    if results:
        first = next(iter(results.values()))
        ax.plot(first.r_values, first.k_expected, color="black", lw=1.5, ls="--", label="CSR (2r)")

    ax.set_xlabel("r")
    ax.set_ylabel("K(r)")
    ax.set_title("Group K function comparison")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    binned_intensity: pd.DataFrame,
    poisson_result: Optional[PoissonModelResult],
    k_result: Optional[KFunctionResult],
    output_dir: str,
    format: str = "html",
    title: str = "Dendritic Spine Spatial Analysis",
) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    N = len(spines)
    L = graph.total_length
    density = N / L if L > 1e-12 else 0.0
    mean_dist = float(np.mean([s.distance_to_soma for s in spines])) if spines else 0.0

    summary_lines = [
        f"# {title}",
        "",
        "## Summary Statistics",
        "",
        f"- **Number of spines (N):** {N}",
        f"- **Total network length (L):** {L:.2f}",
        f"- **Homogeneous density (ρ = N/L):** {density:.4f}",
        f"- **Mean soma distance:** {mean_dist:.2f}",
        f"- **Number of nodes:** {graph.n_nodes}",
        f"- **Number of edges:** {graph.n_edges}",
        "",
    ]

    poisson_lines: List[str] = []
    if poisson_result is not None:
        poisson_lines += [
            "## Inhomogeneous Poisson Model",
            "",
            f"- **Log-likelihood:** {poisson_result.log_likelihood:.4f}",
            f"- **AIC:** {poisson_result.aic:.4f}",
            f"- **BIC:** {poisson_result.bic:.4f}",
            "",
            "| Covariate | Coefficient | Std Error |",
            "|-----------|-------------|-----------|",
        ]
        for name, coef, se in zip(
            poisson_result.covariate_names,
            poisson_result.coefficients,
            poisson_result.standard_errors,
        ):
            poisson_lines.append(f"| {name} | {coef:.4f} | {se:.4f} |")
        poisson_lines += [""]

    k_lines: List[str] = []
    if k_result is not None:
        p_str = f"{k_result.p_value:.4f}" if k_result.p_value is not None else "N/A"
        k_lines += [
            "## Ripley K Function",
            "",
            f"- **Method:** {k_result.method}",
            f"- **p-value:** {p_str}",
            f"- **Interpretation:** {k_result.interpretation}",
            "",
        ]

    plots_dir = out_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_paths: Dict[str, str] = {}

    if _MPL:
        smooth_d, smooth_lam = estimate_smooth_intensity(graph, spines)
        fig_int = plot_intensity_vs_distance(
            binned_intensity,
            smooth_d=smooth_d,
            smooth_lambda=smooth_lam,
            save_path=str(plots_dir / "intensity_vs_distance.png"),
        )
        if fig_int is not None:
            plot_paths["intensity"] = "plots/intensity_vs_distance.png"
            _plt.close(fig_int)

        if k_result is not None:
            fig_k = plot_ripley_k(
                k_result,
                save_path=str(plots_dir / "ripley_k.png"),
            )
            if fig_k is not None:
                plot_paths["ripley_k"] = "plots/ripley_k.png"
                _plt.close(fig_k)

            fig_dev = plot_k_deviation(
                k_result,
                save_path=str(plots_dir / "k_deviation.png"),
            )
            if fig_dev is not None:
                plot_paths["k_deviation"] = "plots/k_deviation.png"
                _plt.close(fig_dev)

    # 3-D network figure (plotly)
    if _PLOTLY:
        net_fig_path = str(plots_dir / "network_3d.html")
        plot_3d_network(graph, spines, save_path=net_fig_path)
        plot_paths["network_3d"] = "plots/network_3d.html"

    # -- Assemble report -----------------------------------------------------
    lines = summary_lines + poisson_lines + k_lines

    if "intensity" in plot_paths:
        lines += [
            "## Intensity Plot",
            "",
            f"![Intensity vs Distance]({plot_paths['intensity']})",
            "",
        ]
    if "ripley_k" in plot_paths:
        lines += [
            "## Ripley K Plot",
            "",
            f"![Ripley K]({plot_paths['ripley_k']})",
            "",
        ]
    if "k_deviation" in plot_paths:
        lines += [
            "## K Deviation",
            "",
            f"![K deviation]({plot_paths['k_deviation']})",
            "",
        ]

    md_content = "\n".join(lines)

    if format == "md":
        report_file = out_path / "report.md"
        report_file.write_text(md_content, encoding="utf-8")
        return str(report_file)

    try:
        import markdown as _md_lib
        html_body = _md_lib.markdown(md_content, extensions=["tables"])
    except ImportError:
        # Fallback: simple <pre> wrapping
        html_body = f"<pre>{md_content}</pre>"

    iframe_html = ""
    if "network_3d" in plot_paths:
        rel = plot_paths["network_3d"]
        iframe_html = (
            f'<h2>Interactive 3-D Network</h2>'
            f'<iframe src="{rel}" width="100%" height="720" frameborder="0"></iframe>'
        )

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        f'<meta charset="utf-8">\n<title>{title}</title>\n'
        "<style>body{font-family:sans-serif;max-width:960px;margin:auto;padding:2em}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:4px 8px}"
        "</style>\n</head>\n<body>\n"
        + html_body
        + iframe_html
        + "\n</body>\n</html>"
    )

    report_file = out_path / "report.html"
    report_file.write_text(html, encoding="utf-8")
    return str(report_file)


# ===========================================================================
# Config loading and high-level pipeline
# ===========================================================================

def load_config(yaml_path: str) -> Dict[str, Any]:
    if not _YAML:
        raise ImportError(
            "PyYAML is required for load_config.  "
            "Install it with: pip install pyyaml"
        )
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with path.open("r", encoding="utf-8") as fh:
        config = _yaml.safe_load(fh)
    return config if isinstance(config, dict) else {}


def run_analysis(
    mesh_path: Optional[str] = None,
    config_path: Optional[str] = None,
    dendrite_mesh: Optional[Any] = None,
    spine_points: Optional[Dict[str, np.ndarray]] = None,
    soma_point: Optional[np.ndarray] = None,
    output_dir: str = "output_network_analysis",
    **kwargs: Any,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if config_path is not None:
        try:
            config = load_config(config_path)
        except Exception as exc:
            warnings.warn(f"Failed to load config {config_path!r}: {exc}", stacklevel=2)
    config.update(kwargs)

    if dendrite_mesh is None and mesh_path is not None:
        if not _TRIMESH:
            raise ImportError("trimesh is required to load mesh files.")
        import trimesh as _tr
        tm = _tr.load(mesh_path, process=False)
        if _CGAL:
            try:
                from spine_analysis.mesh.utils import v_f_to_mesh  
                verts = np.asarray(tm.vertices, dtype=float)
                faces = np.asarray(tm.faces, dtype=int)
                dendrite_mesh = v_f_to_mesh(verts, faces)
            except Exception as exc:
                warnings.warn(f"CGAL mesh conversion failed: {exc}; using trimesh directly.", stacklevel=2)
                dendrite_mesh = tm
        else:
            dendrite_mesh = tm

    if dendrite_mesh is None:
        raise ValueError("Either dendrite_mesh or mesh_path must be provided.")

    if spine_points is None or len(spine_points) == 0:
        raise ValueError("spine_points must be a non-empty dict of spine attachment points.")

    dendrite_id = config.get("dendrite_id", "d0")
    dendrite_type = config.get("dendrite_type", "unknown")
    snap_threshold = float(config.get("snap_threshold", 2.0))

    graph = build_dendritic_graph(
        dendrite_mesh,
        soma_point=soma_point,
        dendrite_id=dendrite_id,
        dendrite_type=dendrite_type,
        snap_threshold=snap_threshold,
    )

    max_dist_to_edge = float(config.get("max_distance_to_edge", 5.0))
    projected_spines, unassigned_ids = project_spines_to_graph(
        graph,
        spine_points,
        max_distance_to_edge=max_dist_to_edge,
    )

    if not projected_spines:
        warnings.warn(
            f"No spines could be projected (max_distance_to_edge={max_dist_to_edge}). "
            "Consider increasing max_distance_to_edge.",
            stacklevel=2,
        )

    _ = graph.soma_distances()

    bin_size = float(config.get("bin_size", 25.0))
    binned_intensity = estimate_binned_intensity(graph, projected_spines, bin_size=bin_size)

    bandwidth = config.get("bandwidth", None)
    smooth_d, smooth_lambda = estimate_smooth_intensity(
        graph, projected_spines, bandwidth=bandwidth
    )

    covariates = config.get(
        "covariates", ["intercept", "distance_to_soma", "distance_to_soma_squared"]
    )
    poisson_result: Optional[PoissonModelResult] = None
    if len(projected_spines) >= 3:
        try:
            poisson_result = fit_inhomogeneous_poisson(graph, projected_spines, covariates=covariates)
        except Exception as exc:
            warnings.warn(f"Poisson model fitting failed: {exc}", stacklevel=2)

    n_r = int(config.get("n_r_values", 20))
    r_max = float(config.get("r_max", graph.total_length * 0.3))
    r_values = np.linspace(0.0, r_max, n_r + 1)[1:]

    n_sim = int(config.get("n_simulations", 99))
    k_result: Optional[KFunctionResult] = None
    if len(projected_spines) >= 2:
        try:
            k_result = compute_simulation_envelopes(
                graph,
                projected_spines,
                r_values,
                n_simulations=n_sim,
                intensity_model=poisson_result,
            )
        except Exception as exc:
            warnings.warn(f"K function computation failed: {exc}", stacklevel=2)
            try:
                k_result = ripley_k_network(graph, projected_spines, r_values)
            except Exception:
                pass

    report_format = config.get("report_format", "html")
    try:
        report_path = generate_report(
            graph=graph,
            spines=projected_spines,
            binned_intensity=binned_intensity,
            poisson_result=poisson_result,
            k_result=k_result,
            output_dir=output_dir,
            format=report_format,
        )
    except Exception as exc:
        warnings.warn(f"Report generation failed: {exc}", stacklevel=2)
        report_path = ""

    return {
        "graph": graph,
        "projected_spines": projected_spines,
        "unassigned_ids": unassigned_ids,
        "binned_intensity": binned_intensity,
        "smooth_d": smooth_d,
        "smooth_lambda": smooth_lambda,
        "poisson_result": poisson_result,
        "k_result": k_result,
        "report_path": report_path,
    }
