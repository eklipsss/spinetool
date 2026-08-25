from __future__ import annotations

import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import chi2, ks_2samp


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
# API
# ---------------------------------------------------------------------------
__all__ = [
    # dataclasses
    "ProjectedSpine",
    "PoissonModelResult",
    "KFunctionResult",
    
    # graph
    "DendriticGraph",
    "build_dendritic_graph_from_vertices_edges",
    "build_dendritic_graph",
    "build_dendritic_graph_from_skeleton",
    "build_graph_from_meshes",
    "load_standard_network",
    "save_standard_network",

    # spine attachment
    "project_spines_to_graph",
    "projected_spines_dataframe",

    # distances
    "compute_soma_distances",
    "compute_spine_pairwise_distances",
    "compute_spine_pairwise_distances_mesh",
    "compute_distance_to_branching",
    "compute_distance_to_terminals",
    "network_circumradius",
    "resolve_k_r_values",
    "auto_bin_size_for_network",

    # intensity
    "estimate_binned_intensity",
    "estimate_smooth_intensity",
    "test_intensity_dependence",
    "test_intensity_dependence_cdf",

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
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Преобразует результат K-анализа в таблицу.

        Входные данные: поля `KFunctionResult`.
        Выходные данные: `pandas.DataFrame`.
        """
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

class DendriticGraph:
    """Графовое представление дендритной сети.
    Хранит `networkx.Graph`, координаты узлов, длины рёбер и индекс
    узла сомы.

    Входные данные: объект создаётся пустым и заполняется функциями построения
    графа из skeleton или mesh.
    Выходные данные: объект для расчёта расстояний, интенсивности и K-функции
    на дендритной сети.
    """

    def __init__(self) -> None:
        """Инициализирует пустой дендритный граф.
        Создаёт пустой `networkx.Graph`, сбрасывает soma node и кэш
        расстояний от сомы.

        Выходные данные: пустой объект `DendriticGraph`.
        """
        self.G: nx.Graph = nx.Graph()
        self.soma_node: Optional[int] = None
        self._soma_distances: Optional[Dict[int, float]] = None
        self._sample_points_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    def __getstate__(self) -> Dict[str, Any]:
        """Подготавливает граф к сериализации между процессами.
        Исключает кэш сэмплированных точек, чтобы не копировать
        крупные массивы в дочерние процессы.

        Входные данные: текущий объект `DendriticGraph`.
        Выходные данные: словарь состояния объекта.
        """
        state = self.__dict__.copy()
        state["_sample_points_cache"] = {}
        return state

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_length(self) -> float:
        """Возвращает суммарную длину всех рёбер графа.

        Входные данные: текущий граф с атрибутом ребра `length`.
        Выходные данные: общая длина дендритной сети.
        """
        return float(sum(d["length"] for _, _, d in self.G.edges(data=True)))

    @property
    def n_nodes(self) -> int:
        """Возвращает число узлов графа.

        Входные данные: текущий `networkx.Graph`.
        Выходные данные: количество узлов.
        """
        return self.G.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """Возвращает число рёбер графа.

        Входные данные: текущий `networkx.Graph`.
        Выходные данные: количество рёбер.
        """
        return self.G.number_of_edges()

    @property
    def nodes_df(self) -> pd.DataFrame:
        """Экспортирует узлы графа в таблицу.
        Формирует строки с id, координатами, типом узла,
        dendrite_id и dendrite_type.

        Входные данные: текущий граф с координатами и атрибутами узлов.
        Выходные данные: `pandas.DataFrame` узлов.
        """
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
        """Экспортирует рёбра графа в таблицу.
        Формирует строки source, target и length.

        Входные данные: текущий граф с длинами рёбер.
        Выходные данные: `pandas.DataFrame` рёбер.
        """
        rows = []
        for u, v, data in self.G.edges(data=True):
            rows.append({"source": u, "target": v, "length": data.get("length", 0.0)})
        return pd.DataFrame(rows)

    # ===========================================================================

    def node_position(self, node: int) -> np.ndarray:
        """Возвращает координаты узла графа.

        Входные данные: id узла.
        Выходные данные: трёхмерная координата узла.
        """
        pos = self.G.nodes[node].get("pos", np.zeros(3))
        return np.asarray(pos, dtype=float)

    def all_node_positions(self) -> np.ndarray:
        """Возвращает координаты всех узлов графа.

        Входные данные: текущий граф.
        Выходные данные: массив формы `(n_nodes, 3)`.
        """
        nodes = list(self.G.nodes())
        if not nodes:
            return np.empty((0, 3))
        return np.stack([self.node_position(n) for n in nodes])

    # ===========================================================================

    def soma_distances(self, recompute: bool = False) -> Dict[int, float]:
        """Вычисляет расстояния от сомы до всех узлов графа.
        Запускает Dijkstra от `soma_node`, 
        если soma node не задан, использует первый узел и выдаёт предупреждение.

        Входные данные: флаг принудительного пересчёта.
        Выходные данные: словарь `{node_id: distance_to_soma}`.
        """
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
        """Вычисляет кратчайшие расстояния между всеми узлами графа.

        Входные данные: текущий граф с длинами рёбер.
        Выходные данные: вложенный словарь расстояний между узлами.
        """
        return dict(nx.all_pairs_dijkstra_path_length(self.G, weight="length"))

    def sample_points_on_graph(
        self, step: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сэмплирует точки вдоль рёбер дендритного графа.
        Размещает точки в серединах равномерных сегментов каждого
        ребра и вычисляет расстояние каждой точки от сомы.

        Входные данные: шаг сэмплирования вдоль рёбер.
        Выходные данные: массив координат точек и массив расстояний от сомы.
        """
        step = float(max(step, 1e-9))
        if not hasattr(self, "_sample_points_cache"):
            self._sample_points_cache = {}
        cache_key = round(step, 9)
        cached = self._sample_points_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.G.number_of_edges() == 0:
            return np.empty((0, 3)), np.empty((0,))

        sd = self.soma_distances()
        all_pts: List[np.ndarray] = []
        all_sd: List[np.ndarray] = []

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
            pts = p_u[None, :] + ts[:, None] * (p_v - p_u)[None, :]
            dists = np.minimum(d_u + ts * length, d_v + (1.0 - ts) * length)
            all_pts.append(pts)
            all_sd.append(dists)

        if not all_pts:
            return np.empty((0, 3)), np.empty((0,))

        result = (np.vstack(all_pts), np.concatenate(all_sd))
        self._sample_points_cache[cache_key] = result
        return result


# ===========================================================================

def _graph_from_skeleton_segments(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    dendrite_id: str = "d0",
    dendrite_type: str = "unknown",
) -> DendriticGraph:
    """Строит дендритный граф из списка skeleton-сегментов.
    Создаёт узлы по координатам концов сегментов, добавляет рёбра с
    длинами и классифицирует узлы как terminal, branch или intermediate.

    Входные данные: список пар трёхмерных точек, идентификатор дендрита и тип
    дендрита.
    Выходные данные: объект `DendriticGraph`.
    """
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


def _polyline_segments(points: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Преобразует полилинию в список соседних сегментов.

    Входные данные: массив точек формы `(n, >=3)`.
    Выходные данные: список сегментов `(start, end)`.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
        return []
    pts = pts[:, :3]
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if len(pts) < 2:
        return []
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _segments_from_skeleton_object(skeleton: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Извлекает сегменты из skeleton-объекта произвольного поддерживаемого формата.
    Ищет пары точек, полилинии или структуру `points`/`edges` и
    рекурсивно преобразует их в сегменты.

    Входные данные: skeleton как массив, словарь, список или вложенная
    объектная структура.
    Выходные данные: список трёхмерных сегментов.
    """
    if skeleton is None:
        return []

    if isinstance(skeleton, np.ndarray) and skeleton.dtype == object:
        if skeleton.shape == ():
            return _segments_from_skeleton_object(skeleton.item())
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
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
                    return [
                        (points[int(u), :3], points[int(v), :3])
                        for u, v in valid_edges
                    ]
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


def _coord_columns(frame: pd.DataFrame) -> Tuple[str, str, str]:
    """Определяет имена координатных столбцов.

    Входные данные: таблица с координатами.
    Выходные данные: tuple имён столбцов `(x, y, z)`.
    """
    lower_to_column = {str(column).lower(): column for column in frame.columns}
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        if all(name.lower() in lower_to_column for name in names):
            return tuple(lower_to_column[name.lower()] for name in names)  # type: ignore[return-value]
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    numeric_columns = [column for column in numeric_columns if str(column).lower() not in {"vertex_id", "spine_id"}]
    if len(numeric_columns) >= 3:
        return str(numeric_columns[0]), str(numeric_columns[1]), str(numeric_columns[2])
    raise ValueError("Cannot infer x/y/z coordinate columns.")


def build_dendritic_graph_from_vertices_edges(
    vertices: Union[pd.DataFrame, np.ndarray],
    edges: Union[pd.DataFrame, np.ndarray],
    root_vertex_id: Optional[int] = None,
    soma_point: Optional[np.ndarray] = None,
    dendrite_id: str = "d0",
    dendrite_type: str = "unknown",
    snap_threshold: float = 2.0,
) -> DendriticGraph:
    """Строит дендритный граф из таблиц vertices/edges.

    Входные данные: таблица/массив вершин, таблица/массив рёбер, id корневой
    вершины или точка сомы, идентификатор и тип дендрита.
    Выходные данные: объект `DendriticGraph`.
    """
    if isinstance(vertices, pd.DataFrame):
        vertices_frame = vertices.copy()
        x_col, y_col, z_col = _coord_columns(vertices_frame)
        if "vertex_id" in vertices_frame.columns:
            vertex_ids = vertices_frame["vertex_id"].astype(int).to_numpy()
        else:
            vertex_ids = np.arange(len(vertices_frame), dtype=int)
        coords = vertices_frame[[x_col, y_col, z_col]].to_numpy(dtype=float)
    else:
        coords = np.asarray(vertices, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 3:
            raise ValueError("vertices array must have shape (n_vertices, >=3).")
        coords = coords[:, :3]
        vertex_ids = np.arange(len(coords), dtype=int)

    if len(vertex_ids) != len(coords):
        raise ValueError("Number of vertex ids does not match number of coordinates.")
    if not np.isfinite(coords).all():
        raise ValueError("vertices contain NaN or Inf coordinates.")

    id_to_pos = {int(vertex_id): np.asarray(coord, dtype=float) for vertex_id, coord in zip(vertex_ids, coords)}
    graph = DendriticGraph()
    for vertex_id, coord in id_to_pos.items():
        graph.G.add_node(
            int(vertex_id),
            pos=np.asarray(coord, dtype=float),
            node_type="intermediate",
            dendrite_id=dendrite_id,
            dendrite_type=dendrite_type,
        )

    if isinstance(edges, pd.DataFrame):
        if not {"source", "target"}.issubset(edges.columns):
            raise ValueError("edges DataFrame must contain 'source' and 'target' columns.")
        edge_array = edges[["source", "target"]].to_numpy(dtype=int)
    else:
        edge_array = np.asarray(edges, dtype=int)
        if edge_array.ndim != 2 or edge_array.shape[1] < 2:
            raise ValueError("edges array must have shape (n_edges, >=2).")
        edge_array = edge_array[:, :2]

    for source, target in edge_array:
        source = int(source)
        target = int(target)
        if source == target or source not in id_to_pos or target not in id_to_pos:
            continue
        length = float(np.linalg.norm(id_to_pos[target] - id_to_pos[source]))
        if length <= 0:
            continue
        if graph.G.has_edge(source, target):
            graph.G[source][target]["length"] = min(float(graph.G[source][target]["length"]), length)
        else:
            graph.G.add_edge(source, target, length=length)

    for node in graph.G.nodes():
        degree = graph.G.degree(node)
        if degree == 1:
            graph.G.nodes[node]["node_type"] = "terminal"
        elif degree >= 3:
            graph.G.nodes[node]["node_type"] = "branch"
        else:
            graph.G.nodes[node]["node_type"] = "intermediate"

    if root_vertex_id is not None and int(root_vertex_id) in graph.G:
        graph.soma_node = int(root_vertex_id)
        graph.G.nodes[graph.soma_node]["node_type"] = "soma"
    elif soma_point is not None and graph.G.number_of_nodes() > 0:
        soma_pt = np.asarray(soma_point, dtype=float)
        node_ids = list(graph.G.nodes())
        positions = np.asarray([graph.node_position(node) for node in node_ids], dtype=float)
        distances = np.linalg.norm(positions - soma_pt, axis=1)
        nearest_idx = int(np.argmin(distances))
        if distances[nearest_idx] <= snap_threshold:
            graph.soma_node = int(node_ids[nearest_idx])
            graph.G.nodes[graph.soma_node]["node_type"] = "soma"
    elif 0 in graph.G:
        graph.soma_node = 0
        graph.G.nodes[0]["node_type"] = "soma"

    return graph


def load_standard_network(
    network_dir: Union[str, Path],
    root_vertex_id: Optional[int] = None,
    dendrite_id: Optional[str] = None,
    dendrite_type: Optional[str] = None,
    project_spines: bool = True,
    max_distance_to_edge: float = np.inf,
) -> Tuple[DendriticGraph, Dict[str, np.ndarray], Dict[str, Any]]:
    """Загружает 3D-сеть.
    Строит граф из vertices/edges и читает точки шипиков X.

    Входные данные: папка с `vertices.csv`, `edges.csv`, `spines.csv` и
    опциональным `metadata.json`; параметры корня и проекции шипиков.
    Выходные данные: `(graph, spine_points, metadata)`.
    """
    network_dir = Path(network_dir)
    vertices_path = network_dir / "vertices.csv"
    edges_path = network_dir / "edges.csv"
    spines_path = network_dir / "spines.csv"
    metadata_path = network_dir / "metadata.json"
    if not vertices_path.exists() or not edges_path.exists() or not spines_path.exists():
        raise FileNotFoundError(
            f"Standard network folder must contain vertices.csv, edges.csv and spines.csv: {network_dir}"
        )

    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            metadata = loaded if isinstance(loaded, dict) else {}
    if root_vertex_id is None:
        root_value = metadata.get("root_vertex_id", metadata.get("soma_vertex_id", 0))
        root_vertex_id = None if root_value is None else int(root_value)
    soma_point = metadata.get("soma_point", None)
    soma_point_array = None if soma_point is None else np.asarray(soma_point, dtype=float)
    dendrite_id = dendrite_id or str(metadata.get("network_name", network_dir.name))
    dendrite_type = dendrite_type or str(metadata.get("group", metadata.get("dendrite_type", "unknown")))

    vertices = pd.read_csv(vertices_path)
    edges = pd.read_csv(edges_path)
    spines = pd.read_csv(spines_path)
    graph = build_dendritic_graph_from_vertices_edges(
        vertices,
        edges,
        root_vertex_id=root_vertex_id,
        soma_point=soma_point_array,
        dendrite_id=dendrite_id,
        dendrite_type=dendrite_type,
    )

    x_col, y_col, z_col = _coord_columns(spines)
    if "spine_id" in spines.columns:
        spine_ids = spines["spine_id"].astype(str).to_numpy()
    else:
        spine_ids = np.asarray([f"spine_{index}" for index in range(len(spines))], dtype=object)
    spine_points = {
        str(spine_id): np.asarray(point, dtype=float)
        for spine_id, point in zip(spine_ids, spines[[x_col, y_col, z_col]].to_numpy(dtype=float))
    }

    metadata = {
        **metadata,
        "standard_network_dir": str(network_dir),
        "root_vertex_id": root_vertex_id,
        "dendrite_id": dendrite_id,
        "dendrite_type": dendrite_type,
        "spines_are_projected_to_graph": bool(project_spines),
    }
    if project_spines:
        projected, unassigned = project_spines_to_graph(
            graph,
            spine_points,
            max_distance_to_edge=max_distance_to_edge,
        )
        metadata["projection_unassigned_count"] = int(len(unassigned))
        spine_points = {spine.spine_id: spine.original_point for spine in projected}
    return graph, spine_points, metadata


def save_standard_network(
    graph: DendriticGraph,
    spine_points: Union[Dict[str, np.ndarray], Sequence[ProjectedSpine]],
    output_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    network_name: Optional[str] = None,
    dendrite_type: str = "unknown",
    soma_point: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Сохраняет граф и точки шипиков в каноническом формате.
    Записывает `vertices.csv`, `edges.csv`, `spines.csv`,
    `matrix.npy` и `metadata.json`.

    Входные данные: `DendriticGraph`, точки шипиков или `ProjectedSpine`,
    папка вывода и metadata.
    Выходные данные: metadata сохранённой сети.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    network_name = network_name or output_dir.name
    metadata = dict(metadata or {})

    node_ids = list(graph.G.nodes())
    vertices_frame = pd.DataFrame(
        [
            {
                "vertex_id": int(node),
                "x": float(graph.node_position(node)[0]),
                "y": float(graph.node_position(node)[1]),
                "z": float(graph.node_position(node)[2]),
            }
            for node in node_ids
        ]
    )
    edges_frame = pd.DataFrame(
        [
            {
                "edge_id": index,
                "source": int(source),
                "target": int(target),
                "weight": 1.0,
                "length": float(data.get("length", np.linalg.norm(graph.node_position(source) - graph.node_position(target)))),
            }
            for index, (source, target, data) in enumerate(graph.G.edges(data=True))
        ]
    )

    if isinstance(spine_points, dict):
        spine_rows = [
            {
                "spine_id": str(spine_id),
                "x": float(np.asarray(point, dtype=float)[0]),
                "y": float(np.asarray(point, dtype=float)[1]),
                "z": float(np.asarray(point, dtype=float)[2]),
            }
            for spine_id, point in spine_points.items()
        ]
    else:
        spine_rows = [
            {
                "spine_id": str(spine.spine_id),
                "x": float(spine.original_point[0]),
                "y": float(spine.original_point[1]),
                "z": float(spine.original_point[2]),
                "projected_x": float(spine.projected_point[0]),
                "projected_y": float(spine.projected_point[1]),
                "projected_z": float(spine.projected_point[2]),
                "edge_source": int(spine.edge_source),
                "edge_target": int(spine.edge_target),
                "edge_position": float(spine.edge_position),
                "distance_to_edge": float(spine.distance_to_edge),
            }
            for spine in spine_points
        ]
    spines_frame = pd.DataFrame(spine_rows)

    vertices_frame.to_csv(output_dir / "vertices.csv", index=False)
    edges_frame.to_csv(output_dir / "edges.csv", index=False)
    spines_frame.to_csv(output_dir / "spines.csv", index=False)

    node_to_idx = {node: index for index, node in enumerate(node_ids)}
    matrix = np.zeros((len(node_ids), len(node_ids)), dtype=int)
    for source, target in graph.G.edges():
        matrix[node_to_idx[source], node_to_idx[target]] = 1
        matrix[node_to_idx[target], node_to_idx[source]] = 1
    np.save(output_dir / "matrix.npy", matrix)

    if soma_point is not None:
        soma_array = np.asarray(soma_point, dtype=float)
        soma_value = [float(soma_array[0]), float(soma_array[1]), float(soma_array[2])]
    else:
        soma_value = metadata.get("soma_point", None)

    saved_metadata = {
        **metadata,
        "network_name": network_name,
        "dendrite_type": dendrite_type,
        "root_vertex_id": graph.soma_node,
        "soma_point": soma_value,
        "soma_point_is_graph_vertex": bool(graph.soma_node is not None and soma_value is not None and np.allclose(
            graph.node_position(graph.soma_node),
            np.asarray(soma_value, dtype=float),
        )),
        "vertex_count": int(len(vertices_frame)),
        "edge_count": int(len(edges_frame)),
        "spine_count": int(len(spines_frame)),
        "coordinate_columns": ["x", "y", "z"],
        "matrix_shape": list(matrix.shape),
        "matrix_format": "adjacency_matrix",
        "indexing": "graph_vertex_ids",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(saved_metadata, handle, ensure_ascii=False, indent=2)
    return saved_metadata


def build_dendritic_graph_from_skeleton(
    skeleton: Any,
    soma_point: Optional[np.ndarray] = None,
    dendrite_id: str = "d0",
    dendrite_type: str = "unknown",
    snap_threshold: float = 2.0,
) -> DendriticGraph:
    """Строит дендритный граф напрямую из skeleton.
    Извлекает сегменты skeleton, строит граф и назначает soma node,
    если ближайший узел находится в пределах `snap_threshold`.

    Входные данные: skeleton, опциональная точка сомы, идентификатор и тип
    дендрита, радиус привязки сомы.
    Выходные данные: объект `DendriticGraph`.
    """
    segments = _segments_from_skeleton_object(skeleton)
    if not segments:
        raise ValueError("Skeleton does not contain any valid 3-D segments.")

    dg = _graph_from_skeleton_segments(
        segments,
        dendrite_id=dendrite_id,
        dendrite_type=dendrite_type,
    )

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


def build_dendritic_graph(
    dendrite_mesh: Any,
    soma_point: Optional[np.ndarray] = None,
    dendrite_id: str = "d0",
    dendrite_type: str = "unknown",
    snap_threshold: float = 2.0,
) -> DendriticGraph:
    """Строит дендритный граф из mesh через CGAL-skeletonization.

    Входные данные: mesh дендрита, опциональная точка сомы, идентификатор и тип
    дендрита, радиус привязки сомы.
    Выходные данные: объект `DendriticGraph`.
    """
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
    """Строит общий дендритный граф из набора mesh-объектов.
    Строит подграф для каждого mesh, объединяет их и добавляет рёбра
    между близкими узлами разных дендритов.

    Входные данные: словарь mesh-объектов дендритов, опциональный mesh сомы,
    карта типов дендритов и радиус сшивания.
    Выходные данные: единый объект `DendriticGraph`.
    """
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
    """Проецирует точку на отрезок.
    Вычисляет ближайшую точку на сегменте, параметр положения `t` и
    евклидово расстояние до сегмента.

    Входные данные: трёхмерная точка и два конца сегмента.
    Выходные данные: `(projected_point, t, distance)`.
    """
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
    """Проецирует точки крепления шипиков на дендритный граф.
    Для каждого шипика ищет ближайшее ребро графа, вычисляет
    проекцию на него и расстояние от сомы вдоль сети.

    Входные данные: граф, словарь точек шипиков, максимальное расстояние до
    ребра, число ближайших рёбер-кандидатов и режим обработки неподходящих
    точек.
    Выходные данные: список `ProjectedSpine` и список id неспроецированных
    шипиков.
    """
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
    """Преобразует список спроецированных шипиков в таблицу.
    Извлекает исходные координаты, координаты проекции, id ребра,
    положение на ребре, расстояние до ребра и расстояние от сомы.

    Входные данные: список объектов `ProjectedSpine`.
    Выходные данные: `pandas.DataFrame` с одной строкой на шипик.
    """
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
    """Возвращает расстояния от сомы до узлов графа.

    Входные данные: объект `DendriticGraph`.
    Выходные данные: словарь `{node_id: distance_to_soma}`.
    """
    return graph.soma_distances()


def compute_spine_pairwise_distances(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
) -> np.ndarray:
    """Вычисляет попарные расстояния между шипиками по дендритной сети.

    Входные данные: дендритный граф и список спроецированных шипиков.
    Выходные данные: квадратная матрица сетевых расстояний между шипиками.
    """
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


def network_circumradius(graph: DendriticGraph) -> float:
    """Для каждой связной компоненты оценивает диаметр двумя запусками
    Dijkstra и возвращает половину максимального диаметра.

    Входные данные: дендритный граф с длинами рёбер.
    Выходные данные: радиус сети в единицах длины графа.
    """
    if graph.G.number_of_edges() == 0:
        return 0.0

    radii: List[float] = []
    for component_nodes in nx.connected_components(graph.G):
        nodes = list(component_nodes)
        if len(nodes) < 2:
            radii.append(0.0)
            continue
        subgraph = graph.G.subgraph(nodes)
        start = nodes[0]
        first_dist = nx.single_source_dijkstra_path_length(subgraph, start, weight="length")
        if not first_dist:
            radii.append(0.0)
            continue
        u = max(first_dist, key=first_dist.get)
        second_dist = nx.single_source_dijkstra_path_length(subgraph, u, weight="length")
        diameter = max(second_dist.values()) if second_dist else 0.0
        radii.append(0.5 * float(diameter))
    return float(max(radii)) if radii else 0.0


def resolve_k_r_values(
    graph: DendriticGraph,
    n_r_values: int,
    r_max: Optional[Any] = "circumradius",
) -> Tuple[np.ndarray, float, str]:
    """Формирует сетку радиусов для сетевой K-функции.

    Входные данные: граф, число радиусов и значение `r_max`.
    Выходные данные: массив радиусов, использованный максимум и источник
    выбора радиуса.
    """
    n_r_values = int(max(1, n_r_values))
    source = "explicit"
    if r_max is None or str(r_max).lower() in {"auto", "circumradius", "reference"}:
        resolved_r_max = network_circumradius(graph)
        source = "circumradius"
    else:
        resolved_r_max = float(r_max)
    if not np.isfinite(resolved_r_max) or resolved_r_max <= 0:
        resolved_r_max = float(graph.total_length)
        source = "fallback_total_length"
    if not np.isfinite(resolved_r_max) or resolved_r_max <= 0:
        resolved_r_max = 1.0
        source = "fallback_unit"
    r_values = np.linspace(0.0, float(resolved_r_max), n_r_values + 1)[1:]
    return r_values, float(resolved_r_max), source


def compute_spine_pairwise_distances_mesh(
    dendrite_mesh: Any,
    spines: List[ProjectedSpine],
) -> np.ndarray:
    """Вычисляет попарные расстояния между шипиками по mesh дендрита.

    Входные данные: mesh дендрита и список спроецированных шипиков.
    Выходные данные: квадратная матрица расстояний.
    """
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
    """Вычисляет расстояния от шипиков до ближайшего узла ветвления.

    Входные данные: дендритный граф и список спроецированных шипиков.
    Выходные данные: массив расстояний до ближайшего ветвления.
    """
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
    """Вычисляет расстояния от шипиков до ближайшего терминального узла.

    Входные данные: дендритный граф и список спроецированных шипиков.
    Выходные данные: массив расстояний до ближайшей терминали.
    """
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

def _adaptive_graph_sample_step(
    graph: DendriticGraph,
    requested_step: float,
    max_samples: int = 50_000,
) -> float:
    """Подбирает шаг сэмплирования графа с ограничением числа точек.
    Увеличивает шаг, если заданный шаг создаёт слишком много
    точек на длинной дендритной сети.

    Входные данные: граф, желаемый шаг и максимальное число sample-точек.
    Выходные данные: эффективный шаг сэмплирования.
    """
    requested_step = float(max(requested_step, 1e-9))
    total_len = float(graph.total_length)
    if total_len <= 0 or max_samples <= 0:
        return requested_step
    min_step_for_budget = total_len / float(max_samples)
    return float(max(requested_step, min_step_for_budget))


def _print_network_timing(label: str, stage: str, seconds: float) -> None:
    """Печатает timing-log для этапов сетевого анализа.

    Входные данные: подпись анализа, название этапа и длительность в секундах.
    Выходные данные: строка timing-log в stdout.
    """
    print(f"[time][network] {label}: {stage} took {seconds:.3f}s", flush=True)


def auto_bin_size_for_network(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    min_bins: int = 8,
    max_bins: int = 30,
    min_expected_spines_per_bin: float = 3.0,
) -> float:
    """Подбирает диагностический bin_size для расстояния от сомы.
    Выбирает ширину бина по протяжённости distance-to-soma оси,
    числу шипиков и локальным расстояниям между соседними значениями
    distance-to-soma.

    Входные данные: дендритный граф, спроецированные шипики и ограничения
    на число бинов/ожидаемое число шипиков в бине.
    Выходные данные: положительная ширина бина.
    """
    soma_distances = np.asarray(list(graph.soma_distances().values()), dtype=float)
    spine_distances = np.asarray([spine.distance_to_soma for spine in spines], dtype=float)
    soma_distances = soma_distances[np.isfinite(soma_distances)]
    spine_distances = spine_distances[np.isfinite(spine_distances)]

    if len(soma_distances) > 0:
        distance_extent = float(np.max(soma_distances))
    elif len(spine_distances) > 0:
        distance_extent = float(np.max(spine_distances))
    else:
        distance_extent = float(graph.total_length)
    distance_extent = max(distance_extent, 1e-9)

    n_spines = int(len(spine_distances))
    if n_spines <= 0:
        return distance_extent / max(1, int(min_bins))

    target_n_bins = int(np.clip(np.sqrt(n_spines), int(min_bins), int(max_bins)))
    bin_size_from_bins = distance_extent / max(target_n_bins, 1)
    bin_size_from_counts = (
        float(min_expected_spines_per_bin) * distance_extent / max(n_spines, 1)
    )

    bin_size_from_local_spacing = 0.0
    if len(spine_distances) >= 2:
        sorted_distances = np.sort(spine_distances)
        local_gaps = np.diff(sorted_distances)
        local_gaps = local_gaps[np.isfinite(local_gaps) & (local_gaps > 0)]
        if len(local_gaps) > 0:
            bin_size_from_local_spacing = float(np.quantile(local_gaps, 0.75))

    return float(max(bin_size_from_bins, bin_size_from_counts, bin_size_from_local_spacing, 1e-9))


def estimate_binned_intensity(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    bin_size: float = 25.0,
    max_network_samples: int = 50_000,
) -> pd.DataFrame:
    """Оценивает биннированную линейную интенсивность шипиков.
    Делит ось расстояния от сомы на интервалы, считает число шипиков
    и длину сети в каждом интервале.

    Входные данные: граф, спроецированные шипики и размер бина по расстоянию
    от сомы.
    Выходные данные: таблица с количеством шипиков, длиной сети и интенсивностью
    в каждом бине.
    """
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
    sample_step = _adaptive_graph_sample_step(graph, requested_step=1.0, max_samples=max_network_samples)
    _, sample_sd = graph.sample_points_on_graph(step=sample_step)

    max_dist = max(
        spine_dists.max() if len(spine_dists) else 0.0,
        sample_sd.max() if len(sample_sd) else 0.0,
    )
    n_bins = max(1, int(math.ceil(max_dist / bin_size)))

    total_len = graph.total_length
    n_samples = len(sample_sd)

    bin_edges = np.arange(0.0, (n_bins + 1) * bin_size, bin_size)
    spine_counts, _ = np.histogram(spine_dists, bins=bin_edges)
    if n_samples > 0:
        sample_counts, _ = np.histogram(sample_sd, bins=bin_edges)
    else:
        sample_counts = np.zeros(n_bins, dtype=int)

    rows = []
    for k in range(n_bins):
        bin_start = k * bin_size
        bin_end = (k + 1) * bin_size

        if n_samples > 0:
            frac = float(sample_counts[k]) / n_samples
            net_len = frac * total_len
        else:
            net_len = 0.0

        count = int(spine_counts[k])
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
    max_eval_points: int = 50_000,
    max_network_samples: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Оценивает сглаженную интенсивность шипиков вдоль расстояния от сомы.
    Строит KDE по расстояниям шипиков от сомы и нормирует её на
    оценку длины сети в окрестности каждого расстояния.

    Входные данные: граф, спроецированные шипики, bandwidth и шаг сетки.
    Выходные данные: массив расстояний и массив сглаженной интенсивности.
    """
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
    if max_eval_points > 0 and d_max / max(eval_step, 1e-9) > max_eval_points:
        eval_step = d_max / float(max_eval_points)
    d_grid = np.arange(d_min, d_max, eval_step)

    if len(d_grid) == 0:
        return d_grid, np.array([])

    diff = d_grid[:, None] - spine_dists[None, :]        
    kde = np.sum(np.exp(-0.5 * (diff / h) ** 2), axis=1) / (n * h * math.sqrt(2 * math.pi))

    sample_step = _adaptive_graph_sample_step(
        graph,
        requested_step=eval_step / 2.0,
        max_samples=max_network_samples,
    )
    _, sample_sd = graph.sample_points_on_graph(step=sample_step)
    total_len = graph.total_length

    rho_net = np.zeros(len(d_grid))
    if len(sample_sd) > 0 and total_len > 0:
        window = h
        sample_sd_sorted = np.sort(sample_sd)
        left = np.searchsorted(sample_sd_sorted, d_grid - window, side="left")
        right = np.searchsorted(sample_sd_sorted, d_grid + window, side="right")
        counts = right - left
        frac = counts.astype(float) / len(sample_sd_sorted)
        rho_net = frac * total_len / (2.0 * window) if window > 0 else rho_net

    lambda_hat = np.zeros_like(kde, dtype=float)
    np.divide(kde, rho_net, out=lambda_hat, where=rho_net > 1e-12)

    return d_grid, lambda_hat


def test_intensity_dependence(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    n_bins: int = 10,
    max_network_samples: int = 50_000,
) -> Dict[str, Any]:
    """Проверяет зависимость интенсивности шипиков от расстояния до сомы.
    Сравнивает постоянную пуассоновскую модель и модель с линейной
    зависимостью от расстояния до сомы через тест отношения правдоподобия.

    Входные данные: граф, спроецированные шипики и число бинов.
    Выходные данные: словарь статистики LR-теста, p-значения, AIC и
    интерпретации.
    """
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
    bin_df = estimate_binned_intensity(
        graph,
        spines,
        bin_size=bin_size,
        max_network_samples=max_network_samples,
    )
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


def test_intensity_dependence_cdf(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    sample_step: float = 1.0,
    max_network_samples: int = 50_000,
) -> Dict[str, Any]:
    """Проверка зависимости от расстояния до сомы 
    с использованием сравнения функций распределения.
    Сравнивает распределение расстояний шипиков от сомы с
    распределением расстояний равномерно сэмплированных точек сети.


    Входные данные: граф, спроецированные шипики и шаг сэмплирования сети.
    Выходные данные: словарь KS-статистики, p-значения, размеров выборок и
    интерпретации.
    """
    if not spines:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_spines": 0,
            "n_network_samples": 0,
            "interpretation": "Insufficient data.",
        }

    sample_step = _adaptive_graph_sample_step(
        graph,
        requested_step=sample_step,
        max_samples=max_network_samples,
    )
    _, sample_sd = graph.sample_points_on_graph(step=sample_step)
    spine_sd = np.array([s.distance_to_soma for s in spines], dtype=float)
    spine_sd = spine_sd[np.isfinite(spine_sd)]
    sample_sd = sample_sd[np.isfinite(sample_sd)]

    if len(spine_sd) < 2 or len(sample_sd) < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_spines": int(len(spine_sd)),
            "n_network_samples": int(len(sample_sd)),
            "interpretation": "Insufficient data for CDF test.",
        }

    result = ks_2samp(spine_sd, sample_sd, alternative="two-sided", mode="auto")
    p_value = float(result.pvalue)
    statistic = float(result.statistic)
    if p_value < 0.05:
        interp = (
            f"Significant dependence of spine intensity on soma distance "
            f"(KS statistic={statistic:.4f}, p={p_value:.4g})."
        )
    else:
        interp = (
            f"No significant dependence on soma distance detected by CDF/KS test "
            f"(KS statistic={statistic:.4f}, p={p_value:.4g})."
        )

    return {
        "statistic": statistic,
        "p_value": p_value,
        "n_spines": int(len(spine_sd)),
        "n_network_samples": int(len(sample_sd)),
        "interpretation": interp,
    }


# ===========================================================================
# Inhomogeneous Poisson model
# ===========================================================================

def _build_covariates(soma_distances: np.ndarray, covariate_names: List[str]) -> np.ndarray:
    """Строит матрицу ковариат для пуассоновской модели.

    Входные данные: массив расстояний от сомы и список имён ковариат.
    Выходные данные: матрица дизайна формы `(n_points, n_covariates)`.
    """
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
    max_network_samples: int = 50_000,
) -> Tuple[float, np.ndarray]:
    """Аппроксимирует интеграл интенсивности по дендритной сети.
    Сэмплирует точки на графе, вычисляет интенсивность и градиент
    интеграла по коэффициентам.

    Входные данные: коэффициенты модели, имена ковариат, граф и плотность
    сэмплирования.
    Выходные данные: значение интеграла и его градиент.
    """
    step = max(1.0 / n_samples_per_unit, 1e-6)
    step = _adaptive_graph_sample_step(
        graph,
        requested_step=step,
        max_samples=max_network_samples,
    )
    _, sample_sd = graph.sample_points_on_graph(step=step)

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
    integration_sample_step: float = 0.5,
    max_integration_samples: int = 50_000,
    verbose_timing: bool = False,
    timing_label: str = "poisson",
) -> PoissonModelResult:
    """Подгоняет неоднородную пуассоновскую модель интенсивности шипиков.
    Максимизирует логарифм функции правдоподобия точечного процесса на сети,
    вычисляет коэффициенты, стандартные ошибки, AIC, BIC и residuals.

    Входные данные: дендритный граф, спроецированные шипики и список ковариат.
    Дополнительно принимает шаг и лимит quadrature-сэмплирования сети,
    а также параметры timing-лога.
    Выходные данные: объект `PoissonModelResult`.
    """
    covariate_names = list(covariates)
    if not spines:
        raise ValueError("Cannot fit model: no spines provided.")

    spine_dists = np.array([s.distance_to_soma for s in spines], dtype=float)
    N = len(spines)
    K = len(covariate_names)

    X_spines = _build_covariates(spine_dists, covariate_names)  # (N, K)
    sum_X = X_spines.sum(axis=0)  # (K,)

    total_start = perf_counter()
    stage_start = perf_counter()
    effective_step = _adaptive_graph_sample_step(
        graph,
        requested_step=integration_sample_step,
        max_samples=max_integration_samples,
    )
    _, sample_sd = graph.sample_points_on_graph(step=effective_step)
    X_samp = _build_covariates(sample_sd, covariate_names) if len(sample_sd) else np.empty((0, K))
    total_len = graph.total_length
    actual_step = total_len / len(sample_sd) if len(sample_sd) else 0.0
    preparation_seconds = perf_counter() - stage_start
    if verbose_timing:
        _print_network_timing(
            timing_label,
            f"poisson_prepare_quadrature(samples={len(sample_sd)}, step={effective_step:.4f})",
            preparation_seconds,
        )

    def integral_and_grad_precomputed(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        if len(sample_sd) == 0:
            return 0.0, np.zeros(K)
        log_lambda = X_samp @ theta
        lam = np.exp(np.clip(log_lambda, -500, 500))
        integral = float(np.sum(lam) * actual_step)
        grad_integral = (X_samp.T @ lam) * actual_step
        return integral, grad_integral

    def neg_logL_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        sum_log_lambda = float(np.sum(X_spines @ theta))
        integral, grad_integral = integral_and_grad_precomputed(theta)
        neg_ll = -(sum_log_lambda - integral)
        grad = -(sum_X - grad_integral)
        return float(neg_ll), grad

    L = total_len
    theta0 = np.zeros(K)
    if "intercept" in covariate_names:
        idx0 = covariate_names.index("intercept")
        theta0[idx0] = math.log(max(N, 1) / max(L, 1e-6))

    stage_start = perf_counter()
    result = minimize(
        neg_logL_and_grad,
        x0=theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    optimization_seconds = perf_counter() - stage_start
    if verbose_timing:
        _print_network_timing(timing_label, "poisson_optimization", optimization_seconds)

    theta_hat = result.x

    stage_start = perf_counter()
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
    hessian_seconds = perf_counter() - stage_start
    if verbose_timing:
        _print_network_timing(timing_label, "poisson_hessian", hessian_seconds)

    log_likelihood = -float(result.fun)
    aic = -2.0 * log_likelihood + 2.0 * K
    bic = -2.0 * log_likelihood + K * math.log(max(N, 1))

    log_lambda_spines = X_spines @ theta_hat
    fitted_intensity = np.exp(np.clip(log_lambda_spines, -500, 500))

    n_bins = min(10, max(2, N // 5))
    stage_start = perf_counter()
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
    residuals_seconds = perf_counter() - stage_start
    if verbose_timing:
        _print_network_timing(timing_label, "poisson_residuals", residuals_seconds)

    diagnostics = {
        "converged": bool(result.success),
        "message": result.message,
        "n_iterations": result.get("nit", None),
        "n_spines": N,
        "network_length": L,
        "integration_sample_step": float(effective_step),
        "n_integration_samples": int(len(sample_sd)),
        "preparation_seconds": float(preparation_seconds),
        "optimization_seconds": float(optimization_seconds),
        "hessian_seconds": float(hessian_seconds),
        "residuals_seconds": float(residuals_seconds),
        "total_seconds": float(perf_counter() - total_start),
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

def _source_node_distances(graph: DendriticGraph, spine: ProjectedSpine) -> Dict[int, float]:
    """Вычисляет расстояния от спроецированного шипика до всех узлов графа.

    Входные данные: дендритный граф и один объект `ProjectedSpine`.
    Выходные данные: словарь `{node_id: distance_from_spine}`.
    """
    u0, v0 = spine.edge_source, spine.edge_target
    if graph.G.has_edge(u0, v0):
        edge_len = float(graph.G[u0][v0].get("length", 0.0))
    else:
        edge_len = float(np.linalg.norm(graph.node_position(v0) - graph.node_position(u0)))
    edge_len = max(edge_len, 0.0)
    t = float(spine.edge_position)

    dist_from_u = nx.single_source_dijkstra_path_length(graph.G, u0, weight="length")
    dist_from_v = nx.single_source_dijkstra_path_length(graph.G, v0, weight="length")

    node_distances: Dict[int, float] = {}
    for node in graph.G.nodes():
        du = dist_from_u.get(node, np.inf)
        dv = dist_from_v.get(node, np.inf)
        node_distances[node] = float(min(t * edge_len + du, (1.0 - t) * edge_len + dv))
    return node_distances


def _network_sphere_multiplicity(
    graph: DendriticGraph,
    source_node_distances: Dict[int, float],
    distance: float,
    tol: float = 1e-8,
) -> int:
    """Считает количество точек на метрическом графе, находящихся точно на расстоянии `distance`.

    Входные данные: граф, расстояния от источника до узлов и радиус.
    Выходные данные: целое число точек сетевой сферы.
    """
    if not np.isfinite(distance) or distance <= tol:
        return 0

    count = 0
    for u, v, edata in graph.G.edges(data=True):
        length = float(edata.get("length", 0.0))
        if length <= tol:
            continue

        du = source_node_distances.get(u, np.inf)
        dv = source_node_distances.get(v, np.inf)
        if not np.isfinite(du) and not np.isfinite(dv):
            continue

        candidates: List[float] = []

        if np.isfinite(du):
            s = (distance - du) / length
            if -tol <= s <= 1.0 + tol:
                s_clipped = min(1.0, max(0.0, float(s)))
                via_u = du + s_clipped * length
                via_v = dv + (1.0 - s_clipped) * length if np.isfinite(dv) else np.inf
                if abs(via_u - distance) <= max(tol, tol * distance) and via_u <= via_v + tol:
                    candidates.append(s_clipped)

        if np.isfinite(dv):
            s = (dv + length - distance) / length
            if -tol <= s <= 1.0 + tol:
                s_clipped = min(1.0, max(0.0, float(s)))
                via_u = du + s_clipped * length if np.isfinite(du) else np.inf
                via_v = dv + (1.0 - s_clipped) * length
                if abs(via_v - distance) <= max(tol, tol * distance) and via_v <= via_u + tol:
                    candidates.append(s_clipped)

        unique_candidates = []
        for s in candidates:
            if not any(abs(s - seen) <= 1e-7 for seen in unique_candidates):
                unique_candidates.append(s)
        count += len(unique_candidates)

    return int(count)


def _graph_edge_geometry_arrays(
    graph: DendriticGraph,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Подготавливает массивы рёбер для быстрого подсчёта multiplicity."""
    node_ids = list(graph.G.nodes())
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    edge_u_idx: List[int] = []
    edge_v_idx: List[int] = []
    edge_lengths: List[float] = []
    for u, v, edata in graph.G.edges(data=True):
        edge_u_idx.append(node_to_index[u])
        edge_v_idx.append(node_to_index[v])
        edge_lengths.append(float(edata.get("length", 0.0)))
    return (
        node_ids,
        np.asarray(edge_u_idx, dtype=int),
        np.asarray(edge_v_idx, dtype=int),
        np.asarray(edge_lengths, dtype=float),
    )


def _source_node_distances_array(
    graph: DendriticGraph,
    spine: ProjectedSpine,
    node_ids: Sequence[int],
) -> np.ndarray:
    """Вычисляет расстояния от шипика до узлов в порядке `node_ids`."""
    source_distances = _source_node_distances(graph, spine)
    return np.asarray([source_distances.get(node, np.inf) for node in node_ids], dtype=float)


def _all_pairs_node_distance_matrix(
    graph: DendriticGraph,
    node_ids: Sequence[int],
    max_entries: int = 10_000_000,
) -> Tuple[Optional[Dict[int, int]], Optional[np.ndarray]]:
    """Предрассчитывает shortest-path расстояния между вершинами, если это разумно по памяти."""
    n_nodes = len(node_ids)
    if n_nodes <= 0 or n_nodes * n_nodes > int(max_entries):
        return None, None

    node_to_index = {node: index for index, node in enumerate(node_ids)}
    distances = np.full((n_nodes, n_nodes), np.inf, dtype=float)
    for source, lengths in nx.all_pairs_dijkstra_path_length(graph.G, weight="length"):
        source_index = node_to_index.get(source)
        if source_index is None:
            continue
        distances[source_index, source_index] = 0.0
        for target, distance in lengths.items():
            target_index = node_to_index.get(target)
            if target_index is not None:
                distances[source_index, target_index] = float(distance)
    return node_to_index, distances


def _source_node_distances_from_matrix(
    graph: DendriticGraph,
    spine: ProjectedSpine,
    node_ids: Sequence[int],
    node_to_index: Optional[Dict[int, int]],
    node_distance_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """Быстро вычисляет расстояния от шипика до узлов через кэш расстояний между вершинами."""
    if node_to_index is None or node_distance_matrix is None:
        return _source_node_distances_array(graph, spine, node_ids)

    u0, v0 = spine.edge_source, spine.edge_target
    u_index = node_to_index.get(u0)
    v_index = node_to_index.get(v0)
    if u_index is None or v_index is None:
        return _source_node_distances_array(graph, spine, node_ids)

    if graph.G.has_edge(u0, v0):
        edge_len = float(graph.G[u0][v0].get("length", 0.0))
    else:
        edge_len = float(np.linalg.norm(graph.node_position(v0) - graph.node_position(u0)))
    edge_len = max(edge_len, 0.0)
    t = float(spine.edge_position)
    return np.minimum(
        t * edge_len + node_distance_matrix[u_index],
        (1.0 - t) * edge_len + node_distance_matrix[v_index],
    )


def _network_sphere_multiplicity_from_arrays(
    edge_u_idx: np.ndarray,
    edge_v_idx: np.ndarray,
    edge_lengths: np.ndarray,
    source_node_distances: np.ndarray,
    distance: float,
    tol: float = 1e-8,
) -> int:
    """Векторизованно считает число точек сетевой сферы радиуса `distance`."""
    if not np.isfinite(distance) or distance <= tol or len(edge_lengths) == 0:
        return 0

    length = edge_lengths
    valid_length = length > tol
    if not np.any(valid_length):
        return 0

    du = source_node_distances[edge_u_idx]
    dv = source_node_distances[edge_v_idx]
    finite_du = np.isfinite(du)
    finite_dv = np.isfinite(dv)

    with np.errstate(divide="ignore", invalid="ignore"):
        s_u = (distance - du) / length
    s_u_clipped = np.clip(s_u, 0.0, 1.0)
    via_u = du + s_u_clipped * length
    via_v_from_u = np.where(finite_dv, dv + (1.0 - s_u_clipped) * length, np.inf)
    valid_u = (
        valid_length
        & finite_du
        & (s_u >= -tol)
        & (s_u <= 1.0 + tol)
        & (np.abs(via_u - distance) <= max(tol, tol * distance))
        & (via_u <= via_v_from_u + tol)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        s_v = (dv + length - distance) / length
    s_v_clipped = np.clip(s_v, 0.0, 1.0)
    via_u_from_v = np.where(finite_du, du + s_v_clipped * length, np.inf)
    via_v = dv + (1.0 - s_v_clipped) * length
    valid_v = (
        valid_length
        & finite_dv
        & (s_v >= -tol)
        & (s_v <= 1.0 + tol)
        & (np.abs(via_v - distance) <= max(tol, tol * distance))
        & (via_v <= via_u_from_v + tol)
    )

    duplicate = valid_u & valid_v & (np.abs(s_u_clipped - s_v_clipped) <= 1e-7)
    return int(np.count_nonzero(valid_u) + np.count_nonzero(valid_v) - np.count_nonzero(duplicate))


def _geometric_multiplicity_rows(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    dist_matrix: np.ndarray,
    row_indices: Sequence[int],
    max_distance: Optional[float],
    node_ids: Sequence[int],
    node_to_index: Optional[Dict[int, int]],
    node_distance_matrix: Optional[np.ndarray],
    edge_u_idx: np.ndarray,
    edge_v_idx: np.ndarray,
    edge_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Считает строки матрицы multiplicity для независимой пачки шипиков."""
    n = len(spines)
    rows = np.ones((len(row_indices), n), dtype=float)
    max_distance_value = None if max_distance is None else float(max_distance)

    for local_index, i in enumerate(row_indices):
        rows[local_index, i] = np.inf
        source_distances = _source_node_distances_from_matrix(
            graph,
            spines[i],
            node_ids,
            node_to_index,
            node_distance_matrix,
        )
        finite_distances = sorted(
            {
                float(d)
                for d in dist_matrix[i]
                if np.isfinite(d) and d > 1e-8
                and (max_distance_value is None or d <= max_distance_value + 1e-8)
            }
        )
        cache = {
            d: max(
                1,
                _network_sphere_multiplicity_from_arrays(
                    edge_u_idx,
                    edge_v_idx,
                    edge_lengths,
                    source_distances,
                    d,
                ),
            )
            for d in finite_distances
        }
        for j in range(n):
            d = float(dist_matrix[i, j])
            if np.isfinite(d) and d > 1e-8:
                rows[local_index, j] = float(cache.get(d, 1))

    return np.asarray(row_indices, dtype=int), rows


def _geometric_multiplicity_chunk(args: Tuple[Any, List[ProjectedSpine], np.ndarray, np.ndarray, Optional[float], Sequence[int], Optional[Dict[int, int]], Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Worker-wrapper для параллельного подсчёта строк multiplicity."""
    (
        graph,
        spines,
        dist_matrix,
        row_indices,
        max_distance,
        node_ids,
        node_to_index,
        node_distance_matrix,
        edge_u_idx,
        edge_v_idx,
        edge_lengths,
    ) = args
    return _geometric_multiplicity_rows(
        graph,
        spines,
        dist_matrix,
        row_indices,
        max_distance,
        node_ids,
        node_to_index,
        node_distance_matrix,
        edge_u_idx,
        edge_v_idx,
        edge_lengths,
    )


def _geometric_multiplicity_matrix(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    dist_matrix: np.ndarray,
    max_distance: Optional[float] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Строит матрицу геометрических множителей для K-функции на сети.
    Для каждой пары шипиков оценивает число направлений/точек
    сетевой сферы на соответствующем расстоянии.

    Входные данные: граф, список шипиков, матрица сетевых расстояний и
    опциональный максимальный радиус K-функции.
    Выходные данные: матрица multiplicity той же формы, что и матрица
    расстояний.
    """
    n = len(spines)
    multiplicity = np.ones((n, n), dtype=float)
    np.fill_diagonal(multiplicity, np.inf)
    if n == 0:
        return multiplicity

    node_ids, edge_u_idx, edge_v_idx, edge_lengths = _graph_edge_geometry_arrays(graph)
    node_to_index, node_distance_matrix = _all_pairs_node_distance_matrix(graph, node_ids)
    n_jobs_effective = _resolve_n_jobs(n_jobs, n)
    row_indices = np.arange(n, dtype=int)

    if n_jobs_effective <= 1:
        _, rows = _geometric_multiplicity_rows(
            graph,
            spines,
            dist_matrix,
            row_indices,
            max_distance,
            node_ids,
            node_to_index,
            node_distance_matrix,
            edge_u_idx,
            edge_v_idx,
            edge_lengths,
        )
        multiplicity[row_indices] = rows
        return multiplicity

    chunks = [chunk for chunk in np.array_split(row_indices, n_jobs_effective) if len(chunk) > 0]
    worker_args = [
        (
            graph,
            spines,
            dist_matrix,
            chunk,
            max_distance,
            node_ids,
            node_to_index,
            node_distance_matrix,
            edge_u_idx,
            edge_v_idx,
            edge_lengths,
        )
        for chunk in chunks
    ]
    try:
        with ProcessPoolExecutor(max_workers=n_jobs_effective) as executor:
            for indices, rows in executor.map(_geometric_multiplicity_chunk, worker_args):
                multiplicity[indices] = rows
    except Exception as exc:
        warnings.warn(
            f"Parallel geometric multiplicity failed ({exc}); falling back to sequential execution.",
            stacklevel=2,
        )
        _, rows = _geometric_multiplicity_rows(
            graph,
            spines,
            dist_matrix,
            row_indices,
            max_distance,
            node_ids,
            node_to_index,
            node_distance_matrix,
            edge_u_idx,
            edge_v_idx,
            edge_lengths,
        )
        multiplicity[row_indices] = rows

    return multiplicity


def ripley_k_network(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    r_values: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    dist_matrix: Optional[np.ndarray] = None,
    correction: str = "geometric",
    n_jobs: int = 1,
) -> KFunctionResult:
    """Вычисляет сетевую K-функцию Рипли для шипиков на дендритном графе.
    Считает число пар шипиков в пределах каждого радиуса с
    homogeneous или inhomogeneous нормировкой и геометрической поправкой сети.

    Входные данные: граф, спроецированные шипики, радиусы `r_values`,
    опциональная интенсивность, матрица расстояний, тип поправки и число
    процессов для геометрической поправки.
    Выходные данные: объект `KFunctionResult` с наблюдаемой и ожидаемой
    K-кривой.
    """
    total_start = perf_counter()
    diagnostics: Dict[str, Any] = {}
    n = len(spines)
    r_values = np.asarray(r_values, dtype=float)
    correction = correction.lower()
    use_geometric = correction in {"geometric", "reference", "kl", "k_l"}
    k_expected = r_values.copy() if use_geometric else 2.0 * r_values

    if n < 2:
        k_obs = np.zeros_like(r_values)
        interp = "Too few spines for K function estimation."
        return KFunctionResult(
            r_values=r_values,
            k_observed=k_obs,
            k_expected=k_expected,
            method="homogeneous" if intensity is None else "inhomogeneous",
            interpretation=interp,
        )

    if dist_matrix is None:
        stage_start = perf_counter()
        dist_matrix = compute_spine_pairwise_distances(graph, spines)
        diagnostics["pairwise_distance_seconds"] = perf_counter() - stage_start
    else:
        diagnostics["pairwise_distance_seconds"] = 0.0

    L = graph.total_length
    method = "homogeneous" if intensity is None else "inhomogeneous"

    k_obs = np.zeros(len(r_values))
    multiplicity: Optional[np.ndarray] = None
    if use_geometric:
        stage_start = perf_counter()
        max_k_distance = float(np.max(r_values)) if len(r_values) else None
        multiplicity = _geometric_multiplicity_matrix(
            graph,
            spines,
            dist_matrix,
            max_distance=max_k_distance,
            n_jobs=n_jobs,
        )
        diagnostics["geometric_multiplicity_seconds"] = perf_counter() - stage_start
    else:
        diagnostics["geometric_multiplicity_seconds"] = 0.0

    stage_start = perf_counter()
    if intensity is None:
        pair_weights = np.ones_like(dist_matrix, dtype=float)
        np.fill_diagonal(pair_weights, 0.0)
        if multiplicity is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                pair_weights = np.where(multiplicity > 0, pair_weights / multiplicity, 0.0)
            np.fill_diagonal(pair_weights, 0.0)
        for ri, r in enumerate(r_values):
            mask = (dist_matrix <= r).astype(float)
            np.fill_diagonal(mask, 0.0)
            k_obs[ri] = (L / (n * n)) * float(np.sum(mask * pair_weights))
    else:
        lam = np.asarray(intensity, dtype=float)
        lam_outer = np.outer(lam, lam)    
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(lam_outer > 1e-300, 1.0 / lam_outer, 0.0)
        np.fill_diagonal(weights, 0.0) 
        if multiplicity is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = np.where(multiplicity > 0, weights / multiplicity, 0.0)
            np.fill_diagonal(weights, 0.0)

        for ri, r in enumerate(r_values):
            mask = (dist_matrix <= r).astype(float)
            np.fill_diagonal(mask, 0.0)
            k_obs[ri] = (1.0 / L) * float(np.sum(mask * weights))
    diagnostics["k_accumulation_seconds"] = perf_counter() - stage_start

    dev = k_obs - k_expected
    max_pos_r = float(r_values[np.argmax(dev)]) if len(dev) else 0.0
    max_neg_r = float(r_values[np.argmin(dev)]) if len(dev) else 0.0
    if np.max(np.abs(dev)) < 0.1 * float(np.max(k_expected)) if np.max(k_expected) > 0 else True:
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
        k_expected=k_expected,
        method=f"{'geometrically_corrected_' if use_geometric else ''}{method}",
        interpretation=interp,
        diagnostics={
            **diagnostics,
            "n_spines": int(n),
            "n_r_values": int(len(r_values)),
            "correction": correction,
            "geometric_multiplicity_n_jobs": int(_resolve_n_jobs(n_jobs, n)) if use_geometric else 1,
            "total_seconds": perf_counter() - total_start,
        },
    )


def simulate_poisson_on_graph(
    graph: DendriticGraph,
    n_points: int,
    intensity_fn: Optional[Callable[[float], float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[ProjectedSpine]:
    """Симулирует точки пуассоновского процесса на дендритном графе.
    Выбирает рёбра пропорционально их длине и размещает точки
    равномерно либо по неоднородной интенсивности методом thinning.

    Входные данные: граф дендритной сети, число точек, опциональная функция
    интенсивности и генератор случайных чисел.
    Выходные данные: список объектов `ProjectedSpine`, представляющих
    симулированные точки на графе.
    """
    if rng is None:
        rng = np.random.default_rng()

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
        chosen_edges = rng.choice(len(edge_list), size=n_points, p=probs)
        results = []
        for k, eidx in enumerate(chosen_edges):
            u, v, L = edge_list[eidx]
            t = float(rng.uniform(0.0, 1.0))
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
            chosen_edges = rng.choice(len(edge_list), size=batch_size, p=probs)
            ts = rng.uniform(0.0, 1.0, size=batch_size)
            uniforms = rng.uniform(0.0, 1.0, size=batch_size)

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


def _resolve_n_jobs(n_jobs: Optional[int], n_tasks: int) -> int:
    """Определяет эффективное число рабочих процессов.

    Входные данные: желаемое число процессов и число задач.
    Выходные данные: число процессов от 1 до `n_tasks`.
    """
    n_tasks = int(max(1, n_tasks))
    if n_jobs is None:
        n_jobs = 1
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    if n_jobs == 0:
        n_jobs = 1
    return int(max(1, min(n_jobs, n_tasks)))


def _print_k_timing(label: str, stage: str, seconds: float) -> None:
    """Печатает timing-log этапа K-анализа.

    Входные данные: подпись K-анализа, имя этапа и длительность в секундах.
    Выходные данные: строка timing-log в stdout.
    """
    print(f"[time][K] {label}: {stage} took {seconds:.3f}s", flush=True)


def _sum_timing_dicts(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Суммирует одноимённые поля времени из нескольких словарей.

    Входные данные: последовательность словарей timing-метрик.
    Выходные данные: словарь суммарных длительностей.
    """
    total: Dict[str, float] = {}
    for record in records:
        for key, value in record.items():
            total[key] = total.get(key, 0.0) + float(value)
    return total


def _simulate_k_envelope_chunk(args: Tuple[Any, int, np.ndarray, str, np.ndarray, Optional[List[str]], Optional[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], int]:
    """Выполняет пачку Monte Carlo симуляций K-функции.

    Входные данные: tuple с графом, числом точек, радиусами, типом поправки,
    seeds, параметрами интенсивности и ожидаемой K-кривой.
    Выходные данные: матрица симулированных K-кривых, timing-метрики пачки и
    число выполненных симуляций.
    """
    (
        graph,
        n_points,
        r_values,
        correction,
        seeds,
        covariate_names,
        coefficients,
        k_expected,
    ) = args
    k_rows: List[np.ndarray] = []
    timing = {
        "simulate_poisson_seconds": 0.0,
        "simulation_pairwise_distance_seconds": 0.0,
        "simulation_k_function_seconds": 0.0,
        "simulation_geometric_multiplicity_seconds": 0.0,
        "simulation_k_accumulation_seconds": 0.0,
    }

    if covariate_names is not None and coefficients is not None:
        coeffs = np.asarray(coefficients, dtype=float)

        def intensity_fn(distance: float) -> float:
            x = _build_covariates(np.array([distance]), covariate_names)[0]
            return float(np.exp(np.clip(np.dot(x, coeffs), -500, 500)))
    else:
        intensity_fn = None

    for seed in np.asarray(seeds, dtype=np.int64):
        rng = np.random.default_rng(int(seed))
        stage_start = perf_counter()
        sim_spines = simulate_poisson_on_graph(graph, n_points=n_points, intensity_fn=intensity_fn, rng=rng)
        timing["simulate_poisson_seconds"] += perf_counter() - stage_start
        if len(sim_spines) < 2:
            k_rows.append(np.asarray(k_expected, dtype=float))
            continue

        sim_intensity: Optional[np.ndarray] = None
        if covariate_names is not None and coefficients is not None:
            sim_dists = np.array([s.distance_to_soma for s in sim_spines])
            X_sim = _build_covariates(sim_dists, covariate_names)
            sim_intensity = np.exp(np.clip(X_sim @ np.asarray(coefficients, dtype=float), -500, 500))

        stage_start = perf_counter()
        sim_dist_mat = compute_spine_pairwise_distances(graph, sim_spines)
        timing["simulation_pairwise_distance_seconds"] += perf_counter() - stage_start

        stage_start = perf_counter()
        sim_k = ripley_k_network(
            graph,
            sim_spines,
            r_values,
            intensity=sim_intensity,
            dist_matrix=sim_dist_mat,
            correction=correction,
            n_jobs=1,
        )
        timing["simulation_k_function_seconds"] += perf_counter() - stage_start
        timing["simulation_geometric_multiplicity_seconds"] += float(
            sim_k.diagnostics.get("geometric_multiplicity_seconds", 0.0)
        )
        timing["simulation_k_accumulation_seconds"] += float(
            sim_k.diagnostics.get("k_accumulation_seconds", 0.0)
        )
        k_rows.append(sim_k.k_observed)

    if not k_rows:
        return np.empty((0, len(r_values)), dtype=float), timing, 0
    return np.asarray(k_rows, dtype=float), timing, len(k_rows)


def compute_simulation_envelopes(
    graph: DendriticGraph,
    spines: List[ProjectedSpine],
    r_values: np.ndarray,
    n_simulations: int = 99,
    alpha: float = 0.05,
    intensity_model: Optional[PoissonModelResult] = None,
    correction: str = "geometric",
    envelope_type: str = "global_constant_width",
    require_inhomogeneous: bool = True,
    fit_intensity_if_missing: bool = True,
    random_state: Optional[int] = None,
    n_jobs: int = 8,
    timing_label: str = "K",
    verbose_timing: bool = True,
) -> KFunctionResult:
    """Строит Monte Carlo envelope для сетевой функции Рипли.
    Вычисляет наблюдаемую K-кривую, симулирует пуассоновские точки
    на том же графе, строит нижнюю/верхнюю envelope и Monte Carlo p-значение.

    Входные данные: дендритный граф, наблюдаемые шипики, значения радиуса,
    число симуляций, модель интенсивности, способ поправки, тип envelope,
    флаг обязательной неоднородной модели, seed, число параллельных процессов
    и параметры timing-лога.
    Выходные данные: объект `KFunctionResult` с наблюдаемой, ожидаемой и
    симулированными границами K-кривой.
    """
    total_start = perf_counter()
    rng = np.random.default_rng(random_state)
    n = len(spines)
    r_values = np.asarray(r_values, dtype=float)
    use_geometric = correction.lower() in {"geometric", "reference", "kl", "k_l"}
    envelope_type_normalized = str(envelope_type).lower()
    n_simulations = int(max(0, n_simulations))
    observed_n_jobs_effective = _resolve_n_jobs(n_jobs, n)
    simulation_n_jobs_effective = _resolve_n_jobs(n_jobs, max(1, n_simulations))

    if n < 2:
        k_exp = r_values.copy() if use_geometric else 2.0 * r_values
        return KFunctionResult(
            r_values=r_values,
            k_observed=np.zeros_like(r_values),
            k_expected=k_exp,
            k_lower=np.zeros_like(r_values),
            k_upper=k_exp.copy(),
            p_value=1.0,
            method="inhomogeneous",
            interpretation="Too few spines for simulation envelopes.",
            diagnostics={
                "n_jobs": int(simulation_n_jobs_effective),
                "observed_n_jobs": int(observed_n_jobs_effective),
                "n_simulations": int(n_simulations),
                "envelope_type": envelope_type_normalized,
                "require_inhomogeneous": bool(require_inhomogeneous),
                "total_seconds": perf_counter() - total_start,
            },
        )

    if intensity_model is None and fit_intensity_if_missing:
        try:
            stage_start = perf_counter()
            intensity_model = fit_inhomogeneous_poisson(graph, spines)
            if verbose_timing:
                _print_k_timing(timing_label, "fit_intensity_if_missing", perf_counter() - stage_start)
        except Exception as exc:
            if require_inhomogeneous:
                raise RuntimeError(
                    "Inhomogeneous K analysis requires a fitted intensity model, "
                    f"but model fitting failed: {type(exc).__name__}: {exc!r}"
                ) from exc
            warnings.warn(f"Intensity model fitting failed: {exc}; using homogeneous CSR.", stacklevel=2)
            intensity_model = None
    if intensity_model is None and require_inhomogeneous:
        raise ValueError("Inhomogeneous K analysis requires `intensity_model`.")

    if intensity_model is not None:
        obs_intensity = intensity_model.fitted_intensity
    else:
        obs_intensity = None

    stage_start = perf_counter()
    obs_dist = compute_spine_pairwise_distances(graph, spines)
    observed_pairwise_seconds = perf_counter() - stage_start
    if verbose_timing:
        _print_k_timing(timing_label, "observed_pairwise_distances", observed_pairwise_seconds)

    stage_start = perf_counter()
    k_obs_result = ripley_k_network(
        graph,
        spines,
        r_values,
        intensity=obs_intensity,
        dist_matrix=obs_dist,
        correction=correction,
        n_jobs=observed_n_jobs_effective,
    )
    observed_k_seconds = perf_counter() - stage_start
    observed_geometric_seconds = float(k_obs_result.diagnostics.get("geometric_multiplicity_seconds", 0.0))
    if verbose_timing:
        _print_k_timing(timing_label, "observed_k_function", observed_k_seconds)
        _print_k_timing(timing_label, "observed_geometric_multiplicity", observed_geometric_seconds)
    k_observed = k_obs_result.k_observed
    k_expected = k_obs_result.k_expected

    if n_simulations <= 0:
        return KFunctionResult(
            r_values=r_values,
            k_observed=k_observed,
            k_expected=k_expected,
            k_lower=None,
            k_upper=None,
            p_value=np.nan,
            method=k_obs_result.method,
            interpretation="K function computed without Monte Carlo simulations.",
            diagnostics={
                "n_jobs": int(simulation_n_jobs_effective),
                "observed_n_jobs": int(observed_n_jobs_effective),
                "n_simulations": int(n_simulations),
                "envelope_type": envelope_type_normalized,
                "require_inhomogeneous": bool(require_inhomogeneous),
                "observed_pairwise_distance_seconds": observed_pairwise_seconds,
                "observed_k_function_seconds": observed_k_seconds,
                "observed_geometric_multiplicity_seconds": observed_geometric_seconds,
                "total_seconds": perf_counter() - total_start,
            },
        )

    simulation_start = perf_counter()
    seeds = rng.integers(
        0,
        np.iinfo(np.int32).max,
        size=n_simulations,
        dtype=np.int64,
    )
    if intensity_model is not None:
        worker_covariates = list(intensity_model.covariate_names)
        worker_coefficients = np.asarray(intensity_model.coefficients, dtype=float)
    else:
        worker_covariates = None
        worker_coefficients = None

    seed_chunks = [
        chunk
        for chunk in np.array_split(seeds, simulation_n_jobs_effective)
        if len(chunk) > 0
    ]
    worker_args = [
        (
            graph,
            n,
            r_values,
            correction,
            chunk,
            worker_covariates,
            worker_coefficients,
            k_expected,
        )
        for chunk in seed_chunks
    ]

    chunk_results: List[Tuple[np.ndarray, Dict[str, float], int]] = []
    if simulation_n_jobs_effective == 1:
        iter_range: Any = worker_args
        if _TQDM:
            iter_range = _tqdm(iter_range, desc=f"{timing_label} simulations", leave=False)
        for args in iter_range:
            chunk_results.append(_simulate_k_envelope_chunk(args))
    else:
        try:
            with ProcessPoolExecutor(max_workers=simulation_n_jobs_effective) as executor:
                mapped: Any = executor.map(_simulate_k_envelope_chunk, worker_args)
                if _TQDM:
                    mapped = _tqdm(
                        mapped,
                        total=len(worker_args),
                        desc=f"{timing_label} simulations",
                        leave=False,
                    )
                for result in mapped:
                    chunk_results.append(result)
        except Exception as exc:
            warnings.warn(
                f"Parallel K simulations failed ({exc}); falling back to sequential execution.",
                stacklevel=2,
            )
            simulation_n_jobs_effective = 1
            chunk_results = [_simulate_k_envelope_chunk(args) for args in worker_args]

    if chunk_results:
        valid_chunks = [result[0] for result in chunk_results if len(result[0]) > 0]
        k_sim_all = np.vstack(valid_chunks) if valid_chunks else np.empty((0, len(r_values)), dtype=float)
    else:
        k_sim_all = np.empty((0, len(r_values)), dtype=float)
    if len(k_sim_all) < n_simulations:
        missing = n_simulations - len(k_sim_all)
        k_sim_all = np.vstack([
            k_sim_all,
            np.repeat(k_expected.reshape(1, -1), missing, axis=0),
        ])
    simulation_seconds = perf_counter() - simulation_start
    simulation_timing = _sum_timing_dicts([result[1] for result in chunk_results])
    if verbose_timing:
        _print_k_timing(timing_label, f"simulations_total(n={n_simulations}, jobs={simulation_n_jobs_effective})", simulation_seconds)
        for key in (
            "simulate_poisson_seconds",
            "simulation_pairwise_distance_seconds",
            "simulation_k_function_seconds",
            "simulation_geometric_multiplicity_seconds",
            "simulation_k_accumulation_seconds",
        ):
            _print_k_timing(timing_label, key, simulation_timing.get(key, 0.0))

    stage_start = perf_counter()
    obs_dev = float(np.max(np.abs(k_observed - k_expected)))
    sim_devs = np.max(np.abs(k_sim_all - k_expected[None, :]), axis=1)
    if envelope_type_normalized in {"global_constant_width", "global", "constant_width", "reference"}:
        if len(sim_devs):
            sorted_devs = np.sort(sim_devs)
            envelope_rank = int(math.ceil((1.0 - alpha) * (n_simulations + 1)))
            envelope_rank = int(np.clip(envelope_rank, 1, n_simulations))
            w_max = float(sorted_devs[envelope_rank - 1])
        else:
            envelope_rank = 0
            w_max = 0.0
        k_lower = k_expected - w_max
        k_upper = k_expected + w_max
        envelope_width = w_max
    elif envelope_type_normalized in {"pointwise_quantile", "pointwise", "quantile"}:
        lo_idx = int(math.floor(alpha / 2.0 * n_simulations))
        hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_simulations))
        lo_idx = max(0, lo_idx)
        hi_idx = min(n_simulations - 1, hi_idx)
        k_sim_sorted = np.sort(k_sim_all, axis=0)
        k_lower = k_sim_sorted[lo_idx]
        k_upper = k_sim_sorted[hi_idx]
        envelope_width = float(np.max(k_upper - k_lower) / 2.0) if len(k_upper) else 0.0
    else:
        raise ValueError(
            "envelope_type must be one of: 'global_constant_width', 'pointwise_quantile'."
        )
    p_value = float((1 + np.sum(sim_devs >= obs_dev)) / (1 + n_simulations))
    envelope_seconds = perf_counter() - stage_start
    if verbose_timing:
        _print_k_timing(timing_label, "envelope_and_p_value", envelope_seconds)

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
        diagnostics={
            "n_jobs": int(simulation_n_jobs_effective),
            "observed_n_jobs": int(observed_n_jobs_effective),
            "n_simulations": int(n_simulations),
            "n_chunks": int(len(seed_chunks)),
            "envelope_type": envelope_type_normalized,
            "require_inhomogeneous": bool(require_inhomogeneous),
            "r_min": float(r_values[0]) if len(r_values) else np.nan,
            "r_max": float(r_values[-1]) if len(r_values) else np.nan,
            "global_envelope_width": float(envelope_width),
            "global_envelope_rank": int(envelope_rank) if "envelope_rank" in locals() else np.nan,
            "observed_pairwise_distance_seconds": observed_pairwise_seconds,
            "observed_k_function_seconds": observed_k_seconds,
            "observed_geometric_multiplicity_seconds": observed_geometric_seconds,
            "simulation_total_wall_seconds": simulation_seconds,
            "simulation_envelope_seconds": envelope_seconds,
            **simulation_timing,
            "total_seconds": perf_counter() - total_start,
        },
    )


# ===========================================================================
# Group comparison
# ===========================================================================

def group_k_analysis(
    graphs_and_spines: Dict[str, Tuple[DendriticGraph, List[ProjectedSpine]]],
    r_values: np.ndarray,
    group_by: str = "group",
) -> Dict[str, KFunctionResult]:
    """Вычисляет K-функции для нескольких групп графов и шипиков.

    Входные данные: словарь `{group_name: (graph, spines)}`, радиусы и имя
    группирующего признака.
    Выходные данные: словарь `{group_name: KFunctionResult}`.
    """
    results: Dict[str, KFunctionResult] = {}
    for group_name, (graph, spines) in graphs_and_spines.items():
        try:
            results[group_name] = ripley_k_network(graph, spines, r_values)
        except Exception as exc:
            warnings.warn(f"K function failed for group {group_name!r}: {exc}", stacklevel=2)
    return results


def permutation_test_groups(
    group_a: List[Tuple[DendriticGraph, List[ProjectedSpine]]],
    group_b: List[Tuple[DendriticGraph, List[ProjectedSpine]]],
    r_values: np.ndarray,
    n_permutations: int = 999,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Выполняет перестановочное сравнение средних K-кривых двух групп.
    Считает интегральное абсолютное различие средних K-кривых и
    сравнивает его с перестановочным распределением.

    Входные данные: две группы пар `(graph, spines)`, радиусы, число
    перестановок и seed генератора случайных чисел.
    Выходные данные: словарь p-value, наблюдаемого различия, распределения
    перестановок и радиусов.
    """
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

    rng = np.random.default_rng(random_state)
    for _ in iter_range:
        perm = rng.permutation(len(all_data))
        perm_a = [all_data[i] for i in perm[:na]]
        perm_b = [all_data[i] for i in perm[na:]]
        k_pa = _mean_k(perm_a)
        k_pb = _mean_k(perm_b)
        perm_diffs.append(float(np.sum(np.abs(k_pa - k_pb)) * dr))

    perm_diffs_arr = np.array(perm_diffs)
    p_value = float((1 + np.sum(perm_diffs_arr >= observed_diff)) / (1 + n_permutations))

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
    """Строит интерактивную 3D-визуализацию дендритной сети.

    Входные данные: граф, опциональный список шипиков, способ окраски,
    интенсивность, путь сохранения и заголовок.
    Выходные данные: объект Plotly figure или `None`, если Plotly недоступен.
    """
    if not _PLOTLY:
        warnings.warn("plotly is required for plot_3d_network.", stacklevel=2)
        return None

    fig = _go.Figure()

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
    """Строит график интенсивности шипиков от расстояния до сомы.

    Входные данные: биннированная интенсивность, опциональная сглаженная
    интенсивность и путь сохранения.
    Выходные данные: объект Matplotlib figure или `None`, если Matplotlib
    недоступен.
    """
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
    """Строит график сетевой K-функции Рипли.

    Входные данные: `KFunctionResult`, заголовок и путь сохранения.
    Выходные данные: объект Matplotlib figure или `None`.
    """
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
    """Строит график отклонения K-кривой от ожидаемой.

    Входные данные: `KFunctionResult` и путь сохранения.
    Выходные данные: объект Matplotlib figure или `None`.
    """
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
    """Строит график сравнения K-кривых нескольких групп.

    Входные данные: словарь результатов K-функции по группам и путь сохранения.
    Выходные данные: объект Matplotlib figure или `None`.
    """
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
    """Создаёт HTML или Markdown отчёт по сетевому анализу.

    Входные данные: граф, шипики, таблица интенсивности, результаты
    пуассоновской модели и K-функции, директория вывода, формат и заголовок.
    Выходные данные: путь к созданному файлу отчёта.
    """
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
    """Загружает YAML-конфигурацию сетевого анализа.

    Входные данные: путь к YAML-файлу.
    Выходные данные: словарь конфигурации.
    """
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
    graph: Optional[DendriticGraph] = None,
    standard_network_dir: Optional[Union[str, Path]] = None,
    spine_points: Optional[Dict[str, np.ndarray]] = None,
    soma_point: Optional[np.ndarray] = None,
    output_dir: str = "output_network_analysis",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Запускает автономный анализ шипиков на одной дендритной сети.
    Принимает готовый граф, каноническую папку или mesh. Строит/загружает
    граф дендрита, проецирует шипики, считает интенсивность,
    тесты зависимости от сомы, пуассоновскую модель, K-функцию и отчёт.

    Входные данные: готовый граф, каноническая папка, путь к mesh или готовый
    mesh, словарь точек шипиков, опциональная точка сомы, директория вывода
    и параметры анализа.
    Выходные данные: словарь с графом, результатами проекции, таблицами,
    моделями, K-результатом и путём отчёта.
    """
    config: Dict[str, Any] = {}
    if config_path is not None:
        try:
            config = load_config(config_path)
        except Exception as exc:
            warnings.warn(f"Failed to load config {config_path!r}: {exc}", stacklevel=2)
    config.update(kwargs)

    standard_metadata: Dict[str, Any] = {}
    if graph is None and standard_network_dir is None:
        standard_network_dir = config.get("standard_network_dir", None)
    if graph is None and standard_network_dir is not None:
        graph, loaded_spine_points, standard_metadata = load_standard_network(
            standard_network_dir,
            root_vertex_id=config.get("root_vertex_id", None),
            dendrite_id=config.get("dendrite_id", None),
            dendrite_type=config.get("dendrite_type", None),
            project_spines=False,
        )
        if spine_points is None:
            spine_points = loaded_spine_points

    if graph is None and dendrite_mesh is None and mesh_path is not None:
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

    if graph is None and dendrite_mesh is None:
        raise ValueError("Either graph, standard_network_dir, dendrite_mesh or mesh_path must be provided.")

    if spine_points is None or len(spine_points) == 0:
        raise ValueError("spine_points must be a non-empty dict of spine attachment points.")

    dendrite_id = config.get("dendrite_id", "d0")
    dendrite_type = config.get("dendrite_type", "unknown")
    snap_threshold = float(config.get("snap_threshold", 2.0))

    if graph is None:
        graph = build_dendritic_graph(
            dendrite_mesh,
            soma_point=soma_point,
            dendrite_id=dendrite_id,
            dendrite_type=dendrite_type,
            snap_threshold=snap_threshold,
        )

    max_dist_config = config.get("max_distance_to_edge", np.inf if standard_network_dir is not None else 5.0)
    max_dist_to_edge = float(max_dist_config)
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

    bin_size_config = config.get("bin_size", "auto")
    max_network_samples = int(config.get("max_network_samples", 50_000))
    max_integration_samples = int(config.get("max_integration_samples", 50_000))
    if bin_size_config is None or (
        isinstance(bin_size_config, str)
        and bin_size_config.lower() in {"auto", "auto_diagnostic_only", "reference"}
    ):
        bin_size = auto_bin_size_for_network(graph, projected_spines)
        print(
            f"[network intensity] bin_size=auto -> {bin_size:.4f} "
            "(diagnostic binned intensity only)",
            flush=True,
        )
    else:
        bin_size = float(bin_size_config)
    binned_intensity = estimate_binned_intensity(
        graph,
        projected_spines,
        bin_size=bin_size,
        max_network_samples=max_network_samples,
    )
    intensity_cdf_test = test_intensity_dependence_cdf(
        graph,
        projected_spines,
        sample_step=float(config.get("cdf_sample_step", 1.0)),
        max_network_samples=max_network_samples,
    )
    intensity_lr_test = test_intensity_dependence(
        graph,
        projected_spines,
        n_bins=int(config.get("intensity_test_bins", 10)),
        max_network_samples=max_network_samples,
    )

    bandwidth = config.get("bandwidth", None)
    smooth_d, smooth_lambda = estimate_smooth_intensity(
        graph,
        projected_spines,
        bandwidth=bandwidth,
        max_network_samples=max_network_samples,
    )

    covariates = config.get(
        "covariates", ["intercept", "distance_to_soma", "distance_to_soma_squared"]
    )
    poisson_result: Optional[PoissonModelResult] = None
    if len(projected_spines) >= 3:
        try:
            poisson_result = fit_inhomogeneous_poisson(
                graph,
                projected_spines,
                covariates=covariates,
                max_integration_samples=max_integration_samples,
                verbose_timing=bool(config.get("verbose_timing", True)),
                timing_label=str(config.get("timing_label", "network")),
            )
        except Exception as exc:
            warnings.warn(
                f"Poisson model fitting failed: {type(exc).__name__}: {exc!r}",
                stacklevel=2,
            )

    n_r = int(config.get("n_r_values", 20))
    r_values, resolved_r_max, r_max_source = resolve_k_r_values(
        graph,
        n_r_values=n_r,
        r_max=config.get("r_max", "circumradius"),
    )
    print(
        f"[network K] r_max={resolved_r_max:.4f} "
        f"(source={r_max_source}), n_r_values={len(r_values)}",
        flush=True,
    )

    n_sim = int(config.get("n_simulations", 99))
    n_jobs = int(config.get("n_jobs", 8))
    k_correction = str(config.get("k_correction", "geometric"))
    envelope_type = str(config.get("envelope_type", "global_constant_width"))
    k_inhomogeneous = bool(config.get("k_inhomogeneous", True))
    random_state_value = config.get("random_state", 42)
    random_state = None if random_state_value is None else int(random_state_value)
    k_result: Optional[KFunctionResult] = None
    if len(projected_spines) >= 2:
        try:
            k_result = compute_simulation_envelopes(
                graph,
                projected_spines,
                r_values,
                n_simulations=n_sim,
                n_jobs=n_jobs,
                intensity_model=poisson_result,
                correction=k_correction,
                envelope_type=envelope_type,
                require_inhomogeneous=k_inhomogeneous,
                random_state=random_state,
                timing_label=str(config.get("timing_label", "network")),
            )
        except Exception as exc:
            warnings.warn(
                f"K function computation failed: {type(exc).__name__}: {exc!r}",
                stacklevel=2,
            )

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
        "intensity_cdf_test": intensity_cdf_test,
        "intensity_lr_test": intensity_lr_test,
        "poisson_result": poisson_result,
        "k_result": k_result,
        "standard_metadata": standard_metadata,
        "report_path": report_path,
    }
