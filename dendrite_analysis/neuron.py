from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import trimesh
    _TRIMESH = True
except ImportError:
    trimesh = None  # type: ignore
    _TRIMESH = False

from .network import (
    DendriticGraph,
    KFunctionResult,
    PoissonModelResult,
    build_dendritic_graph_from_skeleton,
    compute_simulation_envelopes,
    estimate_binned_intensity,
    estimate_smooth_intensity,
    fit_inhomogeneous_poisson,
    project_spines_to_graph,
    projected_spines_dataframe,
    test_intensity_dependence,
    test_intensity_dependence_cdf,
)

from .dendrite import Dendrite
from .config import (
    reset_saved_data,
    set_output_dir,
)

try:
    from spine_analysis.shape_metric.utils import register_attachment_center, register_dendrite_skeleton
except Exception:
    register_attachment_center = None  # type: ignore
    register_dendrite_skeleton = None  # type: ignore


def _load_trimesh(path: Path) -> Optional[Any]:
    """Загружает mesh-файл через `trimesh`.

    Входные данные: путь к mesh-файлу.
    Выходные данные: объект `trimesh.Trimesh` или `None`, если файл недоступен
    или формат не поддержан.
    """
    if not _TRIMESH or not path.exists():
        return None
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return None
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    return mesh


def _load_npy_object(path: Path) -> Optional[Any]:
    """Загружает объект skeleton из `.npy`-файла.

    Входные данные: путь к `.npy`-файлу.
    Выходные данные: загруженный объект или `None`, если файл отсутствует.
    """
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _safe_register_attachment_center(spine_mesh: Any, attachment_center: np.ndarray) -> None:
    """Регистрирует точку крепления шипика.

    Входные данные: меш шипика и координата точки крепления.
    Выходные данные: функция не возвращает значение; ошибки регистрации не
    прерывают основной анализ.
    """
    if register_attachment_center is None:
        return
    try:
        register_attachment_center(spine_mesh, attachment_center)
    except Exception:
        pass


def _safe_register_dendrite_skeleton(dendrite_mesh: Any, skeleton: Any) -> None:
    """Регистрирует skeleton дендрита.

    Входные данные: меш дендрита и объект skeleton.
    Выходные данные: функция не возвращает значение; ошибки регистрации не
    прерывают основной анализ.
    """
    if register_dendrite_skeleton is None or dendrite_mesh is None or skeleton is None:
        return
    try:
        register_dendrite_skeleton(dendrite_mesh, skeleton)
    except Exception:
        pass


def _safe_mesh_metrics(mesh: Optional[Any]) -> Dict[str, float]:
    """Вычисляет базовые геометрические метрики mesh-объекта.

    Входные данные: mesh или `None`.
    Действие: оценивает объём, площадь поверхности, размеры bounding box и
    эквивалентный радиус; при нулевом объёме пробует convex hull.
    Выходные данные: словарь числовых mesh-метрик.
    """
    if mesh is None:
        return {
            "volume": np.nan,
            "surface_area": np.nan,
            "bbox_x": np.nan,
            "bbox_y": np.nan,
            "bbox_z": np.nan,
            "equivalent_radius": np.nan,
        }

    volume = float(getattr(mesh, "volume", np.nan))
    if not np.isfinite(volume) or abs(volume) <= 1e-12:
        try:
            volume = float(mesh.convex_hull.volume)
        except Exception:
            volume = np.nan
    volume = abs(volume) if np.isfinite(volume) else np.nan

    surface_area = float(getattr(mesh, "area", np.nan))
    bounds = np.asarray(getattr(mesh, "bounds", np.full((2, 3), np.nan)), dtype=float)
    if bounds.shape == (2, 3) and np.isfinite(bounds).all():
        bbox = bounds[1] - bounds[0]
    else:
        bbox = np.full(3, np.nan)
    equivalent_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0) if np.isfinite(volume) and volume > 0 else np.nan

    return {
        "volume": volume,
        "surface_area": surface_area,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "equivalent_radius": float(equivalent_radius),
    }


def _polyline_length(points: Any) -> float:
    """Вычисляет длину ломаной линии.

    Входные данные: массив точек формы `(n, >=3)`.
    Выходные данные: длина линии; `0.0` при недостатке валидных точек.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        return 0.0
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _skeleton_length(skeleton: Any) -> float:
    """Вычисляет суммарную длину skeleton-структуры.

    Входные данные: skeleton в формате массива, словаря, списка сегментов или
    вложенной объектной структуры.
    Выходные данные: суммарная длина skeleton; `0.0`, если сегменты не найдены.
    """
    if skeleton is None:
        return 0.0
    if isinstance(skeleton, np.ndarray) and skeleton.dtype == object:
        if skeleton.shape == ():
            return _skeleton_length(skeleton.item())
        return float(sum(_skeleton_length(item) for item in skeleton.tolist()))
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
                    if len(valid_edges) > 0:
                        segments = points[valid_edges[:, 1], :3] - points[valid_edges[:, 0], :3]
                        return float(np.linalg.norm(segments, axis=1).sum())
        return float(sum(_skeleton_length(value) for value in skeleton.values()))
    if isinstance(skeleton, (list, tuple)):
        try:
            numeric = np.asarray(skeleton, dtype=float)
            if numeric.ndim >= 2:
                return _skeleton_length(numeric)
        except Exception:
            pass
        return float(sum(_skeleton_length(item) for item in skeleton))
    try:
        array = np.asarray(skeleton, dtype=float)
    except Exception:
        return 0.0
    if array.ndim == 2 and array.shape[1] >= 3:
        return _polyline_length(array)
    if array.ndim == 3 and array.shape[-1] >= 3:
        if array.shape[1] == 2:
            segments = array[:, 1, :3] - array[:, 0, :3]
            return float(np.linalg.norm(segments, axis=1).sum())
        return float(sum(_polyline_length(polyline) for polyline in array))
    return 0.0


def _attachment_point_from_spine_mesh(mesh: Any) -> np.ndarray:
    """Определяет точку крепления шипика по его mesh.

    Входные данные: mesh шипика в глобальных координатах.
    Действие: пытается найти дырку у основания шипика и возвращает
    среднюю координату вершин выбранного кольца; при ошибке возвращает centroid.
    Выходные данные: трёхмерная координата точки крепления.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return np.zeros(3, dtype=float)
    try:
        from spine_analysis.shape_metric.junction_metric import select_trimesh_junction_boundary_loop

        selected_loop = select_trimesh_junction_boundary_loop(mesh)
        if selected_loop is not None:
            loop_vertices = vertices[np.asarray(selected_loop["vertex_indices"], dtype=int)]
            if len(loop_vertices) > 0:
                return loop_vertices.mean(axis=0)
    except Exception:
        pass
    return np.asarray(mesh.centroid if hasattr(mesh, "centroid") else vertices.mean(axis=0), dtype=float)


def _compose_graphs(graphs: Sequence[DendriticGraph], snap_threshold: float = 2.0) -> DendriticGraph:
    """Объединяет несколько дендритных графов в один граф нейрона.
    Переиндексирует узлы, объединяет графы и добавляет короткие
    рёбра между узлами, расстояние между которыми не превышает `snap_threshold`.

    Входные данные: последовательность графов и радиус сшивания близких узлов.
    Выходные данные: единый объект `DendriticGraph`.
    """
    merged = DendriticGraph()
    node_offset = 0
    for graph in graphs:
        mapping = {old: old + node_offset for old in graph.G.nodes()}
        relabelled = nx.relabel_nodes(graph.G, mapping)
        merged.G = nx.compose(merged.G, relabelled)
        if graph.soma_node is not None and merged.soma_node is None:
            merged.soma_node = graph.soma_node + node_offset
        node_offset += graph.G.number_of_nodes()

    if merged.G.number_of_nodes() > 1 and snap_threshold > 0:
        from scipy.spatial import cKDTree

        node_list = list(merged.G.nodes())
        positions = np.array([merged.G.nodes[n]["pos"] for n in node_list])
        kd = cKDTree(positions)
        for i, j in kd.query_pairs(snap_threshold):
            u, v = node_list[i], node_list[j]
            if merged.G.has_edge(u, v):
                continue
            length = float(np.linalg.norm(positions[i] - positions[j]))
            merged.G.add_edge(u, v, length=length)

    for node in merged.G.nodes():
        deg = merged.G.degree(node)
        if merged.G.nodes[node].get("node_type") == "soma":
            continue
        if deg == 1:
            merged.G.nodes[node]["node_type"] = "terminal"
        elif deg >= 3:
            merged.G.nodes[node]["node_type"] = "branch"
        else:
            merged.G.nodes[node]["node_type"] = "intermediate"
    return merged


def _projection_distance_summary(graph: DendriticGraph, spine_points: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Оценивает расстояния от точек шипиков до ближайших рёбер графа.
    Для каждого шипика приближённо ищет ближайшее ребро графа и
    считает расстояние до проекции на это ребро.

    Входные данные: дендритный граф и словарь точек крепления шипиков.
    Выходные данные: словарь диагностических статистик расстояний и bounding
    box координат графа/шипиков.
    """
    edges = list(graph.G.edges(data=True))
    graph_positions = (
        np.asarray([graph.G.nodes[node]["pos"] for node in graph.G.nodes()], dtype=float)
        if graph.G.number_of_nodes() > 0
        else np.empty((0, 3), dtype=float)
    )
    spine_positions = (
        np.asarray(list(spine_points.values()), dtype=float)
        if spine_points
        else np.empty((0, 3), dtype=float)
    )
    graph_bbox_min = graph_positions.min(axis=0).tolist() if len(graph_positions) else [np.nan, np.nan, np.nan]
    graph_bbox_max = graph_positions.max(axis=0).tolist() if len(graph_positions) else [np.nan, np.nan, np.nan]
    spine_bbox_min = spine_positions.min(axis=0).tolist() if len(spine_positions) else [np.nan, np.nan, np.nan]
    spine_bbox_max = spine_positions.max(axis=0).tolist() if len(spine_positions) else [np.nan, np.nan, np.nan]
    if not edges or not spine_points:
        return {
            "n_spines": len(spine_points),
            "n_edges": len(edges),
            "min": np.nan,
            "q10": np.nan,
            "median": np.nan,
            "q90": np.nan,
            "max": np.nan,
            "distances": [],
            "worst_spine_id": None,
            "graph_bbox_min": graph_bbox_min,
            "graph_bbox_max": graph_bbox_max,
            "spine_bbox_min": spine_bbox_min,
            "spine_bbox_max": spine_bbox_max,
        }

    midpoints = []
    edge_list: List[Tuple[int, int]] = []
    for u, v, _edata in edges:
        pu = graph.node_position(u)
        pv = graph.node_position(v)
        midpoints.append(0.5 * (pu + pv))
        edge_list.append((u, v))

    kd = cKDTree(np.asarray(midpoints, dtype=float))
    k = min(10, len(edge_list))
    nearest_distances = []
    worst_spine_id = None
    worst_distance = -np.inf

    for spine_id, point in spine_points.items():
        point = np.asarray(point, dtype=float)
        _dist_to_midpoint, idxs = kd.query(point, k=k)
        idxs = [int(idxs)] if np.isscalar(idxs) else [int(idx) for idx in idxs]
        best_distance = np.inf
        for idx in idxs:
            u, v = edge_list[idx]
            pu = graph.node_position(u)
            pv = graph.node_position(v)
            direction = pv - pu
            segment_length_sq = float(np.dot(direction, direction))
            if segment_length_sq < 1e-20:
                projection = pu
            else:
                t = float(np.dot(point - pu, direction) / segment_length_sq)
                t = max(0.0, min(1.0, t))
                projection = pu + t * direction
            best_distance = min(best_distance, float(np.linalg.norm(point - projection)))
        nearest_distances.append(best_distance)
        if best_distance > worst_distance:
            worst_distance = best_distance
            worst_spine_id = spine_id

    distances = np.asarray(nearest_distances, dtype=float)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return {
            "n_spines": len(spine_points),
            "n_edges": len(edges),
            "min": np.nan,
            "q10": np.nan,
            "median": np.nan,
            "q90": np.nan,
            "max": np.nan,
            "distances": [],
            "worst_spine_id": worst_spine_id,
            "graph_bbox_min": graph_bbox_min,
            "graph_bbox_max": graph_bbox_max,
            "spine_bbox_min": spine_bbox_min,
            "spine_bbox_max": spine_bbox_max,
        }
    return {
        "n_spines": len(spine_points),
        "n_edges": len(edges),
        "min": float(np.min(distances)),
        "q10": float(np.quantile(distances, 0.10)),
        "median": float(np.median(distances)),
        "q90": float(np.quantile(distances, 0.90)),
        "max": float(np.max(distances)),
        "distances": distances.tolist(),
        "worst_spine_id": worst_spine_id,
        "graph_bbox_min": graph_bbox_min,
        "graph_bbox_max": graph_bbox_max,
        "spine_bbox_min": spine_bbox_min,
        "spine_bbox_max": spine_bbox_max,
    }


def _spine_expected_branch_id(spine_id: str) -> Optional[str]:
    """Извлекает ожидаемый branch-id из пути шипика.

    Входные данные: `spine_id`, обычно вида
    `limb_000/branch_000/spines/spine_000.off`.
    Действие: находит компоненты `limb_*` и `branch_*`.
    Выходные данные: строка `limb_*/branch_*` или `None`.
    """
    parts = Path(str(spine_id).replace("\\", "/")).parts
    limb_part = next((part for part in parts if part.startswith("limb_")), None)
    branch_part = next((part for part in parts if part.startswith("branch_")), None)
    if limb_part is None or branch_part is None:
        return None
    return f"{limb_part}/{branch_part}"


def _edge_dendrite_id(graph: DendriticGraph, u: int, v: int) -> str:
    """Возвращает dendrite_id ребра графа.

    Входные данные: граф и id двух концов ребра.
    Действие: сравнивает `dendrite_id` конечных узлов; если они совпадают,
    возвращает это значение, иначе формирует смешанный id.
    Выходные данные: строковый идентификатор dendrite/branch.
    """
    u_id = str(graph.G.nodes[u].get("dendrite_id", ""))
    v_id = str(graph.G.nodes[v].get("dendrite_id", ""))
    if u_id == v_id:
        return u_id
    return f"{u_id}|{v_id}"


def _project_point_to_segment_local(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Проецирует точку на 3D-сегмент.

    Входные данные: точка и два конца сегмента.
    Действие: вычисляет ближайшую точку на сегменте, параметр `t` и
    евклидово расстояние.
    Выходные данные: `(projected_point, t, distance)`.
    """
    direction = seg_end - seg_start
    segment_length_sq = float(np.dot(direction, direction))
    if segment_length_sq < 1e-20:
        return seg_start.copy(), 0.0, float(np.linalg.norm(point - seg_start))
    t = float(np.dot(point - seg_start, direction) / segment_length_sq)
    t = max(0.0, min(1.0, t))
    projection = seg_start + t * direction
    return projection, t, float(np.linalg.norm(point - projection))


def _nearest_edge_on_expected_branch(
    graph: DendriticGraph,
    point: np.ndarray,
    expected_branch_id: Optional[str],
) -> Dict[str, Any]:
    """Ищет ближайшее ребро на ожидаемой branch шипика.

    Входные данные: граф, точка шипика и expected branch-id.
    Действие: фильтрует рёбра по `dendrite_id`, проецирует точку на каждое
    подходящее ребро и выбирает минимальное расстояние.
    Выходные данные: словарь с ближайшим ребром, точкой проекции и расстоянием.
    """
    best: Dict[str, Any] = {
        "branch_edge_source": None,
        "branch_edge_target": None,
        "branch_edge_id": None,
        "branch_projected_point": np.array([np.nan, np.nan, np.nan], dtype=float),
        "branch_edge_position": np.nan,
        "branch_distance_to_edge": np.nan,
    }
    if expected_branch_id is None:
        return best

    best_distance = np.inf
    point = np.asarray(point, dtype=float)
    for u, v, _edata in graph.G.edges(data=True):
        if _edge_dendrite_id(graph, u, v) != expected_branch_id:
            continue
        pu = graph.node_position(u)
        pv = graph.node_position(v)
        projected_point, t, distance = _project_point_to_segment_local(point, pu, pv)
        if distance < best_distance:
            best_distance = distance
            best = {
                "branch_edge_source": u,
                "branch_edge_target": v,
                "branch_edge_id": f"{u}_{v}",
                "branch_projected_point": projected_point,
                "branch_edge_position": t,
                "branch_distance_to_edge": distance,
            }
    return best


def _build_projection_branch_diagnostics(
    graph: DendriticGraph,
    spine_points: Dict[str, np.ndarray],
    projected_spines: Sequence[Any],
    unassigned_ids: Sequence[str],
) -> pd.DataFrame:
    """Формирует таблицу проверки соответствия шипика и branch проекции.

    Входные данные: граф, исходные точки шипиков, список спроецированных
    шипиков и список неспроецированных id.
    Действие: для каждого шипика извлекает expected branch из пути, определяет
    branch фактически выбранного ребра и отдельно считает ближайшее расстояние
    до skeleton ожидаемой branch.
    Выходные данные: `DataFrame` с диагностикой проекции.
    """
    projected_by_id = {spine.spine_id: spine for spine in projected_spines}
    unassigned_set = set(unassigned_ids)
    rows = []
    for spine_id, point in spine_points.items():
        point = np.asarray(point, dtype=float)
        expected_branch_id = _spine_expected_branch_id(spine_id)
        projected = projected_by_id.get(spine_id)
        projected_branch_id = (
            _edge_dendrite_id(graph, projected.edge_source, projected.edge_target)
            if projected is not None
            else None
        )
        expected_projection = _nearest_edge_on_expected_branch(graph, point, expected_branch_id)
        branch_projected_point = np.asarray(expected_projection["branch_projected_point"], dtype=float)
        rows.append(
            {
                "spine_id": spine_id,
                "expected_branch_id": expected_branch_id,
                "projected_branch_id": projected_branch_id,
                "same_branch": (
                    bool(expected_branch_id == projected_branch_id)
                    if projected is not None and expected_branch_id is not None
                    else False
                ),
                "is_projected": projected is not None,
                "is_unassigned": spine_id in unassigned_set,
                "network_distance_to_edge": projected.distance_to_edge if projected is not None else np.nan,
                "expected_branch_distance_to_edge": expected_projection["branch_distance_to_edge"],
                "expected_branch_edge_id": expected_projection["branch_edge_id"],
                "expected_branch_edge_source": expected_projection["branch_edge_source"],
                "expected_branch_edge_target": expected_projection["branch_edge_target"],
                "original_x": point[0],
                "original_y": point[1],
                "original_z": point[2],
                "expected_projected_x": branch_projected_point[0],
                "expected_projected_y": branch_projected_point[1],
                "expected_projected_z": branch_projected_point[2],
            }
        )
    return pd.DataFrame(rows)


def _plot_branch_projection_debug(
    graph: DendriticGraph,
    branch: Branch,
    spine_id: str,
    spine_mesh: Any,
    original_point: np.ndarray,
    projected_point: np.ndarray,
    distance_to_edge: float,
    save_path: Path,
    show: bool = True,
) -> Optional[Any]:
    """Строит 3D-график расстояния от шипика до skeleton его branch.

    Входные данные: полный граф, объект branch, id шипика, mesh шипика,
    исходная точка крепления, ближайшая точка на skeleton branch, расстояние,
    путь сохранения и флаг отображения.
    Действие: рисует рёбра skeleton соответствующей branch, mesh шипика,
    точку крепления, точку проекции и отрезок кратчайшего расстояния.
    Выходные данные: Plotly figure или `None`, если Plotly недоступен.
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        warnings.warn("plotly is required for branch projection debug plots.", stacklevel=2)
        return None

    branch_id = f"{branch.limb_name}/{branch.name}"
    fig = go.Figure()

    for u, v, _edata in graph.G.edges(data=True):
        if _edge_dendrite_id(graph, u, v) != branch_id:
            continue
        pu = graph.node_position(u)
        pv = graph.node_position(v)
        fig.add_trace(
            go.Scatter3d(
                x=[pu[0], pv[0], None],
                y=[pu[1], pv[1], None],
                z=[pu[2], pv[2], None],
                mode="lines",
                line=dict(color="steelblue", width=5),
                name="branch skeleton",
                showlegend=False,
            )
        )

    if spine_mesh is not None and hasattr(spine_mesh, "vertices") and hasattr(spine_mesh, "faces"):
        vertices = np.asarray(spine_mesh.vertices, dtype=float)
        faces = np.asarray(spine_mesh.faces, dtype=int)
        if vertices.ndim == 2 and vertices.shape[1] >= 3 and faces.ndim == 2 and faces.shape[1] >= 3:
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="lightpink",
                    opacity=0.55,
                    name="spine mesh",
                )
            )

    original_point = np.asarray(original_point, dtype=float)
    projected_point = np.asarray(projected_point, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=[original_point[0]],
            y=[original_point[1]],
            z=[original_point[2]],
            mode="markers",
            marker=dict(size=6, color="crimson"),
            name="attachment point",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[projected_point[0]],
            y=[projected_point[1]],
            z=[projected_point[2]],
            mode="markers",
            marker=dict(size=6, color="black"),
            name="nearest skeleton point",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[original_point[0], projected_point[0]],
            y=[original_point[1], projected_point[1]],
            z=[original_point[2], projected_point[2]],
            mode="lines",
            line=dict(color="red", width=7),
            name=f"shortest distance = {distance_to_edge:.2f}",
        )
    )
    fig.update_layout(
        title=f"{spine_id}<br>branch={branch_id}, distance={distance_to_edge:.2f}",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    if show:
        fig.show()
    return fig


@dataclass
class Soma:
    name: str
    path: Path
    mesh_path: Path
    mesh: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_microns_folder(cls, neuron_path: Path) -> "Soma":
        """Создаёт объект сомы из папки нейрона MICrONS.

        Входные данные: путь к папке нейрона.
        Действие: ищет `soma/soma_mesh.off`, загружает mesh и считает базовые
        mesh-метрики.
        Выходные данные: объект `Soma`.
        """
        soma_path = neuron_path / "soma"
        mesh_path = soma_path / "soma_mesh.off"
        mesh = _load_trimesh(mesh_path)
        return cls(
            name="soma",
            path=soma_path,
            mesh_path=mesh_path,
            mesh=mesh,
            metrics=_safe_mesh_metrics(mesh),
        )

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Возвращает центр mesh сомы.

        Входные данные: поле `self.mesh`.
        Действие: извлекает centroid mesh-объекта.
        Выходные данные: трёхмерная координата centroid или `None`.
        """
        if self.mesh is None:
            return None
        return np.asarray(self.mesh.centroid, dtype=float)


@dataclass
class Branch:
    name: str
    path: Path
    limb_name: str
    mesh_path: Path
    skeleton_path: Path
    spines_dir: Path
    dendrite_type: str = "unknown"
    mesh: Optional[Any] = None
    skeleton: Optional[Any] = None
    spine_mesh_paths: List[Path] = field(default_factory=list)
    spine_meshes: Dict[str, Any] = field(default_factory=dict)
    spine_points: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_microns_folder(
        cls,
        branch_path: Path,
        limb_name: str,
        dendrite_type: str = "unknown",
        spine_file_pattern: str = "*.off",
        load_spine_points: bool = True,
    ) -> "Branch":
        """Создаёт объект ветви из папки `branch_*` датасета MICrONS.

        Входные данные: путь к папке ветви, имя limb, тип дендрита, паттерн
        файлов шипиков и флаг загрузки точек шипиков.
        Действие: загружает `branch_mesh.off`, `branch_skeleton.npy`, при
        необходимости загружает mesh-и шипиков из папки `spines`.
        Выходные данные: объект `Branch`.
        """
        branch = cls(
            name=branch_path.name,
            path=branch_path,
            limb_name=limb_name,
            mesh_path=branch_path / "branch_mesh.off",
            skeleton_path=branch_path / "branch_skeleton.npy",
            spines_dir=branch_path / "spines",
            dendrite_type=dendrite_type,
            mesh=_load_trimesh(branch_path / "branch_mesh.off"),
            skeleton=_load_npy_object(branch_path / "branch_skeleton.npy"),
        )
        _safe_register_dendrite_skeleton(branch.mesh, branch.skeleton)
        if load_spine_points:
            branch.load_spine_points(spine_file_pattern=spine_file_pattern)
        return branch

    @property
    def length(self) -> float:
        """Возвращает длину ветви.

        Входные данные: skeleton ветви или mesh ветви.
        Действие: считает длину по skeleton; если skeleton недоступен,
        использует максимальный размер oriented bounding box как fallback.
        Выходные данные: длина ветви или `0.0`.
        """
        length = _skeleton_length(self.skeleton)
        if length > 0:
            return length
        if self.mesh is not None:
            try:
                return float(self.mesh.bounding_box_oriented.primitive.extents.max())
            except Exception:
                return 0.0
        return 0.0

    def load_spine_points(self, spine_file_pattern: str = "*.off") -> Dict[str, np.ndarray]:
        """Загружает шипики ветви и определяет их точки крепления.
        Читает mesh каждого шипика, вычисляет точку крепления и
        регистрирует её для downstream-метрик.

        Входные данные: паттерн файлов шипиков в папке `spines`.
        Выходные данные: словарь `{spine_id: attachment_point}`.
        """
        self.spine_points = {}
        self.spine_mesh_paths = []
        self.spine_meshes = {}
        if not self.spines_dir.exists():
            return self.spine_points
        for spine_path in sorted(self.spines_dir.glob(spine_file_pattern)):
            mesh = _load_trimesh(spine_path)
            if mesh is None:
                warnings.warn(f"Cannot read spine mesh {spine_path}", stacklevel=2)
                continue
            key = str(spine_path.relative_to(self.path.parents[1]))
            self.spine_mesh_paths.append(spine_path)
            self.spine_meshes[key] = mesh
            attachment_point = _attachment_point_from_spine_mesh(mesh)
            self.spine_points[key] = attachment_point
            _safe_register_attachment_center(mesh, attachment_point)
        return self.spine_points


@dataclass
class Limb:
    name: str
    path: Path
    mesh_path: Path
    skeleton_path: Path
    dendrite_type: str = "unknown"
    mesh: Optional[Any] = None
    skeleton: Optional[Any] = None
    branches: List[Branch] = field(default_factory=list)

    @classmethod
    def from_microns_folder(
        cls,
        limb_path: Path,
        dendrite_type: str = "unknown",
        branch_type_map: Optional[Dict[str, str]] = None,
        spine_file_pattern: str = "*.off",
        load_spine_points: bool = True,
    ) -> "Limb":
        """Создаёт объект limb из папки `limb_*` датасета MICrONS.

        Входные данные: путь к limb, тип дендрита, карта типов branch,
        паттерн файлов шипиков и флаг загрузки точек шипиков.
        Действие: загружает `limb_mesh.off`, `limb_skeleton.npy` и все
        дочерние `branch_*`, содержащие папку `spines`.
        Выходные данные: объект `Limb` со списком ветвей.
        """
        branch_type_map = branch_type_map or {}
        limb = cls(
            name=limb_path.name,
            path=limb_path,
            mesh_path=limb_path / "limb_mesh.off",
            skeleton_path=limb_path / "limb_skeleton.npy",
            dendrite_type=dendrite_type,
            mesh=_load_trimesh(limb_path / "limb_mesh.off"),
            skeleton=_load_npy_object(limb_path / "limb_skeleton.npy"),
        )
        _safe_register_dendrite_skeleton(limb.mesh, limb.skeleton)
        for branch_path in sorted(limb_path.glob("branch_*")):
            if not branch_path.is_dir() or not (branch_path / "spines").exists():
                continue
            branch_key = f"{limb_path.name}/{branch_path.name}"
            branch_type = branch_type_map.get(branch_key, dendrite_type)
            limb.branches.append(
                Branch.from_microns_folder(
                    branch_path,
                    limb_name=limb.name,
                    dendrite_type=branch_type,
                    spine_file_pattern=spine_file_pattern,
                    load_spine_points=load_spine_points,
                )
            )
        return limb

    @property
    def length(self) -> float:
        """Возвращает длину limb.

        Входные данные: skeleton limb и длины дочерних ветвей.
        Выходные данные: длина limb или `0.0`.
        """
        length = _skeleton_length(self.skeleton)
        if length > 0:
            return length
        branch_lengths = [branch.length for branch in self.branches if branch.length > 0]
        return float(np.sum(branch_lengths)) if branch_lengths else 0.0


@dataclass
class NeuronNetworkAnalysisResult:
    graph: DendriticGraph
    projected_spines: List[Any]
    unassigned_ids: List[str]
    projection_branch_diagnostics: pd.DataFrame
    binned_intensity: pd.DataFrame
    smooth_d: np.ndarray
    smooth_lambda: np.ndarray
    intensity_cdf_test: Dict[str, Any]
    intensity_lr_test: Dict[str, Any]
    poisson_result: Optional[PoissonModelResult]
    k_result: Optional[KFunctionResult]
    compartment_k_results: Dict[str, KFunctionResult]
    metrics_vector: Dict[str, Any]
    metadata_vector: Dict[str, Any]


@dataclass
class NeuronBranchAnalysisResult:
    dendrites: List[Dendrite]
    skipped_branches: List[str] = field(default_factory=list)


@dataclass
class NeuronFullAnalysisResult:
    network_result: Optional[NeuronNetworkAnalysisResult] = None
    branch_result: Optional[NeuronBranchAnalysisResult] = None


@dataclass
class Neuron:
    name: str
    path: Path
    soma: Soma
    limbs: List[Limb] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_microns_folder(
        cls,
        neuron_path: str | Path,
        dendrite_type_map: Optional[Dict[str, str]] = None,
        spine_file_pattern: str = "*.off",
        load_spine_points: bool = True,
    ) -> "Neuron":
        """Создаёт объект нейрона из папки датасета MICrONS.
        Загружает сому и все папки `limb_*` с дочерними ветвями.

        Входные данные: путь к папке нейрона, карта типов дендритов, паттерн
        файлов шипиков и флаг загрузки точек шипиков.
        Выходные данные: объект `Neuron`.
        """
        neuron_path = Path(neuron_path)
        if not neuron_path.exists():
            raise FileNotFoundError(f"Neuron folder not found: {neuron_path}")
        if not neuron_path.is_dir():
            raise NotADirectoryError(f"Neuron path is not a directory: {neuron_path}")
        dendrite_type_map = dendrite_type_map or {}
        soma = Soma.from_microns_folder(neuron_path)
        limbs = []
        for limb_path in sorted(neuron_path.glob("limb_*")):
            if not limb_path.is_dir():
                continue
            limb_type = dendrite_type_map.get(limb_path.name, "unknown")
            limbs.append(
                Limb.from_microns_folder(
                    limb_path,
                    dendrite_type=limb_type,
                    branch_type_map=dendrite_type_map,
                    spine_file_pattern=spine_file_pattern,
                    load_spine_points=load_spine_points,
                )
            )
        return cls(name=neuron_path.name, path=neuron_path, soma=soma, limbs=limbs)

    @property
    def branches(self) -> List[Branch]:
        """Возвращает все branch-объекты нейрона.

        Входные данные: список limb-объектов `self.limbs`.
        Выходные данные: плоский список `Branch`.
        """
        return [branch for limb in self.limbs for branch in limb.branches]

    def branch_by_id(self) -> Dict[str, Branch]:
        """Возвращает branch-объекты по id `limb_*/branch_*`.

        Входные данные: список branch-объектов нейрона.
        Действие: формирует ключ из имени limb и имени branch.
        Выходные данные: словарь `{branch_id: Branch}`.
        """
        return {f"{branch.limb_name}/{branch.name}": branch for branch in self.branches}

    @property
    def all_spine_points(self) -> Dict[str, np.ndarray]:
        """Возвращает точки всех загруженных шипиков без фильтрации ветвей.

        Входные данные: `spine_points` всех branch-объектов.
        Действие: объединяет словари точек шипиков.
        Выходные данные: словарь `{spine_id: attachment_point}`.
        """
        points: Dict[str, np.ndarray] = {}
        for branch in self.branches:
            points.update(branch.spine_points)
        return points

    def get_spine_points(self, min_valid_branch_spines: int = 3) -> Dict[str, np.ndarray]:
        """Возвращает точки шипиков только с валидных ветвей.
        Исключает ветви, где число шипиков меньше порога.

        Входные данные: минимальное число шипиков на ветви.
        Выходные данные: словарь `{spine_id: attachment_point}` для анализа.
        """
        points: Dict[str, np.ndarray] = {}
        for branch in self.branches:
            if len(branch.spine_points) < min_valid_branch_spines:
                continue
            points.update(branch.spine_points)
        return points

    @property
    def spine_points(self) -> Dict[str, np.ndarray]:
        """Возвращает точки шипиков для стандартного сетевого анализа.

        Входные данные: загруженные ветви нейрона.
        Выходные данные: словарь валидных точек шипиков.
        """
        return self.get_spine_points(min_valid_branch_spines=3)

    def build_network_graph(self, snap_threshold: float = 2.0) -> DendriticGraph:
        """Строит дендритный граф всего нейрона.

        Входные данные: skeleton limb/branch, centroid сомы и радиус сшивания
        близких узлов.
        Выходные данные: объект `DendriticGraph`.
        """
        soma_point = self.soma.centroid
        graphs: List[DendriticGraph] = []
        for limb in self.limbs:
            limb_branch_graphs: List[DendriticGraph] = []
            for branch in limb.branches:
                if branch.skeleton is None:
                    continue
                try:
                    limb_branch_graphs.append(
                        build_dendritic_graph_from_skeleton(
                            branch.skeleton,
                            soma_point=soma_point,
                            dendrite_id=f"{limb.name}/{branch.name}",
                            dendrite_type=branch.dendrite_type,
                            snap_threshold=snap_threshold,
                        )
                    )
                except Exception as exc:
                    warnings.warn(f"Cannot build graph from {branch.skeleton_path}: {exc}", stacklevel=2)
            if limb_branch_graphs:
                graphs.extend(limb_branch_graphs)
                continue

            if limb.skeleton is not None:
                try:
                    graphs.append(
                        build_dendritic_graph_from_skeleton(
                            limb.skeleton,
                            soma_point=soma_point,
                            dendrite_id=limb.name,
                            dendrite_type=limb.dendrite_type,
                            snap_threshold=snap_threshold,
                        )
                    )
                    continue
                except Exception as exc:
                    warnings.warn(f"Cannot build graph from {limb.skeleton_path}: {exc}", stacklevel=2)
        if not graphs:
            raise ValueError(f"Neuron {self.name!r} has no valid limb or branch skeletons.")
        graph = _compose_graphs(graphs, snap_threshold=snap_threshold)
        if graph.soma_node is None and soma_point is not None and graph.G.number_of_nodes() > 0:
            node_ids = list(graph.G.nodes())
            positions = np.array([graph.G.nodes[n]["pos"] for n in node_ids])
            nearest_idx = int(np.argmin(np.linalg.norm(positions - soma_point, axis=1)))
            graph.soma_node = node_ids[nearest_idx]
            graph.G.nodes[graph.soma_node]["node_type"] = "soma"
        return graph

    def infer_apical_limb_candidate(self) -> Optional[str]:
        """Предлагает кандидат на апикальный limb по грубой геометрической эвристике.
        Выбирает limb с максимальным произведением длины на удаление
        centroid limb от сомы

        Входные данные: centroid сомы, skeleton или mesh каждого limb.
        Выходные данные: имя limb-кандидата или `None`; результат не заменяет
        ручную или metadata-разметку apical/basal.
        """
        if len(self.limbs) < 2 or self.soma.centroid is None:
            return self.limbs[0].name if self.limbs else None
        soma = self.soma.centroid
        candidates = []
        for limb in self.limbs:
            points = []
            if limb.skeleton is not None:
                try:
                    arr = np.asarray(limb.skeleton, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        points = arr[:, :3]
                    elif arr.ndim == 3 and arr.shape[-1] >= 3:
                        points = arr.reshape(-1, arr.shape[-1])[:, :3]
                except Exception:
                    points = []
            if len(points) == 0 and limb.mesh is not None:
                points = np.asarray(limb.mesh.vertices, dtype=float)
            if len(points) == 0:
                continue
            centroid = np.asarray(points, dtype=float).mean(axis=0)
            length = limb.length
            distance = float(np.linalg.norm(centroid - soma))
            candidates.append((length * distance, limb.name))
        if not candidates:
            return None
        return max(candidates)[1]

    @staticmethod
    def _safe_branch_key(branch: Branch) -> str:
        return f"{branch.limb_name}/{branch.name}"

    def run_branch_analysis(
        self,
        output_dir: str | Path = "output_dendrite_metrics",
        reset_output: bool = False,
        min_valid_spines: int = 3,
        print_structural_vectors: bool = True,
        calculate_cluster_metrics: bool = True,
        calculate_comprehensive_spatial_analysis: bool = True,
        spatial_morphology_permutation_count: int = 199,
        spatial_morphology_random_state: int = 42,
        calculate_graph_metrics: bool = True,
        save_structural_organization_vector: bool = True,
    ) -> NeuronBranchAnalysisResult:
        """Запускает анализ всех валидных дендритных ветвей нейрона.
        Для каждой branch с достаточным числом шипиков создаёт
        объект `Dendrite`, считает branch-level метрики и сохраняет выбранные
        таблицы.

        Входные данные: директория вывода, параметры фильтрации ветвей,
        параметры пространственного анализа и флаги сохранения результатов.
        Выходные данные: `NeuronBranchAnalysisResult` со списком рассчитанных
        дендритов и списком пропущенных ветвей.
        """
        set_output_dir(str(output_dir))
        if reset_output:
            reset_saved_data()

        dendrites: List[Dendrite] = []
        skipped: List[str] = []

        for branch in self.branches:
            branch_key = self._safe_branch_key(branch)
            dendrite_name = f"{self.name}/{branch_key}"
            if branch.mesh is None or not branch.spine_meshes:
                skipped.append(f"{dendrite_name}: missing mesh or spines")
                continue
            if len(branch.spine_meshes) < min_valid_spines:
                skipped.append(f"{dendrite_name}: too few spines ({len(branch.spine_meshes)} < {min_valid_spines})")
                print(
                    f"[branches] skipped {dendrite_name}: too few spines "
                    f"({len(branch.spine_meshes)} < {min_valid_spines})",
                    flush=True,
                )
                continue

            print(
                f"[branches] analyzing {dendrite_name}: "
                f"n_spines={len(branch.spine_meshes)}",
                flush=True,
            )
            dendrite = Dendrite(
                dendrite_name,
                dendrite_meshes={dendrite_name: branch.mesh},
                spine_meshes=dict(branch.spine_meshes),
            )
            dendrite.save_init_metrics()

            if calculate_cluster_metrics:
                dendrite.calculate_cluster_metrics()

            if calculate_comprehensive_spatial_analysis:
                dendrite.calculate_comprehensive_spatial_analysis(
                    permutation_count=spatial_morphology_permutation_count,
                    random_state=spatial_morphology_random_state,
                )
                dendrite.save_spatial_morphology_analysis()

            if calculate_graph_metrics:
                dendrite.graph_analysis()

            if save_structural_organization_vector:
                structural_record = dendrite.save_structural_organization_vector()
                if print_structural_vectors:
                    print(f"[branches] structural vector {dendrite_name}", flush=True)
                    print(pd.Series(structural_record).to_string(), flush=True)

            dendrites.append(dendrite)

        return NeuronBranchAnalysisResult(dendrites=dendrites, skipped_branches=skipped)

    def run_full_analysis(
        self,
        mode: str = "both",
        network_output_dir: str | Path = "output_neuron_network_analysis",
        branch_output_dir: str | Path = "output_dendrite_metrics",
        reset_branch_output: bool = False,
        **kwargs: Any,
    ) -> NeuronFullAnalysisResult:
        """Запускает сетевой и/или branch-level анализ нейрона.

        Входные данные: режим `network`, `branches` или `both`, директории
        вывода и словари параметров для соответствующих подпроцедур.
        Выходные данные: `NeuronFullAnalysisResult` с результатами выбранных
        этапов.
        """
        mode = mode.lower()
        if mode not in {"network", "branches", "both"}:
            raise ValueError("mode must be one of: 'network', 'branches', 'both'")

        network_kwargs = dict(kwargs.pop("network_kwargs", {}))
        branch_kwargs = dict(kwargs.pop("branch_kwargs", {}))
        if kwargs:
            raise ValueError(f"Unknown keyword arguments: {sorted(kwargs)}")

        network_result = None
        branch_result = None
        if mode in {"network", "both"}:
            network_result = self.run_network_analysis(
                output_dir=network_output_dir,
                **network_kwargs,
            )
        if mode in {"branches", "both"}:
            branch_result = self.run_branch_analysis(
                output_dir=branch_output_dir,
                reset_output=reset_branch_output,
                **branch_kwargs,
            )
        return NeuronFullAnalysisResult(network_result=network_result, branch_result=branch_result)

    def run_network_analysis(
        self,
        output_dir: str | Path = "output_neuron_network_analysis",
        snap_threshold: float = 2.0,
        max_distance_to_edge: Optional[float] = None,
        auto_max_distance_to_edge_quantile: float = 1.0,
        auto_max_distance_to_edge_margin: float = 1.05,
        min_valid_branch_spines: int = 3,
        bin_size: float = 25.0,
        covariates: Sequence[str] = ("intercept", "distance_to_soma", "distance_to_soma_squared"),
        n_r_values: int = 20,
        r_max: Optional[float] = None,
        n_simulations: int = 99,
        k_correction: str = "geometric",
        random_state: Optional[int] = 42,
        save_outputs: bool = True,
        log_projection_diagnostics: bool = True,
        save_projection_branch_debug_plots: bool = True,
        show_projection_branch_debug_plots: bool = True,
        projection_branch_debug_plot_count: int = 3,
    ) -> NeuronNetworkAnalysisResult:
        """Запускает полный анализ пространственной организации шипиков на сети нейрона.
        Строит дендритный граф, проецирует шипики на сеть, считает
        интенсивность вдоль расстояния от сомы, статистические тесты,
        неоднородную пуассоновскую модель и сетевую функцию Рипли.

        Входные данные: параметры построения графа, проекции шипиков,
        интенсивности, пуассоновской модели, K-функции и сохранения.
        Выходные данные: объект `NeuronNetworkAnalysisResult` с графом,
        projected spines, таблицами анализа, ML-вектором и metadata-вектором.
        """
        output_dir = Path(output_dir)
        graph = self.build_network_graph(snap_threshold=snap_threshold)
        spine_points = self.get_spine_points(min_valid_branch_spines=min_valid_branch_spines)
        projection_distance_summary = _projection_distance_summary(graph, spine_points)
        projection_threshold_is_auto = max_distance_to_edge is None
        if projection_threshold_is_auto:
            nearest_distances = np.asarray(projection_distance_summary.get("distances", []), dtype=float)
            nearest_distances = nearest_distances[np.isfinite(nearest_distances)]
            if len(nearest_distances) > 0:
                quantile = float(np.clip(auto_max_distance_to_edge_quantile, 0.0, 1.0))
                margin = float(auto_max_distance_to_edge_margin)
                max_distance_to_edge_effective = float(np.quantile(nearest_distances, quantile) * margin)
            else:
                max_distance_to_edge_effective = 0.0
        else:
            max_distance_to_edge_effective = float(max_distance_to_edge)
        projected_spines, unassigned_ids = project_spines_to_graph(
            graph,
            spine_points,
            max_distance_to_edge=max_distance_to_edge_effective,
        )
        projection_branch_diagnostics = _build_projection_branch_diagnostics(
            graph,
            spine_points,
            projected_spines,
            unassigned_ids,
        )
        if log_projection_diagnostics:
            threshold_label = (
                f"auto={max_distance_to_edge_effective:.4f} "
                f"(q={auto_max_distance_to_edge_quantile}, margin={auto_max_distance_to_edge_margin})"
                if projection_threshold_is_auto
                else f"{max_distance_to_edge_effective:.4f}"
            )
            print(
                "[network projection] "
                f"{self.name}: threshold={threshold_label}, "
                f"projected={len(projected_spines)}/{len(spine_points)}, "
                f"nearest_edge_distance min={projection_distance_summary['min']:.4f}, "
                f"q10={projection_distance_summary['q10']:.4f}, "
                f"median={projection_distance_summary['median']:.4f}, "
                f"q90={projection_distance_summary['q90']:.4f}, "
                f"max={projection_distance_summary['max']:.4f}",
                flush=True,
            )
            if len(projection_branch_diagnostics) > 0:
                projected_diag = projection_branch_diagnostics[
                    projection_branch_diagnostics["is_projected"].astype(bool)
                ]
                same_branch_count = int(projected_diag["same_branch"].sum()) if len(projected_diag) else 0
                mismatch_count = int(len(projected_diag) - same_branch_count)
                missing_branch_count = int(
                    projection_branch_diagnostics["expected_branch_distance_to_edge"].isna().sum()
                )
                print(
                    "[network projection branch-check] "
                    f"{self.name}: same_branch={same_branch_count}/{len(projected_diag)}, "
                    f"mismatch={mismatch_count}, "
                    f"missing_expected_branch_edges={missing_branch_count}",
                    flush=True,
                )
                if mismatch_count > 0:
                    mismatch_preview = projected_diag[~projected_diag["same_branch"].astype(bool)].head(5)
                    print(
                        "[network projection branch-check] mismatches preview:\n"
                        + mismatch_preview[
                            [
                                "spine_id",
                                "expected_branch_id",
                                "projected_branch_id",
                                "network_distance_to_edge",
                                "expected_branch_distance_to_edge",
                            ]
                        ].to_string(index=False),
                        flush=True,
                    )
            if len(projected_spines) == 0 and len(spine_points) > 0:
                print(
                    "[network projection] "
                    f"{self.name}: no spines passed the threshold. "
                    "Most likely max_distance_to_edge is smaller than the "
                    "distance from surface attachment points to the dendrite skeleton, "
                    "or skeleton/spine coordinates are in different coordinate systems.",
                    flush=True,
                )
                print(
                    "[network projection] "
                    f"{self.name}: graph_bbox={projection_distance_summary['graph_bbox_min']}.."
                    f"{projection_distance_summary['graph_bbox_max']}, "
                    f"spine_bbox={projection_distance_summary['spine_bbox_min']}.."
                    f"{projection_distance_summary['spine_bbox_max']}",
                    flush=True,
                )

        if save_projection_branch_debug_plots and len(projection_branch_diagnostics) > 0:
            branch_map = self.branch_by_id()
            debug_dir = Path(output_dir) / self.name / "projection_branch_debug"
            finite_diag = projection_branch_diagnostics[
                projection_branch_diagnostics["expected_branch_distance_to_edge"].notna()
            ].copy()
            finite_diag = finite_diag.sort_values("expected_branch_distance_to_edge", ascending=False)
            for rank, row in enumerate(finite_diag.head(int(projection_branch_debug_plot_count)).itertuples(index=False), start=1):
                branch = branch_map.get(row.expected_branch_id)
                if branch is None:
                    continue
                spine_mesh = branch.spine_meshes.get(row.spine_id)
                original_point = np.array([row.original_x, row.original_y, row.original_z], dtype=float)
                projected_point = np.array(
                    [row.expected_projected_x, row.expected_projected_y, row.expected_projected_z],
                    dtype=float,
                )
                safe_spine_name = str(row.spine_id).replace("\\", "_").replace("/", "_").replace(":", "_")
                save_path = debug_dir / f"top_{rank}_{safe_spine_name}.html"
                fig = _plot_branch_projection_debug(
                    graph=graph,
                    branch=branch,
                    spine_id=row.spine_id,
                    spine_mesh=spine_mesh,
                    original_point=original_point,
                    projected_point=projected_point,
                    distance_to_edge=float(row.expected_branch_distance_to_edge),
                    save_path=save_path,
                    show=show_projection_branch_debug_plots,
                )
                if fig is not None:
                    print(f"[network projection branch-debug] saved: {save_path}", flush=True)

        binned_intensity = estimate_binned_intensity(graph, projected_spines, bin_size=bin_size)
        smooth_d, smooth_lambda = estimate_smooth_intensity(graph, projected_spines)
        intensity_cdf_test = test_intensity_dependence_cdf(graph, projected_spines)
        intensity_lr_test = test_intensity_dependence(graph, projected_spines)

        poisson_result: Optional[PoissonModelResult] = None
        if len(projected_spines) >= 3:
            try:
                poisson_result = fit_inhomogeneous_poisson(graph, projected_spines, covariates=covariates)
            except Exception as exc:
                warnings.warn(f"Poisson model fitting failed for neuron {self.name}: {exc}", stacklevel=2)

        if r_max is None:
            r_max = graph.total_length * 0.3
        r_values = np.linspace(0.0, float(r_max), int(n_r_values) + 1)[1:]

        k_result: Optional[KFunctionResult] = None
        if len(projected_spines) >= 2:
            try:
                k_result = compute_simulation_envelopes(
                    graph,
                    projected_spines,
                    r_values,
                    n_simulations=n_simulations,
                    intensity_model=poisson_result,
                    correction=k_correction,
                    random_state=random_state,
                )
            except Exception as exc:
                warnings.warn(f"K analysis failed for neuron {self.name}: {exc}", stacklevel=2)

        compartment_k_results = self._run_compartment_k_analysis(
            graph=graph,
            projected_spines=projected_spines,
            n_r_values=n_r_values,
            n_simulations=n_simulations,
            k_correction=k_correction,
            covariates=covariates,
            random_state=random_state,
        )

        metrics_vector = self.build_structural_network_vector(
            graph=graph,
            projected_spines=projected_spines,
            projected_count=len(projected_spines),
            unassigned_count=len(unassigned_ids),
            intensity_cdf_test=intensity_cdf_test,
            intensity_lr_test=intensity_lr_test,
            poisson_result=poisson_result,
            k_result=k_result,
            compartment_k_results=compartment_k_results,
        )
        metadata_vector = self.build_neuron_metadata_vector(
            graph=graph,
            projected_count=len(projected_spines),
            unassigned_count=len(unassigned_ids),
            poisson_result=poisson_result,
            k_result=k_result,
        )
        projected_diag = projection_branch_diagnostics[
            projection_branch_diagnostics["is_projected"].astype(bool)
        ] if len(projection_branch_diagnostics) else projection_branch_diagnostics
        same_branch_projected_count = int(projected_diag["same_branch"].sum()) if len(projected_diag) else 0
        branch_mismatch_count = int(len(projected_diag) - same_branch_projected_count)
        metadata_vector.update(
            {
                "projection_nearest_edge_distance_min": projection_distance_summary["min"],
                "projection_nearest_edge_distance_q10": projection_distance_summary["q10"],
                "projection_nearest_edge_distance_median": projection_distance_summary["median"],
                "projection_nearest_edge_distance_q90": projection_distance_summary["q90"],
                "projection_nearest_edge_distance_max": projection_distance_summary["max"],
                "projection_worst_spine_id": projection_distance_summary["worst_spine_id"],
                "projection_max_distance_to_edge": max_distance_to_edge_effective,
                "projection_threshold_auto": projection_threshold_is_auto,
                "projection_threshold_quantile": auto_max_distance_to_edge_quantile if projection_threshold_is_auto else np.nan,
                "projection_threshold_margin": auto_max_distance_to_edge_margin if projection_threshold_is_auto else np.nan,
                "projection_same_branch_count": same_branch_projected_count,
                "projection_branch_mismatch_count": branch_mismatch_count,
                "projection_same_branch_rate": (
                    same_branch_projected_count / len(projected_diag)
                    if len(projected_diag)
                    else np.nan
                ),
            }
        )

        result = NeuronNetworkAnalysisResult(
            graph=graph,
            projected_spines=projected_spines,
            unassigned_ids=unassigned_ids,
            projection_branch_diagnostics=projection_branch_diagnostics,
            binned_intensity=binned_intensity,
            smooth_d=smooth_d,
            smooth_lambda=smooth_lambda,
            intensity_cdf_test=intensity_cdf_test,
            intensity_lr_test=intensity_lr_test,
            poisson_result=poisson_result,
            k_result=k_result,
            compartment_k_results=compartment_k_results,
            metrics_vector=metrics_vector,
            metadata_vector=metadata_vector,
        )

        if save_outputs:
            self.save_network_analysis_result(result, output_dir=output_dir)

        return result

    def build_structural_network_vector(
        self,
        graph: DendriticGraph,
        projected_spines: Sequence[Any],
        projected_count: int,
        unassigned_count: int,
        intensity_cdf_test: Dict[str, Any],
        intensity_lr_test: Dict[str, Any],
        poisson_result: Optional[PoissonModelResult],
        k_result: Optional[KFunctionResult],
        compartment_k_results: Optional[Dict[str, KFunctionResult]] = None,
    ) -> Dict[str, Any]:
        """Формирует ML-вектор признаков пространственной организации нейрона.

        Входные данные: дендритный граф, спроецированные шипики, результаты
        тестов интенсивности, пуассоновской модели и K-анализа.
        Выходные данные: словарь признаков одного нейрона для последующего
        анализа или обучения модели.
        """
        branch_nodes = sum(1 for _, data in graph.G.nodes(data=True) if data.get("node_type") == "branch")
        terminal_nodes = sum(1 for node in graph.G.nodes() if graph.G.degree(node) == 1)
        total_length = graph.total_length
        has_length = total_length > 1e-12
        type_metrics = self._calculate_dendrite_type_network_metrics(graph, projected_spines)

        row: Dict[str, Any] = {
            "neuron_id": self.name,
            "network_branch_node_linear_density": branch_nodes / total_length if has_length else np.nan,
            "network_terminal_node_linear_density": terminal_nodes / total_length if has_length else np.nan,
            "network_spine_linear_density": projected_count / total_length if has_length else np.nan,
            "apical_spine_linear_density": type_metrics["apical_spine_linear_density"],
            "basal_spine_linear_density": type_metrics["basal_spine_linear_density"],
            "apical_branch_node_linear_density": type_metrics["apical_branch_node_linear_density"],
            "basal_branch_node_linear_density": type_metrics["basal_branch_node_linear_density"],
            "apical_terminal_node_linear_density": type_metrics["apical_terminal_node_linear_density"],
            "basal_terminal_node_linear_density": type_metrics["basal_terminal_node_linear_density"],
            "soma_volume": self.soma.metrics.get("volume", np.nan),
            "soma_surface_area": self.soma.metrics.get("surface_area", np.nan),
            "intensity_cdf_statistic": intensity_cdf_test.get("statistic", np.nan),
            "intensity_cdf_p_value": intensity_cdf_test.get("p_value", np.nan),
            "intensity_lr_statistic": intensity_lr_test.get("lr_statistic", np.nan),
            "intensity_lr_p_value": intensity_lr_test.get("p_value", np.nan),
        }

        if poisson_result is not None:
            row.update(
                {
                    "poisson_log_likelihood": poisson_result.log_likelihood,
                    "poisson_aic": poisson_result.aic,
                    "poisson_bic": poisson_result.bic,
                }
            )
            for name, coef, se in zip(
                poisson_result.covariate_names,
                poisson_result.coefficients,
                poisson_result.standard_errors,
            ):
                row[f"poisson_coef_{name}"] = float(coef)
                row[f"poisson_se_{name}"] = float(se)

        if k_result is not None:
            self._add_k_summary_to_row(row, "k", k_result)
        for compartment, compartment_k_result in (compartment_k_results or {}).items():
            self._add_k_summary_to_row(row, f"{compartment}_k", compartment_k_result)
        return row

    @staticmethod
    def _add_k_summary_to_row(row: Dict[str, Any], prefix: str, k_result: KFunctionResult) -> None:
        """Добавляет scalar summaries K-кривой в строку признаков.

        Входные данные: словарь признаков, префикс полей и результат K-анализа.
        Выходные данные: обновлённый словарь `row`.
        """
        deviation = k_result.k_observed - k_result.k_expected
        max_pos_idx = int(np.argmax(deviation)) if len(deviation) else 0
        max_neg_idx = int(np.argmin(deviation)) if len(deviation) else 0
        max_abs_idx = int(np.argmax(np.abs(deviation))) if len(deviation) else 0
        row.update(
            {
                f"{prefix}_p_value": k_result.p_value,
                f"{prefix}_mean_deviation": float(np.mean(deviation)) if len(deviation) else np.nan,
                f"{prefix}_max_positive_deviation": float(np.max(deviation)) if len(deviation) else np.nan,
                f"{prefix}_max_negative_deviation": float(np.min(deviation)) if len(deviation) else np.nan,
                f"{prefix}_max_abs_deviation": float(np.max(np.abs(deviation))) if len(deviation) else np.nan,
                f"{prefix}_r_at_max_positive_deviation": (
                    float(k_result.r_values[max_pos_idx]) if len(deviation) else np.nan
                ),
                f"{prefix}_r_at_max_negative_deviation": (
                    float(k_result.r_values[max_neg_idx]) if len(deviation) else np.nan
                ),
                f"{prefix}_r_at_max_abs_deviation": (
                    float(k_result.r_values[max_abs_idx]) if len(deviation) else np.nan
                ),
            }
        )

    @staticmethod
    def _normalized_dendrite_type(value: Any) -> str:
        """Нормализует значение типа дендрита.

        Входные данные: произвольное значение типа.
        Выходные данные: нормализованная строка типа дендрита.
        """
        value = str(value or "unknown").lower()
        return value if value in {"apical", "basal"} else "unknown"

    @classmethod
    def _edge_dendrite_type(cls, graph: DendriticGraph, u: int, v: int) -> str:
        """Определяет тип дендрита для ребра графа.
        Сравнивает типы концов ребра и разрешает `unknown`, если
        один из концов не размечен.

        Входные данные: граф и два узла ребра.
        Выходные данные: `apical`, `basal` или `unknown`.
        """
        u_type = cls._normalized_dendrite_type(graph.G.nodes[u].get("dendrite_type", "unknown"))
        v_type = cls._normalized_dendrite_type(graph.G.nodes[v].get("dendrite_type", "unknown"))
        if u_type == v_type:
            return u_type
        if u_type == "unknown":
            return v_type
        if v_type == "unknown":
            return u_type
        return "unknown"

    @classmethod
    def _calculate_dendrite_type_network_metrics(
        cls,
        graph: DendriticGraph,
        projected_spines: Sequence[Any],
    ) -> Dict[str, float]:
        """Вычисляет density-метрики отдельно для apical и basal частей.
        Суммирует длины рёбер, считает узлы ветвления, терминальные
        узлы и шипики по типам дендритов, затем нормирует счётчики на длину
        соответствующей части сети.

        Входные данные: дендритный граф и список спроецированных шипиков.
        Выходные данные: словарь compartment-specific длин и линейных
        плотностей.
        """
        lengths = {"apical": 0.0, "basal": 0.0, "unknown": 0.0}
        branch_nodes = {"apical": 0, "basal": 0, "unknown": 0}
        terminal_nodes = {"apical": 0, "basal": 0, "unknown": 0}
        spine_counts = {"apical": 0, "basal": 0, "unknown": 0}

        for u, v, data in graph.G.edges(data=True):
            dtype = cls._edge_dendrite_type(graph, u, v)
            lengths[dtype] += float(data.get("length", 0.0) or 0.0)

        for node, data in graph.G.nodes(data=True):
            dtype = cls._normalized_dendrite_type(data.get("dendrite_type", "unknown"))
            if data.get("node_type") == "branch":
                branch_nodes[dtype] += 1
            if graph.G.degree(node) == 1:
                terminal_nodes[dtype] += 1

        for spine in projected_spines:
            dtype = cls._edge_dendrite_type(graph, spine.edge_source, spine.edge_target)
            spine_counts[dtype] += 1

        def density(counts: Dict[str, int], dtype: str) -> float:
            return counts[dtype] / lengths[dtype] if lengths[dtype] > 1e-12 else np.nan

        return {
            "apical_length": lengths["apical"],
            "basal_length": lengths["basal"],
            "unknown_length": lengths["unknown"],
            "apical_spine_linear_density": density(spine_counts, "apical"),
            "basal_spine_linear_density": density(spine_counts, "basal"),
            "apical_branch_node_linear_density": density(branch_nodes, "apical"),
            "basal_branch_node_linear_density": density(branch_nodes, "basal"),
            "apical_terminal_node_linear_density": density(terminal_nodes, "apical"),
            "basal_terminal_node_linear_density": density(terminal_nodes, "basal"),
        }

    @classmethod
    def _subgraph_for_dendrite_type(cls, graph: DendriticGraph, dendrite_type: str) -> DendriticGraph:
        """Выделяет подграф заданного типа дендритов.
        Оставляет только рёбра выбранного типа и переносит доступные
        расстояния от сомы.

        Входные данные: полный дендритный граф и тип `apical` или `basal`.
        Выходные данные: объект `DendriticGraph` для выбранного compartment.
        """
        selected_edges = [
            (u, v)
            for u, v in graph.G.edges()
            if cls._edge_dendrite_type(graph, u, v) == dendrite_type
        ]
        subgraph = DendriticGraph()
        subgraph.G = graph.G.edge_subgraph(selected_edges).copy()
        if graph.soma_node in subgraph.G:
            subgraph.soma_node = graph.soma_node
        soma_distances = graph.soma_distances()
        subgraph._soma_distances = {
            node: distance
            for node, distance in soma_distances.items()
            if node in subgraph.G
        }
        return subgraph

    @classmethod
    def _projected_spines_for_dendrite_type(
        cls,
        graph: DendriticGraph,
        projected_spines: Sequence[Any],
        dendrite_type: str,
    ) -> List[Any]:
        """Фильтрует спроецированные шипики по типу дендрита.
        Оставляет шипики, лежащие на рёбрах заданного типа.

        Входные данные: полный граф, список спроецированных шипиков и тип
        дендрита.
        Выходные данные: список `ProjectedSpine` для выбранного compartment.
        """
        return [
            spine
            for spine in projected_spines
            if cls._edge_dendrite_type(graph, spine.edge_source, spine.edge_target) == dendrite_type
        ]

    def _run_compartment_k_analysis(
        self,
        graph: DendriticGraph,
        projected_spines: Sequence[Any],
        n_r_values: int,
        n_simulations: int,
        k_correction: str,
        covariates: Sequence[str],
        random_state: Optional[int],
    ) -> Dict[str, KFunctionResult]:
        """Выполняет K-анализ отдельно для апикальной и базальной частей.
        Выделяет подграфы `apical` и `basal`, фильтрует шипики по
        типу рёбер и считает compartment-specific K-кривые.

        Входные данные: полный граф нейрона, спроецированные шипики, параметры
        K-функции, ковариаты пуассоновской модели и seed.
        Выходные данные: словарь `{тип дендрита: KFunctionResult}`.
        """
        results: Dict[str, KFunctionResult] = {}
        rng = np.random.default_rng(random_state)
        for dendrite_type in ("apical", "basal"):
            compartment_graph = self._subgraph_for_dendrite_type(graph, dendrite_type)
            compartment_spines = self._projected_spines_for_dendrite_type(
                graph,
                projected_spines,
                dendrite_type,
            )
            if compartment_graph.total_length <= 1e-12 or len(compartment_spines) < 3:
                continue
            r_max = compartment_graph.total_length * 0.3
            r_values = np.linspace(0.0, float(r_max), int(n_r_values) + 1)[1:]
            compartment_poisson: Optional[PoissonModelResult] = None
            if len(compartment_spines) >= 3:
                try:
                    compartment_poisson = fit_inhomogeneous_poisson(
                        compartment_graph,
                        compartment_spines,
                        covariates=covariates,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"{dendrite_type} Poisson model fitting failed for neuron {self.name}: {exc}",
                        stacklevel=2,
                    )
            try:
                results[dendrite_type] = compute_simulation_envelopes(
                    compartment_graph,
                    compartment_spines,
                    r_values,
                    n_simulations=n_simulations,
                    intensity_model=compartment_poisson,
                    correction=k_correction,
                    fit_intensity_if_missing=False,
                    random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
            except Exception as exc:
                warnings.warn(
                    f"{dendrite_type} K analysis failed for neuron {self.name}: {exc}",
                    stacklevel=2,
                )
        return results

    def build_neuron_metadata_vector(
        self,
        graph: DendriticGraph,
        projected_count: int,
        unassigned_count: int,
        poisson_result: Optional[PoissonModelResult],
        k_result: Optional[KFunctionResult],
    ) -> Dict[str, Any]:
        """Формирует metadata-вектор технических и абсолютных характеристик нейрона.
        Собирает счётчики, абсолютные длины, характеристики
        проекции, размеры выборки и диагностические поля, не входящие в
        ML-вектор.

        Входные данные: граф, число спроецированных и неспроецированных
        шипиков, результат пуассоновской модели и K-анализа.
        Выходные данные: словарь metadata одной записи нейрона.
        """
        limb_lengths = np.array([limb.length for limb in self.limbs if limb.length > 0], dtype=float)
        branch_lengths = np.array([branch.length for branch in self.branches if branch.length > 0], dtype=float)
        branch_spine_counts = np.array([len(branch.spine_points) for branch in self.branches], dtype=float)
        dendrite_types = [limb.dendrite_type for limb in self.limbs]
        branch_types = [branch.dendrite_type for branch in self.branches]

        branch_nodes = sum(1 for _, data in graph.G.nodes(data=True) if data.get("node_type") == "branch")
        terminal_nodes = sum(1 for node in graph.G.nodes() if graph.G.degree(node) == 1)
        total_spines = int(sum(len(branch.spine_points) for branch in self.branches))
        valid_spines = len(self.spine_points)
        type_metrics = self._calculate_dendrite_type_network_metrics(graph, [])

        row: Dict[str, Any] = {
            "neuron_id": self.name,
            "n_limbs": len(self.limbs),
            "n_branches": len(self.branches),
            "n_spines_total": total_spines,
            "n_spines_valid_for_analysis": valid_spines,
            "n_spines_projected": projected_count,
            "n_spines_unassigned": unassigned_count,
            "projection_success_rate": projected_count / valid_spines if valid_spines else np.nan,
            "network_total_length": graph.total_length,
            "network_n_nodes": graph.n_nodes,
            "network_n_edges": graph.n_edges,
            "network_n_branch_nodes": branch_nodes,
            "network_n_terminal_nodes": terminal_nodes,
            "network_apical_length": type_metrics["apical_length"],
            "network_basal_length": type_metrics["basal_length"],
            "network_unknown_length": type_metrics["unknown_length"],
            "n_apical_limbs": int(sum(dtype == "apical" for dtype in dendrite_types)),
            "n_basal_limbs": int(sum(dtype == "basal" for dtype in dendrite_types)),
            "n_unknown_limbs": int(sum(dtype not in {"apical", "basal"} for dtype in dendrite_types)),
            "n_apical_branches": int(sum(dtype == "apical" for dtype in branch_types)),
            "n_basal_branches": int(sum(dtype == "basal" for dtype in branch_types)),
            "n_unknown_branches": int(sum(dtype not in {"apical", "basal"} for dtype in branch_types)),
            "limb_length_mean": float(limb_lengths.mean()) if len(limb_lengths) else np.nan,
            "limb_length_median": float(np.median(limb_lengths)) if len(limb_lengths) else np.nan,
            "limb_length_std": float(limb_lengths.std(ddof=1)) if len(limb_lengths) > 1 else 0.0,
            "branch_length_mean": float(branch_lengths.mean()) if len(branch_lengths) else np.nan,
            "branch_length_median": float(np.median(branch_lengths)) if len(branch_lengths) else np.nan,
            "branch_length_std": float(branch_lengths.std(ddof=1)) if len(branch_lengths) > 1 else 0.0,
            "branch_spine_count_mean": float(branch_spine_counts.mean()) if len(branch_spine_counts) else np.nan,
            "branch_spine_count_median": float(np.median(branch_spine_counts)) if len(branch_spine_counts) else np.nan,
            "branch_spine_count_std": float(branch_spine_counts.std(ddof=1)) if len(branch_spine_counts) > 1 else 0.0,
        }
        if poisson_result is not None:
            row["poisson_converged"] = poisson_result.diagnostics.get("converged", np.nan)
        if k_result is not None:
            row["k_method"] = k_result.method
        return row

    def save_network_analysis_result(
        self,
        result: NeuronNetworkAnalysisResult,
        output_dir: str | Path,
        summary_filename: str = "neuron_structural_network_vectors.csv",
        metadata_filename: str = "neuron_metadata.csv",
    ) -> None:
        """Сохраняет результаты сетевого анализа нейрона.
        Сохраняет per-neuron таблицы, K-кривые, общий ML-вектор и
        metadata-вектор; при повторном запуске заменяет строку текущего
        нейрона.

        Входные данные: объект `NeuronNetworkAnalysisResult`, директория вывода
        и имена summary-файлов.
        Выходные данные: набор CSV-файлов в директории анализа.
        """
        output_dir = Path(output_dir)
        neuron_dir = output_dir / self.name
        neuron_dir.mkdir(parents=True, exist_ok=True)

        result.binned_intensity.to_csv(neuron_dir / "binned_intensity.csv", index=False)
        projected_spines_dataframe(result.projected_spines).to_csv(neuron_dir / "projected_spines.csv", index=False)
        result.projection_branch_diagnostics.to_csv(neuron_dir / "projection_branch_diagnostics.csv", index=False)
        pd.DataFrame([result.intensity_cdf_test]).to_csv(neuron_dir / "intensity_cdf_test.csv", index=False)
        pd.DataFrame([result.intensity_lr_test]).to_csv(neuron_dir / "intensity_lr_test.csv", index=False)
        if result.k_result is not None:
            result.k_result.to_dataframe().to_csv(neuron_dir / "ripley_k_network.csv", index=False)
        for compartment, k_result in result.compartment_k_results.items():
            k_result.to_dataframe().to_csv(neuron_dir / f"ripley_k_network_{compartment}.csv", index=False)

        summary_path = output_dir / summary_filename
        new_row = pd.DataFrame([result.metrics_vector])
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            old = old[old["neuron_id"] != self.name] if "neuron_id" in old.columns else old
            combined = pd.concat([old, new_row], ignore_index=True, sort=False)
        else:
            combined = new_row
        combined.to_csv(summary_path, index=False)

        metadata_path = output_dir / metadata_filename
        new_metadata_row = pd.DataFrame([result.metadata_vector])
        if metadata_path.exists():
            old_metadata = pd.read_csv(metadata_path)
            old_metadata = (
                old_metadata[old_metadata["neuron_id"] != self.name]
                if "neuron_id" in old_metadata.columns
                else old_metadata
            )
            combined_metadata = pd.concat([old_metadata, new_metadata_row], ignore_index=True, sort=False)
        else:
            combined_metadata = new_metadata_row
        combined_metadata.to_csv(metadata_path, index=False)


def load_microns_neurons(
    root_path: str | Path,
    dendrite_type_map: Optional[Dict[str, Dict[str, str]] | Dict[str, str]] = None,
    spine_file_pattern: str = "*.off",
    load_spine_points: bool = True,
) -> List[Neuron]:
    """Загружает все нейроны из корневой папки MICrONS-датасета.

    Входные данные: корневая папка, карта типов дендритов, паттерн файлов
    шипиков и флаг загрузки точек шипиков.
    Выходные данные: список загруженных объектов `Neuron`.
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"MICrONS root folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"MICrONS root path is not a directory: {root}")
    neurons: List[Neuron] = []
    dendrite_type_map = dendrite_type_map or {}
    for neuron_path in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (neuron_path / "soma").exists():
            continue
        per_neuron_map: Dict[str, str]
        if neuron_path.name in dendrite_type_map and isinstance(dendrite_type_map[neuron_path.name], dict):  # type: ignore[index]
            per_neuron_map = dendrite_type_map[neuron_path.name]  # type: ignore[index,assignment]
        else:
            per_neuron_map = dendrite_type_map  # type: ignore[assignment]
        neurons.append(
            Neuron.from_microns_folder(
                neuron_path,
                dendrite_type_map=per_neuron_map,
                spine_file_pattern=spine_file_pattern,
                load_spine_points=load_spine_points,
            )
        )
    return neurons


def run_microns_neuron_analyses(
    root_path: str | Path,
    mode: str = "both",
    dendrite_type_map: Optional[Dict[str, Dict[str, str]] | Dict[str, str]] = None,
    spine_file_pattern: str = "*.off",
    network_output_dir: str | Path = "output_neuron_network_analysis",
    branch_output_dir: str | Path = "output_dendrite_metrics",
    reset_branch_output: bool = True,
    network_kwargs: Optional[Dict[str, Any]] = None,
    branch_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Neuron], Dict[str, NeuronFullAnalysisResult]]:
    """Запускает batch-анализ нейронов MICrONS.

    Входные данные: корневая папка датасета, режим анализа, карта типов
    дендритов, директории вывода и параметры network/branch подпроцедур.
    Выходные данные: список объектов `Neuron` и словарь результатов по имени
    нейрона.
    """
    mode = mode.lower()
    if mode not in {"network", "branches", "both"}:
        raise ValueError("mode must be one of: 'network', 'branches', 'both'")

    neurons = load_microns_neurons(
        root_path,
        dendrite_type_map=dendrite_type_map,
        spine_file_pattern=spine_file_pattern,
        load_spine_points=True,
    )

    if mode in {"branches", "both"} and reset_branch_output:
        set_output_dir(str(branch_output_dir))
        reset_saved_data()

    results: Dict[str, NeuronFullAnalysisResult] = {}
    for neuron in neurons:
        results[neuron.name] = neuron.run_full_analysis(
            mode=mode,
            network_output_dir=network_output_dir,
            branch_output_dir=branch_output_dir,
            reset_branch_output=False,
            network_kwargs=network_kwargs or {},
            branch_kwargs=branch_kwargs or {},
        )

    return neurons, results
