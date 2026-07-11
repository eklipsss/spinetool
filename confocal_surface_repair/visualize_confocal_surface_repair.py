from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from confocal_surface_repair.run_confocal_surface_repair import (
    DATASETS,
    Z_SCALE,
    combine_pickles,
    convex_bridge,
    distance3d,
    get_lookup_name,
    get_pickle_dir,
    get_spine_long,
    project_point_to_mesh_surface,
    union_meshes,
)
from spine_analysis.mesh.utils import v_f_to_mesh
from spine_analysis.shape_metric.utils import _calculate_junction_center, _get_junction_triangles


DATASET_BY_FOLDER = {dataset.folder: dataset for dataset in DATASETS}


def get_dataset_for_dir(project_root: Path, dataset_dir: Path):
    dataset_folder = dataset_dir.relative_to(project_root).as_posix()
    return DATASET_BY_FOLDER[dataset_folder]


def load_name_mapping(dataset_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    mapping_path = dataset_dir / "name_mapping.tsv"
    if not mapping_path.exists():
        return {}, {}

    new_to_original: dict[str, str] = {}
    original_to_new: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            original_name = Path(row["original_name"]).stem
            new_name = Path(row["new_name"]).stem
            new_to_original[new_name] = original_name
            original_to_new[original_name] = new_name
    return new_to_original, original_to_new


def resolve_spine_identifier(project_root: Path, spine_id: str) -> tuple[Path, str, str]:
    matching_folders = [
        folder for folder in DATASET_BY_FOLDER if spine_id.startswith(f"{folder}/")
    ]
    if not matching_folders:
        raise ValueError(
            "Use a known dataset path followed by a mesh name, for example "
            "'Mouse_basal/spine_0' or 'Mouse_Apical/Mouse_Apical_1/spine_0'."
        )

    dataset_folder = max(matching_folders, key=len)
    spine_name = spine_id[len(dataset_folder) + 1 :]
    dataset = DATASET_BY_FOLDER[dataset_folder]

    dataset_dir = project_root / dataset_folder
    new_to_original, original_to_new = load_name_mapping(dataset_dir)

    if (dataset_dir / f"{spine_name}.off").exists():
        current_name = spine_name
        lookup_name = get_lookup_name(dataset, spine_name, new_to_original)
        return dataset_dir, current_name, lookup_name

    if spine_name in original_to_new:
        current_name = original_to_new[spine_name]
        lookup_name = spine_name
        return dataset_dir, current_name, lookup_name

    raise FileNotFoundError(
        f"Could not resolve '{spine_id}' to a mesh in {dataset_dir}. "
        f"Expected current mesh name or original name from name_mapping.tsv."
    )


def get_spine_points(project_root: Path, spine_id: str) -> tuple[np.ndarray, Path, Path]:
    dataset_dir, current_name, lookup_name = resolve_spine_identifier(project_root, spine_id)
    dataset = get_dataset_for_dir(project_root, dataset_dir)
    longs_df = combine_pickles(get_pickle_dir(dataset, project_root), dataset.pkl_files)
    path_points = np.asarray(longs_df["longs"][lookup_name], dtype=float).copy()
    path_points[:, 2] *= Z_SCALE

    source_mesh = dataset_dir / f"{current_name}.off"
    repaired_mesh = dataset_dir / "z-corr" / f"{current_name}.off"
    return path_points, source_mesh, repaired_mesh


def get_saved_attachment_center(project_root: Path, spine_id: str) -> np.ndarray | None:
    dataset_dir, current_name, _ = resolve_spine_identifier(project_root, spine_id)
    centers_path = dataset_dir / "attachment_centers.pkl"
    if not centers_path.exists():
        return None
    centers = pd.read_pickle(centers_path)
    key = current_name
    if isinstance(centers, pd.Series):
        if key not in centers.index:
            return None
        return np.asarray(centers[key], dtype=float)
    if key not in centers:
        return None
    return np.asarray(centers[key], dtype=float)


def load_off_mesh(mesh_path: Path, apply_z_correction: bool = False) -> tuple[np.ndarray, np.ndarray]:
    with mesh_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header not in {"OFF", "COFF"}:
            raise ValueError(f"Unsupported OFF header in {mesh_path}: {header}")

        counts_line = handle.readline().strip()
        while counts_line.startswith("#") or not counts_line:
            counts_line = handle.readline().strip()
        num_vertices, num_faces, *_ = map(int, counts_line.split())

        vertices = []
        for _ in range(num_vertices):
            parts = handle.readline().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

        faces = []
        for _ in range(num_faces):
            parts = handle.readline().split()
            face_size = int(parts[0])
            if face_size != 3:
                continue
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    vertices = np.asarray(vertices, dtype=float)
    if apply_z_correction:
        vertices[:, 2] *= Z_SCALE
    return vertices, np.asarray(faces, dtype=int)


def _compute_junction_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    spine_mesh = v_f_to_mesh(vertices, faces)
    for i, vertex in enumerate(spine_mesh.vertices()):
        vertex.set_id(i)

    junction_triangles = _get_junction_triangles(spine_mesh)
    junction_faces = []
    for facet in junction_triangles:
        circulator = facet.facet_begin()
        begin = facet.facet_begin()
        face = []
        while circulator.hasNext():
            halfedge = circulator.next()
            face.append(halfedge.vertex().id())
            if circulator == begin or len(face) == 3:
                break
        if len(face) == 3:
            junction_faces.append(face)

    junction_center = _calculate_junction_center(spine_mesh)
    center = np.array([junction_center.x(), junction_center.y(), junction_center.z()], dtype=float)
    if len(junction_faces) == 0:
        return np.empty((0, 3), dtype=int), center
    return np.asarray(junction_faces, dtype=int), center


def _compute_projected_attachment_junction_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    attachment_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    projected_point, face_index, _ = project_point_to_mesh_surface(mesh, attachment_point)

    seed_face = faces[face_index]
    seed_vertices = set(int(v) for v in seed_face.tolist())
    junction_face_indices = []
    for idx, face in enumerate(faces):
        if seed_vertices.intersection(int(v) for v in face.tolist()):
            junction_face_indices.append(idx)

    if len(junction_face_indices) == 0:
        junction_faces = np.asarray([seed_face], dtype=int)
    else:
        junction_faces = faces[np.asarray(junction_face_indices, dtype=int)]

    return junction_faces, projected_point


def add_mesh_to_axis(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str,
    points: np.ndarray,
    show_junction: bool = False,
    show_projected_junction: bool = False,
    projected_attachment_point: np.ndarray | None = None,
) -> None:
    triangles = vertices[faces]
    collection = Poly3DCollection(
        triangles,
        alpha=0.55,
        facecolor="#8cb6d9",
        edgecolor="#4f6f8c",
        linewidths=0.1,
    )
    ax.add_collection3d(collection)
    if show_junction:
        junction_faces, junction_center = _compute_junction_geometry(vertices, faces)
        if len(junction_faces) > 0:
            junction_collection = Poly3DCollection(
                vertices[junction_faces],
                alpha=0.82,
                facecolor="#ffd166",
                edgecolor="#a06b00",
                linewidths=0.2,
            )
            ax.add_collection3d(junction_collection)
        _add_junction_center_matplotlib(ax, junction_center)
    if show_projected_junction:
        attachment_point = points[-1] if projected_attachment_point is None else projected_attachment_point
        junction_faces, junction_center = _compute_projected_attachment_junction_geometry(vertices, faces, attachment_point)
        projected_collection = Poly3DCollection(
            vertices[junction_faces],
            alpha=0.82,
            facecolor="#39ff14",
            edgecolor="#158f00",
            linewidths=0.2,
        )
        ax.add_collection3d(projected_collection)
        _add_projected_attachment_matplotlib(ax, points[-1], junction_center)
    _add_points_matplotlib(ax, points)
    ax.set_title(title)

    all_points = np.vstack([vertices, points])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius == 0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_repair_comparison(
    project_root: str | Path,
    spine_id: str,
    show_junction: bool = False,
    show_projected_junction: bool = False,
):
    project_root = Path(project_root)
    points, source_mesh_path, repaired_mesh_path = get_spine_points(project_root, spine_id)
    projected_attachment_center = get_saved_attachment_center(project_root, spine_id)
    junction_attachment_point = projected_attachment_center if projected_attachment_center is not None else points[-1]
    print("Attachment point (last skeleton point):", points[-1].tolist())
    print(
        "Projected attachment point:",
        None if projected_attachment_center is None else projected_attachment_center.tolist(),
    )

    if not source_mesh_path.exists():
        raise FileNotFoundError(f"Source mesh not found: {source_mesh_path}")
    if not repaired_mesh_path.exists():
        raise FileNotFoundError(
            f"Repaired mesh not found: {repaired_mesh_path}. "
            "Run the repair step first or choose a spine that was restored."
        )

    source_vertices, source_faces = load_off_mesh(source_mesh_path, apply_z_correction=True)
    repaired_vertices, repaired_faces = load_off_mesh(repaired_mesh_path, apply_z_correction=False)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    add_mesh_to_axis(
        ax1,
        source_vertices,
        source_faces,
        "Original mesh (z-corr) + PKL points",
        points,
        show_junction=show_junction,
        show_projected_junction=show_projected_junction,
        projected_attachment_point=junction_attachment_point,
    )
    add_mesh_to_axis(
        ax2,
        repaired_vertices,
        repaired_faces,
        "Saved z-corr mesh + PKL points",
        points,
        show_junction=show_junction,
        show_projected_junction=show_projected_junction,
        projected_attachment_point=junction_attachment_point,
    )
    add_overlay_to_axis(
        ax3,
        source_vertices,
        source_faces,
        repaired_vertices,
        repaired_faces,
        points,
        show_junction=show_junction,
        show_projected_junction=show_projected_junction,
        projected_attachment_point=junction_attachment_point,
    )

    fig.suptitle(spine_id)
    fig.tight_layout()
    return fig


def add_overlay_to_axis(
    ax,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    repaired_vertices: np.ndarray,
    repaired_faces: np.ndarray,
    points: np.ndarray,
    show_junction: bool = False,
    show_projected_junction: bool = False,
    projected_attachment_point: np.ndarray | None = None,
) -> None:
    source_triangles = source_vertices[source_faces]
    repaired_triangles = repaired_vertices[repaired_faces]

    repaired_collection = Poly3DCollection(
        repaired_triangles,
        alpha=0.22,
        facecolor="#f28e2b",
        edgecolor="#9a5a1a",
        linewidths=0.1,
    )
    ax.add_collection3d(repaired_collection)
    source_collection = Poly3DCollection(
        source_triangles,
        alpha=0.5,
        facecolor="#4c78a8",
        edgecolor="#35516e",
        linewidths=0.15,
    )
    ax.add_collection3d(source_collection)
    if show_junction:
        repaired_junction_faces, repaired_junction_center = _compute_junction_geometry(repaired_vertices, repaired_faces)
        source_junction_faces, source_junction_center = _compute_junction_geometry(source_vertices, source_faces)
        if len(repaired_junction_faces) > 0:
            repaired_junction = Poly3DCollection(
                repaired_vertices[repaired_junction_faces],
                alpha=0.8,
                facecolor="#ffd166",
                edgecolor="#a06b00",
                linewidths=0.2,
            )
            ax.add_collection3d(repaired_junction)
        if len(source_junction_faces) > 0:
            source_junction = Poly3DCollection(
                source_vertices[source_junction_faces],
                alpha=0.85,
                facecolor="#7bd389",
                edgecolor="#2f7f3f",
                linewidths=0.2,
            )
            ax.add_collection3d(source_junction)
        _add_junction_center_matplotlib(ax, repaired_junction_center)
        _add_junction_center_matplotlib(ax, source_junction_center)
    if show_projected_junction:
        attachment_point = points[-1] if projected_attachment_point is None else projected_attachment_point
        repaired_projected_faces, repaired_projected_center = _compute_projected_attachment_junction_geometry(
            repaired_vertices, repaired_faces, attachment_point
        )
        source_projected_faces, source_projected_center = _compute_projected_attachment_junction_geometry(
            source_vertices, source_faces, attachment_point
        )
        repaired_projected = Poly3DCollection(
            repaired_vertices[repaired_projected_faces],
            alpha=0.82,
            facecolor="#39ff14",
            edgecolor="#158f00",
            linewidths=0.2,
        )
        source_projected = Poly3DCollection(
            source_vertices[source_projected_faces],
            alpha=0.9,
            facecolor="#39ff14",
            edgecolor="#158f00",
            linewidths=0.2,
        )
        ax.add_collection3d(repaired_projected)
        ax.add_collection3d(source_projected)
        _add_projected_attachment_matplotlib(ax, points[-1], repaired_projected_center)
        _add_projected_attachment_matplotlib(ax, points[-1], source_projected_center)
    _add_points_matplotlib(ax, points)
    ax.set_title("Overlay: original + repaired")

    all_points = np.vstack([source_vertices, repaired_vertices, points])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius == 0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _add_points_matplotlib(ax, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    if len(points) > 1:
        mask = np.ones(len(points), dtype=bool)
        mask[-1] = False
        other_points = points[mask]
        if len(other_points) > 0:
            ax.scatter(
                other_points[:, 0],
                other_points[:, 1],
                other_points[:, 2],
                c="red",
                s=18,
                depthshade=False,
            )
        attachment_point = points[-1]
        ax.scatter(
            [attachment_point[0]],
            [attachment_point[1]],
            [attachment_point[2]],
            c="purple",
            s=32,
            depthshade=False,
        )
    else:
        ax.scatter(
            [points[-1, 0]],
            [points[-1, 1]],
            [points[-1, 2]],
            c="purple",
            s=32,
            depthshade=False,
        )


def _add_junction_center_matplotlib(ax, center: np.ndarray | None) -> None:
    if center is None:
        return
    ax.scatter(
        [center[0]],
        [center[1]],
        [center[2]],
        c="cyan",
        s=42,
        depthshade=False,
    )


def _add_projected_attachment_matplotlib(
    ax,
    original_attachment_point: np.ndarray,
    projected_center: np.ndarray | None,
) -> None:
    if projected_center is None:
        return
    ax.plot(
        [original_attachment_point[0], projected_center[0]],
        [original_attachment_point[1], projected_center[1]],
        [original_attachment_point[2], projected_center[2]],
        color="#ff8c00",
        linewidth=3.5,
    )
    ax.scatter(
        [original_attachment_point[0]],
        [original_attachment_point[1]],
        [original_attachment_point[2]],
        c="#ff1493",
        s=64,
        depthshade=False,
    )
    ax.scatter(
        [projected_center[0]],
        [projected_center[1]],
        [projected_center[2]],
        c="#ff8c00",
        s=80,
        depthshade=False,
    )
    ax.text(
        original_attachment_point[0],
        original_attachment_point[1],
        original_attachment_point[2],
        "src",
        color="#ff1493",
    )
    ax.text(
        projected_center[0],
        projected_center[1],
        projected_center[2],
        "proj",
        color="#ff8c00",
    )


def _point_traces_plotly(points: np.ndarray, showlegend: bool) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    if len(points) == 0:
        return traces

    if len(points) > 1:
        mask = np.ones(len(points), dtype=bool)
        mask[-1] = False
        other_points = points[mask]
        if len(other_points) > 0:
            traces.append(
                go.Scatter3d(
                    x=other_points[:, 0],
                    y=other_points[:, 1],
                    z=other_points[:, 2],
                    mode="markers",
                    marker={"size": 4, "color": "red"},
                    name="PKL points",
                    showlegend=showlegend,
                )
            )
        attachment_point = points[-1]
        traces.append(
            go.Scatter3d(
                x=[attachment_point[0]],
                y=[attachment_point[1]],
                z=[attachment_point[2]],
                mode="markers",
                marker={"size": 6, "color": "purple"},
                name="Attachment point",
                showlegend=showlegend,
            )
        )
    else:
        traces.append(
            go.Scatter3d(
                x=[points[-1, 0]],
                y=[points[-1, 1]],
                z=[points[-1, 2]],
                mode="markers",
                marker={"size": 6, "color": "purple"},
                name="Attachment point",
                showlegend=showlegend,
            )
        )

    return traces


def _junction_traces_plotly(
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_name: str,
    showlegend: bool,
    junction_color: str,
) -> list[go.BaseTraceType]:
    traces: list[go.BaseTraceType] = []
    junction_faces, junction_center = _compute_junction_geometry(vertices, faces)
    if len(junction_faces) > 0:
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=junction_faces[:, 0],
                j=junction_faces[:, 1],
                k=junction_faces[:, 2],
                color=junction_color,
                opacity=0.92,
                name=f"{mesh_name} junction",
                showscale=False,
                showlegend=showlegend,
            )
        )
    if junction_center is not None:
        traces.append(
            go.Scatter3d(
                x=[junction_center[0]],
                y=[junction_center[1]],
                z=[junction_center[2]],
                mode="markers",
                marker={"size": 7, "color": "cyan"},
                name=f"{mesh_name} junction center",
                showlegend=showlegend,
            )
        )
    return traces


def _projected_junction_traces_plotly(
    vertices: np.ndarray,
    faces: np.ndarray,
    attachment_point: np.ndarray,
    mesh_name: str,
    showlegend: bool,
    junction_color: str,
) -> list[go.BaseTraceType]:
    traces: list[go.BaseTraceType] = []
    junction_faces, junction_center = _compute_projected_attachment_junction_geometry(vertices, faces, attachment_point)
    traces.append(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=junction_faces[:, 0],
            j=junction_faces[:, 1],
            k=junction_faces[:, 2],
            color=junction_color,
            opacity=0.92,
            name=f"{mesh_name} projected junction",
            showscale=False,
            showlegend=showlegend,
        )
    )
    traces.append(
        go.Scatter3d(
            x=[attachment_point[0], junction_center[0]],
            y=[attachment_point[1], junction_center[1]],
            z=[attachment_point[2], junction_center[2]],
            mode="lines",
            line={"width": 10, "color": "#ff8c00"},
            name=f"{mesh_name} projection line",
            showlegend=showlegend,
        )
    )
    traces.append(
        go.Scatter3d(
            x=[attachment_point[0]],
            y=[attachment_point[1]],
            z=[attachment_point[2]],
            mode="markers+text",
            marker={"size": 7, "color": "#ff1493"},
            text=["src"],
            textposition="top center",
            name=f"{mesh_name} source attachment",
            showlegend=showlegend,
        )
    )
    traces.append(
        go.Scatter3d(
            x=[junction_center[0]],
            y=[junction_center[1]],
            z=[junction_center[2]],
            mode="markers+text",
            marker={"size": 8, "color": "#ff8c00"},
            text=["proj"],
            textposition="top center",
            name=f"{mesh_name} projected junction center",
            showlegend=showlegend,
        )
    )
    return traces


def _scene_axes(all_points: np.ndarray) -> dict:
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius == 0:
        radius = 1.0
    return {
        "xaxis": {"range": [center[0] - radius, center[0] + radius], "title": "X"},
        "yaxis": {"range": [center[1] - radius, center[1] + radius], "title": "Y"},
        "zaxis": {"range": [center[2] - radius, center[2] + radius], "title": "Z"},
        "aspectmode": "cube",
    }


def plot_repair_comparison_interactive(
    project_root: str | Path,
    spine_id: str,
    show_junction: bool = False,
    show_projected_junction: bool = False,
):
    project_root = Path(project_root)
    points, source_mesh_path, repaired_mesh_path = get_spine_points(project_root, spine_id)
    projected_attachment_center = get_saved_attachment_center(project_root, spine_id)
    junction_attachment_point = projected_attachment_center if projected_attachment_center is not None else points[-1]
    print("Attachment point (last skeleton point):", points[-1].tolist())
    print(
        "Projected attachment point:",
        None if projected_attachment_center is None else projected_attachment_center.tolist(),
    )

    if not source_mesh_path.exists():
        raise FileNotFoundError(f"Source mesh not found: {source_mesh_path}")
    if not repaired_mesh_path.exists():
        raise FileNotFoundError(
            f"Repaired mesh not found: {repaired_mesh_path}. "
            "Run the repair step first or choose a spine that was restored."
        )

    source_vertices, source_faces = load_off_mesh(source_mesh_path, apply_z_correction=True)
    repaired_vertices, repaired_faces = load_off_mesh(repaired_mesh_path, apply_z_correction=False)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Original mesh (z-corr) + PKL points",
            "Saved z-corr mesh + PKL points",
            "Overlay: original + repaired",
        ),
    )

    fig.add_trace(
        go.Mesh3d(
            x=source_vertices[:, 0],
            y=source_vertices[:, 1],
            z=source_vertices[:, 2],
            i=source_faces[:, 0],
            j=source_faces[:, 1],
            k=source_faces[:, 2],
            color="#8cb6d9",
            opacity=0.6,
            name="Original mesh",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    for trace in _point_traces_plotly(points, showlegend=True):
        fig.add_trace(trace, row=1, col=1)
    if show_junction:
        for trace in _junction_traces_plotly(source_vertices, source_faces, "Original", True, "#ffd166"):
            fig.add_trace(trace, row=1, col=1)
    if show_projected_junction:
        for trace in _projected_junction_traces_plotly(
            source_vertices, source_faces, junction_attachment_point, "Original", True, "#7bd389"
        ):
            fig.add_trace(trace, row=1, col=1)

    fig.add_trace(
        go.Mesh3d(
            x=repaired_vertices[:, 0],
            y=repaired_vertices[:, 1],
            z=repaired_vertices[:, 2],
            i=repaired_faces[:, 0],
            j=repaired_faces[:, 1],
            k=repaired_faces[:, 2],
            color="#8cb6d9",
            opacity=0.6,
            name="Repaired mesh",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    for trace in _point_traces_plotly(points, showlegend=False):
        fig.add_trace(trace, row=1, col=2)
    if show_junction:
        for trace in _junction_traces_plotly(repaired_vertices, repaired_faces, "Repaired", False, "#ffd166"):
            fig.add_trace(trace, row=1, col=2)
    if show_projected_junction:
        for trace in _projected_junction_traces_plotly(
            repaired_vertices, repaired_faces, junction_attachment_point, "Repaired", False, "#7bd389"
        ):
            fig.add_trace(trace, row=1, col=2)
    fig.add_trace(
        go.Mesh3d(
            x=repaired_vertices[:, 0],
            y=repaired_vertices[:, 1],
            z=repaired_vertices[:, 2],
            i=repaired_faces[:, 0],
            j=repaired_faces[:, 1],
            k=repaired_faces[:, 2],
            color="#f28e2b",
            opacity=0.22,
            name="Repaired mesh overlay",
            showscale=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Mesh3d(
            x=source_vertices[:, 0],
            y=source_vertices[:, 1],
            z=source_vertices[:, 2],
            i=source_faces[:, 0],
            j=source_faces[:, 1],
            k=source_faces[:, 2],
            color="#4c78a8",
            opacity=0.5,
            name="Original mesh overlay",
            showscale=False,
        ),
        row=1,
        col=3,
    )
    for trace in _point_traces_plotly(points, showlegend=False):
        trace.name = "PKL points overlay" if trace.name == "PKL points" else "Attachment point overlay"
        fig.add_trace(trace, row=1, col=3)
    if show_junction:
        for trace in _junction_traces_plotly(repaired_vertices, repaired_faces, "Repaired", False, "#ffd166"):
            fig.add_trace(trace, row=1, col=3)
        for trace in _junction_traces_plotly(source_vertices, source_faces, "Original", False, "#7bd389"):
            fig.add_trace(trace, row=1, col=3)
    if show_projected_junction:
        for trace in _projected_junction_traces_plotly(
            repaired_vertices, repaired_faces, junction_attachment_point, "Repaired", False, "#98df8a"
        ):
            fig.add_trace(trace, row=1, col=3)
        for trace in _projected_junction_traces_plotly(
            source_vertices, source_faces, junction_attachment_point, "Original", False, "#17becf"
        ):
            fig.add_trace(trace, row=1, col=3)

    fig.update_layout(
        title=spine_id,
        scene=_scene_axes(np.vstack([source_vertices, points])),
        scene2=_scene_axes(np.vstack([repaired_vertices, points])),
        scene3=_scene_axes(np.vstack([source_vertices, repaired_vertices, points])),
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h"},
    )
    return fig


def debug_spine_repair(project_root: str | Path, spine_id: str, project_attachment_to_surface: bool = False) -> dict:
    project_root = Path(project_root)
    dataset_dir, current_name, lookup_name = resolve_spine_identifier(project_root, spine_id)
    dataset = get_dataset_for_dir(project_root, dataset_dir)
    longs_df = combine_pickles(get_pickle_dir(dataset, project_root), dataset.pkl_files)

    mesh_path = dataset_dir / f"{current_name}.off"
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected a mesh, got {type(mesh).__name__}")

    mesh = mesh.copy()
    mesh.vertices[:, 2] *= Z_SCALE
    components_before = mesh.split(only_watertight=False)

    debug = {
        "spine_id": spine_id,
        "mesh_path": str(mesh_path),
        "lookup_name": lookup_name,
        "components_before": len(components_before),
        "components_after_first_step": None,
        "attachment_distance_before": None,
        "attachment_distance_after_first_step": None,
        "attachment_point": None,
        "projected_attachment_point_before": None,
        "projected_attachment_distance_before": None,
        "projected_attachment_point_after_first_step": None,
        "projected_attachment_distance_after_first_step": None,
        "first_step_performed": False,
        "first_step_error": None,
    }

    attachment_point = get_spine_long(longs_df, lookup_name)
    debug["attachment_point"] = attachment_point.tolist()
    debug["attachment_distance_before"] = float(np.min(np.linalg.norm(mesh.vertices - attachment_point, axis=1)))
    if project_attachment_to_surface:
        projected_before, _, projected_distance_before = project_point_to_mesh_surface(mesh, attachment_point)
        debug["projected_attachment_point_before"] = projected_before.tolist()
        debug["projected_attachment_distance_before"] = projected_distance_before

    if len(components_before) == 2:
        mesh_a, mesh_b = components_before
        centroid_a = np.mean(mesh_a.vertices, axis=0)
        centroid_b = np.mean(mesh_b.vertices, axis=0)
        bridge_a = mesh_a.vertices[np.argmin(np.linalg.norm(mesh_a.vertices - centroid_b, axis=1))]
        bridge_b = mesh_b.vertices[np.argmin(np.linalg.norm(mesh_b.vertices - centroid_a, axis=1))]
        bridge = convex_bridge(bridge_a, bridge_b, dataset.radius)
        debug["first_step_performed"] = True
        try:
            mesh = union_meshes([mesh, bridge])
        except Exception as exc:
            debug["first_step_error"] = str(exc)
            debug["components_after_first_step"] = len(components_before)
            return debug

    components_after = mesh.split(only_watertight=False)
    debug["components_after_first_step"] = len(components_after)
    debug["attachment_distance_after_first_step"] = float(
        np.min(np.linalg.norm(mesh.vertices - attachment_point, axis=1))
    )
    if project_attachment_to_surface:
        projected_after, _, projected_distance_after = project_point_to_mesh_surface(mesh, attachment_point)
        debug["projected_attachment_point_after_first_step"] = projected_after.tolist()
        debug["projected_attachment_distance_after_first_step"] = projected_distance_after
    return debug
