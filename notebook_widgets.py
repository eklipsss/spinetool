from copy import deepcopy
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import trimesh
from ipywidgets import widgets
from sklearn.decomposition import PCA

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from typing import List, Tuple, Dict, Set, Iterable, Callable, Optional

from spine_analysis.clusterization.hierarchial_clusterizer import HierarchicalSpineClusterizer
from spine_analysis.mesh.utils import MeshDataset, LineSet, preprocess_meshes, _mesh_to_v_f, polylines_to_line_set, \
    rotate, write_off
from spine_analysis.mesh.vizualization import _add_line_set_to_viewer, _add_mesh_to_viewer_as_wireframe
from spine_analysis.shape_metric import OldChordDistributionSpineMetric, HistogramSpineMetric, FloatSpineMetric, \
    SpineMetric, JunctionCenterSpineMetric
from spine_analysis.shape_metric.junction_metric import (
    select_trimesh_junction_boundary_loop,
    trimesh_boundary_edge_loops,
)
from spine_analysis.shape_metric.io_metric import SpineMetricDataset
from spine_analysis.shape_metric.utils import calculate_metrics, _get_junction_triangles, get_facet_norm, \
    get_rotation_matrix, get_attachment_center, register_attachment_center, register_dendrite_skeleton
from spine_analysis.spine.grouping import SpineGrouping
from spine_segmentation import point_2_list, list_2_point, hash_point, \
    Segmentation, segmentation_by_distance, local_threshold_3d,\
    spines_to_segmentation, correct_segmentation, get_spine_meshes, apply_scale
import meshplot as mp
from IPython.display import display
from scipy.ndimage import measurements
from scipy.ndimage.measurements import label
from spine_analysis.clusterization import SpineClusterizer, KMeansSpineClusterizer, DBSCANSpineClusterizer, \
    KmeansKernelSpineClusterizer
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
from functools import cmp_to_key
from spine_analysis.clusterization.utils import ks_test
from CGAL.CGAL_Polygon_mesh_processing import Polylines, face_area
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization
from scipy.spatial.distance import euclidean
import csv
import tempfile #
import shutil #


Color = Tuple[float, float, float]

RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
WHITE = (1, 1, 1)
YELLOW = (1, 0.8, 0)
GRAY = (0.69, 0.69, 0.69)
DARK_GRAY = (0.30, 0.30, 0.30)
BLACK = (0.0, 0.0, 0.0)

V_F = Tuple[np.ndarray, np.ndarray]


def _mesh_to_v_f_any(mesh) -> V_F:
    if isinstance(mesh, trimesh.Trimesh):
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int)
    return _mesh_to_v_f(mesh)


class SpineMeshDataset:
    # spine name -> Polyhedron_3
    spine_meshes: MeshDataset
    # dendrite name -> Polyhedron_3
    dendrite_meshes: MeshDataset
    spine_to_dendrite: Dict[str, str]
    dendrite_to_spines: Dict[str, Set[str]]
    # spine name -> v f
    spine_v_f: Dict[str, V_F]
    # dendrite name -> v f
    dendrite_v_f: Dict[str, V_F]
    spine_attachment_centers: Dict[str, np.ndarray]
    dataset_root_path: Path

    def __init__(self, spine_meshes: MeshDataset = None, dendrite_meshes: MeshDataset = None,
                 spine_to_dendrite: Dict[str, str] = None,
                 spine_attachment_centers: Dict[str, np.ndarray] = None,
                 dataset_root_path: Path = None) -> None:
        if spine_meshes is None:
            spine_meshes = {}
        if dendrite_meshes is None:
            dendrite_meshes = {}
        if spine_to_dendrite is None:
            spine_to_dendrite = {}
        if spine_attachment_centers is None:
            spine_attachment_centers = {}
        if dataset_root_path is None:
            dataset_root_path = Path.cwd()
            
        # set fields
        self.spine_meshes = spine_meshes
        self.dendrite_meshes = dendrite_meshes
        self.spine_to_dendrite = spine_to_dendrite
        self.spine_attachment_centers = spine_attachment_centers
        self.dataset_root_path = Path(dataset_root_path)

        # generate mapping of dendrites to their spines
        self.dendrite_to_spines = {name: set() for name in dendrite_meshes.keys()}
        for (spine_name, dendrite_name) in spine_to_dendrite.items():
            self.dendrite_to_spines[dendrite_name].add(spine_name)

        # calculate 'meshplot' mesh representations
        self._calculate_v_f()

    @property
    def spine_names(self) -> Set[str]:
        return set(self.spine_meshes.keys())

    @property
    def dendrite_names(self) -> Set[str]:
        return set(self.dendrite_meshes.keys())

    def get_dendrite_mesh(self, spine_name: str) -> Polyhedron_3:
        return self.dendrite_meshes[self.spine_to_dendrite[spine_name]]

    def get_dendrite_v_f(self, spine_name: str) -> V_F:
        return self.dendrite_v_f[self.spine_to_dendrite[spine_name]]

    def apply_scale(self, scale: Color) -> None:
        def _apply_scale(mesh_dataset: MeshDataset) -> None:
            for (name, mesh) in mesh_dataset.items():
                mesh_dataset[name] = apply_scale(mesh, scale)
        _apply_scale(self.spine_meshes)
        _apply_scale(self.dendrite_meshes)
        self._calculate_v_f()

    @staticmethod
    def _orient_spine(mesh):
        target_normal = [0, -1, 0]

        print("##### ORIENT DEBUG START #####", flush=True)
        attachment_center = None
        try:
            from spine_analysis.shape_metric.utils import get_attachment_center
            attachment_center = get_attachment_center(mesh)
        except Exception:
            attachment_center = None
        print("attachment_center:", None if attachment_center is None else attachment_center.tolist(), flush=True)
        junction_triangles = _get_junction_triangles(mesh)
        print("junction_triangles_count:", len(junction_triangles), flush=True)

        mean_norm = np.zeros(3)
        valid_triangle_count = 0
        invalid_triangle_count = 0
        total_junction_area = 0.0
        for triangle in junction_triangles:
            cur_norm = get_facet_norm(triangle)
            cur_area = face_area(triangle, mesh)
            if np.isnan(cur_norm).any():
                invalid_triangle_count += 1
                print("triangle_norm: invalid", flush=True)
                continue
            valid_triangle_count += 1
            total_junction_area += cur_area
#             print(f"triangle_norm: {cur_norm.tolist()} area: {cur_area}", flush=True)
            mean_norm += cur_norm * cur_area

        mean_norm_length = np.linalg.norm(mean_norm)
        print("valid_triangle_count:", valid_triangle_count, flush=True)
        print("invalid_triangle_count:", invalid_triangle_count, flush=True)
        print("total_junction_area:", total_junction_area, flush=True)
        print("mean_norm_before_normalization:", mean_norm.tolist(), flush=True)
        print("mean_norm_length:", float(mean_norm_length), flush=True)

        if not np.isfinite(mean_norm_length) or mean_norm_length <= 1e-12:
            print("cant calculate norm: junction normal length is zero or non-finite", flush=True)
            print("##### ORIENT DEBUG END #####", flush=True)
            return mesh, False

        mean_norm = mean_norm / mean_norm_length

        if np.isnan(mean_norm).any():
            print("cant calculate norm: normalized mean norm contains NaN", flush=True)
            print("##### ORIENT DEBUG END #####", flush=True)
            return mesh, False

        print("mean_norm_normalized:", mean_norm.tolist(), flush=True)

        rotation_axis = np.cross(mean_norm, target_normal)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        print("rotation_axis_before_normalization:", rotation_axis.tolist(), flush=True)
        print("rotation_axis_length:", float(rotation_axis_length), flush=True)

        t = np.arcsin(rotation_axis_length)
        if np.dot(mean_norm, target_normal) < 0:
            t = np.pi - t
        if mean_norm[0] < target_normal[0] < 0:
            t = 2 * np.pi - t

        if not np.isfinite(rotation_axis_length) or rotation_axis_length <= 1e-12:
            print("rotation_axis is zero or non-finite, skip first rotation", flush=True)
            print("##### ORIENT DEBUG END #####", flush=True)
            return mesh, False

        rotation_axis = rotation_axis / rotation_axis_length
        print("rotation_axis_normalized:", rotation_axis.tolist(), flush=True)
        print("rotation_angle_t:", float(t), flush=True)
        r_matrix = get_rotation_matrix(t, rotation_axis)

        result_mesh = rotate(mesh, r_matrix)

        data = np.ndarray((mesh.size_of_vertices(), 2))
        for i, vertex in enumerate(mesh.vertices()):
            data[i, :] = [vertex.point().x(), vertex.point().z()]
        pca = PCA(n_components=1)
        pca.fit(data)
        t = np.arctan2(pca.components_[0][0], pca.components_[0][1])
        print("pca_component:", pca.components_[0].tolist(), flush=True)
        print("second_rotation_angle_t:", float(t), flush=True)

        r_matrix = get_rotation_matrix(t, target_normal)
        final_mesh = rotate(result_mesh, r_matrix)

        ##### ORIENT VISUALIZATION START #####
#         try:
#             import plotly.graph_objects as go
#             from plotly.subplots import make_subplots
#             from IPython.display import display
#             import plotly.io as pio

#             try:
#                 if pio.renderers.default in ("", None):
#                     pio.renderers.default = "notebook_connected"
#             except Exception:
#                 pass

#             original_vertices, original_faces = _mesh_to_v_f(mesh)
#             oriented_vertices, oriented_faces = _mesh_to_v_f(final_mesh)

#             fig = make_subplots(
#                 rows=1,
#                 cols=2,
#                 specs=[[{"type": "scene"}, {"type": "scene"}]],
#                 subplot_titles=("Original spine", "Oriented spine"),
#             )
#             fig.add_trace(
#                 go.Mesh3d(
#                     x=original_vertices[:, 0],
#                     y=original_vertices[:, 1],
#                     z=original_vertices[:, 2],
#                     i=original_faces[:, 0],
#                     j=original_faces[:, 1],
#                     k=original_faces[:, 2],
#                     color="#8cb6d9",
#                     opacity=0.7,
#                     name="original",
#                     showscale=False,
#                 ),
#                 row=1,
#                 col=1,
#             )
#             fig.add_trace(
#                 go.Mesh3d(
#                     x=oriented_vertices[:, 0],
#                     y=oriented_vertices[:, 1],
#                     z=oriented_vertices[:, 2],
#                     i=oriented_faces[:, 0],
#                     j=oriented_faces[:, 1],
#                     k=oriented_faces[:, 2],
#                     color="#f28e2b",
#                     opacity=0.7,
#                     name="oriented",
#                     showscale=False,
#                 ),
#                 row=1,
#                 col=2,
#             )

#             original_all = original_vertices
#             oriented_all = oriented_vertices

#             def _scene(points):
#                 mins = points.min(axis=0)
#                 maxs = points.max(axis=0)
#                 center = (mins + maxs) / 2.0
#                 radius = np.max(maxs - mins) / 2.0
#                 if radius == 0:
#                     radius = 1.0
#                 return {
#                     "xaxis": {"range": [center[0] - radius, center[0] + radius], "title": "X"},
#                     "yaxis": {"range": [center[1] - radius, center[1] + radius], "title": "Y"},
#                     "zaxis": {"range": [center[2] - radius, center[2] + radius], "title": "Z"},
#                     "aspectmode": "cube",
#                 }

#             fig.update_layout(
#                 title="Orientation debug: original vs oriented",
#                 scene=_scene(original_all),
#                 scene2=_scene(oriented_all),
#                 margin={"l": 0, "r": 0, "t": 40, "b": 0},
#             )
#             display(fig)
#             fig.show()
#         except Exception:
#             pass
        ##### ORIENT VISUALIZATION END #####

        print("##### ORIENT DEBUG END #####", flush=True)
        return final_mesh, True

    def orient_spines(self) -> None:
        """ Rotate the spikes so the polygons of the dendrite junction are maximally parallel to the xy plane,
        and the most wide axis is directed along x """
        orient_dir = self.dataset_root_path / "orient"
        orient_dir.mkdir(parents=True, exist_ok=True)
        original_names = []
        for (name, mesh) in self.spine_meshes.items():
            oriented_mesh, was_oriented = self._orient_spine(mesh)
            self.spine_meshes[name] = oriented_mesh
            if not was_oriented:
                original_names.append(Path(name).name)
            vertices, facets = _mesh_to_v_f(oriented_mesh)
            output_path = orient_dir / Path(name).name
            with output_path.open("w", encoding="utf-8") as fd:
                write_off(fd, vertices, facets)
        original_file = orient_dir / "original.txt"
        with original_file.open("w", encoding="utf-8") as fd:
            for spine_name in sorted(original_names):
                fd.write(f"{spine_name}\n")


    @staticmethod
    def capture_native_stderr(callable_, *args, **kwargs):
        fd = 2  # stderr
        saved = os.dup(fd)
        try:
            with tempfile.TemporaryFile(mode="w+b") as tmp:
                os.dup2(tmp.fileno(), fd)  # stderr -> tmp
                try:
                    result = callable_(*args, **kwargs)
                finally:
                    os.dup2(saved, fd)      # вернуть stderr
                tmp.seek(0)
                err = tmp.read().decode("utf-8", errors="replace")
            return result, err
        finally:
            os.close(saved)

    @staticmethod
    def _vector_to_array(vector) -> np.ndarray:
        return np.asarray([vector.x(), vector.y(), vector.z()], dtype=float)

    @staticmethod
    @staticmethod
    def _loop_center_radius(mesh: trimesh.Trimesh, loop: Dict[str, object]) -> Tuple[np.ndarray, float]:
        vertices = np.asarray(mesh.vertices, dtype=float)
        loop_vertex_indices = np.asarray(loop["vertex_indices"], dtype=int)
        loop_vertices = vertices[loop_vertex_indices]
        center = loop_vertices.mean(axis=0)
        distances = np.linalg.norm(loop_vertices - center, axis=1)
        radius = float(np.median(distances)) if len(distances) else 0.0
        return center, radius

    @staticmethod
    def _is_probably_dendrite_tube_fragment(mesh: trimesh.Trimesh) -> Tuple[bool, str]:
        loops = trimesh_boundary_edge_loops(mesh)
        closed_loops = [loop for loop in loops if loop["is_closed"]]
        if len(closed_loops) < 2:
            return False, ""

        ranked_loops = sorted(
            closed_loops,
            key=lambda loop: (int(loop["n_edges"]), len(loop["vertex_indices"])),
            reverse=True,
        )
        first, second = ranked_loops[:2]
        first_edges = int(first["n_edges"])
        second_edges = int(second["n_edges"])
        if first_edges < 8 or second_edges < 8:
            return False, ""

        edge_count_ratio = second_edges / max(first_edges, 1)
        if edge_count_ratio < 0.6:
            return False, ""

        first_center, first_radius = SpineMeshDataset._loop_center_radius(mesh, first)
        second_center, second_radius = SpineMeshDataset._loop_center_radius(mesh, second)
        max_radius = max(first_radius, second_radius)
        min_radius = min(first_radius, second_radius)
        if max_radius <= 1e-12 or min_radius / max_radius < 0.45:
            return False, ""

        vertices = np.asarray(mesh.vertices, dtype=float)
        centered = vertices - vertices.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            return False, ""

        principal_axis = vh[0]
        axial_positions = centered @ principal_axis
        axial_extent = float(axial_positions.max() - axial_positions.min()) if len(axial_positions) else 0.0
        radial_vectors = centered - np.outer(axial_positions, principal_axis)
        radial_extent = float(np.percentile(np.linalg.norm(radial_vectors, axis=1), 95)) if len(vertices) else 0.0
        elongation = axial_extent / max(radial_extent, 1e-12)
        if elongation < 2.0:
            return False, ""

        loop_center_distance = float(np.linalg.norm(first_center - second_center))
        if loop_center_distance < 1.5 * max_radius:
            return False, ""

        reason = (
            "likely dendrite tube fragment: "
            f"{len(closed_loops)} closed boundary loops; "
            f"two largest loops have edges=({first_edges}, {second_edges}), "
            f"edge_ratio={edge_count_ratio:.2f}, "
            f"radii=({first_radius:.3g}, {second_radius:.3g}), "
            f"elongation={elongation:.2f}, "
            f"loop_center_distance={loop_center_distance:.3g}"
        )
        return True, reason

    @staticmethod
    def _validate_trimesh_before_native_load(
        mesh_path: Path,
        require_faces: bool = True,
        reject_tube_like_spine: bool = False,
    ) -> Tuple[bool, str]:
        try:
            mesh = trimesh.load_mesh(mesh_path, process=False)
        except Exception as exc:
            return False, f"trimesh cannot read mesh: {exc}"

        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                return False, "trimesh scene has no geometry"
            try:
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            except Exception as exc:
                return False, f"trimesh cannot concatenate scene geometry: {exc}"

        if not isinstance(mesh, trimesh.Trimesh):
            return False, f"trimesh returned unsupported object: {type(mesh).__name__}"

        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            return False, f"invalid vertices shape/count: {vertices.shape}"
        if not np.isfinite(vertices).all():
            return False, "vertices contain NaN or Inf"
        if require_faces and (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0):
            return False, f"invalid triangular faces shape/count: {faces.shape}"
        if len(faces) > 0:
            if faces.min() < 0 or faces.max() >= len(vertices):
                return False, "faces reference missing vertices"
            triangles = vertices[faces]
            areas2 = np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            valid_area_mask = np.isfinite(areas2) & (areas2 > 1e-12)
            if require_faces and not valid_area_mask.any():
                return False, "all faces are degenerate"
        if reject_tube_like_spine:
            is_tube_fragment, tube_reason = SpineMeshDataset._is_probably_dendrite_tube_fragment(mesh)
            if is_tube_fragment:
                return False, tube_reason
        return True, ""

    @staticmethod
    def _load_trimesh_mesh(mesh_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"trimesh returned unsupported object: {type(mesh).__name__}")
        return mesh

    @staticmethod
    def _copy_invalid_mesh(mesh_path: Path, incorrect_dir: Path) -> None:
        incorrect_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(mesh_path, incorrect_dir / mesh_path.name)
        except Exception as copy_err:
            print(f"⚠️ Не удалось скопировать {mesh_path} в '{incorrect_dir}': {copy_err}", flush=True)

    @staticmethod
    def _load_attachment_centers(folder_path: Path, load_attachment_centers: bool) -> Dict[str, np.ndarray]:
        if not load_attachment_centers:
            return {}
        attachment_centers_path = folder_path.parent / "attachment_centers.pkl"
        if not attachment_centers_path.exists():
            return {}
        attachment_obj = pd.read_pickle(attachment_centers_path)
        if hasattr(attachment_obj, "to_dict"):
            attachment_obj = attachment_obj.to_dict()
        if not isinstance(attachment_obj, dict):
            return {}
        return {
            str(key): np.asarray(value, dtype=float)
            for key, value in attachment_obj.items()
        }

    @staticmethod
    def _resolve_attachment_center_for_debug(spine_mesh: Polyhedron_3) -> Tuple[np.ndarray, str]:
        registered_center = get_attachment_center(spine_mesh)
        if registered_center is not None:
            return np.asarray(registered_center, dtype=float), "attachment_centers.pkl"

        center_vec = JunctionCenterSpineMetric(spine_mesh)._value
        source = "JunctionCenterSpineMetric heuristic"
        if isinstance(spine_mesh, trimesh.Trimesh):
            loops = trimesh_boundary_edge_loops(spine_mesh)
            selected_loop = select_trimesh_junction_boundary_loop(spine_mesh)
            closed_count = sum(1 for loop in loops if loop["is_closed"])
            if selected_loop is None:
                source += " (no boundary loops; fallback)"
            else:
                source += (
                    f" ({len(loops)} boundary component(s), {closed_count} closed; "
                    f"selected n_edges={selected_loop['n_edges']}, closed={selected_loop['is_closed']})"
                )
        return SpineMeshDataset._vector_to_array(center_vec), source

    @staticmethod
    def _plot_attachment_debug(
        spine_name: str,
        spine_mesh: Polyhedron_3,
        dendrite_name: str,
        dendrite_mesh: Polyhedron_3,
        attachment_center: np.ndarray,
        source: str,
    ) -> None:
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            try:
                if pio.renderers.default in ("", None):
                    pio.renderers.default = "notebook_connected"
            except Exception:
                pass

            def add_mesh(fig, mesh, name: str, color: str, opacity: float) -> np.ndarray:
                vertices, faces = _mesh_to_v_f_any(mesh)
                fig.add_trace(
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=color,
                        opacity=opacity,
                        name=name,
                        showscale=False,
                    )
                )
                return vertices

            def add_boundary_loops(fig, mesh) -> None:
                if not isinstance(mesh, trimesh.Trimesh):
                    return
                vertices = np.asarray(mesh.vertices, dtype=float)
                loops = trimesh_boundary_edge_loops(mesh)
                selected_loop = select_trimesh_junction_boundary_loop(mesh)
                selected_edges = set()
                if selected_loop is not None:
                    selected_edges = {
                        tuple(sorted((int(edge[0]), int(edge[1]))))
                        for edge in np.asarray(selected_loop["edges"], dtype=int)
                    }
                colors = [
                    "#d62728",
                    "#2ca02c",
                    "#9467bd",
                    "#ff7f0e",
                    "#17becf",
                    "#e377c2",
                    "#bcbd22",
                    "#8c564b",
                ]
                for loop_index, loop in enumerate(loops):
                    edges = np.asarray(loop["edges"], dtype=int)
                    current_edges = {
                        tuple(sorted((int(edge[0]), int(edge[1]))))
                        for edge in edges
                    }
                    is_selected = bool(selected_edges) and current_edges == selected_edges
                    xs, ys, zs = [], [], []
                    for start, end in edges:
                        segment = vertices[[int(start), int(end)]]
                        xs.extend([segment[0, 0], segment[1, 0], None])
                        ys.extend([segment[0, 1], segment[1, 1], None])
                        zs.extend([segment[0, 2], segment[1, 2], None])
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="lines",
                            line={
                                "color": colors[loop_index % len(colors)],
                                "width": 8 if is_selected else 5,
                            },
                            name=(
                                f"{'selected ' if is_selected else ''}"
                                f"boundary #{loop_index + 1}: "
                                f"edges={loop['n_edges']}, closed={loop['is_closed']}"
                            ),
                        )
                    )

            def add_attachment_point(fig) -> None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[attachment_center[0]],
                        y=[attachment_center[1]],
                        z=[attachment_center[2]],
                        mode="markers+text",
                        marker={"size": 7, "color": "#d62728"},
                        text=["attachment"],
                        textposition="top center",
                        name="attachment point",
                    )
                )

            def scene_from_points(points: np.ndarray) -> Dict[str, object]:
                points = np.asarray(points, dtype=float)
                finite_points = points[np.isfinite(points).all(axis=1)]
                if len(finite_points) == 0:
                    finite_points = np.zeros((1, 3), dtype=float)
                mins = finite_points.min(axis=0)
                maxs = finite_points.max(axis=0)
                center = (mins + maxs) / 2.0
                radius = float(np.max(maxs - mins) / 2.0)
                if not np.isfinite(radius) or radius <= 0:
                    radius = 1.0
                return {
                    "xaxis": {"range": [center[0] - radius, center[0] + radius], "title": "X"},
                    "yaxis": {"range": [center[1] - radius, center[1] + radius], "title": "Y"},
                    "zaxis": {"range": [center[2] - radius, center[2] + radius], "title": "Z"},
                    "aspectmode": "cube",
                }

            spine_fig = go.Figure()
            spine_vertices = add_mesh(spine_fig, spine_mesh, "spine", "#8cb6d9", 0.65)
            add_boundary_loops(spine_fig, spine_mesh)
            add_attachment_point(spine_fig)
            spine_points = np.vstack([spine_vertices, attachment_center.reshape(1, 3)])
            spine_fig.update_layout(
                title=f"Attachment point on spine<br>{spine_name}<br>source: {source}",
                scene=scene_from_points(spine_points),
                margin={"l": 0, "r": 0, "t": 80, "b": 0},
            )
            display(spine_fig)

            combined_fig = go.Figure()
            dendrite_vertices = add_mesh(combined_fig, dendrite_mesh, "dendrite", "#bdbdbd", 0.28)
            spine_vertices = add_mesh(combined_fig, spine_mesh, "spine", "#8cb6d9", 0.72)
            add_boundary_loops(combined_fig, spine_mesh)
            add_attachment_point(combined_fig)
            combined_points = np.vstack([
                dendrite_vertices,
                spine_vertices,
                attachment_center.reshape(1, 3),
            ])
            combined_fig.update_layout(
                title=f"Attachment point with dendrite<br>{dendrite_name}<br>{spine_name}<br>source: {source}",
                scene=scene_from_points(combined_points),
                margin={"l": 0, "r": 0, "t": 95, "b": 0},
            )
            display(combined_fig)
        except Exception as exc:
            print(f"⚠️ Не удалось построить attachment debug-график для {spine_name}: {exc}", flush=True)

    @staticmethod
    def _maybe_show_attachment_debug(
        spine_name: str,
        spine_mesh: Polyhedron_3,
        dendrite_name: str,
        dendrite_mesh: Polyhedron_3,
        debug_attachment_points: bool,
        debug_index: int,
        attachment_debug_limit: Optional[int],
    ) -> None:
        if not debug_attachment_points:
            return
        if attachment_debug_limit is not None and debug_index >= attachment_debug_limit:
            return

        attachment_center, source = SpineMeshDataset._resolve_attachment_center_for_debug(spine_mesh)
        print(
            f"[attachment] {spine_name}: source={source}, point={attachment_center.tolist()}",
            flush=True,
        )
        if isinstance(spine_mesh, trimesh.Trimesh):
            loops = trimesh_boundary_edge_loops(spine_mesh)
            selected_loop = select_trimesh_junction_boundary_loop(spine_mesh)
            selected_edges = set()
            if selected_loop is not None:
                selected_edges = {
                    tuple(sorted((int(edge[0]), int(edge[1]))))
                    for edge in np.asarray(selected_loop["edges"], dtype=int)
                }
            if not loops:
                print(f"[boundary] {spine_name}: no boundary edges found", flush=True)
            for loop_index, loop in enumerate(loops):
                current_edges = {
                    tuple(sorted((int(edge[0]), int(edge[1]))))
                    for edge in np.asarray(loop["edges"], dtype=int)
                }
                is_selected = bool(selected_edges) and current_edges == selected_edges
                print(
                    f"[boundary] {spine_name}: component={loop_index + 1}, "
                    f"edges={loop['n_edges']}, vertices={len(loop['vertex_indices'])}, "
                    f"closed={loop['is_closed']}, selected={is_selected}",
                    flush=True,
                )
        SpineMeshDataset._plot_attachment_debug(
            spine_name=spine_name,
            spine_mesh=spine_mesh,
            dendrite_name=dendrite_name,
            dendrite_mesh=dendrite_mesh,
            attachment_center=attachment_center,
            source=source,
        )

    def _load_labid(
        self,
        folder_path: str,
        spine_file_pattern: str,
        load_attachment_centers: bool,
        debug_attachment_points: bool,
        attachment_debug_limit: Optional[int],
    ) -> "SpineMeshDataset":
        spine_meshes = {}
        dendrite_meshes = {}
        spine_to_dendrite = {}
        spine_attachment_centers = {}

        failed_spines = []

        path = Path(folder_path)
        spine_names = list(path.glob(spine_file_pattern))
        attachment_centers = self._load_attachment_centers(path, load_attachment_centers)

        # папка для некорректных шипов (создадим только если появятся ошибки)
        incorrect_dir = path / "incorrect spines"
        incorrect_dir_created = False

        debug_index = 0
        for spine_name in spine_names:
            spine_path = str(spine_name).replace('\\', '/')
            print("spine_name", spine_name, flush=True)

            is_valid, validation_error = self._validate_trimesh_before_native_load(
                spine_name,
                reject_tube_like_spine=True,
            )
            if not is_valid:
                print(f"\n❌ Mesh validation failed before CGAL for SPINE: {spine_name}\n{validation_error}\n", flush=True)
                failed_spines.append(spine_path)
                self._copy_invalid_mesh(spine_name, incorrect_dir)
                continue

            poly, err = self.capture_native_stderr(Polyhedron_3, spine_path)
            if "input error" in err or "cannot open file" in err:
                print(f"\n❌ CGAL error while reading SPINE: {spine_name}\n{err}\n", flush=True)
                failed_spines.append(spine_path)

                self._copy_invalid_mesh(spine_name, incorrect_dir)

                continue

            dendrite_path = os.path.join(spine_name.parent, "surface_mesh.off").replace('\\', '/')
            if dendrite_path not in dendrite_meshes:
                dendrite_path_obj = Path(dendrite_path)
                is_valid, validation_error = self._validate_trimesh_before_native_load(dendrite_path_obj)
                if not is_valid:
                    print(f"\n❌ Mesh validation failed before CGAL for DENDRITE: {dendrite_path_obj}\n{validation_error}\n", flush=True)
                    failed_spines.append(spine_path)
                    self._copy_invalid_mesh(spine_name, incorrect_dir)
                    continue
                dendrite_meshes[dendrite_path] = Polyhedron_3(dendrite_path)

            spine_meshes[spine_path] = poly
            if load_attachment_centers:
                spine_key = spine_name.stem
                if spine_key in attachment_centers:
                    attachment_center = np.asarray(attachment_centers[spine_key], dtype=float)
                    spine_attachment_centers[spine_path] = attachment_center
                    register_attachment_center(poly, attachment_center)
            spine_to_dendrite[spine_path] = dendrite_path

            self._maybe_show_attachment_debug(
                spine_name=spine_path,
                spine_mesh=poly,
                dendrite_name=dendrite_path,
                dendrite_mesh=dendrite_meshes[dendrite_path],
                debug_attachment_points=debug_attachment_points,
                debug_index=debug_index,
                attachment_debug_limit=attachment_debug_limit,
            )
            debug_index += 1

        if failed_spines:
            print("\n====== Итог: пропущенные шипы (ошибка чтения / validation / tube-like fragment) ======", flush=True)
            for p in failed_spines:
                print(p, flush=True)
            print(f"Всего: {len(failed_spines)}", flush=True)
            print("======================================================================================\n", flush=True)
        else:
            print("\n✅ Итог: ошибок чтения шипов не было.\n", flush=True)

        self.__init__(spine_meshes, dendrite_meshes, spine_to_dendrite, spine_attachment_centers, path)
        return self

    def _load_microns(
        self,
        folder_path: str,
        spine_file_pattern: str,
        load_attachment_centers: bool,
        debug_attachment_points: bool,
        attachment_debug_limit: Optional[int],
    ) -> "SpineMeshDataset":
        spine_meshes = {}
        dendrite_meshes = {}
        spine_to_dendrite = {}
        spine_attachment_centers = {}
        failed_spines = []

        branch_path = Path(folder_path)
        spines_dir = branch_path / "spines"
        dendrite_mesh_path = branch_path / "branch_mesh.off"
        skeleton_path = branch_path / "branch_skeleton.npy"

        if not spines_dir.exists():
            print(f"⏭️ MICrONS branch пропущен, нет папки spines: {branch_path}", flush=True)
            self.__init__({}, {}, {}, {}, branch_path)
            return self
        if not dendrite_mesh_path.exists():
            print(f"⏭️ MICrONS branch пропущен, нет branch_mesh.off: {branch_path}", flush=True)
            self.__init__({}, {}, {}, {}, branch_path)
            return self

        dendrite_path = str(dendrite_mesh_path).replace('\\', '/')
        is_valid, validation_error = self._validate_trimesh_before_native_load(dendrite_mesh_path)
        if not is_valid:
            print(f"\n❌ Mesh validation failed for MICrONS DENDRITE: {dendrite_mesh_path}\n{validation_error}\n", flush=True)
            self.__init__({}, {}, {}, {}, branch_path)
            return self
        try:
            dendrite_mesh = self._load_trimesh_mesh(dendrite_mesh_path)
        except Exception as exc:
            print(f"\n❌ trimesh error while reading MICrONS DENDRITE: {dendrite_mesh_path}\n{exc}\n", flush=True)
            self.__init__({}, {}, {}, {}, branch_path)
            return self
        dendrite_meshes[dendrite_path] = dendrite_mesh

        if skeleton_path.exists():
            try:
                skeleton = np.load(skeleton_path, allow_pickle=True)
                register_dendrite_skeleton(dendrite_mesh, skeleton)
                print(f"[microns] registered branch_skeleton.npy: {skeleton_path}", flush=True)
            except Exception as exc:
                print(f"⚠️ Не удалось загрузить branch_skeleton.npy для {branch_path}: {exc}", flush=True)

        attachment_centers = self._load_attachment_centers(spines_dir, load_attachment_centers)
        incorrect_dir = spines_dir / "incorrect spines"
        incorrect_dir_created = False

        spine_names = sorted(spines_dir.glob(spine_file_pattern))
        debug_index = 0
        for spine_name in spine_names:
            if not spine_name.is_file():
                continue
            spine_path = str(spine_name).replace('\\', '/')
            print("spine_name", spine_name, flush=True)

            is_valid, validation_error = self._validate_trimesh_before_native_load(
                spine_name,
                reject_tube_like_spine=True,
            )
            if not is_valid:
                print(f"\n❌ Mesh validation failed for MICrONS SPINE: {spine_name}\n{validation_error}\n", flush=True)
                failed_spines.append(spine_path)
                self._copy_invalid_mesh(spine_name, incorrect_dir)
                continue

            try:
                spine_mesh = self._load_trimesh_mesh(spine_name)
            except Exception as exc:
                print(f"\n❌ trimesh error while reading MICrONS SPINE: {spine_name}\n{exc}\n", flush=True)
                failed_spines.append(spine_path)
                self._copy_invalid_mesh(spine_name, incorrect_dir)
                continue

            spine_meshes[spine_path] = spine_mesh
            if load_attachment_centers:
                spine_key = spine_name.stem
                if spine_key in attachment_centers:
                    attachment_center = np.asarray(attachment_centers[spine_key], dtype=float)
                    spine_attachment_centers[spine_path] = attachment_center
                    register_attachment_center(spine_mesh, attachment_center)
            spine_to_dendrite[spine_path] = dendrite_path

            self._maybe_show_attachment_debug(
                spine_name=spine_path,
                spine_mesh=spine_mesh,
                dendrite_name=dendrite_path,
                dendrite_mesh=dendrite_mesh,
                debug_attachment_points=debug_attachment_points,
                debug_index=debug_index,
                attachment_debug_limit=attachment_debug_limit,
            )
            debug_index += 1

        if failed_spines:
            print("\n====== Итог: пропущенные шипы (ошибка чтения / validation / tube-like fragment) ======", flush=True)
            for p in failed_spines:
                print(p, flush=True)
            print(f"Всего: {len(failed_spines)}", flush=True)
            print("======================================================================================\n", flush=True)
        else:
            print("\n✅ Итог: ошибок чтения шипов не было.\n", flush=True)

        self.__init__(spine_meshes, dendrite_meshes, spine_to_dendrite, spine_attachment_centers, branch_path)
        return self

    def load(self, folder_path: str = "output",
             spine_file_pattern: str = "**/spine_*.off",
             load_attachment_centers: bool = False,
             dataset_format: str = "labid",
             debug_attachment_points: bool = False,
             attachment_debug_limit: Optional[int] = None) -> "SpineMeshDataset":
        dataset_format = dataset_format.lower()
        if dataset_format == "labid":
            return self._load_labid(
                folder_path=folder_path,
                spine_file_pattern=spine_file_pattern,
                load_attachment_centers=load_attachment_centers,
                debug_attachment_points=debug_attachment_points,
                attachment_debug_limit=attachment_debug_limit,
            )
        if dataset_format == "microns":
            return self._load_microns(
                folder_path=folder_path,
                spine_file_pattern=spine_file_pattern,
                load_attachment_centers=load_attachment_centers,
                debug_attachment_points=debug_attachment_points,
                attachment_debug_limit=attachment_debug_limit,
            )
        raise ValueError(f"Unknown dataset_format={dataset_format!r}; expected 'labid' or 'microns'")

    def _calculate_v_f(self) -> None:
        self.spine_v_f = preprocess_meshes(self.spine_meshes)
        self.dendrite_v_f = preprocess_meshes(self.dendrite_meshes)


def create_dir(dir_name: str) -> None:
    try:
        os.mkdir(dir_name)
    except OSError as _:
        pass


def remove_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def preprocess_meshes(spine_meshes: MeshDataset) -> Dict[str, V_F]:
    output = {}
    for (spine_name, spine_mesh) in spine_meshes.items():
        output[spine_name] = _mesh_to_v_f_any(spine_mesh)
    return output


def show_3d_mesh(mesh: Polyhedron_3, scale: Color = (1, 1, 1)) -> None:
    shown_mesh = apply_scale(mesh, scale)
    v, f = _mesh_to_v_f(shown_mesh)
    mp.plot(v, f)


def polylines_to_line_set(polylines: Polylines) -> LineSet:
    output = []
    for line in polylines:
        for i in range(len(line) - 1):
            output.append((line[i], line[i + 1]))
    return output


def show_polylines(polylines: Polylines, mesh: Polyhedron_3 = None) -> None:
    show_line_set(polylines_to_line_set(polylines), mesh)


def _add_line_set_to_viewer(viewer: mp.Viewer, lines: LineSet) -> None:
    viewer.add_lines(np.array([point_2_list(line[0]) for line in lines]),
                     np.array([point_2_list(line[1]) for line in lines]),
                     shading={"line_color": "red"})


def _add_mesh_to_viewer_as_wireframe(viewer: mp.Viewer, mesh_v_f: V_F) -> None:
    (v, f) = mesh_v_f
    starts = []
    ends = []
    for facet in f:
        starts.append(v[facet[0]])
        starts.append(v[facet[1]])
        starts.append(v[facet[2]])
        ends.append(v[facet[1]])
        ends.append(v[facet[2]])
        ends.append(v[facet[0]])
    viewer.add_lines(np.array(starts), np.array(ends),
                     shading={"line_color": "gray"})


def show_line_set(lines: LineSet, mesh: Polyhedron_3 = None) -> None:
    view = mp.Viewer({})
    _add_line_set_to_viewer(view, lines)
    if mesh:
        _add_mesh_to_viewer_as_wireframe(view, _mesh_to_v_f(mesh))
    display(view._renderer)


def _show_image(ax, image, mask=None, mask_opacity=0.5,
                cmap="gray", title=None):
    if mask is not None:
        indices = mask > 0
        mask = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], -1)
        image = np.stack([image, image, image], -1)
        image[indices] = image[indices] * (1 - mask_opacity) + mask[indices] * mask_opacity

    ax.imshow(image, norm=Normalize(0, 255), cmap=cmap)

    if title:
        ax.set_title(title)


def _show_cross_planes(ax, coord_1, coord_2, shape, color_1, color_2, border_color) -> None:
    # show plane 1
    ax.plot((coord_1, coord_1), (0, shape[0] - 1), color=color_1, lw=3)
    # show plane 2
    ax.plot((0, shape[1] - 1), (coord_2, coord_2), color=color_2, lw=3)
    # show border
    # horizontal
    ax.plot((0, shape[1] - 1), (0, 0), color=border_color, lw=3)
    ax.plot((0, shape[1] - 1), (shape[0] - 1, shape[0] - 1), color=border_color, lw=3)
    # vertical
    ax.plot((0, 0), (0, shape[0] - 1), color=border_color, lw=3)
    ax.plot((shape[1] - 1, shape[1] - 1), (0, shape[0] - 1), color=border_color, lw=3)


def make_viewer(width: int = 600, height: int = 600) -> mp.Viewer:
    return mp.Viewer({"width": width, "height": height})


def make_colors(v_f: V_F, color: Color) -> np.ndarray:
    num_of_v = v_f[0].shape[0]
    colors = np.ndarray((num_of_v, 3))
    for i in range(num_of_v):
        colors[i] = color
    return colors


def _segmentation_to_colors(vertices: np.ndarray,
                            segmentation: Segmentation,
                            dendrite_color: Color = GREEN,
                            spine_color: Color = RED) -> np.ndarray:
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        if hash_point(list_2_point(vertex)) in segmentation:
            colors[i] = spine_color
        else:
            colors[i] = dendrite_color
    return colors


def _grouping_to_colors(vertices: np.ndarray,
                        spine_meshes: MeshDataset,
                        grouping: SpineGrouping,
                        dendrite_color: Color = GREEN) -> np.ndarray:
    # generate segmentation for each group
    group_segmentations = []
    outlier_group = grouping.outlier_group
    group_colors = []
    for (label, group) in grouping.groups.items():
        group_segmentations.append(spines_to_segmentation([spine_meshes[name] for name in group]))
        group_colors.append(grouping.colors[label])
    group_segmentations.append(spines_to_segmentation([spine_meshes[name] for name in outlier_group]))
    group_colors.append(BLACK)

    # for each vertex check if belongs to group
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        hp = hash_point(list_2_point(vertex))
        colors[i] = dendrite_color
        for segmentation, color in zip(group_segmentations, group_colors):
            if hp in segmentation:
                colors[i] = color[:3]
                break
    return colors


class SpinePreview:
    widget: widgets.Widget
    spine_viewer: mp.Viewer
    dendrite_viewer: mp.Viewer

    spine_mesh: Polyhedron_3
    spine_name: str
    metrics: List[SpineMetric]

    _spine_color: Color

    _spine_colors: np.ndarray

    _dendrite_colors: np.ndarray

    _spine_v_f: V_F
    _dendrite_v_f: V_F

    _metrics_box: widgets.VBox

    _spine_mesh_id: int

    def __init__(self, spine_mesh: Polyhedron_3,
                 spine_v_f: V_F,
                 dendrite_v_f: V_F,
                 metrics: List[SpineMetric],
                 spine_name: str,
                 spine_color: Color = RED) -> None:
        self._spine_color = spine_color
        self._spine_mesh_id = 0
        self._dendrite_v_f = dendrite_v_f
        self._set_spine_mesh(spine_mesh, spine_v_f, metrics)
        self.spine_name = spine_name
        self.create_views()

    @property
    def spine_color(self) -> Color:
        return self._spine_color

    @spine_color.setter
    def spine_color(self, new_spine_color: Color) -> None:
        self._spine_color = new_spine_color
        self._make_colors()
        self.spine_viewer.update_object(colors=self._get_spine_colors())
        self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())

    def create_views(self) -> None:
        preview_panel = widgets.HBox(children=[self._make_dendrite_view(),
                                               self._make_spine_panel()],
                                     layout=widgets.Layout(align_items="flex-start"))
        self.widget = widgets.VBox([widgets.Label(self.spine_name), preview_panel])

    def _set_spine_mesh(self, spine_mesh: Polyhedron_3, spine_v_f, metrics: List[SpineMetric]) -> None:
        self.spine_mesh = spine_mesh
        self._spine_v_f = spine_v_f

        self._make_colors()

        if hasattr(self, "spine_viewer"):
            # self.spine_viewer.update_object(vertices=self._spine_v_f[0],
            #                                 faces=self._spine_v_f[1],
            #                                 colors=self._get_spine_colors())
            # self.spine_viewer.update_object(colors=self._get_spine_colors())
            self.spine_viewer.remove_object(self._spine_mesh_id)
            self._spine_mesh_id += 1
            self.spine_viewer.add_mesh(*self._spine_v_f, self._get_spine_colors())
        if hasattr(self, "dendrite_viewer"):
            self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())

        self.metrics = metrics
        if hasattr(self, "_metrics_box"):
            self._fill_metrics_box()

    def _make_colors(self) -> None:
        # dendrite view colors
        self._dendrite_colors = np.ndarray((len(self._dendrite_v_f[0]), 3))
        self._dendrite_colors[:] = \
            _segmentation_to_colors(self._dendrite_v_f[0],
                                    spines_to_segmentation([self.spine_mesh]),
                                    GRAY, self.spine_color)
        # spine view colors
        self._spine_colors = np.ndarray((self.spine_mesh.size_of_vertices(), 3))
        self._spine_colors[:] = self.spine_color

    def _get_dendrite_colors(self):
        return self._dendrite_colors

    def _get_spine_colors(self):
        return self._spine_colors

    def _make_dendrite_view(self) -> widgets.Widget:
        # title
        title = widgets.Label("Full View")

        # make mesh viewer
        if len(self._dendrite_v_f[0]) < 1:
            return widgets.VBox(children=[title])

        self.dendrite_viewer = make_viewer(400, 600)
        self.dendrite_viewer.add_mesh(*self._dendrite_v_f, self._get_dendrite_colors())

        # set layout
        self.dendrite_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        return widgets.VBox(children=[title, self.dendrite_viewer._renderer])

    def _make_spine_view(self) -> widgets.Widget:
        # make mesh viewer
        self.spine_viewer = make_viewer(200, 200)
        self.spine_viewer.add_mesh(*self._spine_v_f, self._get_spine_colors())

        # set layout
        self.spine_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        # title
        title = widgets.Label("Spine View")

        return widgets.VBox(children=[title, self.spine_viewer._renderer])

    def _fill_metrics_box(self) -> None:
        self._metrics_box.children = [widgets.VBox([widgets.Label(metric.name),
                                                    metric.show()],
                                                   layout=widgets.Layout(border="solid 1px"))
                                      for metric in self.metrics]

    def _make_metrics_panel(self) -> widgets.Widget:
        # TODO: figure out scrolling
        self._metrics_box = widgets.VBox([], layout=widgets.Layout())
        self._fill_metrics_box()

        return widgets.VBox(children=[widgets.Label("Metrics"), self._metrics_box])

    def _make_spine_panel(self) -> widgets.Widget:
        # convert spine mesh to meshplot format
        return widgets.VBox(children=[self._make_spine_view(),
                                      self._make_metrics_panel()],
                            layout=widgets.Layout(align_items="flex-start"))


class SelectableSpinePreview(SpinePreview):
    is_selected_checkbox: widgets.Checkbox
    is_selected: bool
    on_selected: Callable[["SelectableSpinePreview"], None] = None

    _unselected_spine_colors: np.ndarray
    _unselected_dendrite_colors: np.ndarray

    metrics: List[SpineMetric]
    _metric_names: List[str]
    _metric_params: List[Dict]

    _correction_slider: widgets.IntSlider

    _dendrite_mesh: Polyhedron_3

    _initial_segmentation: Segmentation

    def __init__(self, spine_mesh: Polyhedron_3,
                 spine_v_f: V_F,
                 dendrite_v_f: V_F,
                 dendrite_mesh: Polyhedron_3,
                 metric_names: List[str],
                 metric_params: List[Dict] = None) -> None:
        self._initial_segmentation = spines_to_segmentation([spine_mesh])
        self.is_selected = True
        self._make_is_selected()
        self._dendrite_mesh = dendrite_mesh
        self._make_correction_slider()
        self._metric_names = metric_names
        self._metric_params = metric_params
        self._metrics = calculate_metrics(spine_mesh, self._metric_names,
                                          self._metric_params)
        super().__init__(spine_mesh, spine_v_f, dendrite_v_f, self._metrics, "")

    def create_views(self) -> None:
        super().create_views()
        self.widget = widgets.VBox([self.is_selected_checkbox,
                                    self._correction_slider,
                                    self.widget])

    def _make_correction_slider(self) -> None:
        def observe_correction(change: Dict) -> None:
            if change["name"] == "value":
                self._correct_spine(change["new"])
        self._correction_slider = widgets.IntSlider(min=-6, max=6, value=0,
                                                    continuous_update=False,
                                                    description="Correction")
        self._correction_slider.observe(observe_correction)

    def _correct_spine(self, correction_value: int) -> None:
        new_segm = correct_segmentation(self._initial_segmentation,
                                        self._dendrite_mesh, correction_value)
        meshes = get_spine_meshes(self._dendrite_mesh, new_segm)
        # TODO: handle spine-splitting through correction slider
        if len(meshes) != 1:
            print(f"Oops, split this spine into {len(meshes)} spines.")
        self._set_spine_mesh(meshes[0], _mesh_to_v_f(meshes[0]),
                             calculate_metrics(meshes[0], self._metric_names, self._metric_params))

    def _make_is_selected(self) -> None:
        def update_is_selected(change: Dict) -> None:
            if change["name"] == "value":
                self.set_selected(change["new"])
        self.is_selected_checkbox = widgets.Checkbox(value=self.is_selected,
                                                     description="Valid spine")
        self.is_selected_checkbox.observe(update_is_selected)

    def _make_colors(self) -> None:
        super()._make_colors()
        # dendrite view colors
        self._unselected_dendrite_colors = np.ndarray(
            (len(self._dendrite_v_f[0]), 3))
        self._unselected_dendrite_colors[:] = \
            _segmentation_to_colors(self._dendrite_v_f[0],
                                    spines_to_segmentation([self.spine_mesh]),
                                    GRAY, DARK_GRAY)
        # spine view colors
        self._unselected_spine_colors = np.ndarray(
            (self.spine_mesh.size_of_vertices(), 3))
        self._unselected_spine_colors[:] = GRAY

    def _get_dendrite_colors(self):
        if self.is_selected:
            return self._dendrite_colors
        return self._unselected_dendrite_colors

    def _get_spine_colors(self):
        if self.is_selected:
            return self._spine_colors
        return self._unselected_spine_colors

    def set_selected(self, value: bool) -> None:
        self.is_selected = value
        self.spine_viewer.update_object(self._spine_mesh_id, colors=self._get_spine_colors())
        self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())
        if self.on_selected is not None:
            self.on_selected(self)


def _make_navigation_widget(slider: widgets.IntSlider, step=1) -> widgets.Widget:
    next_button = widgets.Button(description=">")
    prev_button = widgets.Button(description="<")

    def disable_buttons(_=None) -> None:
        next_button.disabled = slider.value >= slider.max
        prev_button.disabled = slider.value <= slider.min
        
    disable_buttons()
    slider.observe(disable_buttons)

    def next_callback(_: widgets.Button) -> None:
        slider.value += step
        disable_buttons()

    def prev_callback(_: widgets.Button) -> None:
        slider.value -= step
        disable_buttons()

    next_button.on_click(next_callback)
    prev_button.on_click(prev_callback)

    box = widgets.HBox([prev_button, next_button])
    return box
    

def select_spines_widget(spine_meshes: List[Polyhedron_3],
                         dendrite_mesh: Polyhedron_3,
                         metric_names: List[str],
                         metric_params: List[Dict] = None) -> widgets.Widget:
    # create selectable 3d previews for each spine
    dendrite_v_f: V_F = _mesh_to_v_f(dendrite_mesh)
    spine_previews = [SelectableSpinePreview(spine_mesh, _mesh_to_v_f(spine_mesh),
                                             dendrite_v_f, dendrite_mesh, metric_names,
                                             metric_params)
                      for spine_mesh in spine_meshes]

    # set callbacks to update selected spines on checkbox toggle
    selection = []

    def on_selected_callback(_: SelectableSpinePreview) -> None:
        selection.clear()
        selection.extend([(selected_preview.spine_mesh, selected_preview.metrics)
                          for selected_preview in spine_previews if selected_preview.is_selected])

    for preview in spine_previews:
        preview.on_selected = on_selected_callback

    # show previews
    def show_spine_by_index(index: int) -> List[Tuple[Polyhedron_3, List[SpineMetric]]]:
        # keeping old views caused bugs when switching between spines
        # this sacrifices saving camera position but oh well
        spine_previews[index].create_views()
        display(spine_previews[index].widget)

        # selected spine meshes and metrics
        selection.clear()
        selection.extend([(selected_preview.spine_mesh, selected_preview.metrics)
                          for selected_preview in spine_previews if selected_preview.is_selected])
        return selection

    slider = widgets.IntSlider(min=0, max=len(spine_meshes) - 1)
    navigation_buttons = _make_navigation_widget(slider)

    return widgets.VBox([navigation_buttons,
                         widgets.interactive(show_spine_by_index,
                                             index=slider)])


def interactive_segmentation(mesh: Polyhedron_3, correspondence,
                             reverse_correspondence,
                             skeleton_graph) -> widgets.Widget:
    vertices, facets = _mesh_to_v_f(mesh)

    slider = widgets.FloatLogSlider(min=-3.0, max=0.0, step=0.01, value=-1.0,
                                    continuous_update=False)
    correction_slider = widgets.IntSlider(min=-6, max=6, value=0,
                                          continuous_update=False)
    plot = mp.plot(vertices, facets)

    def do_segmentation(sensitivity=0.15, correction=0):
        segmentation = segmentation_by_distance(mesh, correspondence,
                                                reverse_correspondence,
                                                skeleton_graph, 1 - sensitivity)
        segmentation = correct_segmentation(segmentation, mesh, correction)
        plot.update_object(colors=_segmentation_to_colors(vertices, segmentation))

        return segmentation

    return widgets.interactive(do_segmentation, sensitivity=slider,
                               correction=correction_slider)


def show_segmented_mesh(mesh: Polyhedron_3, segmentation: Segmentation):
    vertices, facets = _mesh_to_v_f(mesh)
    colors = _segmentation_to_colors(vertices, segmentation)
    mp.plot(vertices, facets, c=colors)


def show_sliced_image(image: np.ndarray, x: int, y: int, z: int,
                      mask: np.ndarray = None, mask_opacity=0.5,
                      cmap="gray", title=""):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10),
                           gridspec_kw={
                               'width_ratios': [image.shape[2],
                                                image.shape[1]],
                               'height_ratios': [image.shape[2],
                                                 image.shape[0]]
                           })

    if title != "":
        fig.suptitle(title)

    data_x = image[:, x, :]
    data_y = image[y, :, :].transpose()
    data_z = image[:, :, z]

    mask_x = None
    mask_y = None
    mask_z = None
    if mask is not None:
        mask_x = mask[:, x, :]
        mask_y = mask[y, :, :].transpose()
        mask_z = mask[:, :, z]

    ax[0, 0].axis("off")
    _show_image(ax[1, 0], data_x, mask=mask_x, mask_opacity=mask_opacity, title=f"X = {x}", cmap=cmap)
    _show_image(ax[0, 1], data_y, mask=mask_y, mask_opacity=mask_opacity, title=f"Y = {y}", cmap=cmap)
    _show_image(ax[1, 1], data_z, mask=mask_z, mask_opacity=mask_opacity, title=f"Z = {z}", cmap=cmap)

    _show_cross_planes(ax[1, 0], z, y, data_x.shape, "blue", "green", "red")
    _show_cross_planes(ax[0, 1], x, z, data_y.shape, "red", "blue", "green")
    _show_cross_planes(ax[1, 1], x, y, data_z.shape, "red", "green", "blue")

    plt.tight_layout()
    plt.show()


def show_3d_image(data: np.ndarray, cmap="gray"):
    @widgets.interact(
        x=widgets.IntSlider(min=0, max=data.shape[1] - 1, continuous_update=False),
        y=widgets.IntSlider(min=0, max=data.shape[0] - 1, continuous_update=False),
        z=widgets.IntSlider(min=0, max=data.shape[2] - 1, continuous_update=False),
        layout=widgets.Layout(width='500px'))
    def display_slice(x, y, z):
        show_sliced_image(data, x, y, z, cmap=cmap)

    return display_slice


class Image3DRenderer:
    image: np.ndarray
    title: str

    _x: int
    _y: int
    _z: int

    def __init__(self, image: np.ndarray = np.zeros(0), title: str = "Title"):
        self._x = -1
        self._y = -1
        self._z = -1
        self.image = image
        self.title = title

    def show(self, cmap="gray"):
        shape = self.image.shape

        if self._x < 0:
            self._x = shape[1] // 2
        if self._y < 0:
            self._y = shape[0] // 2
        if self._z < 0:
            self._z = shape[2] // 2

        @widgets.interact(x=widgets.IntSlider(value=self._x, min=0,
                                              max=shape[1] - 1,
                                              continuous_update=False),
                          y=widgets.IntSlider(value=self._y, min=0,
                                              max=shape[0] - 1,
                                              continuous_update=False),
                          z=widgets.IntSlider(value=self._z, min=0,
                                              max=shape[2] - 1,
                                              continuous_update=False))
        def display_slice(x, y, z):
            self._x = x
            self._y = y
            self._z = z
            self._display_slice(x, y, z, cmap)

        return display_slice

    def _display_slice(self, x, y, z, cmap):
        show_sliced_image(self.image, x, y, z, cmap=cmap, title=self.title)


class MaskedImage3DRenderer(Image3DRenderer):
    mask: np.ndarray
    _mask_opacity: float

    def __init__(self, image: np.ndarray = np.zeros(0),
                 mask: np.ndarray = np.zeros(0), title: str = "Title"):
        super().__init__(image, title)
        self.mask = mask
        self._mask_opacity = 1

    def _display_slice(self, x, y, z, cmap):
        @widgets.interact(mask_opacity=widgets.FloatSlider(min=0, max=1,
                                                           value=self._mask_opacity,
                                                           step=0.1,
                                                           continuous_update=False))
        def display_slice_with_mask(mask_opacity):
            self._mask_opacity = mask_opacity
            show_sliced_image(self.image, x, y, z,
                              mask=self.mask, mask_opacity=mask_opacity,
                              cmap=cmap, title=self.title)


def interactive_binarization(image: np.ndarray) -> widgets.Widget:
    base_threshold_slider = widgets.IntSlider(min=0, max=255, value=127,
                                              continuous_update=False)
    weight_slider = widgets.IntSlider(min=0, max=100, value=5,
                                      continuous_update=False)
    block_size_slider = widgets.IntSlider(min=1, max=31, value=3, step=2,
                                          continuous_update=False)

    image_renderer = MaskedImage3DRenderer(title="Binarization Result")

    def show_binarization(base_threshold: int, weight: int, block_size: int) -> np.ndarray:
        result = local_threshold_3d(image, base_threshold=base_threshold,
                                    weight=weight / 100, block_size=block_size)
        image_renderer.image = image
        image_renderer.mask = result
        image_renderer.show()

        return result

    return widgets.interactive(show_binarization,
                               base_threshold=base_threshold_slider,
                               weight=weight_slider,
                               block_size=block_size_slider)


def select_connected_component_widget(binary_image: np.ndarray) -> widgets.Widget:
    # find connected components
    labels, num_of_components = measurements.label(binary_image)

    # sort labels by size
    unique, counts = np.unique(labels, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()
    unique.sort(key=lambda x: counts[x], reverse=True)
    counts.sort(reverse=True)

    # filter background and too small labels
    used_labels = []
    for i, count in enumerate(counts):
        if count >= 10 and unique[i] != 0:
            used_labels.append(unique[i])

    image_renderer = Image3DRenderer(title="Selected Connected Component")

    label_index_slider = widgets.IntSlider(min=0, max=len(used_labels) - 1,
                                           continuous_update=False)
    navigation_buttons = _make_navigation_widget(label_index_slider)
    display(navigation_buttons)

    def show_component(label_index: int) -> np.ndarray:
        lbl = used_labels[label_index]

        component = np.zeros_like(binary_image)
        component[labels == lbl] = 255

        preview = binary_image.copy()
        preview[preview > 0] = 64
        preview[labels == lbl] = 255

        image_renderer.image = preview
        image_renderer.show()

        return component

    return widgets.interactive(show_component, label_index=label_index_slider)


def grouping_in_3d_widget(grouping: SpineGrouping,
                          spine_dataset: SpineMeshDataset,
                          metrics_dataset: SpineMetricDataset,
                          distance_metric=None) -> widgets.Widget:
    # TODO: show only representative spines?
    spine_previews_by_cluster = {label: {} for label in grouping.group_labels}
    colors = grouping.colors

    def show_spine_by_group_label(label: str):
        def show_spine_by_name(spine_name: str):
            spine_previews_by_cluster[label] = {}
            if spine_name not in spine_previews_by_cluster[label]:
                preview = SpinePreview(spine_dataset.spine_meshes[spine_name],
                                       spine_dataset.spine_v_f[spine_name],
                                       spine_dataset.get_dendrite_v_f(
                                           spine_name),
                                       metrics_dataset.row(spine_name),
                                       spine_name, colors[label][:3])
                spine_previews_by_cluster[label][spine_name] = preview

            # keeping old views caused bugs when switching between spines
            # this sacrifices saving camera position but oh well
            spine_previews_by_cluster[label][spine_name].create_views()
            display(spine_previews_by_cluster[label][spine_name].widget)
        spine_name_dropdown = widgets.Dropdown(options=grouping.get_sorted_group(label),
                                               description="Spine:")
        display(widgets.interactive(show_spine_by_name, spine_name=spine_name_dropdown))

    group_label_dropdown = widgets.Dropdown(options=grouping.sorted_group_labels,
                                            description="Group:")

    return widgets.interactive(show_spine_by_group_label, label=group_label_dropdown)


# def new_clusterization_widget(clusterizer: SpineClusterizer,
#                               spine_meshes: Dict[str, V_F]) -> widgets.Widget:
#     def show_grid_by_cluster_index(cluster_index: int):
#         # extract representative cluster meshes
#         cluster = [(clusterizer.get_spine_reduced_coord(spine_name),
#                     spine_meshes[spine_name])
#                    for spine_name in clusterizer.get_representative_samples(cluster_index, 9)]
#         # sort by y
#         cluster.sort(key=lambda x: x[0][1])
#         # separate into rows
#         grid_size = int(np.ceil(np.sqrt(len(cluster))))
#         rows = [cluster[i:i + grid_size] for i in range(0, len(cluster), grid_size)]
#
#         # make rendering grid
#         grid_widget = widgets.VBox(
#             [widgets.HBox([make_viewer(*spine_mesh, None, 100, 100)._renderer
#                            for (_, spine_mesh) in row])
#              for row in rows])
#
#         display(widgets.HBox([clusterizer.show({cluster_index}),
#                               grid_widget]))
#
#     cluster_slider = widgets.IntSlider(min=0, max=max(clusterizer.num_of_clusters - 1, 0))
#     cluster_navigation_buttons = _make_navigation_widget(cluster_slider)
#
#     return widgets.VBox([cluster_navigation_buttons,
#                          widgets.interactive(show_grid_by_cluster_index,
#                                              cluster_index=cluster_slider)])


def new_new_clusterization_widget(grouping: SpineGrouping,
                                  spine_dataset: SpineMeshDataset) -> widgets.Widget:
    colors = {}

    def show_dendrite_by_name(dendrite_name: str):
        if dendrite_name not in colors:
            colors[dendrite_name] = _grouping_to_colors(
                spine_dataset.dendrite_v_f[dendrite_name][0],
                spine_dataset.spine_meshes,
                grouping)

        viewer = make_viewer()
        viewer.add_mesh(*spine_dataset.dendrite_v_f[dendrite_name], colors[dendrite_name])
        # for spine_name in spine_dataset.dendrite_to_spines[dendrite_name]:
        #     viewer.add_mesh(*spine_dataset.spine_v_f[spine_name])

        display(viewer._renderer)

    dendrite_name_dropdown = widgets.Dropdown(
        options=list(spine_dataset.dendrite_meshes.keys()),
        description="Dendrite:"
    )

    return widgets.interactive(show_dendrite_by_name, dendrite_name=dendrite_name_dropdown)


def representative_spines_widget(grouping: SpineGrouping,
                                 spine_dataset: SpineMeshDataset,
                                 metrics_dataset: SpineMetricDataset,
                                 num_of_samples: int = 3,
                                 distance: Callable = euclidean) -> widgets.Widget:
    representatives = grouping.get_representative_samples(metrics_dataset, num_of_samples, distance)
    colors = grouping.colors

    rows = []
    for label in grouping.sorted_group_labels:
        row = [widgets.Label(f"{label}:")]
        for name in representatives[label]:
            viewer = make_viewer(100, 100)
            v_f = spine_dataset.spine_v_f[name]
            viewer.add_mesh(*v_f, make_colors(v_f, colors[label][:3]))
            row.append(viewer._renderer)
        rows.append(widgets.HBox(row))
    return widgets.HBox([grouping.show(metrics_dataset), widgets.VBox(rows)])


def clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                 every_spine_metrics: SpineMetricDataset,
                                 spine_dataset: SpineMeshDataset,
                                 clusterizer_type,
                                 param_slider_type, param_name,
                                 param_min_value, param_max_value, param_step,
                                 static_params: Dict,
                                 score_function: Callable[[SpineClusterizer], float],
                                 dim_reduction: str = "pca",
                                 show_method: str = "tsne",
                                 classification: SpineGrouping = None,
                                 save_folder: str = "output/clusterization") -> widgets.Widget:
    # calculate score graph
    reduced_dim = 2 if dim_reduction else -1

    scores = {reduced_dim: []}

    # scores = {-1: [], 2: []}
    # pca_dim = spine_metrics.row_as_array(spine_metrics.spine_names[0]).size // 2
    # while pca_dim >= 2:
    #     scores[pca_dim] = []
    #     pca_dim //= 2

    num_of_steps = int(np.ceil((param_max_value - param_min_value + 1) / param_step))
    param_values = [np.clip(param_min_value + param_step * i,
                    param_min_value, param_max_value) for i in range(num_of_steps)]
    clusterized = {}
    for (dim, dim_scores) in scores.items():
        for value in param_values:
            scored_clusterizer = clusterizer_type(**{param_name: value}, **static_params, dim=dim,
                                                  reduction=dim_reduction)
            scored_clusterizer.set_show_method(show_method)
            scored_clusterizer.fit(spine_metrics)
            clusterized[value] = deepcopy(scored_clusterizer)
            dim_scores.append(score_function(scored_clusterizer))

    peak = np.nanargmax(scores[reduced_dim])

    def export_score_graph(_: widgets.Button):
        create_dir(save_folder)
        filename = f"{save_folder}/score_graph.csv"
        with open(filename, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow([param_name] + param_values)
            writer.writerow(["score"] + scores[reduced_dim])
        print(f"Saved score graph to '{filename}'.")

    export_score_button = widgets.Button(description="Export Score Graph")
    export_score_button.on_click(export_score_graph)
    display(export_score_button)

    # reg = LinearRegression().fit(np.reshape(param_values, (-1, 1)), scores[2])

    param_slider = param_slider_type(min=min(param_values),
                                     max=max(param_values),
                                     value=param_values[peak],
                                     step=param_step,
                                     continuous_update=False)

    def show_clusterization(param_value) -> None:
        #clusterizer = clusterizer_type(**{param_name: param_value}, **static_params, dim=reduced_dim,
        #                               reduction=dim_reduction)
        clusterizer = deepcopy(clusterized[param_value])
        clusterizer.set_show_method(show_method)
        #clusterizer.fit(spine_metrics)

        score_graph = widgets.Output()
        with score_graph:
            plt.axvline(x=param_value, color='g', linestyle='-')
            plt.axhline(y=0, color='r', linestyle='-')
            for (dim, dim_scores) in scores.items():
                if dim == -1:
                    plot_label = f"no {dim_reduction}"
                else:
                    plot_label = f"{dim}d with {dim_reduction} "
                plt.plot(param_values, dim_scores, label=plot_label)

            # plt.plot(param_values, reg.predict([[param] for param in param_values]))

            plt.title(clusterizer_type.__name__)
            plt.xlabel(param_name)
            plt.ylabel("Score")
            # plt.ylim([-1, 1])
            plt.legend()
            # plt.rcParams["figure.figsize"] = (10, 10)
            plt.show()

        # export clusterization button
        def export_clusterization(_: widgets.Button):
            create_dir(save_folder)
            save_path = f"{save_folder}/{param_name}={param_value}" \
                        f"_{clusterizer.reduction}={clusterizer.pca_dim}" \
                        f"_{clusterizer.grouping.num_of_groups}_clusters"
            create_dir(save_path)
            save_path += "/"

            clusterization_save_path = save_path + "clusterization.json"
            clusterizer.grouping.save(clusterization_save_path)
            print(f"Saved clusterization to \"{clusterization_save_path}\".")

            if dim_reduction:
                reduced_save_path = save_path + f"reduced_{dim_reduction}.csv"
                clusterizer.grouping.save_reduced(spine_metrics, reduced_save_path, dim_reduction)
                print(f"Saved reduced coordinates to \"{reduced_save_path}\".")

            classification_save_path = save_path + "classification.json"
            classification.save(classification_save_path)
            print(f"Saved classification to \"{classification_save_path}\".")

            score_values_path = save_path + "score_values.json"
            with open(score_values_path, 'w') as f:
                scores_info = scores.copy()
                scores_info['x_labels'] = np.array(param_values).tolist()
                json.dump(scores_info, f)
            print(f"Saved scores values to \"{score_values_path}\".")

            if dim_reduction:
                classification_save_reduced_path = save_path + f"classification_reduced_{dim_reduction}.csv"
                classification.save_reduced(spine_metrics, classification_save_reduced_path, dim_reduction)
                print(f"Saved classification reduced coordinates to \"{classification_save_reduced_path}\".")

            distribution_save_path = save_path + "metric_distributions.csv"
            clusterizer.grouping.save_metric_distribution(every_spine_metrics, distribution_save_path)
            print(f"Saved metric distributions to \"{distribution_save_path}\".")

            clust_over_class_save_path = save_path + "intersection_clust_over_class.csv"
            clusterizer.grouping.intersection_ratios(classification, False).save(clust_over_class_save_path)
            class_over_clust_save_path = save_path + "intersection_class_over_clust.csv"
            classification.intersection_ratios(clusterizer.grouping, False).save(class_over_clust_save_path)
            class_over_clust_norm_save_path = save_path + "intersection_class_over_clust_norm.csv"
            classification.intersection_ratios(clusterizer.grouping, True).save(class_over_clust_norm_save_path)
            print(f'Saved intersection ratios to "{clust_over_class_save_path}", '
                  f'"{class_over_clust_save_path}", "{class_over_clust_norm_save_path}".')

        export_button = widgets.Button(description="Export Clusterization")
        export_button.on_click(export_clusterization)

        # clusterization inspector
        inspector = inspect_grouping_widget(clusterizer.grouping, spine_dataset, spine_metrics,
                                            every_spine_metrics, classification)

        display(widgets.VBox([widgets.HBox([clusterizer.grouping.show(spine_metrics), score_graph]),
                              export_button, inspector]))

    clusterization_result = widgets.interactive(show_clusterization,
                                                param_value=param_slider)

    navigation_buttons = _make_navigation_widget(param_slider, param_step)

    return widgets.VBox([navigation_buttons, clusterization_result])


def kernel_k_means_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                                every_spine_metrics: SpineMetricDataset,
                                                spine_dataset: SpineMeshDataset,
                                                score_function: Callable[[SpineClusterizer], float],
                                                min_num_of_clusters: int = 2,
                                                max_num_of_clusters: int = 20,
                                                metric="euclidean",
                                                dim_reduction: str = "pca",
                                                show_method: str = "tsne",
                                                classification: SpineGrouping = None,
                                                filename_prefix: str = "") -> widgets.Widget:
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        KmeansKernelSpineClusterizer,
                                        widgets.IntSlider, "num_of_clusters",
                                        min_num_of_clusters, max_num_of_clusters,
                                        1, {"metric": metric}, score_function,
                                        dim_reduction, show_method, classification, f"{filename_prefix}_kernel_kmeans")


def kernel_hierarchical_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                                     every_spine_metrics: SpineMetricDataset,
                                                     spine_dataset: SpineMeshDataset,
                                                     score_function: Callable[[SpineClusterizer], float],
                                                     min_num_of_clusters: int = 2,
                                                     max_num_of_clusters: int = 20,
                                                     metric="euclidean",
                                                     dim_reduction: str = "pca",
                                                     show_method: str = "tsne",
                                                     classification: SpineGrouping = None,
                                                     filename_prefix: str = "") -> widgets.Widget:
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        HierarchicalSpineClusterizer,
                                        widgets.IntSlider, "num_of_clusters",
                                        min_num_of_clusters, max_num_of_clusters,
                                        1, {"metric": metric}, score_function,
                                        dim_reduction, show_method, classification, f"{filename_prefix}_kernel_hierarchical")


def k_means_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                         every_spine_metrics: SpineMetricDataset,
                                         spine_dataset: SpineMeshDataset,
                                         score_function: Callable[[SpineClusterizer], float],
                                         min_num_of_clusters: int = 2,
                                         max_num_of_clusters: int = 20,
                                         metric="euclidean",
                                         dim_reduction: str = "pca",
                                         show_method: str = "tsne",
                                         classification: SpineGrouping = None,
                                         save_folder: str = "output/clustering") -> widgets.Widget:
    min_num_of_clusters = max(min_num_of_clusters, 2)
    max_num_of_clusters = min(max_num_of_clusters, spine_metrics.num_of_spines)
    create_dir(save_folder)
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        KMeansSpineClusterizer,
                                        widgets.IntSlider, "num_of_clusters",
                                        min_num_of_clusters, max_num_of_clusters,
                                        1, {"metric": metric}, score_function,
                                        dim_reduction, show_method, classification, f"{save_folder}/kmeans")


def dbscan_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                        every_spine_metrics: SpineMetricDataset,
                                        spine_dataset: SpineMeshDataset,
                                        score_function: Callable[[SpineClusterizer], float],
                                        metric="euclidean",
                                        min_eps: float = 2,
                                        max_eps: float = 20,
                                        eps_step: float = 0.1,
                                        dim_reduction: str = "pca",
                                        show_method: str = "tsne",
                                        classification: SpineGrouping = None,
                                        save_folder: str = "output/clustering") -> widgets.Widget:
    create_dir(save_folder)
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        DBSCANSpineClusterizer,
                                        widgets.FloatSlider, "eps",
                                        min_eps, max_eps, eps_step,
                                        {"metric": metric}, score_function,
                                        dim_reduction, show_method, classification, f"{save_folder}/dbscan")


def grouping_metric_distribution_widget(grouping: SpineGrouping,
                                        metrics: SpineMetricDataset) -> widgets.Widget:
    metric_distributions = []
    for metric in metrics.row(list(metrics.spine_names)[0]):
        distribution_graph = widgets.Output()
        with distribution_graph:
            data = []
            #colors = grouping.colors
            for label, cluster in grouping.groups.items():
                cluster_metrics = metrics.get_spines_subset(cluster)
                metric_column = list(cluster_metrics.column(metric.name).values())
                if not issubclass(metric.__class__, FloatSpineMetric):
                    metric._show_distribution(metric_column)
                else:
                    data.append(metric.get_distribution(metric_column))
                # if issubclass(metric.__class__, HistogramSpineMetric):
                #     value = metric.get_distribution(metric_column)
                #     left_edges = [(int(label) - 1) + j / len(value) for j in range(len(value))]
                #     width = left_edges[1] - left_edges[0]
                #     plt.bar(left_edges, value, align='edge', width=width, color=colors[label])
            if issubclass(metric.__class__, FloatSpineMetric):
                plt.boxplot(data)
            plt.title(metric.name)
            plt.show()
        metric_distributions.append(distribution_graph)

    # slider = widgets.IntSlider(min=0, max=max(0, len(metric_distributions) - 1))
    # navigation_buttons = _make_navigation_widget(slider)
    # return widgets.VBox([navigation_buttons,
    #                      widgets.interactive(lambda index: display(metric_distributions[index]),
    #                                          index=slider)])

    return widgets.HBox(metric_distributions, layout=widgets.Layout(width='3000px'))


def color_to_hex(color: Tuple[float, float, float, float]) -> str:
    b = [int(c * 255) for c in color]
    c = (b[0] << 16) + (b[1] << 8) + b[2]
    return "#" + f"{c:06x}"


def clasters_spines_widget(meshes: SpineMeshDataset, clasterization: SpineGrouping) -> widgets.Widget:
    clusters = []
    spines = {}
    for label, group in clasterization.groups.items():
        spines[label] = list(group)

        def show_spine(label):
            def inter_fun(spine_index: int):
                print(label)
                name = spines[label][spine_index]
                spine_viewer = make_viewer(width=200, height=200)
                spine_viewer.add_mesh(*meshes.spine_v_f[name])
                spine_viewer._renderer.layout = widgets.Layout(border="solid 1px")
                print(name)
                display(spine_viewer._renderer)
                # display(SpinePreview(meshes.spine_meshes[name], meshes.spine_v_f[name],
                #                  meshes.get_dendrite_v_f(name), [],
                #                  name, clasterization.get_color(name)[:3]).widget)
                return clasterization
            return inter_fun

        spine_index_slider = widgets.IntSlider(max=max(0, len(spines[label]) - 1))
        navigation_buttons = _make_navigation_widget(spine_index_slider)

        spine_classification = widgets.interactive(show_spine(label), spine_index=spine_index_slider)
        clusters.append(widgets.HBox(children=[navigation_buttons, spine_classification]))
    return widgets.VBox(clusters)


def manual_classification_widget(meshes: SpineMeshDataset,
                                 metrics: SpineMetricDataset,
                                 classes: Iterable[str],
                                 initial_classification: SpineGrouping = None) -> widgets.Widget:
    if initial_classification is None:
        initial_classification = SpineGrouping(meshes.spine_names,
                                               {class_name: set() for class_name in classes},
                                               "Unclassified")
    result_grouping = initial_classification
    colors = result_grouping.colors

    spine_names_list = list(meshes.spine_names)
    unclassified = result_grouping.outlier_group
    spine_names_list.sort(key=lambda x: x not in unclassified)

    spine_name = [""]

    preview = []

    def class_button_callback(clicked_button: widgets.Button) -> None:
        class_name = clicked_button.description

        # remove spine from current class
        current_class = result_grouping.get_group(spine_name[0])
        if current_class != result_grouping.outliers_label:
            result_grouping.groups[current_class].remove(spine_name[0])

        # add spine to new class
        result_grouping.groups[class_name].add(spine_name[0])

        # update preview spine color (if last spine, needs to be updated!)
        preview[0].spine_color = colors[class_name][:3]

        # move to next spine
        spine_index_slider.value += 1

    # create classification buttons
    class_buttons = []
    for class_name in classes:
        button = widgets.Button(description=class_name)
        button.style.button_color = color_to_hex(colors[class_name])
        button.style.text_color = "#FFFFFF"
        button.on_click(class_button_callback)
        class_buttons.append(button)
    class_buttons_box = widgets.HBox(class_buttons)

    def show_spine(spine_index: int) -> SpineGrouping:
        spine_name[0] = spine_names_list[spine_index]
        name = spine_name[0]
        preview.clear()
        preview.append(SpinePreview(meshes.spine_meshes[name], meshes.spine_v_f[name],
                                    meshes.get_dendrite_v_f(name), metrics.row(name),
                                    name, result_grouping.get_color(name)[:3]))
        display(preview[0].widget)
        return result_grouping

    spine_index_slider = widgets.IntSlider(max=max(0, len(spine_names_list) - 1))
    spine_classification = widgets.interactive(show_spine, spine_index=spine_index_slider)

    navigation_buttons = _make_navigation_widget(spine_index_slider)

    return widgets.VBox([widgets.VBox([class_buttons_box, navigation_buttons]),
                         spine_classification])


def intersection_ratios_mean_distance(a: SpineGrouping, b: SpineGrouping, normalize: bool = True) -> float:
    intersections = a.intersection_ratios(b, normalize)

    a_labels = list(intersections.keys())
    b_labels = list(b.group_labels_with_outliers)

    mean_distance = 0
    num = 0
    for i in range(len(a_labels) - 1):
        for j in range(i + 1, len(a_labels)):
            row_i = np.array([intersections[a_labels[i]][b_label] for b_label in b_labels])
            row_j = np.array([intersections[a_labels[j]][b_label] for b_label in b_labels])
            mean_distance += np.linalg.norm(row_i - row_j)
            num += 1
    mean_distance /= num

    return mean_distance


def grouping_intersection_widget(a: SpineGrouping, b: SpineGrouping, normalize: bool = True) -> widgets.Widget:
    intersections = a.intersection_ratios(b, normalize)

    # calculate mean distance
    mean_distance = intersection_ratios_mean_distance(a, b, normalize)
    mean_distance_label = widgets.Label(f"Mean distance b/w distributions: {mean_distance:.2f}")

    # generate pie charts
    b_colors = b.colors_with_outliers

    pie_charts = {}
    for a_label in intersections.keys():
        pie_chart_widget = widgets.Output()
        with pie_chart_widget:
            a_intersection = intersections[a_label].copy()
            plt.bar(range(len(a_intersection)), list(a_intersection.values()),
                    tick_label=list(a_intersection.keys()),
                    color=[b_colors[label] for label in a_intersection.keys()])
            plt.show()
            # remove zero-length segments
            for key, value in list(a_intersection.items()):
                if value == 0:
                    del a_intersection[key]
            plt.pie(list(a_intersection.values()),
                    labels=list(a_intersection.keys()), autopct='%1.1f%%', pctdistance=0.85,
                    colors=[b_colors[label] for label in a_intersection.keys()],
                    normalize=True)
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.show()
        pie_charts[a_label] = pie_chart_widget

    all_charts_widget = widgets.HBox([widgets.VBox([widgets.Label(f"{a_label}:"), pie_chart_widget])
                                      for a_label, pie_chart_widget in pie_charts.items()])

    return widgets.VBox([mean_distance_label, all_charts_widget])

    # # generate table
    # for class_label in class_labels:
    #     class_label_widget = widgets.Label(str(class_label))
    #     class_label_widget.st = b_colors[class_label][:3]
    #     grid_items.append(class_label_widget)
    # for a_label in clustering.group_labels:
    #     grid_items.append(widgets.Label(str(a_label)))
    #     for class_label in class_labels:
    #         grid_items.append(widgets.Label(f"{intersections[a_label][class_label]:.2f}"))
    #
    # return widgets.GridBox(grid_items, layout=widgets.Layout(grid_template_columns=f"repeat({classification.num_of_groups + 1}, 100px)"))


def consensus_widget(groupings: List[SpineGrouping]) -> widgets.Widget:
    def compare(a, b) -> int:
        label_a = merged_grouping.get_group(a)
        votes_a = sum(1 for grouping in groupings
                      if grouping.get_group(a) == label_a)
        label_b = merged_grouping.get_group(b)
        votes_b = sum(1 for grouping in groupings
                      if grouping.get_group(b) == label_b)
        if votes_a < votes_b:
            return -1
        if votes_a > votes_b:
            return 1

        # sort by size if equal votes
        samples_a = merged_grouping.get_group_size(label_a)
        samples_b = merged_grouping.get_group_size(label_b)
        return np.sign(samples_a - samples_b)

    for grouping in groupings:
        grouping.outliers_label = "Unclassified"

    merged_grouping = SpineGrouping.merge(groupings, outliers_label="Unclassified")
    labels = list(merged_grouping.group_labels)
    labels.sort(key=lambda label: len(merged_grouping.groups[label]), reverse=True)

    sorted_spines = list(merged_grouping.samples)
    sorted_spines.sort(key=cmp_to_key(compare), reverse=True)

    colors = merged_grouping.colors_with_outliers

    # legend = []
    # for label in merged_grouping.groups:
    #     button = widgets.Button(description=label)
    #     button.style.button_color = color_to_hex(colors[label])
    #     button.style.text_color = "#FFFFFF"
    #     legend.append(button)
    # class_buttons_box = widgets.HBox(class_buttons)

    grid_items = [widgets.Widget()]
    for i, _ in enumerate(groupings):
        grid_items.append(widgets.Label(str(i + 1)))

    for i, spine_name in enumerate(sorted_spines):
        spine_name_label = widgets.Label(str(i + 1))
        spine_name_label.layout.width = "30px"
        grid_items.append(spine_name_label)
        for grouping in groupings:
            group_label = grouping.get_group(spine_name)
            color = color_to_hex(colors[group_label])
            rect = widgets.HTML(value=f'<svg width="30" height="15"><rect width="30" height="15" style="fill:{color};stroke:black;stroke-width:2"/></svg>')
            grid_items.append(rect)
            # button = widgets.Button(disabled=True)
            # button.style.button_color = color_to_hex(colors[group_label])
            # button.layout.width = "10px"
            # row.append(button)

    return widgets.GridBox(grid_items,
                           layout=widgets.Layout(grid_template_columns=f"repeat({len(groupings) + 1}, 30px)"))


def dendrite_segmentation_view_widget(spine_dataset: SpineMeshDataset) -> widgets.Widget:
    def show_dendrite_by_name(dendrite_name: str):
        dendrite_v_f = spine_dataset.dendrite_v_f[dendrite_name]
        dendrite_colors = np.ndarray((len(dendrite_v_f[0]), 3))
        dendrite_colors[:] = \
            _segmentation_to_colors(dendrite_v_f[0],
                                    spines_to_segmentation([spine_dataset.spine_meshes[spine] for spine in spine_dataset.dendrite_to_spines[dendrite_name]]))

        mesh_viewer = make_viewer(600, 600)
        mesh_viewer.add_mesh(*dendrite_v_f, dendrite_colors)

        display(widgets.HBox([mesh_viewer._renderer]))

    names = list(spine_dataset.dendrite_names)
    names.sort()
    dendrite_names_dropdown = widgets.Dropdown(options=names, description="Dendrite:")

    return widgets.interactive(show_dendrite_by_name, dendrite_name=dendrite_names_dropdown)


def spine_dataset_view_widget(spine_dataset: SpineMeshDataset,
                              metrics_dataset: SpineMetricDataset,
                              spine_color: Color = RED) -> widgets.Widget:
    def show_spine_by_name(spine_name: str):
        spine_mesh = spine_dataset.spine_meshes[spine_name]
        spine_v_f = spine_dataset.spine_v_f[spine_name]
        dendrite_v_f = spine_dataset.get_dendrite_v_f(spine_name)
        metrics = metrics_dataset.row(spine_name)
        spine_preview = SpinePreview(spine_mesh, spine_v_f, dendrite_v_f, metrics,
                                     spine_name, spine_color)
        display(spine_preview.widget)
    names = list(spine_dataset.spine_names)
    names.sort()
    spine_names_dropdown = widgets.Dropdown(options=names,
                                            description="Spine:")
    return widgets.interactive(show_spine_by_name, spine_name=spine_names_dropdown)


def spine_chords_widget(spine_dataset: SpineMeshDataset, scaled_spine_dataset: SpineMeshDataset,
                        dataset_path: str, num_of_chords: int = 3000,
                        num_of_bins: int = 100) -> widgets.Widget:
    chord_metrics = {}
    scaled_chord_metrics = {}

    def show_spine_by_name(spine_name: str):
        if spine_name in chord_metrics:
            chord_metric = chord_metrics[spine_name]
            scaled_chord_metric = scaled_chord_metrics[spine_name]
        else:
            chord_metric = OldChordDistributionSpineMetric(spine_dataset.spine_meshes[spine_name],
                                                           num_of_chords=num_of_chords, num_of_bins=num_of_bins)
            chord_metrics[spine_name] = [chord_metric]
            scaled_chord_metric = OldChordDistributionSpineMetric(scaled_spine_dataset.spine_meshes[spine_name],
                                                                  num_of_chords=num_of_chords, num_of_bins=num_of_bins)
            scaled_chord_metrics[spine_name] = [chord_metric]
        view = mp.Viewer({})
        _add_line_set_to_viewer(view, scaled_chord_metric.chords)
        _add_mesh_to_viewer_as_wireframe(view, scaled_spine_dataset.spine_v_f[spine_name])

        display(widgets.HBox([view._renderer, chord_metric.show()]))

    def export_callback(_: widgets.Button) -> None:
        save_path = f"{dataset_path}/chords_{num_of_chords}_chords_{num_of_bins}_bins.csv"
        SpineMetricDataset(chord_metrics).save_as_array(save_path)
        print(f"Saved histograms to \"{save_path}\"")

    export_button = widgets.Button(description="Export Histograms")
    export_button.on_click(export_callback)

    names = list(spine_dataset.spine_names)
    names.sort()
    spine_names_dropdown = widgets.Dropdown(options=names,
                                            description="Spine:")
    return widgets.VBox([widgets.interactive(show_spine_by_name, spine_name=spine_names_dropdown), export_button])


def view_skeleton_widget(scaled_spine_dataset: SpineMeshDataset) -> widgets.Widget:
    def show_dendrite_by_name(dendrite_name: str):
        dendrite_mesh = scaled_spine_dataset.dendrite_meshes[dendrite_name]
        dendrite_v_f = scaled_spine_dataset.dendrite_v_f[dendrite_name]

        # get skeleton
        skeleton_polylines = Polylines()
        correspondence_polylines = Polylines()
        surface_mesh_skeletonization(dendrite_mesh, skeleton_polylines,
                                     correspondence_polylines)

        # make viewers
        w = 600
        h = 600

        mesh_viewer = make_viewer(w, h)
        mesh_viewer.add_mesh(*scaled_spine_dataset.dendrite_v_f[dendrite_name])

        skeleton_viewer = make_viewer(w, h)
        skeleton_line_set = polylines_to_line_set(skeleton_polylines)
        _add_line_set_to_viewer(skeleton_viewer, skeleton_line_set)

        skeleton_mesh_viewer = make_viewer(w, h)
        _add_line_set_to_viewer(skeleton_mesh_viewer, polylines_to_line_set(skeleton_polylines))
        _add_mesh_to_viewer_as_wireframe(skeleton_mesh_viewer, dendrite_v_f)

        display(widgets.HBox([mesh_viewer._renderer, skeleton_viewer._renderer,
                              skeleton_mesh_viewer._renderer]))

    names = list(scaled_spine_dataset.dendrite_names)
    names.sort()
    dendrite_names_dropdown = widgets.Dropdown(options=names, description="Dendrite:")
    
    return widgets.interactive(show_dendrite_by_name, dendrite_name=dendrite_names_dropdown)


def inspect_grouping_widget(grouping: SpineGrouping, spine_dataset: SpineMeshDataset,
                            metrics_dataset: SpineMetricDataset, all_metrics_dataset: SpineMetricDataset,
                            reference_grouping: SpineGrouping) -> widgets.Widget:
    inspectors = {}
    inspector_names = ["Metric Distribution", "Reference Grouping Intersection",
                       "View Spines in 3D", "Representative Spines"]

    def generate_inspector(inspector_index) -> widgets.Widget:
        if inspector_index == 0:
            return grouping_metric_distribution_widget(grouping, all_metrics_dataset)
        elif inspector_index == 1:
            if reference_grouping is not None:
                intersection_widgets = [
                    reference_grouping.show(metrics_dataset),
                    widgets.Label("-- Clusters Over Classes --"),
                    grouping_intersection_widget(grouping, reference_grouping, False),
                    widgets.Label("-- Classes Over Clusters, Non-normalized --"),
                    grouping_intersection_widget(reference_grouping, grouping, False),
                    widgets.Label("-- Clusters Over Classes, Normalized --"),
                    grouping_intersection_widget(reference_grouping, grouping, True)
                ]
            else:
                intersection_widgets = [widgets.Label("No reference grouping provided!")]
            return widgets.VBox(intersection_widgets)
        elif inspector_index == 2:
            return grouping_in_3d_widget(grouping, spine_dataset, metrics_dataset)
        elif inspector_index == 3:
            return representative_spines_widget(grouping, spine_dataset, metrics_dataset)

    def show_inspector(inspector_index):
        if inspector_index not in inspectors:
            inspectors[inspector_index] = generate_inspector(inspector_index)
        display(inspectors[inspector_index])

    inspector_dropdown = widgets.Dropdown(
        options=[(name, i) for i, name in enumerate(inspector_names)])

    return widgets.interactive(show_inspector, inspector_index=inspector_dropdown)


def inspect_saved_groupings_widget(folder_path: str, spine_dataset: SpineMeshDataset,
                                   all_metrics_dataset: SpineMetricDataset,
                                   chord_metric_dataset: SpineMetricDataset,
                                   classic_metrics_dataset: SpineMetricDataset,
                                   reference_grouping: SpineGrouping,
                                   grouping_file_pattern: str = "**/*.json") -> widgets.Widget:
    path = Path(folder_path)

    grouping_paths = [str(grouping_path) for grouping_path in path.glob(grouping_file_pattern)]
    grouping_paths.sort()
    groupings = {grouping_path: SpineGrouping().load(grouping_path)
                 for grouping_path in grouping_paths}

    metrics = [all_metrics_dataset, chord_metric_dataset, classic_metrics_dataset]
    metric_names = ["Combined", "Chord Length Histogram", "Classic Metrics"]

    def inspect_grouping(grouping_path: str):
        if grouping_path is None:
            print("No groupings found!")
            return
        grouping = groupings[grouping_path]
        header = widgets.HBox([widgets.VBox([widgets.Label(f"{name}:"),
                                             grouping.show(dataset)])
                               for name, dataset in zip(metric_names, metrics)])
        display(header)

        def inspect_grouping_by_metric(metrics_set: int):
            display(inspect_grouping_widget(grouping, spine_dataset,
                                            metrics[metrics_set], all_metrics_dataset,
                                            reference_grouping))

        metrics_dropdown = widgets.Dropdown(options=[(name, i) for i, name in enumerate(metric_names)])
        display(widgets.interactive(inspect_grouping_by_metric, metrics_set=metrics_dropdown))

    groupings_dropdown = widgets.Dropdown(options=grouping_paths)

    return widgets.interactive(inspect_grouping, grouping_path=groupings_dropdown)
