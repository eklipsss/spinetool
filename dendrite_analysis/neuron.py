from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

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
    output_path,
    reset_saved_data,
    save_all_dendr_metric_dict,
    set_output_dir,
)

try:
    from spine_analysis.shape_metric.utils import register_attachment_center, register_dendrite_skeleton
except Exception:
    register_attachment_center = None  # type: ignore
    register_dendrite_skeleton = None  # type: ignore


def _load_trimesh(path: Path) -> Optional[Any]:
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
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _safe_register_attachment_center(spine_mesh: Any, attachment_center: np.ndarray) -> None:
    if register_attachment_center is None:
        return
    try:
        register_attachment_center(spine_mesh, attachment_center)
    except Exception:
        pass


def _safe_register_dendrite_skeleton(dendrite_mesh: Any, skeleton: Any) -> None:
    if register_dendrite_skeleton is None or dendrite_mesh is None or skeleton is None:
        return
    try:
        register_dendrite_skeleton(dendrite_mesh, skeleton)
    except Exception:
        pass


def _safe_mesh_metrics(mesh: Optional[Any]) -> Dict[str, float]:
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
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        return 0.0
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _skeleton_length(skeleton: Any) -> float:
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


@dataclass
class Soma:
    name: str
    path: Path
    mesh_path: Path
    mesh: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_microns_folder(cls, neuron_path: Path) -> "Soma":
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
        neuron_path = Path(neuron_path)
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
        return [branch for limb in self.limbs for branch in limb.branches]

    @property
    def spine_points(self) -> Dict[str, np.ndarray]:
        points: Dict[str, np.ndarray] = {}
        for branch in self.branches:
            points.update(branch.spine_points)
        return points

    def build_network_graph(self, snap_threshold: float = 2.0) -> DendriticGraph:
        soma_point = self.soma.centroid
        graphs: List[DendriticGraph] = []
        for limb in self.limbs:
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
            for branch in limb.branches:
                if branch.skeleton is None:
                    continue
                try:
                    graphs.append(
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
        """Heuristic suggestion only; use metadata/manual override for final labels."""
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
        calculate_grouping_metrics: bool = True,
        calculate_cluster_metrics: bool = True,
        calculate_comprehensive_spatial_analysis: bool = True,
        spatial_morphology_permutation_count: int = 199,
        spatial_morphology_random_state: int = 42,
        calculate_graph_metrics: bool = True,
        save_structural_organization_vector: bool = True,
        save_summary_csv: bool = True,
    ) -> NeuronBranchAnalysisResult:
        set_output_dir(str(output_dir))
        if reset_output:
            reset_saved_data()

        dendrites: List[Dendrite] = []
        skipped: List[str] = []

        for branch in self.branches:
            branch_key = self._safe_branch_key(branch)
            dendrite_name = f"{self.name}/{branch_key}"
            if branch.mesh is None or not branch.spine_meshes:
                skipped.append(dendrite_name)
                continue

            dendrite = Dendrite(
                dendrite_name,
                dendrite_meshes={dendrite_name: branch.mesh},
                spine_meshes=dict(branch.spine_meshes),
            )
            dendrite.save_init_metrics()

            if calculate_grouping_metrics:
                dendrite.calculate_grouping_metrics()
                dendrite.save_grouping_metrics()

            if calculate_cluster_metrics:
                dendrite.calculate_cluster_metrics()
                dendrite.save_cluster_metrics()

            if calculate_comprehensive_spatial_analysis:
                dendrite.calculate_comprehensive_spatial_analysis(
                    permutation_count=spatial_morphology_permutation_count,
                    random_state=spatial_morphology_random_state,
                )
                dendrite.save_spatial_morphology_analysis()

            if calculate_graph_metrics:
                dendrite.graph_analysis()
                dendrite.save_graph_metrics()

            if save_structural_organization_vector:
                dendrite.save_structural_organization_vector()

            if save_summary_csv:
                dendrite.save_dendr_metrics_without_class_cluster()

            dendrites.append(dendrite)

        if save_summary_csv:
            pd.DataFrame(save_all_dendr_metric_dict).to_csv(output_path("all_dendr_metrics.csv"), index=False)

        return NeuronBranchAnalysisResult(dendrites=dendrites, skipped_branches=skipped)

    def run_full_analysis(
        self,
        mode: str = "both",
        network_output_dir: str | Path = "output_neuron_network_analysis",
        branch_output_dir: str | Path = "output_dendrite_metrics",
        reset_branch_output: bool = False,
        **kwargs: Any,
    ) -> NeuronFullAnalysisResult:
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
        max_distance_to_edge: float = 5.0,
        bin_size: float = 25.0,
        covariates: Sequence[str] = ("intercept", "distance_to_soma", "distance_to_soma_squared"),
        n_r_values: int = 20,
        r_max: Optional[float] = None,
        n_simulations: int = 99,
        k_correction: str = "geometric",
        save_outputs: bool = True,
    ) -> NeuronNetworkAnalysisResult:
        output_dir = Path(output_dir)
        graph = self.build_network_graph(snap_threshold=snap_threshold)
        spine_points = self.spine_points
        projected_spines, unassigned_ids = project_spines_to_graph(
            graph,
            spine_points,
            max_distance_to_edge=max_distance_to_edge,
        )

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

        result = NeuronNetworkAnalysisResult(
            graph=graph,
            projected_spines=projected_spines,
            unassigned_ids=unassigned_ids,
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
        branch_nodes = sum(1 for _, data in graph.G.nodes(data=True) if data.get("node_type") == "branch")
        terminal_nodes = sum(1 for node in graph.G.nodes() if graph.G.degree(node) == 1)
        total_spines = len(self.spine_points)
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
        value = str(value or "unknown").lower()
        return value if value in {"apical", "basal"} else "unknown"

    @classmethod
    def _edge_dendrite_type(cls, graph: DendriticGraph, u: int, v: int) -> str:
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
    ) -> Dict[str, KFunctionResult]:
        results: Dict[str, KFunctionResult] = {}
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
        limb_lengths = np.array([limb.length for limb in self.limbs if limb.length > 0], dtype=float)
        branch_lengths = np.array([branch.length for branch in self.branches if branch.length > 0], dtype=float)
        branch_spine_counts = np.array([len(branch.spine_points) for branch in self.branches], dtype=float)
        dendrite_types = [limb.dendrite_type for limb in self.limbs]
        branch_types = [branch.dendrite_type for branch in self.branches]

        branch_nodes = sum(1 for _, data in graph.G.nodes(data=True) if data.get("node_type") == "branch")
        terminal_nodes = sum(1 for node in graph.G.nodes() if graph.G.degree(node) == 1)
        total_spines = len(self.spine_points)
        type_metrics = self._calculate_dendrite_type_network_metrics(graph, [])

        row: Dict[str, Any] = {
            "neuron_id": self.name,
            "n_limbs": len(self.limbs),
            "n_branches": len(self.branches),
            "n_spines_total": total_spines,
            "n_spines_projected": projected_count,
            "n_spines_unassigned": unassigned_count,
            "projection_success_rate": projected_count / total_spines if total_spines else np.nan,
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
        output_dir = Path(output_dir)
        neuron_dir = output_dir / self.name
        neuron_dir.mkdir(parents=True, exist_ok=True)

        result.binned_intensity.to_csv(neuron_dir / "binned_intensity.csv", index=False)
        projected_spines_dataframe(result.projected_spines).to_csv(neuron_dir / "projected_spines.csv", index=False)
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
    root = Path(root_path)
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
