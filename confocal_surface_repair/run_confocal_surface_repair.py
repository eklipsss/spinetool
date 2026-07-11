from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import trimesh


Z_SCALE = 0.84
DISTANCE_THRESHOLD = 0.2
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_PROCESSES = 8
WORKER_LONGS_DF: pd.DataFrame | None = None
WORKER_PROJECT_ATTACHMENT_TO_SURFACE = False
PROJECT_ATTACHMENT_TO_SURFACE = False


@dataclass(frozen=True)
class DatasetConfig:
    folder: str
    repair_kind: str
    radius: float
    pkl_files: tuple[str, ...]
    renamed_meshes: bool = False
    pkl_folder: str | None = None
    lookup_prefix: str = ""
    lookup_suffix: str = ""


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig(
        folder="Mouse_Apical/Mouse_Apical_1",
        repair_kind="mouse",
        radius=0.14,
        pkl_files=("SpineID2longs.pkl",),
        pkl_folder="Mouse_Apical",
        lookup_prefix="apical1_",
        lookup_suffix="_R",
    ),
    DatasetConfig(
        folder="Mouse_Apical/Mouse_Apical_2",
        repair_kind="mouse",
        radius=0.14,
        pkl_files=("SpineID2longs.pkl",),
        pkl_folder="Mouse_Apical",
        lookup_prefix="apical2_",
        lookup_suffix="_R",
    ),
    DatasetConfig(
        folder="Mouse_Apical/Mouse_Apical_3",
        repair_kind="mouse",
        radius=0.14,
        pkl_files=("SpineID2longs.pkl",),
        pkl_folder="Mouse_Apical",
        lookup_prefix="apical3_",
        lookup_suffix="_R",
    ),
    DatasetConfig(
        folder="Mouse_basal",
        repair_kind="mouse",
        radius=0.14,
        pkl_files=(
            "SpineID2longs_Basal-Mouse-enviado-Repair.pkl",
            "SpineID2longs_Basal-Mouse2.pkl",
        ),
        renamed_meshes=True,
    ),
    DatasetConfig(
        folder="Human_age_85_basal_OFF",
        repair_kind="human",
        radius=0.17,
        pkl_files=("SpineID2longs_Human85-cing-basal.pkl",),
        renamed_meshes=True,
    ),
    DatasetConfig(
        folder="Human_age40_apical_OFF",
        repair_kind="human",
        radius=0.17,
        pkl_files=("SpineID2longs_Apical-Human.pkl",),
        renamed_meshes=True,
    ),
    DatasetConfig(
        folder="Human_age40_basal_OFF",
        repair_kind="human",
        radius=0.17,
        pkl_files=(
            "SpineID2longs_Basal-Human-Repair.pkl",
            "SpineID2longs_Human40-cing-basal-more20.pkl",
        ),
        renamed_meshes=True,
    ),
    DatasetConfig(
        folder="Human_age85_apical_OFF",
        repair_kind="human",
        radius=0.17,
        pkl_files=("SpineID2longs_Human85-cing-apical.pkl",),
        renamed_meshes=True,
    ),
)


def natural_keys(text: str) -> list[object]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def distance3d(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def generate_sphere_vertices(center: np.ndarray, rad: float, n: int = 9) -> np.ndarray:
    vertices = []
    for phi in np.linspace(0.0, np.pi, n):
        for theta in np.linspace(0.0, 2.0 * np.pi, n * 2, endpoint=False):
            x = center[0] + rad * np.sin(phi) * np.cos(theta)
            y = center[1] + rad * np.sin(phi) * np.sin(theta)
            z = center[2] + rad * np.cos(phi)
            vertices.append((x, y, z))
    return np.asarray(vertices, dtype=float)


def combine_pickles(resource_dir: Path, pickle_names: Iterable[str]) -> pd.DataFrame:
    frames = [pd.read_pickle(resource_dir / name) for name in pickle_names]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=0)


def get_pickle_dir(dataset: DatasetConfig, project_root: Path) -> Path:
    if dataset.pkl_folder is not None:
        return project_root / dataset.pkl_folder
    return RESOURCES_DIR


def get_lookup_name(dataset: DatasetConfig, mesh_name: str, name_mapping: dict[str, str]) -> str:
    lookup_name = name_mapping.get(mesh_name, mesh_name)
    if not dataset.lookup_prefix and not dataset.lookup_suffix:
        return lookup_name

    match = re.fullmatch(r"spine_(\d+)", lookup_name)
    if match is None:
        raise ValueError(
            f"Expected mesh name in format 'spine_N' for {dataset.folder}, got: {mesh_name}"
        )
    return f"{dataset.lookup_prefix}{match.group(1)}{dataset.lookup_suffix}"


def init_worker(resource_dir: str, pickle_names: tuple[str, ...], project_attachment_to_surface: bool) -> None:
    global WORKER_LONGS_DF, WORKER_PROJECT_ATTACHMENT_TO_SURFACE
    WORKER_LONGS_DF = combine_pickles(Path(resource_dir), pickle_names)
    WORKER_PROJECT_ATTACHMENT_TO_SURFACE = project_attachment_to_surface


def load_name_mapping(dataset_dir: Path) -> dict[str, str]:
    mapping_path = dataset_dir / "name_mapping.tsv"
    if not mapping_path.exists():
        return {}

    mapping: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            original_name = Path(row["original_name"]).stem
            new_name = Path(row["new_name"]).stem
            mapping[new_name] = original_name
    return mapping


def get_spine_long(longs_df: pd.DataFrame, spine_key: str) -> np.ndarray:
    longs_series = longs_df["longs"]
    long_path = longs_series[spine_key]
    long_point = np.asarray(long_path[-1], dtype=float).copy()
    long_point[2] *= Z_SCALE
    return long_point


def convex_bridge(point_a: np.ndarray, point_b: np.ndarray, radius: float) -> trimesh.Trimesh:
    verts_a = generate_sphere_vertices(point_a, rad=radius, n=9)
    verts_b = generate_sphere_vertices(point_b, rad=radius, n=9)
    hull = trimesh.PointCloud(np.vstack([verts_a, verts_b])).convex_hull
    vertices, faces = hull.vertices, hull.faces
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def project_point_to_mesh_surface(mesh: trimesh.Trimesh, point: np.ndarray) -> tuple[np.ndarray, int, float]:
    triangles = np.asarray(mesh.triangles, dtype=float)
    if len(triangles) == 0:
        raise RuntimeError("Mesh has no triangles for surface projection.")

    finite_mask = np.isfinite(triangles).all(axis=(1, 2))
    triangle_norms = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    nondegenerate_mask = triangle_norms > 1e-12
    valid_mask = finite_mask & nondegenerate_mask

    if np.any(valid_mask):
        valid_triangles = triangles[valid_mask]
        valid_indices = np.flatnonzero(valid_mask)
        repeated_points = np.repeat(point.reshape(1, 3), len(valid_triangles), axis=0)
        projected_points = trimesh.triangles.closest_point(valid_triangles, repeated_points)
        finite_projected_mask = np.isfinite(projected_points).all(axis=1)
        if np.any(finite_projected_mask):
            projected_points = projected_points[finite_projected_mask]
            candidate_indices = valid_indices[finite_projected_mask]
            candidate_points = repeated_points[finite_projected_mask]
            distances = np.linalg.norm(projected_points - candidate_points, axis=1)
            face_pos = int(np.argmin(distances))
            return projected_points[face_pos], int(candidate_indices[face_pos]), float(distances[face_pos])

    vertices = np.asarray(mesh.vertices, dtype=float)
    finite_vertex_mask = np.isfinite(vertices).all(axis=1)
    if np.any(finite_vertex_mask):
        valid_vertices = vertices[finite_vertex_mask]
        distances = np.linalg.norm(valid_vertices - point.reshape(1, 3), axis=1)
        vertex_pos = int(np.argmin(distances))
        closest_vertex = valid_vertices[vertex_pos]
        vertex_matches = np.all(np.isclose(vertices, closest_vertex, atol=1e-12), axis=1)
        vertex_indices = np.flatnonzero(vertex_matches)
        faces = np.asarray(mesh.faces, dtype=int)
        if len(vertex_indices) > 0 and len(faces) > 0:
            face_candidates = np.flatnonzero(np.any(np.isin(faces, vertex_indices), axis=1))
        else:
            face_candidates = np.array([], dtype=int)
        face_index = int(face_candidates[0]) if len(face_candidates) > 0 else 0
        return closest_vertex, face_index, float(distances[vertex_pos])

    raise RuntimeError("Projection failed: mesh has no finite triangles or vertices.")


def union_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    last_error: Exception | None = None
    for engine in ("blender", None):
        try:
            result = trimesh.boolean.union(meshes, engine=engine)
            if result is None:
                continue
            return result
        except BaseException as exc:  # trimesh may raise non-Exception subclasses
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"Boolean union failed: {last_error}") from last_error
    raise RuntimeError("Boolean union returned no mesh.")


def maybe_subdivide_mouse_mesh(mesh: trimesh.Trimesh, repair_kind: str) -> trimesh.Trimesh:
    if repair_kind != "mouse":
        return mesh
    vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def repair_mesh(
    mesh_path: Path,
    longs_df: pd.DataFrame,
    radius: float,
    repair_kind: str,
    lookup_name: str,
) -> tuple[trimesh.Trimesh | None, str]:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected a mesh, got {type(mesh).__name__}")

    mesh = mesh.copy()
    mesh.vertices[:, 2] *= Z_SCALE
    modified = False

    components = mesh.split(only_watertight=False)
    if len(components) > 2:
        return None, "skipped_gt_2_components"

    if len(components) == 2:
        mesh_a, mesh_b = components
        centroid_a = np.mean(mesh_a.vertices, axis=0)
        centroid_b = np.mean(mesh_b.vertices, axis=0)
        bridge_a = mesh_a.vertices[np.argmin(np.linalg.norm(mesh_a.vertices - centroid_b, axis=1))]
        bridge_b = mesh_b.vertices[np.argmin(np.linalg.norm(mesh_b.vertices - centroid_a, axis=1))]
        bridge = convex_bridge(bridge_a, bridge_b, radius)
        mesh = union_meshes([mesh, bridge])
        modified = True

    components = mesh.split(only_watertight=False)
    if len(components) == 1:
        long_point = get_spine_long(longs_df, lookup_name)
        distances = np.linalg.norm(mesh.vertices - long_point, axis=1)
        if float(np.min(distances)) > DISTANCE_THRESHOLD:
            closest_vertex = mesh.vertices[int(np.argmin(distances))]
            bridge = convex_bridge(long_point, closest_vertex, radius)
            mesh = union_meshes([mesh, bridge])
            modified = True

    if not modified:
        return mesh, "unchanged"

    mesh = maybe_subdivide_mouse_mesh(mesh, repair_kind)
    return mesh, "repaired"


def write_off(mesh: trimesh.Trimesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("COFF\n")
        handle.write(f"{len(vertices)} {len(faces)} 0\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def run_dataset(dataset: DatasetConfig, project_root: Path) -> dict[str, int]:
    dataset_dir = project_root / dataset.folder
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    pickle_dir = get_pickle_dir(dataset, project_root)
    for pickle_name in dataset.pkl_files:
        pickle_path = pickle_dir / pickle_name
        if not pickle_path.is_file():
            raise FileNotFoundError(f"Attachment points file not found: {pickle_path}")

    name_mapping = load_name_mapping(dataset_dir) if dataset.renamed_meshes else {}
    mesh_paths = sorted(dataset_dir.glob("*.off"), key=lambda path: natural_keys(path.name))

    stats = {
        "processed": 0,
        "repaired": 0,
        "unchanged": 0,
        "skipped_gt_2_components": 0,
        "missing_long": 0,
        "failed": 0,
    }
    skipped_files: list[str] = []
    failed_files: list[str] = []
    repaired_files: list[str] = []
    unchanged_files: list[str] = []
    projected_attachment_centers: dict[str, list[float]] = {}

    tasks = [
        (
            str(mesh_path),
            dataset.radius,
            dataset.repair_kind,
            get_lookup_name(dataset, mesh_path.stem, name_mapping),
            str(dataset_dir / "repair" / mesh_path.name),
            dataset.folder,
        )
        for mesh_path in mesh_paths
    ]

    stats["processed"] = len(tasks)
    processes = min(DEFAULT_PROCESSES, os.cpu_count() or DEFAULT_PROCESSES)
    chunk_size = max(1, int(np.ceil(len(tasks) / processes))) if tasks else 1
    print(
        f"[{dataset.folder}] start: total={len(tasks)} processes={processes} "
        f"mode={'parallel' if processes > 1 and len(tasks) > 1 else 'single'}"
    )

    if processes <= 1 or len(tasks) <= 1:
        init_worker(str(pickle_dir), dataset.pkl_files, PROJECT_ATTACHMENT_TO_SURFACE)
        for completed, task in enumerate(tasks, start=1):
            status, mesh_name, projected_attachment_center = process_mesh_task(task)
            stats[status] += 1
            if status == "skipped_gt_2_components":
                skipped_files.append(mesh_name)
            elif status == "repaired":
                repaired_files.append(mesh_name)
            elif status == "unchanged":
                unchanged_files.append(mesh_name)
            elif status == "failed":
                failed_files.append(mesh_name)
            if projected_attachment_center is not None:
                projected_attachment_centers[Path(mesh_name).stem] = projected_attachment_center.tolist()
            report_progress(dataset.folder, completed, len(tasks), stats)
    else:
        with Pool(
            processes=processes,
            initializer=init_worker,
            initargs=(str(pickle_dir), dataset.pkl_files, PROJECT_ATTACHMENT_TO_SURFACE),
        ) as pool:
            for completed, (status, mesh_name, projected_attachment_center) in enumerate(
                pool.imap_unordered(process_mesh_task, tasks, chunksize=chunk_size),
                start=1,
            ):
                stats[status] += 1
                if status == "skipped_gt_2_components":
                    skipped_files.append(mesh_name)
                elif status == "repaired":
                    repaired_files.append(mesh_name)
                elif status == "unchanged":
                    unchanged_files.append(mesh_name)
                elif status == "failed":
                    failed_files.append(mesh_name)
                if projected_attachment_center is not None:
                    projected_attachment_centers[Path(mesh_name).stem] = projected_attachment_center.tolist()
                report_progress(dataset.folder, completed, len(tasks), stats)

    if tasks:
        sys.stdout.write("\n")
        sys.stdout.flush()

    write_status_file(dataset_dir, "skipped_meshes.txt", skipped_files)
    write_status_file(dataset_dir, "failed_meshes.txt", failed_files)
    write_status_file(dataset_dir, "repaired_meshes.txt", repaired_files)
    write_status_file(dataset_dir, "unchanged_meshes.txt", unchanged_files)

    if PROJECT_ATTACHMENT_TO_SURFACE:
        write_attachment_centers(dataset_dir, projected_attachment_centers)

    return stats


def process_mesh_task(task: tuple[str, float, str, str, str, str]) -> tuple[str, str, np.ndarray | None]:
    global WORKER_LONGS_DF, WORKER_PROJECT_ATTACHMENT_TO_SURFACE

    mesh_path_str, radius, repair_kind, lookup_name, output_path_str, dataset_folder = task
    mesh_path = Path(mesh_path_str)
    output_path = Path(output_path_str)

    try:
        if WORKER_LONGS_DF is None:
            raise RuntimeError("Worker pickle data was not initialized.")
        repaired_mesh, status = repair_mesh(
            mesh_path=mesh_path,
            longs_df=WORKER_LONGS_DF,
            radius=radius,
            repair_kind=repair_kind,
            lookup_name=lookup_name,
        )
        projected_attachment_center = None
        if repaired_mesh is not None:
            z_corr_output_path = output_path.parent.parent / "z-corr" / output_path.name
            write_off(repaired_mesh, z_corr_output_path)
            if WORKER_PROJECT_ATTACHMENT_TO_SURFACE:
                attachment_point = get_spine_long(WORKER_LONGS_DF, lookup_name)
                projected_attachment_center, _, _ = project_point_to_mesh_surface(repaired_mesh, attachment_point)
        return status, mesh_path.name, projected_attachment_center
    except KeyError:
        print(f"[{dataset_folder}] missing skeleton points for {mesh_path.name} -> {lookup_name}")
        return "missing_long", mesh_path.name, None
    except Exception as exc:
        print(f"[{dataset_folder}] failed to repair {mesh_path.name}: {exc}")
        return "failed", mesh_path.name, None


def report_progress(dataset_folder: str, completed: int, total: int, stats: dict[str, int]) -> None:
    if total <= 0:
        return
    percent = (completed / total) * 100.0
    message = (
        f"\r[{dataset_folder}] progress {completed}/{total} ({percent:5.1f}%) | "
        f"repaired={stats['repaired']} unchanged={stats['unchanged']} "
        f"skipped={stats['skipped_gt_2_components']} missing={stats['missing_long']} "
        f"failed={stats['failed']}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def write_status_file(dataset_dir: Path, filename: str, mesh_names: list[str]) -> None:
    output_path = dataset_dir / filename
    mesh_names = sorted(mesh_names, key=natural_keys)
    with output_path.open("w", encoding="utf-8") as handle:
        for mesh_name in mesh_names:
            handle.write(f"{mesh_name}\n")


def write_attachment_centers(dataset_dir: Path, centers: dict[str, list[float]]) -> None:
    output_path = dataset_dir / "attachment_centers.pkl"
    center_series = pd.Series(centers, name="attachment_center", dtype=object)
    center_series.to_pickle(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair confocal spine meshes using the original human/mouse restoration logic."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing Mouse_* and Human_* folders.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[dataset.folder for dataset in DATASETS],
        help="Run only the selected dataset(s). Can be repeated.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help="Number of worker processes for per-mesh parallel repair.",
    )
    parser.add_argument(
        "--project-attachment-to-surface",
        action="store_true",
        help="After repair, save a per-spine attachment center projected onto the final mesh surface.",
    )
    return parser.parse_args()


def main() -> int:
    global DEFAULT_PROCESSES, PROJECT_ATTACHMENT_TO_SURFACE
    args = parse_args()
    DEFAULT_PROCESSES = max(1, args.processes)
    PROJECT_ATTACHMENT_TO_SURFACE = args.project_attachment_to_surface
    selected = [dataset for dataset in DATASETS if not args.dataset or dataset.folder in args.dataset]
    total_processed = 0
    total_repaired = 0
    total_unchanged = 0

    for dataset in selected:
        stats = run_dataset(dataset, args.project_root.resolve())
        total_processed += stats["processed"]
        total_repaired += stats["repaired"]
        total_unchanged += stats["unchanged"]
        print(
            f"{dataset.folder}: "
            f"processed={stats['processed']} "
            f"repaired={stats['repaired']} "
            f"unchanged={stats['unchanged']} "
            f"skipped_gt_2_components={stats['skipped_gt_2_components']} "
            f"missing_long={stats['missing_long']} "
            f"failed={stats['failed']}"
        )
        print(f"[{dataset.folder}] Проверено шипов: {stats['processed']}")
        print(f"[{dataset.folder}] Восстановлено шипов: {stats['repaired']}")
        print(f"[{dataset.folder}] Не требовали восстановления: {stats['unchanged']}")

    if selected:
        print("Итог по запуску:")
        print(f"Проверено шипов: {total_processed}")
        print(f"Восстановлено шипов: {total_repaired}")
        print(f"Не требовали восстановления: {total_unchanged}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
