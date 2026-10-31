"""Raw-mesh cleaning before sealing (spine_geometry.clean_input_mesh) and the per-dataset false-spine threshold lookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from src.neuron_model.spine_geometry import clean_input_mesh
from src.neuron_model.spine_preprocessing import SpinePreprocessingConfig


def _open_sphere_with_debris():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=100.0)
    faces = sphere.faces[sphere.vertices[sphere.faces].mean(axis=1)[:, 1] > -80.0]  # an open spine-like cup
    debris_v = np.array([[300.0, 0, 0], [310.0, 0, 0], [300.0, 10.0, 0]])
    vertices = np.vstack([sphere.vertices, debris_v])
    n = len(sphere.vertices)
    faces = np.vstack([faces, faces[:2][:, ::-1], [[n, n + 1, n + 2]]])  # 2 duplicated faces (reversed) + a loose triangle
    return trimesh.Trimesh(vertices, faces, process=False), len(sphere.vertices)


def test_clean_input_mesh_drops_duplicates_and_debris_but_keeps_vertex_indices():
    mesh, n_sphere_vertices = _open_sphere_with_debris()
    cleaned, info = clean_input_mesh(mesh, drop_duplicate_faces=True, min_component_face_fraction=0.05)
    assert info["input_duplicate_faces_dropped"] == 2 and info["input_components_dropped"] == 1 and info["input_faces_dropped"] == 3
    assert len(cleaned.vertices) == len(mesh.vertices)  # indices of the raw mesh stay valid
    assert np.array_equal(cleaned.vertices, mesh.vertices)
    edges, counts = np.unique(cleaned.edges_sorted, axis=0, return_counts=True)
    assert counts.max() <= 2  # no edge shared by more than two faces


def test_clean_input_mesh_is_identity_when_off_or_clean():
    mesh, _ = _open_sphere_with_debris()
    same, info = clean_input_mesh(mesh)
    assert same is mesh and info["input_faces_dropped"] == 0
    clean = trimesh.creation.icosphere(subdivisions=1)
    same, info = clean_input_mesh(clean, drop_duplicate_faces=True, min_component_face_fraction=0.05)
    assert same is clean and info["input_faces_dropped"] == 0


def test_false_spine_threshold_by_dataset_kind_and_minnie_hash_unchanged():
    base = SpinePreprocessingConfig(raw_data_root=Path("."), false_spine_threshold_nm_by_dataset=(("minnie65", 300.0), ("h01", 0.0))).normalized()
    assert base.false_spine_threshold("H01_Seg") == 0.0   # folder name of the segmentation -> kind h01
    assert base.false_spine_threshold("minnie65") == 300.0
    assert base.false_spine_threshold("other") == base.false_spine_threshold_nm
    cleaning = SpinePreprocessingConfig(raw_data_root=Path("."), input_drop_duplicate_faces=True, input_min_component_face_fraction=0.05).normalized()
    plain = SpinePreprocessingConfig(raw_data_root=Path(".")).normalized()
    assert cleaning.stage_hash("seal") != plain.stage_hash("seal")
    assert cleaning.stage_hash("orient") == plain.stage_hash("orient")
