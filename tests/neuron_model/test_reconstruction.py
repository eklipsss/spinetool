import numpy as np
import pytest
import trimesh

from conftest import poisson_backend_works

from src.neuron_model.spine_generation.evaluation.geometry_metrics import compare_meshes
from src.neuron_model.spine_generation.reconstruction.calibration import (
    calibrate_marching_cubes_spine,
    calibrate_poisson_spine,
    poisson_param_grid,
    select_best,
    summarize,
)
from src.neuron_model.spine_generation.reconstruction.estimate_normals import estimate_normals, normal_angle_error
from src.neuron_model.spine_generation.reconstruction.marching_cubes import (
    GridSpec,
    evaluate_sdf_on_grid,
    extract_mesh,
    grid_points,
    sdf_to_mesh,
)
from src.neuron_model.spine_generation.reconstruction.mesh_validation import aggregate_validity, validate_mesh
from src.neuron_model.spine_generation.reconstruction.postprocess import postprocess_mesh
from src.neuron_model.spine_generation.reconstruction.screened_poisson import PoissonParams, screened_poisson


needs_poisson = pytest.mark.skipif(
    not poisson_backend_works(),
    reason="pymeshlab Screened Poisson unusable in this env (pip pymeshlab libomp vs conda libomp clash on macOS) - verify on Windows",
)


def sphere_sdf(radius, center=(0.0, 0.0, 0.0)):
    center = np.asarray(center)
    return lambda q: np.linalg.norm(q - center, axis=1) - radius


def sampled(mesh, n, seed=0):
    points, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points), np.asarray(mesh.face_normals[face_idx])


def test_marching_cubes_sphere_volume_and_orientation():
    grid = GridSpec((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), 48)
    mesh, diag, _ = sdf_to_mesh(sphere_sdf(1.0), grid, chunk_size=5000)
    assert diag["surface_found"] and not diag["surface_touches_grid_boundary"]
    assert mesh.is_watertight
    assert mesh.volume > 0  # negative-inside SDF -> outward-facing triangles
    assert mesh.volume == pytest.approx(4 / 3 * np.pi, rel=0.03)


def test_marching_cubes_flags_clipped_surface_and_missing_surface():
    grid = GridSpec((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 24)
    _, diag, _ = sdf_to_mesh(sphere_sdf(1.5), grid)
    assert diag["surface_touches_grid_boundary"]
    mesh, diag = extract_mesh(np.ones((8, 8, 8), dtype=np.float32), GridSpec((0, 0, 0), (1, 1, 1), 8))
    assert mesh is None and not diag["surface_found"]


def test_chunked_grid_evaluation_matches_grid_layout():
    grid = GridSpec((-1.0, 0.0, 2.0), (1.0, 3.0, 4.0), 7)
    fn = lambda q: q @ np.array([1.0, 10.0, 100.0])
    small = evaluate_sdf_on_grid(fn, grid, chunk_size=10)
    big = evaluate_sdf_on_grid(fn, grid, chunk_size=10**9)
    assert np.array_equal(small, big)
    assert np.allclose(small.reshape(-1), fn(grid_points(grid)))


@pytest.mark.parametrize("shape", ["sphere", "capsule"])
def test_estimated_normals_are_consistent_and_outward(shape):
    mesh = trimesh.creation.icosphere(subdivisions=4) if shape == "sphere" else trimesh.creation.capsule(height=3.0, radius=0.5, count=[32, 32])
    points, true_normals = sampled(mesh, 4096)
    normals, diag = estimate_normals(points, k=30)
    err = normal_angle_error(normals, true_normals)
    assert err["flipped_fraction"] == 0.0
    assert err["oriented_mean_deg"] < 5.0
    assert diag["outward_fraction"] > 0.99


@needs_poisson
def test_screened_poisson_reconstructs_sphere_with_estimated_normals():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    points, _ = sampled(sphere, 4096)
    normals, _ = estimate_normals(points, k=30)
    mesh = screened_poisson(points, normals, PoissonParams(depth=7))
    report = validate_mesh(mesh)
    assert report["is_watertight"] and report["n_connected_components"] == 1 and report["genus"] == 0
    metrics = compare_meshes(mesh, sphere, n_points=4096)
    assert metrics["chamfer_l1"] < 0.02  # radius 1
    assert metrics["volume_rel_diff"] == pytest.approx(0.0, abs=0.03)


def test_mesh_validation_and_aggregate_rates():
    good = trimesh.creation.icosphere(subdivisions=2)
    open_mesh = good.copy()
    open_mesh.update_faces(np.arange(len(open_mesh.faces)) > 3)
    two = trimesh.util.concatenate([good, good.copy().apply_translation([5, 0, 0])])
    reports = [validate_mesh(good), validate_mesh(open_mesh), validate_mesh(two), validate_mesh(None)]
    assert [r["is_valid"] for r in reports] == [True, False, False, False]
    assert validate_mesh(good, bbox=((-1, -1, -1), (1, 1, 1)))["surface_touches_bbox"] is True
    assert validate_mesh(good, bbox=((-2, -2, -2), (2, 2, 2)))["surface_touches_bbox"] is False
    rates = aggregate_validity(reports)
    assert rates["valid_mesh_rate"] == 0.25 and rates["mesh_present_rate"] == 0.75
    assert rates["single_component_rate"] == 0.5  # good + open mesh (one component, just not closed)
    assert rates["watertight_rate"] == 0.5


def test_postprocess_removes_only_tiny_components():
    big = trimesh.creation.icosphere(subdivisions=3)
    tiny = trimesh.creation.icosphere(subdivisions=1, radius=0.01).apply_translation([3, 0, 0])
    medium = trimesh.creation.icosphere(subdivisions=2, radius=0.5).apply_translation([-3, 0, 0])
    mesh = trimesh.util.concatenate([big, tiny, medium])
    post, report = postprocess_mesh(mesh, min_component_area_fraction=0.01)
    assert report["n_components_before"] == 3 and report["n_components_removed"] == 1
    assert len(post.split(only_watertight=False)) == 2


def _synthetic_spine(tmp_path):
    reference = trimesh.creation.capsule(height=2.0, radius=0.5, count=[16, 16])
    mesh_path = tmp_path / "local_sealed_spine_000.off"
    reference.export(mesh_path)
    points, normals = sampled(reference, 2048)
    pc_path = tmp_path / "pointcloud_2048_1.npz"
    np.savez_compressed(pc_path, points=points.astype(np.float32), normals=normals.astype(np.float32))
    return {"spine_key": "synthetic/0/0/0/0", "local_sealed_mesh_path": str(mesh_path), "pointcloud_2048_path": str(pc_path)}


def test_marching_cubes_calibration_on_synthetic_spine(tmp_path):
    spine = _synthetic_spine(tmp_path)
    rows = calibrate_marching_cubes_spine(spine, (-1.0, -1.0, -2.0), (1.0, 1.0, 2.0), [16, 32], compare_kwargs={"n_points": 2048})
    assert [r["resolution"] for r in rows] == [16, 32]
    assert all(r["raw_is_watertight"] for r in rows)
    assert rows[1]["raw_chamfer_l1"] < rows[0]["raw_chamfer_l1"]  # finer grid -> smaller error


@needs_poisson
def test_poisson_calibration_on_synthetic_spine(tmp_path):
    import pandas as pd

    spine = _synthetic_spine(tmp_path)
    params = poisson_param_grid({"depth": [5, 7]}, fixed={"point_weight": 4.0})
    rows = calibrate_poisson_spine(spine, params, k_normals=[20], n_points=2048, compare_kwargs={"n_points": 2048, "f_score_thresholds": [0.05]})
    assert len(rows) == 4  # (estimated k=20, true) x (depth 5, 7)
    assert all(row["raw_is_watertight"] for row in rows)
    summary = summarize(pd.DataFrame(rows), ["normals_source", "k_normal", "depth"])
    best = select_best(summary, error_column="raw_chamfer_l1_median", validity_column="raw_is_valid_mean", min_validity=1.0)
    assert best["selected"]["depth"] in (5, 7)


def test_param_grid_and_selection_logic():
    import pandas as pd

    grid = poisson_param_grid({"depth": [6, 8], "point_weight": [0.0, 4.0]}, fixed={"threads": 2})
    assert len(grid) == 4 and all(p.threads == 2 for p in grid)
    with pytest.raises(ValueError):
        poisson_param_grid({"not_a_param": [1]})
    summary = pd.DataFrame({"depth": [6, 7, 8], "err": [0.1, 0.05, 0.01], "valid": [1.0, 1.0, 0.5]})
    assert select_best(summary, error_column="err", validity_column="valid", min_validity=0.99)["selected"]["depth"] == 7
    fallback = select_best(summary, error_column="err", validity_column="valid", min_validity=1.1)
    assert not fallback["met_validity_threshold"] and fallback["selected"]["depth"] == 7
