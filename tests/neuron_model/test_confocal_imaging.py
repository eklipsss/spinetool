"""Confocal forward model (src/neuron_model/confocal_imaging.py) on synthetic shapes with known answers."""

import numpy as np
import pytest
import trimesh

from src.neuron_model.confocal_imaging import ConfocalParams, gaussian_psf_sigmas, image_mesh, occupancy_grid, spine_seed


def _sphere(r, center=(0.0, 0.0, 0.0)):
    m = trimesh.creation.icosphere(subdivisions=4, radius=r)
    m.apply_translation(center)
    return m


def test_occupancy_grid_gives_the_volume_of_a_sphere():
    m = _sphere(400.0)
    h = 10.0
    origin = np.array([-450.0, -450.0, -450.0])
    occ = occupancy_grid(m.vertices, m.faces, origin, h, (90, 90, 90))
    assert occ.sum() * h ** 3 == pytest.approx(m.volume, rel=0.02)
    assert occ[45, 45, 45] and not occ[0, 0, 0]


def test_psf_sigmas_follow_the_gaussian_approximation():
    wx, wz = gaussian_psf_sigmas(1.4, 520.0, 1.515, confocal=False)
    assert wx == pytest.approx(0.21 * 520 / 1.4) and wz == pytest.approx(0.66 * 520 * 1.515 / 1.96)
    cx, cz = gaussian_psf_sigmas(1.4, 520.0, 1.515, wavelength_ex_nm=488.0)
    assert cx < wx and cz < wz                       # the confocal PSF is narrower


def test_large_sphere_keeps_its_size_and_tiny_sphere_vanishes():
    p = ConfocalParams(sigma_xy_nm=80.0, sigma_z_nm=250.0, threshold=0.5)
    big = _sphere(800.0)
    out = image_mesh(big.vertices, big.faces, p, np.random.default_rng(1))
    assert out["detected"]
    imaged = trimesh.Trimesh(out["vertices"], out["faces"])
    # a large convex object at half-maximum keeps roughly its volume (blur shrinks curved surfaces a little)
    assert imaged.volume == pytest.approx(big.volume, rel=0.15)
    small = _sphere(40.0)
    out = image_mesh(small.vertices, small.faces, p, np.random.default_rng(1))
    assert not out["detected"]                       # 40 nm sphere: peak intensity far below half of the full fill


def _spine(neck_r=40.0, neck_len=600.0, head_r=300.0):
    neck = trimesh.creation.cylinder(radius=neck_r, height=neck_len, sections=24)
    neck.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))   # axis along +y
    neck.apply_translation([0.0, neck_len / 2, 0.0])
    head = _sphere(head_r, (0.0, neck_len + 0.8 * head_r, 0.0))
    return trimesh.boolean.union([neck, head], engine="manifold") if hasattr(trimesh.boolean, "union") else trimesh.util.concatenate([neck, head])


def test_thin_neck_is_thickened_or_lost_and_the_frame_is_kept():
    s = _spine()
    p = ConfocalParams(sigma_xy_nm=80.0, sigma_z_nm=250.0, threshold=0.3)
    out = image_mesh(s.vertices, s.faces, p, np.random.default_rng(3))
    assert out["detected"]
    v = out["vertices"]
    # the result is back in the local frame: the head is still on +y
    assert v[:, 1].max() > 600.0
    # half-width of the imaged neck at mid-height is wider than the 40 nm neck
    mid = np.abs(v[:, 1] - 300.0) < 30.0
    if mid.any():
        assert np.median(np.hypot(v[mid, 0], v[mid, 2])) > 40.0


def test_spine_seed_is_stable_and_differs_between_spines():
    assert spine_seed("a/b", 1) == spine_seed("a/b", 1) != spine_seed("a/c", 1)


def test_axis_angle_changes_the_axial_elongation():
    m = _sphere(250.0)
    p0 = ConfocalParams(sigma_xy_nm=60.0, sigma_z_nm=300.0, threshold=0.3, optical_axis_angle=0.0, random_phase=False)
    out = image_mesh(m.vertices, m.faces, p0, np.random.default_rng(0))
    ext = np.ptp(out["vertices"], axis=0)
    # optical axis = local y at angle 0: the image is stretched along y, not along x
    assert ext[1] > ext[0] * 1.2


def test_with_a_trunk_the_spine_is_cut_at_the_imaged_trunk_and_keeps_its_neck():
    s = _spine(neck_r=50.0, neck_len=500.0, head_r=250.0)
    common = dict(sigma_xy_nm=80.0, sigma_z_nm=250.0, threshold=0.5, optical_axis_angle=0.3, random_phase=False)
    alone = image_mesh(s.vertices, s.faces, ConfocalParams(**common), np.random.default_rng(0))
    joint = image_mesh(s.vertices, s.faces, ConfocalParams(trunk_radius_nm=300.0, **common), np.random.default_rng(0))
    assert alone["detected"] and joint["detected"]
    va, vj = alone["vertices"], joint["vertices"]
    m = trimesh.Trimesh(vj, joint["faces"])
    assert m.is_watertight
    # the trunk does not leak in: nothing deep below the attachment (trunk surface at y ~ 0)
    assert vj[:, 1].min() > -200.0
    # the trunk's light keeps the neck base: the cut spine starts lower than the isolated one
    assert vj[:, 1].min() < va[:, 1].min()


def test_shift_table_splits_total_into_optics_and_rest():
    import pandas as pd

    from src.neuron_model.spine_generation.evaluation.domain_shift import shift_table, summarize

    rng = np.random.default_rng(0)

    def make(shift, n_units=20, per=20):
        unit = np.repeat([f"u{i}" for i in range(n_units)], per)
        lv = rng.normal(0.0, 0.3, n_units)[np.repeat(np.arange(n_units), per)] + rng.normal(0, 0.5, n_units * per) + shift
        return pd.DataFrame({"unit": unit, "Volume": np.exp(lv + 18), "Length": np.exp(lv + 7)})

    sets = {"ref": make(0.0), "img": make(0.6), "tgt": make(1.0)}  # optics 0.6 of the total 1.0
    t = shift_table(sets, "ref", "tgt", ["Volume", "Length"], n_boot=300)
    v = t[t["metric"] == "Volume"].iloc[0]
    assert v.total == pytest.approx(1.0, abs=0.3) and v.optics == pytest.approx(0.6, abs=0.3) and v.rest == pytest.approx(0.4, abs=0.3)
    assert v.total_sig and v.optics_sig
    s = summarize(t, sets, "ref")
    assert 0.2 < s["closed_fraction"].iloc[0] < 0.9


def test_spinetool_surface_keeps_a_large_sphere_and_a_visible_neck_at_low_threshold():
    # SpineTool last step (local threshold on the smoothed 8-bit image, boundary voxels, Poisson in voxel units);
    # binary_mc: the mac cannot run pymeshlab (OpenMP clash) nor CGAL Poisson (no Eigen); the path is the same up to the mesher
    big = _sphere(800.0)
    p = ConfocalParams(sigma_xy_nm=55.0, sigma_z_nm=180.0, threshold=0.5, surface="spinetool", random_phase=False,
                       voxel_nm=(22.17, 22.17, 100.0), st_poisson="binary_mc")
    out = image_mesh(big.vertices, big.faces, p, np.random.default_rng(1))
    assert out["detected"] and out["info"]["poisson_backend"] in ("cgal", "pymeshlab", "binary_mc")
    m = trimesh.Trimesh(out["vertices"], out["faces"])
    assert m.volume == pytest.approx(big.volume, rel=0.2)
    # optical axis across the neck: neck brightness ~ r^2 / (2 sigma_xy sigma_z) = 0.13 for r = 50 nm
    s = _spine(neck_r=50.0, neck_len=600.0, head_r=300.0)
    common = dict(sigma_xy_nm=55.0, sigma_z_nm=180.0, surface="spinetool", optical_axis_angle=np.pi / 2, random_phase=False,
                  trunk_radius_nm=400.0, voxel_nm=(22.17, 22.17, 100.0), st_poisson="binary_mc")
    low = image_mesh(s.vertices, s.faces, ConfocalParams(threshold=0.08, **common), np.random.default_rng(0))
    high = image_mesh(s.vertices, s.faces, ConfocalParams(threshold=0.5, **common), np.random.default_rng(0))
    assert low["detected"] and low["info"]["attached_to_trunk"]          # the 50 nm neck is visible at the LabID threshold
    assert (not high["detected"]) or (not high["info"]["attached_to_trunk"])  # at 0.5 the head comes off (or vanishes)
