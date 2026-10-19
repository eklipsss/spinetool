import math

import numpy as np
import pytest

from src.neuron_model.branch_geometry import (
    FLAG_END_SECTION,
    bending_from_frames,
    fit_sections,
    flag_sections,
    FLAG_KINK,
    Sections,
    initial_normal,
    integrate_bending,
    match_vertices,
    node_grid,
    project_points,
    resample_polyline,
    rotation_minimizing_frames,
    smooth_centerline,
    smoothed_tangents,
    tube_mesh,
    tube_self_contact,
    ellipse_tube_mesh,
)


def _axis(n=400, length=20000.0):
    """Planar S-curve in nm (gentle bends, radius of curvature >> shaft radius)."""
    u = np.linspace(0.0, 1.0, n)
    x = u * length
    y = 1500.0 * np.sin(2 * np.pi * u)
    z = 400.0 * u**2 * 1000.0 / 1000.0
    return np.column_stack([x, y, z])


def _synthetic_branch(rng, n_spines=12, r0=300.0):
    axis = resample_polyline(_axis(), 50.0)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(axis, axis=0), axis=1))]
    radius = r0 * (1.0 - 0.3 * s / s[-1])  # tapering 300 -> 210 nm
    t = np.gradient(axis, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    e1_0, _ = initial_normal(axis, t[0], 0.02)
    t, e1, e2 = rotation_minimizing_frames(axis, t, e1_0)
    verts, faces = tube_mesh(axis, radius, e1, e2, n_theta=48, caps=False)
    # spines: small spheres glued to the surface (their vertices are part of the branch mesh)
    spine_vertices = []
    all_v, all_f = [verts], [faces]
    offset = len(verts)
    for k in range(n_spines):
        j = int(rng.integers(40, len(axis) - 40))
        ang = rng.uniform(-np.pi, np.pi)
        direction = math.cos(ang) * e1[j] + math.sin(ang) * e2[j]
        center = axis[j] + (radius[j] + 250.0) * direction
        phi, th = np.meshgrid(np.linspace(0, np.pi, 8), np.linspace(0, 2 * np.pi, 12, endpoint=False))
        sph = np.column_stack([np.sin(phi).ravel() * np.cos(th).ravel(), np.sin(phi).ravel() * np.sin(th).ravel(), np.cos(phi).ravel()])
        sv = center + 220.0 * sph
        spine_vertices.append(sv)
        all_v.append(sv)
        offset += len(sv)
    mesh_v = np.vstack(all_v)
    skeleton = axis + radius[:, None] * e1  # the raw skeleton runs ON the membrane, like the real data
    spine_v = np.vstack(spine_vertices) if spine_vertices else np.zeros((0, 3))
    return dict(axis=axis, radius=radius, s=s, mesh_v=mesh_v, tube_faces=faces, tube_v=verts, spine_v=spine_v, skeleton=skeleton)


def test_match_vertices_rounds_coordinates():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([[1.0004, 2.0, 3.0], [9.0, 9.0, 9.0]])
    assert match_vertices(a, b, decimals=3).tolist() == [True, False]
    assert match_vertices(a, np.zeros((0, 3))).tolist() == [False, False]


def test_centerline_recovered_from_surface_skeleton():
    rng = np.random.default_rng(0)
    b = _synthetic_branch(rng)
    mask = match_vertices(b["mesh_v"], b["spine_v"])
    assert mask.sum() == len(b["spine_v"])
    shaft = b["mesh_v"][~mask]
    g = resample_polyline(b["skeleton"], 500.0)
    t = smoothed_tangents(g, 3)
    sec = fit_sections(g, t, shaft, slab=150.0, max_r=1200.0, cluster_eps=150.0, robust_k=0.35, min_points=10)
    flags, r_med = flag_sections(sec, min_points=10, large_radius_factor=1.5, kink_factor=0.5, end_distance_factor=2.0)
    ok = flags == 0
    assert ok.mean() > 0.8
    assert abs(r_med - np.median(b["radius"])) / np.median(b["radius"]) < 0.05
    cl = smooth_centerline(sec.centers[ok], smoothing_nm=1000.0)
    # distance of the smoothed centreline to the true axis << radius (the skeleton itself is ~r away)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(b["axis"]).query(cl.dense)
    # the smoothing spline (h = 1 um) cuts bends by ~ h^2 kappa / 2; this S-curve bends with a radius of
    # only ~7 um, real dendrites much less, so the bias here is an upper estimate
    assert np.median(d) < 0.15 * r_med and np.max(d) < 0.3 * r_med
    d_skel, _ = cKDTree(b["axis"]).query(g)
    assert np.median(d_skel) > 0.9 * r_med  # sanity: the input really was off-axis


def test_end_sections_flagged():
    rng = np.random.default_rng(1)
    b = _synthetic_branch(rng, n_spines=0)
    g = resample_polyline(b["skeleton"], 500.0)
    sec = fit_sections(g, smoothed_tangents(g, 3), b["mesh_v"], slab=150.0, max_r=1200.0, cluster_eps=150.0, robust_k=0.35, min_points=10)
    flags, _ = flag_sections(sec, min_points=10, large_radius_factor=1.5, kink_factor=0.5, end_distance_factor=2.0)
    assert flags[0] & FLAG_END_SECTION and flags[-1] & FLAG_END_SECTION


def _helix(n, a=3000.0, c=800.0, turns=1.5):
    th = np.linspace(0, 2 * np.pi * turns, n)
    return np.column_stack([a * np.cos(th), a * np.sin(th), c * th])


def test_rmf_has_no_twist_and_bending_matches_helix_curvature():
    a, c = 3000.0, 800.0
    pts = resample_polyline(_helix(4000, a, c), 20.0)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    e1_0, ref = initial_normal(pts, t[0], 0.02)
    t, e1, e2 = rotation_minimizing_frames(pts, t, e1_0)
    assert np.allclose(np.einsum("ij,ij->i", t, e1), 0, atol=1e-9) and np.allclose(np.linalg.norm(e1, axis=1), 1)
    ds = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    k1, k2, twist = bending_from_frames(t, e1, e2, ds)
    kappa = np.hypot(k1, k2)
    expected = a / (a * a + c * c)
    assert np.median(np.abs(kappa[5:-5] - expected)) / expected < 0.02
    assert np.max(twist) < 1e-3 * expected


def test_integrate_bending_roundtrip():
    pts = resample_polyline(_axis(), 100.0)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    nodes, ds = node_grid(float(s[-1]), 500.0)
    gamma = np.column_stack([np.interp(nodes, s, pts[:, k]) for k in range(3)])
    t = np.gradient(gamma, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    e1_0, _ = initial_normal(gamma, t[0], 0.02)
    t, e1, e2 = rotation_minimizing_frames(gamma, t, e1_0)
    k1, k2, _ = bending_from_frames(t, e1, e2, ds)
    frame0 = np.stack([t[0], e1[0], e2[0]], axis=1)
    rebuilt, frames = integrate_bending(k1, k2, ds, gamma[0], frame0)
    # discrete scheme: error small relative to the branch (and to the step)
    assert np.max(np.linalg.norm(rebuilt - gamma, axis=1)) < 0.05 * ds * math.sqrt(len(nodes))
    assert np.allclose(np.einsum("mij,mik->mjk", frames, frames)[-1], np.eye(3), atol=1e-9)


def test_node_grid_reproduces_length():
    nodes, ds = node_grid(12345.0, 500.0)
    assert nodes[-1] == pytest.approx(12345.0) and len(nodes) == 26 and ds <= 500.0


def test_initial_normal_rotation_shifts_all_angles():
    pts = resample_polyline(_axis(), 100.0)
    nodes, _ = node_grid(float(np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))][-1]), 500.0)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    gamma = np.column_stack([np.interp(nodes, s, pts[:, k]) for k in range(3)])
    t = np.gradient(gamma, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    e1_0, ref = initial_normal(gamma, t[0], 0.02)
    assert ref == "bending"
    t, e1, e2 = rotation_minimizing_frames(gamma, t, e1_0)
    # attachment points at known (s, theta) on a radius-300 tube
    rng = np.random.default_rng(3)
    j = rng.integers(2, len(nodes) - 2, size=20)
    theta_true = rng.uniform(-np.pi, np.pi, size=20)
    p = gamma[j] + 300.0 * (np.cos(theta_true)[:, None] * e1[j] + np.sin(theta_true)[:, None] * e2[j])
    proj = project_points(p, gamma, nodes, t, e1)
    # the node polyline approximates the curve: the foot point moves by ~ r * kappa * ds / 2 (~10 nm here)
    assert np.allclose(proj.s, nodes[j], atol=0.05 * (nodes[1] - nodes[0]))
    assert np.allclose(np.angle(np.exp(1j * (proj.theta - theta_true))), 0, atol=0.02)
    # rotate the reference by phi: every angle shifts by -phi, differences are unchanged
    phi = 0.7
    e1r = np.cos(phi) * e1 + np.sin(phi) * e2
    proj_r = project_points(p, gamma, nodes, t, e1r)
    assert np.allclose(np.angle(np.exp(1j * (proj_r.theta - proj.theta + phi))), 0, atol=1e-3)
    # a point beyond the distal end is reported as such
    far = gamma[-1] + 400.0 * t[-1] + 300.0 * e1[-1]
    pr = project_points(far[None], gamma, nodes, t, e1)
    assert pr.at_end[0] == 1 and pr.beyond[0] == pytest.approx(400.0, abs=1.0)


def test_tube_mesh_closed_outward_and_self_contact():
    import trimesh

    pts = resample_polyline(_axis(), 100.0)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    nodes, ds = node_grid(float(s[-1]), 500.0)
    gamma = np.column_stack([np.interp(nodes, s, pts[:, k]) for k in range(3)])
    t = np.gradient(gamma, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    t, e1, e2 = rotation_minimizing_frames(gamma, t, initial_normal(gamma, t[0], 0.02)[0])
    r = np.full(len(nodes), 300.0)
    v, f = tube_mesh(gamma, r, e1, e2, n_theta=24)
    mesh = trimesh.Trimesh(v, f, process=True)
    assert mesh.is_watertight and mesh.volume > 0
    assert not tube_self_contact(gamma, r, nodes, ds)[0]
    # a helix whose turns are closer than 2r touches itself
    hel = resample_polyline(_helix(4000, 2000.0, 60.0, 2.0), 50.0)
    hs = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(hel, axis=0), axis=1))]
    hn, hds = node_grid(float(hs[-1]), 200.0)
    hg = np.column_stack([np.interp(hn, hs, hel[:, k]) for k in range(3)])
    assert tube_self_contact(hg, np.full(len(hn), 300.0), hn, hds)[0]


def test_kink_limit_has_an_absolute_floor():
    n = 12
    centers = np.column_stack([np.arange(n) * 500.0, np.zeros(n), np.zeros(n)])
    centers[6, 1] = 80.0  # one centre displaced by 80 nm (the midpoint deviation of its own: 80 nm)
    sec = Sections(points=centers.copy(), tangents=np.tile([1.0, 0.0, 0.0], (n, 1)), centers=centers, radii=np.full(n, 100.0), n_points=np.full(n, 50))
    kw = dict(min_points=10, large_radius_factor=1.5, kink_factor=0.5, end_distance_factor=0.0)
    flags, _ = flag_sections(sec, **kw)
    assert flags[6] & FLAG_KINK  # 80 nm > 0.5 * 100 nm
    flags, _ = flag_sections(sec, kink_floor_nm=150.0, **kw)
    assert not (flags & FLAG_KINK).any()  # 80 nm < 150 nm: noise level of the centres, not a kink
    sec.centers[6, 1] = 400.0
    flags, _ = flag_sections(sec, kink_floor_nm=150.0, **kw)
    assert flags[6] & FLAG_KINK


def test_adaptive_smoothing_keeps_a_wiggly_thin_line_inside_the_tube():
    """A thin dendrite (r = 110 nm) winding with a 150 nm amplitude every 1.6 um: a 1 um smoothing cuts the bends and
    leaves the tube, an adaptive width ~2.5 r follows them (the effect seen on real thin Minnie65 branches)."""
    import dataclasses

    from src.neuron_model.branch_preprocessing import BranchPreprocessingConfig
    from src.neuron_model.branch_preprocessing.process import adaptive_node_step_nm, adaptive_smoothing_nm, centerline_geometry, line_inside_fraction

    s = np.linspace(0.0, 16000.0, 1601)
    axis = np.column_stack([s, 150.0 * np.sin(2 * np.pi * s / 1600.0), 60.0 * np.cos(2 * np.pi * s / 2300.0)])
    axis = resample_polyline(axis, 10.0)
    t = np.gradient(axis, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    t, e1, e2 = rotation_minimizing_frames(axis, t, initial_normal(axis, t[0], 0.02)[0])
    r = 110.0
    v, f = tube_mesh(axis, np.full(len(axis), r), e1, e2, n_theta=24, caps=True)
    rng = np.random.default_rng(0)
    centres = resample_polyline(axis, 500.0)  # one section every 500 nm
    centres = centres + rng.normal(0.0, 15.0, centres.shape)
    radii = np.full(len(centres), r)
    cfg = BranchPreprocessingConfig()
    fixed = centerline_geometry(centres, radii, smoothing_nm=1000.0, radius_smoothing_nm=1000.0, node_step_nm=500.0, straight_fraction=0.02)
    acfg = dataclasses.replace(cfg, centerline_smoothing_factor=2.5, node_step_per_smoothing=0.5)
    h = adaptive_smoothing_nm(acfg, r)
    assert h == 275.0 and adaptive_node_step_nm(acfg, h) == 250.0
    assert adaptive_smoothing_nm(acfg, 600.0) == 1000.0 and adaptive_smoothing_nm(cfg, 110.0) == 1000.0  # clipped / fixed by default
    adapt = centerline_geometry(centres, radii, smoothing_nm=h, radius_smoothing_nm=1000.0, node_step_nm=adaptive_node_step_nm(acfg, h), straight_fraction=0.02)
    tri = v[f]
    inside_fixed = line_inside_fraction(fixed["gamma"], tri, margin=220.0)
    inside_adapt = line_inside_fraction(adapt["gamma"], tri, margin=220.0)
    assert inside_fixed < 0.65 and inside_adapt > 0.95
    # the adaptive line resolves the bends: its node step is 250 nm and the bending -> curve round trip stays within
    # 1 % of the length (the drift accumulates along the branch: ~75 nm over 16 um here)
    assert adapt["step"] <= 250.0 + 1e-9 and np.max(np.linalg.norm(adapt["rebuilt"] - adapt["gamma"], axis=1)) < 0.01 * adapt["length"]
    # at most max_nodes nodes
    capped = centerline_geometry(centres, radii, smoothing_nm=h, radius_smoothing_nm=1000.0, node_step_nm=250.0, straight_fraction=0.02, max_nodes=40)
    assert len(capped["nodes"]) <= 40


def test_flattened_tube_recovers_ellipse_profile_and_beats_the_circle():
    """A tube with an elliptic section (a = 300, b = 160 nm, major axis turning slowly in the RMF): the section ellipses and
    the node profile recover a, b and the angle; the elliptic tube describes the surface far better than the round one."""
    from scipy.spatial import cKDTree

    from src.neuron_model.branch_preprocessing.process import centerline_geometry

    s = np.linspace(0.0, 12000.0, 1201)
    axis = resample_polyline(np.column_stack([s, 600.0 * np.sin(np.pi * s / 12000.0), np.zeros_like(s)]), 25.0)
    t = np.gradient(axis, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    t, e1, e2 = rotation_minimizing_frames(axis, t, initial_normal(axis, t[0], 0.02)[0])
    m = len(axis)
    true_angle = 0.4 + 0.3 * np.linspace(0.0, 1.0, m)
    v, f = ellipse_tube_mesh(axis, np.full(m, 300.0), np.full(m, 160.0), true_angle, e1, e2, n_theta=48)
    g = resample_polyline(axis, 500.0)
    sec = fit_sections(g, smoothed_tangents(g, 3), v, slab=150.0, max_r=1200.0, cluster_eps=150.0, robust_k=0.35, min_points=10, fit_ellipse=True)
    ok = np.isfinite(sec.ellipse_a)
    assert ok.mean() > 0.9
    assert np.median(sec.ellipse_a[ok]) == pytest.approx(300.0, rel=0.05) and np.median(sec.ellipse_b[ok]) == pytest.approx(160.0, rel=0.08)
    line = centerline_geometry(sec.centers[ok], sec.radii[ok], smoothing_nm=1000.0, radius_smoothing_nm=1000.0, node_step_nm=500.0,
                               straight_fraction=0.02, ellipse=(sec.ellipse_a[ok], sec.ellipse_b[ok], sec.ellipse_major[ok]))
    assert np.median(line["ellipse_b"] / line["ellipse_a"]) == pytest.approx(160.0 / 300.0, rel=0.1)
    # the angle is measured in the line's own RMF: compare tubes rather than angles (frames may differ by a rotation)
    import trimesh

    surface = lambda vv, ff: cKDTree(trimesh.Trimesh(vv, ff, process=False).sample(40000, seed=0))  # dense points of the tube
    tv, tf = tube_mesh(line["gamma"], line["radius"], line["e1"], line["e2"], n_theta=32, caps=False)
    ev, ef = ellipse_tube_mesh(line["gamma"], line["ellipse_a"], line["ellipse_b"], line["ellipse_angle"], line["e1"], line["e2"], n_theta=32)
    inner = (np.linalg.norm(v - axis[0], axis=1) > 800) & (np.linalg.norm(v - axis[-1], axis=1) > 800)
    d_circle = surface(tv, tf).query(v[inner])[0]
    d_ellipse = surface(ev, ef).query(v[inner])[0]
    assert np.median(d_ellipse) < 0.4 * np.median(d_circle)
