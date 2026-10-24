"""Branch assembly on synthetic trunks and spines (no real data)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("manifold3d")
import trimesh

from src.neuron_model.branch_assembly import AssemblyConfig, SpinePool, assemble_branch, densify, load_tp_set, place, rotation
from src.neuron_model.placement_generation.generate import anchors, pack_records


def _trunk(length=6000.0, radius=300.0, bend=0.0, step=250.0):
    """Planar arc (bend = curvature, 1/nm) or straight line along x, with an RMF (e1 = in-plane normal, e2 = z)."""
    s = np.arange(0.0, length + 1e-9, step)
    if bend > 0:
        phi = s * bend
        gamma = np.column_stack([np.sin(phi) / bend, (1 - np.cos(phi)) / bend, np.zeros_like(s)])
        t = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(s)])
        e1 = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(s)])
    else:
        gamma = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
        t = np.tile([1.0, 0.0, 0.0], (len(s), 1))
        e1 = np.tile([0.0, 1.0, 0.0], (len(s), 1))
    e2 = np.cross(t, e1)
    return {"s": s, "gamma": gamma + 1e5, "radius": np.full(len(s), radius), "tangent": t, "normal": e1, "binormal": e2, "length": float(length)}


def _spine(neck_r=60.0, neck_len=600.0, head_r=200.0):
    """Mushroom spine in the local frame: flat base at y = -20 (the attachment cap faces -y), head along +y."""
    import manifold3d as m3d

    def mf(mesh):
        return m3d.Manifold(m3d.Mesh64(vert_properties=np.asarray(mesh.vertices, float), tri_verts=np.asarray(mesh.faces, np.uint64)))

    neck = trimesh.creation.cylinder(radius=neck_r, height=neck_len, sections=24)
    neck.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # z -> y
    neck.apply_translation([0, neck_len / 2 - 20.0, 0])
    head = trimesh.creation.icosphere(3, head_r)
    head.apply_translation([0, neck_len - 20.0 + 0.7 * head_r, 0])
    out = (mf(neck) + mf(head)).to_mesh64()
    return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)


@pytest.fixture(scope="module")
def pool():
    return SpinePool.from_meshes([("a", _spine()), ("b", _spine(neck_r=80, neck_len=800, head_r=250))])


def test_pool_finds_base(pool):
    assert len(pool) == 2 and not pool.rejected
    sp = pool.spines[0]
    assert abs(sp.center[1] + 20.0) < 5 and np.hypot(sp.center[0], sp.center[2]) < 5
    assert sp.base_mask.any() and (sp.vertices[sp.base_mask, 1] < 15).all()


def test_rotation_maps_local_axes():
    t, n = np.array([1.0, 0.2, 0.0]), np.array([0.0, 0.0, 1.0])
    R = rotation(t, n)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(np.array([0.0, 1.0, 0.0]) @ R, n)  # local +y -> outward normal


@pytest.mark.parametrize("bend", [0.0, 1.0 / 3000.0])
def test_place_on_surface_and_sink_base(pool, bend):
    trunk = densify(_trunk(bend=bend))
    sp = pool.spines[0]
    s, theta = 3000.0, 0.7
    pose = place(sp, trunk, s, theta, AssemblyConfig())
    assert pose["status"] == "ok" and pose["delta_nm"] >= 0
    # the anchor is the P anchor: gamma + r (cos theta e1 + sin theta e2)
    ref = anchors(_trunk(bend=bend), np.array([s]), np.array([theta]))
    assert np.linalg.norm(pose["anchor"] - ref["anchor"][0]) < 5.0
    assert abs(trunk.signed_distance(pose["anchor"][None])[0]) < 2.0
    v = pose["vertices"]
    base_sd = trunk.signed_distance(v[sp.base_mask])
    assert base_sd.max() <= -AssemblyConfig().base_margin_nm + 1.0  # the base is inside
    # the spine grows outward along the normal
    top = v[np.argmax(sp.vertices[:, 1])]
    assert (top - pose["anchor"]) @ pose["normal"] > 0.8 * sp.height


def test_assemble_closed_single_component(pool):
    rec = _trunk()
    s = np.array([1000.0, 2500.0, 4000.0, 5000.0])
    theta = np.array([0.0, 2.0, -2.0, 1.0])
    res = assemble_branch(rec, s, theta, pool, AssemblyConfig(seed=1))
    b = res.branch
    assert b["status"] == "ok", b
    assert b["n_spines_placed"] == 4 and b["watertight"] and b["winding_consistent"] and b["n_components"] == 1
    assert b["volume_nm3"] > b["volume_trunk_nm3"]
    assert np.abs(res.vertices.mean(0) - rec["gamma"].mean(0)).max() < 2000  # shifted back to the input coordinates


def test_wide_base_on_thin_trunk_is_rejected():
    pool = SpinePool.from_meshes([("fat", _spine(neck_r=400.0, neck_len=600.0, head_r=450.0))])
    res = assemble_branch(_trunk(radius=120.0), np.array([3000.0]), np.array([0.0]), pool, AssemblyConfig(on_failure="drop"))
    assert res.spines[0]["status"] == "dropped" and res.spines[0]["failure_reasons"] == "base_not_immersed"
    assert res.branch["n_spines_placed"] == 0 and res.branch["n_components"] == 1


def test_overlapping_spines_collide(pool):
    res = assemble_branch(_trunk(), np.array([3000.0, 3000.0]), np.array([0.5, 0.5]), pool, AssemblyConfig(max_tries=2))
    assert [r["status"] for r in res.spines] == ["placed", "dropped"]
    assert set(res.spines[1]["failure_reasons"].split(";")) == {"collides"}
    assert res.branch["n_components"] == 1


def test_load_tp_set_roundtrip(tmp_path):
    import pandas as pd

    trunks = [_trunk(), _trunk(length=4000.0, bend=1 / 3000.0)]
    np.savez_compressed(tmp_path / "trunks.npz", **pack_records(trunks))
    pd.DataFrame({"branch_index": [0, 1], "length_nm": [6000.0, 4000.0]}).to_parquet(tmp_path / "branches.parquet")
    pd.DataFrame({"branch_index": [1, 1], "spine_index": [1, 0], "s": [2000.0, 1000.0], "theta": [0.1, 0.2]}).to_parquet(tmp_path / "placements.parquet")
    branches, recs, placements = load_tp_set(tmp_path)
    assert len(recs) == 2 and np.allclose(recs[1]["gamma"], trunks[1]["gamma"]) and recs[1]["length"] == pytest.approx(4000.0)
    assert list(placements) == [1] and placements[1]["s"].tolist() == [1000.0, 2000.0]


def test_manifold_simplify_keeps_a_closed_mesh():
    """The degenerate-triangle cleanup (Manifold.simplify, 0.1 nm) must keep a valid closed surface."""
    import manifold3d as m3d
    import trimesh

    a = m3d.Manifold.sphere(500.0, 48)
    b = m3d.Manifold.sphere(300.0, 48).translate([700.0, 0.0, 0.0])
    u = (a + b).simplify(0.1)
    out = u.to_mesh64()
    mesh = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert abs(mesh.volume - u.volume()) / u.volume() < 1e-6


def test_spine_geometry_stats_and_circular_moments():
    from src.neuron_model.branch_assembly.evaluate import circular_moments, spine_geometry_stats

    # three spines: two 300 nm apart along s on the same side, one on the opposite side of a r = 200 nm trunk
    st = spine_geometry_stats(np.array([1000.0, 1300.0, 1300.0]), np.array([0.0, 0.0, np.pi]), np.full(3, 200.0), 2000.0)
    assert np.allclose(st["u"], [0.5, 0.65, 0.65])
    assert st["nn_surface_nm"][0] == pytest.approx(300.0)
    assert st["nn_surface_nm"][2] == pytest.approx(min(np.hypot(300.0, 200.0 * np.pi), 200.0 * np.pi))
    assert np.allclose(st["gap_nm"], [300.0, 0.0])
    rng = np.random.default_rng(0)
    assert circular_moments(rng.uniform(-np.pi, np.pi, 4000))["R1"] < 0.05
    assert circular_moments(np.r_[np.zeros(500), np.full(500, np.pi)])["R2"] == pytest.approx(1.0)


def test_flip_needles_removes_a_zero_area_triangle_and_keeps_the_surface_closed():
    from src.neuron_model.branch_assembly.assemble import flip_needles

    m = trimesh.creation.icosphere(subdivisions=2, radius=1000.0)
    v, f = np.asarray(m.vertices, float).copy(), np.asarray(m.faces).copy()
    a, b, c = f[0]
    v[c] = 0.5 * (v[a] + v[b])  # c on the edge ab: face 0 becomes a needle
    tri = v[f]
    assert (0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) < 1e-6).sum() >= 1
    new, flips = flip_needles(v, f)
    out = trimesh.Trimesh(v, new, process=False)
    assert flips >= 1
    assert (out.area_faces < 1e-6).sum() == 0
    assert out.is_watertight and out.is_winding_consistent


def test_context_selector_fit_recovers_the_base_trunk_relation():
    import pandas as pd

    from src.neuron_model.branch_assembly.selection import ContextSelector

    rng = np.random.default_rng(0)
    r = np.exp(rng.uniform(np.log(150.0), np.log(800.0), 500))
    b = np.exp(np.log(120.0) + 0.6 * np.log(r / 300.0) + rng.normal(0.0, 0.2, len(r)))
    df = pd.DataFrame({"radius_at_s": r, "attachment_loop_area": np.pi * b**2, "has_position": True})
    sel = ContextSelector.fit(df)
    assert sel.a == pytest.approx(np.log(120.0), abs=0.05) and sel.c == pytest.approx(0.6, abs=0.08) and sel.s == pytest.approx(0.2, abs=0.03)
    assert ContextSelector.from_dict(sel.to_dict()).c == sel.c


def test_context_chooser_prefers_narrow_bases_on_thin_trunks():
    from src.neuron_model.branch_assembly.selection import ContextSelector, SpineChooser

    pool = SpinePool.from_meshes([(f"n{nr}", _spine(neck_r=nr, neck_len=600.0, head_r=nr + 120.0)) for nr in (40.0, 80.0, 160.0, 320.0)])
    assert len(pool) == 4 and np.isfinite([sp.base_radius_nm for sp in pool.spines]).all()
    widths = np.array([sp.base_radius_nm for sp in pool.spines])
    assert np.all(np.diff(widths) > 0)  # the found base grows with the neck
    chooser = SpineChooser(pool, ContextSelector(a=np.log(100.0), c=1.0, s=0.25))
    rng = np.random.default_rng(1)
    thin = np.array([chooser.choose(rng, 150.0) for _ in range(400)])
    thick = np.array([chooser.choose(rng, 900.0) for _ in range(400)])
    assert widths[thin].mean() < 0.6 * widths[thick].mean()
    # uniform chooser (default) ignores the radius
    uni = SpineChooser(pool)
    assert len(set(uni.choose(rng, 150.0) for _ in range(200))) == 4
    # fewer impossible placements on a thin trunk than with the uniform choice
    rec = _trunk(radius=150.0, length=20000.0)
    s = np.linspace(1000.0, 19000.0, 10)
    th = np.linspace(-3.0, 3.0, 10)
    cfg = AssemblyConfig(on_failure="drop", seed=3)
    bad_ctx = sum(r["status"] == "dropped" for r in assemble_branch(rec, s, th, pool, cfg, np.random.default_rng(3), chooser=chooser).spines)
    bad_uni = sum(r["status"] == "dropped" for r in assemble_branch(rec, s, th, pool, cfg, np.random.default_rng(3)).spines)
    assert bad_ctx < bad_uni  # 0 vs 4 base_not_immersed of 10 (seed 3)
