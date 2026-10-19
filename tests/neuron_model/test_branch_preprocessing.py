"""End-to-end check of the branch preprocessing on a synthetic mini dataset with a known answer."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.neuron_model import branch_geometry as bg
from src.neuron_model.branch_preprocessing import BranchPreprocessingConfig, run

NEURON = "100000000000000001"
R = 300.0


def _frames(axis):
    t = np.gradient(axis, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return bg.rotation_minimizing_frames(axis, t, bg.initial_normal(axis, t[0], 0.02)[0])


def _write_off(path, vertices, faces):
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices, faces, process=False).export(path)


def _sphere(center, radius=180.0):
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    return np.asarray(s.vertices) + center, np.asarray(s.faces)


@pytest.fixture()
def mini(tmp_path):
    raw = tmp_path / "raw"
    meta = tmp_path / "meta"
    s_root = tmp_path / "s_pre"
    neuron = raw / "minnie65" / NEURON
    # one dendrite: a gentle arc starting 3 um from the soma centre; branches 000 + 001 form merged branch 0
    u = np.linspace(0.0, 1.0, 801)
    axis = np.column_stack([3000.0 + 20000.0 * u, 800.0 * np.sin(np.pi * u), 300.0 * u])
    axis = bg.resample_polyline(axis, 25.0)
    t, e1, e2 = _frames(axis)
    half = len(axis) // 2
    parts = {0: slice(0, half + 1), 1: slice(half, len(axis))}  # members share the junction vertex
    rng = np.random.default_rng(0)
    spines_truth = {}
    rows_manifest, rows_oriented = [], []
    for b, sl in parts.items():
        ax, tt, ee1, ee2 = axis[sl], t[sl], e1[sl], e2[sl]
        v, f = bg.tube_mesh(ax, np.full(len(ax), R), ee1, ee2, n_theta=40, caps=False)
        verts, faces = [v], [f]
        bdir = neuron / "limb_000" / f"branch_{b:03d}"
        specs = [("000", "valid"), ("001", "invalid"), ("002", None)] if b == 0 else [("000", "valid")]
        for sid, status in specs:
            j = int(rng.integers(len(ax) // 4, 3 * len(ax) // 4))
            theta = float(rng.uniform(-math.pi, math.pi))
            normal = math.cos(theta) * ee1[j] + math.sin(theta) * ee2[j]
            base = ax[j] + R * normal  # attachment centre on the membrane
            sv, sf = _sphere(base + 160.0 * normal)
            _write_off(bdir / "spines" / f"spine_{sid}.off", sv, sf)
            faces.append(sf + sum(len(x) for x in verts))
            verts.append(sv)
            spines_truth[(b, sid)] = (base, status)
            if status is not None:
                key = dict(dataset="minnie65", neuron_id=NEURON, limb_id="000", branch_id=f"{b:03d}", spine_id=sid)
                rows_manifest.append({**key, "merged_branch_id": 0, "status": status, "seal_status": "success", "orient_status": "success",
                                      "attachment_ambiguous": False, "attachment_method": "loop", "attachment_region_path": ""})
                rows_oriented.append({**key, "origin_global": base.tolist(), "attachment_loop_area": 3.0e4, "attachment_cap_area": 4.0e4})
        _write_off(bdir / "branch_mesh.off", np.vstack(verts), np.vstack(faces))
        skel = ax + R * ee1  # raw skeleton on the membrane
        np.save(bdir / "branch_skeleton.npy", np.stack([skel[:-1], skel[1:]], axis=1))
    # a short separate branch (merged id 2): too short for geometry
    short = np.column_stack([np.linspace(0, 1000, 21), np.full(21, 5000.0), np.zeros(21)])
    st, se1, se2 = _frames(short)
    v, f = bg.tube_mesh(short, np.full(len(short), R), se1, se2, n_theta=24, caps=False)
    sdir = neuron / "limb_000" / "branch_002"
    _write_off(sdir / "branch_mesh.off", v, f)
    np.save(sdir / "branch_skeleton.npy", np.stack([short[:-1], short[1:]], axis=1))
    # soma and limb skeleton (from the soma centre along the dendrite)
    sv, sf = _sphere(np.zeros(3), radius=2500.0)
    _write_off(neuron / "soma" / "soma_mesh.off", sv, sf)
    limb = np.vstack([np.linspace([0, 0, 0], axis[0], 30), axis[1:]])
    np.save(neuron / "limb_000" / "limb_skeleton.npy", np.stack([limb[:-1], limb[1:]], axis=1))
    # merge mapping (lengths in um, like the real table)
    mapping = pd.DataFrame(
        {
            "limb_id": [0, 0, 0], "branch_id": [0, 1, 2],
            "prefix": [f"{NEURON}_limb_000_branch_{b:03d}" for b in (0, 1, 2)],
            "merged_branch_id": [0, 0, 2], "merged_parent_branch_id": [-1, -1, 0], "merged_member_count": [2, 2, 1],
            "merged_length_um": [20.2, 20.2, 1.0], "merged_branch_depth": [1, 1, 2], "merged_is_terminal": [False, False, True],
            "merged_parent_ambiguous": [False] * 3, "was_merged": [True, True, False],
        }
    )
    (meta / "minnie65" / "merged_branches").mkdir(parents=True)
    mapping.to_parquet(meta / "minnie65" / "merged_branches" / "single_child_merge_mapping.parquet")
    s_root.mkdir()
    pd.DataFrame(rows_manifest).to_parquet(s_root / "manifest.parquet")
    pd.DataFrame(rows_oriented).to_parquet(s_root / "oriented_spines.parquet")
    pd.DataFrame([{k: r[k] for k in ("dataset", "neuron_id", "limb_id", "branch_id", "spine_id")} | {"attachment_loop_area": 1.0, "attachment_cap_area": 1.0} for r in rows_oriented]).to_parquet(s_root / "sealed_spines.parquet")
    cfg = BranchPreprocessingConfig(dataset="minnie65", raw_data_root=raw, metadata_root=meta, s_preprocessed_root=s_root, output_root=tmp_path / "out")
    return dict(cfg=cfg, axis=axis, t=t, e1=e1, spines=spines_truth, tmp=tmp_path)


def test_pipeline_end_to_end(mini):
    cfg = mini["cfg"]
    result = run(cfg)
    assert result["processed"] == 1 and result["failed_neurons"] == 0
    branches = pd.read_parquet(cfg.out_root / "branches.parquet").set_index("merged_branch_id")
    spines = pd.read_parquet(cfg.out_root / "branch_spines.parquet")

    short = branches.loc[2]
    assert short["status"] == "too_short" and not short["train_eligible_T"] and not short["train_eligible_P"]

    b = branches.loc[0]
    assert b["status"] == "ok", b["error"]
    assert b["member_count"] == 2 and b["merge_max_gap_nm"] < 1.0 and b["skeleton_topology_ok"]
    # counts: 4 segments, 1 false spine (stays in the shaft), 1 without a verdict
    assert (b["n_segments_total"], b["n_false"], b["n_unknown"], b["n_spines"]) == (4, 1, 1, 3)
    assert b["n_with_position"] == 2 and b["n_spines_in_range"] == 3
    # vertices removed = the valid and the unknown spine spheres, NOT the false one
    sphere_vertices = len(_sphere(np.zeros(3))[0])
    assert b["n_spine_vertices_removed"] == 3 * sphere_vertices
    assert b["radius_median_nm"] == pytest.approx(R, rel=0.05)
    # the centreline spans the reliable sections; the ends near the cut / branching are trimmed and recorded
    full = float(bg.arclength(mini["axis"])[-1])
    assert b["length_nm"] + b["trim_start_nm"] + b["trim_end_nm"] == pytest.approx(full, rel=0.03)
    assert max(b["trim_start_nm"], b["trim_end_nm"]) <= cfg.trim_max_nm
    assert b["proximal_end_method"] == "limb_graph" and b["frame_reference"] in ("bending", "global_axis")
    # the soma is at the origin and the dendrite starts 3 um away: proximal end distance ~ 3 um (+ trimmed end)
    assert 2500.0 < b["soma_distance_start_nm"] < 4500.0
    assert b["train_eligible_T"]
    assert not b["train_eligible_P"]  # 1 of 4 segments without a verdict: unknown_fraction 0.25 > 0.1

    # geometry file: nodes every ~500 nm, centreline close to the true axis
    geom = np.load(b["geometry_path"])
    from scipy.spatial import cKDTree

    d, _ = cKDTree(mini["axis"]).query(geom["gamma"])
    assert np.median(d) < 0.15 * R
    assert np.all(np.diff(geom["s"]) <= 500.0 + 1e-6)

    # positions of the valid spines: radial distance ~ r, angles consistent with the construction (same frame up to a shift)
    pos = spines[spines["has_position"]]
    assert set(pos["spine_id"]) == {"000"} and len(pos) == 2
    assert np.all(np.abs(pos["dist_perp_minus_radius"]) < 0.2 * R)
    assert set(spines["status"]) == {"valid", "invalid", "unknown"}

    # resume: nothing is recomputed, the marker stays
    marker = cfg.out_root / NEURON / "_done.json"
    before = marker.stat().st_mtime_ns
    again = run(cfg)
    assert again["processed"] == 0 and marker.stat().st_mtime_ns == before
    assert json.loads(marker.read_text())["partial"] is False


def test_branch_subset_is_not_marked_done(mini):
    cfg = mini["cfg"]
    cfg.merged_branch_ids = [2]
    run(cfg)
    assert json.loads((cfg.out_root / NEURON / "_done.json").read_text())["partial"] is True
    cfg.merged_branch_ids = None
    assert run(cfg)["processed"] == 1  # a partial neuron is processed again in a full run


def test_only_s_neurons_skips_neurons_without_s_rows(mini):
    import dataclasses

    from src.neuron_model.branch_preprocessing.inputs import available_neurons, load_members

    cfg = mini["cfg"]
    members = load_members(cfg)
    assert available_neurons(cfg, members) == [NEURON]
    # a second neuron that exists in the raw data and the mapping but has no rows in the S manifest
    (cfg.dataset_root / "100000000000000002").mkdir()
    extra = members.copy()
    extra["neuron_id"] = "100000000000000002"
    import pandas as pd

    members2 = pd.concat([members, extra], ignore_index=True)
    assert sorted(available_neurons(cfg, members2)) == [NEURON, "100000000000000002"]
    only = dataclasses.replace(cfg, only_s_neurons=True)
    assert available_neurons(only, members2) == [NEURON]


def test_reapply_eligibility_changes_flags_without_reprocessing(mini):
    import dataclasses

    from src.neuron_model.branch_preprocessing import reapply_eligibility

    cfg = mini["cfg"]
    run(cfg)
    marker = cfg.out_root / NEURON / "_done.json"
    stamp = marker.stat().st_mtime_ns
    base = pd.read_parquet(cfg.out_root / "branches.parquet").set_index("merged_branch_id")
    assert bool(base.loc[0, "train_eligible_T"]) and not bool(base.loc[0, "train_eligible_P"])  # P: 25 % unknown segments, 2 of 3 spines positioned

    # the eligibility thresholds are not part of the geometry hash
    assert dataclasses.replace(cfg, unknown_fraction_max=0.5, position_fraction_min=0.5, radius_max_nm=200.0).config_hash() == cfg.config_hash()

    dry = reapply_eligibility(dataclasses.replace(cfg, unknown_fraction_max=0.5, position_fraction_min=0.5), write=False)
    assert dry["train_eligible_P"] == 1 and dry["train_eligible_T"] == 1
    assert not bool(pd.read_parquet(cfg.out_root / "branches.parquet").set_index("merged_branch_id").loc[0, "train_eligible_P"])  # nothing written

    res = reapply_eligibility(dataclasses.replace(cfg, unknown_fraction_max=0.5, position_fraction_min=0.5, radius_max_nm=200.0))
    after = pd.read_parquet(cfg.out_root / "branches.parquet").set_index("merged_branch_id")
    assert bool(after.loc[0, "train_eligible_P"]) is False  # radius 300 nm > 200: radius_range
    assert "radius_range" in after.loc[0, "qc_reason"] and res["train_eligible_T"] == 0
    res = reapply_eligibility(dataclasses.replace(cfg, unknown_fraction_max=0.5, position_fraction_min=0.5))
    after = pd.read_parquet(cfg.out_root / "branches.parquet").set_index("merged_branch_id")
    assert bool(after.loc[0, "train_eligible_P"]) and bool(after.loc[0, "train_eligible_T"])
    assert marker.stat().st_mtime_ns == stamp and run(cfg)["processed"] == 0  # the geometry is not recomputed
    assert after.loc[2, "status"] == "too_short" and not bool(after.loc[2, "train_eligible_T"])

    # the absolute floor of the centreline limit
    tight = dataclasses.replace(cfg, spline_rms_factor=1e-3)
    assert reapply_eligibility(tight, write=False)["train_eligible_T"] == 0
    assert reapply_eligibility(dataclasses.replace(tight, spline_rms_floor_nm=100.0), write=False)["train_eligible_T"] == 1
