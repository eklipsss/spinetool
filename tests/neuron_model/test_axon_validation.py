import json

import numpy as np
import pandas as pd

from src.neuron_model.axon_validation import (VOXEL_NM, assign_points, branch_spine_counts, limb_evidence, load_synapse_points,
                                              neuron_index, per_neuron)


def _line(a, b, n=20):
    p = np.linspace(a, b, n)
    return np.stack([p[:-1], p[1:]], axis=1)


def _neuron(tmp_path, nid):
    d = tmp_path / str(nid)
    # limb 0 along +x (axon), limb 1 along +y (dendrite); one branch each
    for limb, end in ((0, [50_000.0, 0, 0]), (1, [0, 50_000.0, 0])):
        seg = _line(np.array([2000.0, 0, 0]) if limb == 0 else np.array([0, 2000.0, 0]), np.array(end))
        (d / f"limb_{limb:03d}" / "branch_000").mkdir(parents=True)
        np.save(d / f"limb_{limb:03d}" / "limb_skeleton.npy", seg)
        np.save(d / f"limb_{limb:03d}" / "branch_000" / "branch_skeleton.npy", seg)
    (d / "summary.json").write_text(json.dumps({"limbs": [{"spines_per_branch": [0]}, {"spines_per_branch": [12]}]}))
    return d


def _vox(p):
    return list(np.asarray(p, float) / VOXEL_NM)


def test_points_land_on_the_right_limb_and_labels_are_scored(tmp_path):
    nid = 864691100000000001
    d = _neuron(tmp_path, nid)
    meta = tmp_path / "meta"
    meta.mkdir()
    pre = [[20_000.0, 300.0, 0.0], [30_000.0, -200.0, 0.0], [40_000.0, 0.0, 400.0]]
    post = [[200.0, 25_000.0, 0.0], [-300.0, 35_000.0, 0.0], [0.0, 45_000.0, 400.0]]
    rows = [{"tag": "shaft", "pre_pt_root_id": nid, "post_pt_root_id": 1, "pre_pt_position": _vox(p), "post_pt_position": _vox(p)} for p in pre]
    rows += [{"tag": "spine", "pre_pt_root_id": 2, "post_pt_root_id": nid, "pre_pt_position": _vox(p), "post_pt_position": _vox(p)} for p in post]
    pd.DataFrame(rows).to_parquet(meta / "vortex_compartment_targets.parquet")

    pts = load_synapse_points(meta, [nid])
    assert sorted(pts["kind"].value_counts().to_dict().items()) == [("post_spine", 3), ("pre", 3)]
    np.testing.assert_allclose(pts.loc[pts["kind"] == "pre", ["x", "y", "z"]].to_numpy(), np.array(pre), atol=1e-6)

    ipts, ilimb, ibr = neuron_index(d, branches=True)
    limb, branch, dist = assign_points(pts[["x", "y", "z"]].to_numpy(), ipts, ilimb, ibr)
    assert list(limb[pts["kind"].to_numpy() == "pre"]) == [0, 0, 0]
    assert list(limb[pts["kind"].to_numpy() == "post_spine"]) == [1, 1, 1]
    assert (dist < 600).all()
    assert branch_spine_counts(d) == {(0, 0): 0, (1, 0): 12}

    assigned = pts.assign(limb_id=limb, branch_id=branch, dist_nm=dist)
    limbs = pd.DataFrame({"neuron_id": [nid, nid], "limb_id": [0, 1], "good": [True, False], "bad": [False, True]})
    ev = limb_evidence(assigned, max_dist_nm=3000)
    assert dict(zip(ev["limb_id"], ev["by_synapses"])) == {0: "axon", 1: "dendrite"}
    nn = per_neuron(assigned, limbs, ["good", "bad"], max_dist_nm=3000).iloc[0]
    assert nn["pre_on_good"] == 1.0 and nn["dendrite_points_on_good"] == 0.0 and nn["top_pre_limb_is_good"]
    assert nn["pre_on_bad"] == 0.0 and nn["dendrite_points_on_bad"] == 1.0 and not nn["top_pre_limb_is_bad"]
