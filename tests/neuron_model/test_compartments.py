"""Apical / basal by direction towards the pia (src/neuron_model/compartments.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neuron_model.compartments import classify_by_direction, estimate_pia_axis, layer_value, limb_features


def test_layer_values_and_pia_axis_from_layers():
    assert layer_value("Layer 2") == 2.0 and layer_value("White matter") == 7.0 and layer_value("5P-IT") == 5.0 and layer_value("BC") is None
    rng = np.random.default_rng(0)
    depth_axis = np.array([0.2, 0.9, 0.1]) / np.linalg.norm([0.2, 0.9, 0.1])  # layers deepen along this axis
    centers = rng.uniform(0, 1e6, (200, 3))
    layers = 1 + 5 * (centers @ depth_axis - (centers @ depth_axis).min()) / np.ptp(centers @ depth_axis) + rng.normal(0, 0.2, 200)
    pia, r2 = estimate_pia_axis(centers, layers)
    assert np.dot(pia, -depth_axis) > 0.99 and r2 > 0.9
    # a thin slab (z spread 1 %): the z slope is noise and must not tilt the axis
    flat_axis = np.array([0.3, 0.95, 0.0]) / np.linalg.norm([0.3, 0.95, 0.0])
    slab = rng.uniform(0, 1e6, (60, 3))
    slab[:, 2] = rng.normal(3e4, 1e4, 60)
    layers = 1 + 5 * (slab @ flat_axis - (slab @ flat_axis).min()) / np.ptp(slab @ flat_axis) + rng.normal(0, 0.4, 60)
    pia, _ = estimate_pia_axis(slab, layers)
    assert pia[2] == 0.0 and np.dot(pia, -flat_axis) > 0.99


def _write_limb(root, limb, start, direction, length=60000.0, step=500.0, branch=None):
    n = int(length / step)
    pts = start + np.outer(np.arange(n + 1) * step, direction)
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    if branch is not None:  # a side branch from the middle
        mid = pts[n // 2]
        side = mid + np.outer(np.arange(1, 40) * step, branch)
        seg = np.concatenate([seg, np.stack([np.vstack([mid, side[:-1]]), side], axis=1)])
    d = root / f"limb_{limb:03d}"
    d.mkdir(parents=True)
    np.save(d / "limb_skeleton.npy", seg)


def test_limb_features_and_direction_labels(tmp_path):
    center = np.zeros(3)
    up = np.array([0.0, -1.0, 0.0])
    _write_limb(tmp_path, 0, center + 7000 * up, up, length=300000.0, branch=np.array([1.0, 0, 0]))   # apical, towards the pia
    _write_limb(tmp_path, 1, center + np.array([0, 7000, 0]), np.array([0.3, 1, 0]) / np.linalg.norm([0.3, 1, 0]))  # basal, down
    _write_limb(tmp_path, 2, center + np.array([7000, 0, 0]), np.array([1.0, 0.2, 0]) / np.linalg.norm([1, 0.2, 0]))  # basal, sideways
    rows = limb_features(tmp_path, center)
    assert [r["limb_id"] for r in rows] == [0, 1, 2]
    assert rows[0]["cable_nm"] == pytest.approx(300000.0 + 39 * 500.0, rel=1e-6)
    assert np.dot(rows[0]["dir_init"], up) > 0.99
    limbs = pd.DataFrame(rows).assign(neuron_id="n1", axon_like=False)
    labelled, neurons = classify_by_direction(limbs, up)
    assert labelled["compartment_dir"].tolist() == ["apical", "basal", "basal"] and bool(neurons["confident"].iloc[0])
    # two limbs towards the pia with similar length -> uncertain
    two = limbs.copy()
    two["dir_init"] = [list(up) if i == 1 else d for i, d in enumerate(two["dir_init"])]
    lab2, n2 = classify_by_direction(two.assign(cable_nm=[1e5, 1e5, 1e5]), up)
    assert set(lab2["compartment_dir"]) == {"uncertain"} and "two limbs" in n2["reason"].iloc[0]
    lab3, n3 = classify_by_direction(limbs, up, applicable={"n1": False})
    assert set(lab3["compartment_dir"]) == {"not_applicable"}


def test_interneurons_get_thin_long_axon_and_inhibitory_dendrites():
    from src.neuron_model.compartments import label_no_apical_neurons

    limbs = pd.DataFrame({"neuron_id": ["i1"] * 3 + ["i2"] * 2 + ["o1"] + ["e1"] * 2,
                          "limb_id": [0, 1, 2, 0, 1, 0, 0, 1],
                          "limb_radius_nm": [150.0, 260.0, 240.0, 250.0, 230.0, 120.0, 300.0, 120.0],
                          "cable_nm": [6e6, 6e5, 4e5, 8e5, 5e5, 5e6, 1e6, 4e6],
                          "compartment_dir": ["not_applicable"] * 6 + ["apical", "axon_like"],
                          "axon_like": [True, True, False, False, True, False, False, True]})  # spine-based flags: wrong on aspiny dendrites
    applicable = {"i1": False, "i2": False, "o1": False, "e1": True}
    out = label_no_apical_neurons(limbs, applicable, non_neuronal=("OLIGO",), cell_type={"o1": "OLIGO", "i1": "BC", "i2": "MC"})
    assert out["compartment_dir"].tolist() == ["axon_like", "inhibitory_dendrite", "inhibitory_dendrite",  # i1: the thin long limb
                                               "not_applicable", "not_applicable",  # i2: no thin long limb -> axon not identified
                                               "not_applicable",  # OLIGO: not a neuron
                                               "apical", "axon_like"]  # excitatory: untouched
    assert out["axon_like"].tolist() == [True, False, False, False, False, False, False, True]


def test_inhibitory_dendrite_in_the_design_is_not_collinear():
    from src.neuron_model.trunk_generation.conditions import DesignEncoder, group_key

    enc = DesignEncoder.fit([5000.0, 50000.0])
    names = enc.names
    row = dict(zip(names, enc.row("mouse", "inhibitory", "inhibitory_dendrite", 20000.0)))
    assert row["cell_type_binary=inhibitory"] == 1.0
    assert all(row[n] == 0.0 for n in names if n.startswith("compartment="))  # not "unknown", no own column
    assert len(names) == len(DesignEncoder.fit([5000.0, 50000.0]).names)  # old model files keep their layout
    assert group_key("mouse", "inhibitory", "inhibitory_dendrite") == "mouse|inhibitory|inhibitory_dendrite"
    assert group_key("mouse", "inhibitory", None) == "mouse|inhibitory|unknown"


def test_alternative_compartment_table_replaces_the_default(tmp_path):
    from src.neuron_model.metadata import load_metadata_tables

    table = pd.DataFrame({"dataset": "minnie65", "neuron": ["n1", "n1", "n1", "n2"], "limb_id": [0, 1, 2, 0],
                          "limb": ["n1_limb_000", "n1_limb_001", "n1_limb_002", "n2_limb_000"],
                          "compartment": ["apical", "axon_like", "basal", "not_applicable"]})
    path = tmp_path / "limb_compartments_direction.parquet"
    table.to_parquet(path)
    meta = load_metadata_tables(tmp_path, path)
    comps = meta["compartments"]["minnie65"]
    assert comps == {("n1", 0): "apical", ("n1", 1): "axon_like", ("n1", 2): "basal"}  # not_applicable = unknown
    assert meta["compartments"]["minnie"] is comps
    h01 = table.assign(dataset="h01", neuron="neuron_5", limb=lambda t: "neuron_5_limb_00" + t["limb_id"].astype(str))
    h01.to_parquet(path)
    assert ("neuron_5", 0) in load_metadata_tables(tmp_path, path)["compartments"]["h01"]


def test_h01_tables_are_found_next_to_h01_seg_when_an_older_merged_branches_exists(tmp_path):
    from src.neuron_model.metadata import h01_merged_branches_dir, load_metadata_tables

    (tmp_path / "H01" / "merged_branches").mkdir(parents=True)  # the older layout on the NAS, without the tables
    seg = tmp_path / "H01" / "H01_Seg"
    (seg / "merged_branches").mkdir(parents=True)
    pd.DataFrame({"neuron_id": [5, 7], "layer": ["Layer 2", "Layer 5"]}).to_csv(seg / "h01_layer.csv", index=False)
    pd.DataFrame({"neuron_id": [5, 7], "cell_type": ["PYRAMIDAL", "INTERNEURON"]}).to_csv(seg / "h01_celltype.csv", index=False)
    meta = load_metadata_tables(tmp_path)
    assert meta["layers"]["h01"] == {"neuron_5": "Layer 2", "neuron_7": "Layer 5"}
    assert meta["cell_types"]["h01"]["neuron_7"] == "INTERNEURON"
    (tmp_path / "H01" / "merged_branches" / "h01_single_child_merge_mapping.parquet").write_bytes(b"")
    found = h01_merged_branches_dir(tmp_path, "h01_single_child_merge_mapping.parquet")  # case-insensitive file systems: h01 = H01
    assert str(found).lower() == str(tmp_path / "H01" / "merged_branches").lower()
    assert h01_merged_branches_dir(tmp_path) == seg / "merged_branches"
