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
