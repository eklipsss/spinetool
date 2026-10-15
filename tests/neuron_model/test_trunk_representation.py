import math

import numpy as np
import pytest

from src.neuron_model import branch_geometry as bg
from src.neuron_model.trunk_generation.conditions import CONDITION_DIM, DesignEncoder, encode_conditions, group_key
from src.neuron_model.trunk_generation.constraints import TrunkLimits, check_trunk
from src.neuron_model.trunk_generation.representation import (
    FeatureStats,
    canonicalize_frames,
    geometry_from_features,
    pad_sequence,
    valid_step_mask,
)


def _arc_branch(n=60, step=500.0, radius=280.0):
    u = np.linspace(0.0, 1.0, n)
    length = (n - 1) * step
    pts = np.column_stack([length * 0.9 * u, 1500.0 * np.sin(1.5 * np.pi * u), 600.0 * np.sin(np.pi * u)])
    pts = bg.resample_polyline(pts, step * 0.2)
    s = bg.arclength(pts)
    nodes, ds = bg.node_grid(float(s[-1]), step)
    gamma = np.column_stack([np.interp(nodes, s, pts[:, k]) for k in range(3)])
    t = np.gradient(gamma, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    e1_0, _ = bg.initial_normal(gamma, t[0], 0.02)
    t, e1, e2 = bg.rotation_minimizing_frames(gamma, t, e1_0)
    k1, k2, _ = bg.bending_from_frames(t, e1, e2, ds)
    return gamma, (t, e1, e2), k1, k2, ds, np.full(len(nodes), radius)


def test_condition_encoding_unknown_and_masks():
    v = encode_conditions("mouse", "inhibitory", "uncertain")
    assert v.shape == (CONDITION_DIM,) and v[-3:].tolist() == [1.0, 1.0, 1.0]
    u = encode_conditions(None, "unknown-type", float("nan"))
    assert u[-3:].tolist() == [0.0, 0.0, 0.0]
    assert u[:3].tolist() == [0.0, 0.0, 1.0]  # species: unknown slot
    assert v[:3].tolist() == [1.0, 0.0, 0.0]


def test_design_encoder_and_group_key():
    enc = DesignEncoder.fit([5000, 20000, 80000])
    row = enc.row("mouse", "inhibitory", "basal", 20000.0)
    assert len(row) == len(enc.names) and row[0] == 1.0
    assert row[enc.names.index("cell_type_binary=inhibitory")] == 1.0 and row[enc.names.index("compartment=basal")] == 1.0
    unk = enc.row(None, None, None, 20000.0)
    assert unk[enc.names.index("species=unknown")] == 1.0 and unk[enc.names.index("compartment=unknown")] == 1.0
    assert abs(enc.z_length(math.exp(enc.log_length_mean))) < 1e-12
    assert DesignEncoder.from_dict(enc.to_dict()).names == enc.names
    assert group_key("mouse", "excitatory", None) == "mouse|excitatory|unknown"


def test_features_roundtrip_preserves_shape_and_canonical_frame():
    gamma, _, k1, k2, ds, radius = _arc_branch()
    geom = geometry_from_features(k1, k2, np.log(radius), ds)
    from scipy.spatial.distance import pdist

    # the canonical placement is arbitrary; the shape (all pairwise distances) must be the same
    assert np.max(np.abs(pdist(geom["gamma"]) - pdist(gamma))) < 0.02 * ds * math.sqrt(len(gamma))
    assert geom["length"] == pytest.approx((len(gamma) - 1) * ds)
    # canonical zero angle: e1(0) is the bending direction, frames stay orthonormal, RMF twist-free
    target, ref = bg.initial_normal(geom["gamma"], geom["tangent"][0], 0.02)
    assert ref == "bending" and np.allclose(geom["e1"][0], target, atol=1e-6)
    f = np.stack([geom["tangent"], geom["e1"], geom["e2"]], axis=2)
    assert np.allclose(np.einsum("mij,mik->mjk", f, f), np.eye(3), atol=1e-8)
    k1b, k2b, tw = bg.bending_from_frames(geom["tangent"], geom["e1"], geom["e2"], ds)
    assert np.max(tw) < 1e-8
    # canonicalising twice changes nothing
    again, _ = canonicalize_frames(geom["gamma"], f)
    assert np.allclose(again, f, atol=1e-8)


def test_feature_stats_and_padding():
    rng = np.random.default_rng(0)
    seqs = [np.column_stack([rng.normal(1e-3, 5e-4, 40), rng.normal(-2e-3, 1e-3, 40), rng.normal(5.6, 0.2, 40)]) for _ in range(5)]
    for s in seqs:
        s[-1, :2] = np.nan
    stats = FeatureStats.fit(seqs)
    x = stats.normalize(seqs[0])
    assert np.allclose(stats.denormalize(x)[:-1], seqs[0][:-1])
    assert abs(float(np.nanmean(np.vstack([stats.normalize(s) for s in seqs])[:, 0]))) < 1e-9
    padded, node, step = pad_sequence(seqs[0], 64)
    assert padded.shape == (64, 3) and node.sum() == 40 and step.sum() == 39 and not np.isnan(padded).any()
    assert valid_step_mask(5).tolist() == [True, True, True, True, False]
    with pytest.raises(ValueError):
        pad_sequence(seqs[0], 30)


def test_check_trunk_limits_and_reasons():
    gamma, _, k1, k2, ds, radius = _arc_branch()
    limits = TrunkLimits.from_data([np.hypot(k1, k2)], [3.0])
    assert check_trunk(gamma, radius, k1, k2, ds, limits)["passed"]
    assert "radius_range" in check_trunk(gamma, radius * 10, k1, k2, ds, limits)["reasons"]
    assert "r_kappa" in check_trunk(gamma, radius, k1 * 40, k2 * 40, ds, limits)["reasons"]
    assert "kappa_max" in check_trunk(gamma, radius, k1 * 3, k2 * 3, ds, TrunkLimits(kappa_max=float(np.hypot(k1, k2).max()), c_r_kappa=10.0))["reasons"]
    bad = radius.copy()
    bad[5] = np.nan
    assert check_trunk(gamma, bad, k1, k2, ds, limits)["reasons"] == ["non_finite"]
    # a tightly wound helix touches itself
    th = np.linspace(0, 2 * np.pi * 2.0, 120)
    helix = np.column_stack([2000 * np.cos(th), 2000 * np.sin(th), 60 * th])
    pts = bg.resample_polyline(helix, 200.0)
    nodes, ds2 = bg.node_grid(float(bg.arclength(pts)[-1]), 200.0)
    g2 = np.column_stack([np.interp(nodes, bg.arclength(pts), pts[:, k]) for k in range(3)])
    res = check_trunk(g2, np.full(len(nodes), 300.0), np.zeros(len(nodes) - 1), np.zeros(len(nodes) - 1), ds2, TrunkLimits(c_r_kappa=10.0))
    assert "self_contact" in res["reasons"]
