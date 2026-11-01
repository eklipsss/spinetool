"""Neuron-level tests of spine morphology by cell type and compartment (evaluation/morphology_by_condition.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.neuron_model.spine_generation.evaluation.morphology_by_condition import bh_qvalues, cliffs_delta, morphology_by_condition


def _synthetic(seed=0, n_neurons=200, spines_per_limb=60):
    """Strong neuron style on every metric; a real effect only on Length (inhibitory +35 %) and on Volume (apical
    +30 % vs basal within the same neuron). Inhibitory neurons have no apical limb (the confound of the real data)."""
    rng = np.random.default_rng(seed)
    rows = []
    types = ["23P", "4P", "BC", "MC"]
    for n in range(n_neurons):
        ctype = types[n % 4]
        binary = "inhibitory" if ctype in ("BC", "MC") else "excitatory"
        style = rng.normal(0.0, 0.2, 3)  # neuron style, larger than the spine noise of the median
        comps = ["basal", "uncertain"] if binary == "inhibitory" else ["apical", "basal"]
        for comp in comps:
            k = spines_per_limb
            length = np.exp(np.log(1000.0) + style[0] + (0.3 if binary == "inhibitory" else 0.0) + rng.normal(0, 0.4, k))
            volume = np.exp(np.log(5e7) + style[1] + (0.26 if comp == "apical" else 0.0) + rng.normal(0, 0.5, k))
            area = np.exp(np.log(2e6) + style[2] + rng.normal(0, 0.4, k))
            for i in range(k):
                rows.append({"neuron_id": f"n{n}", "cell_type": ctype, "cell_type_binary": binary, "compartment": comp,
                             "Length": length[i], "Volume": volume[i], "Area": area[i]})
    return pd.DataFrame(rows)


def test_effects_are_found_only_where_they_were_put():
    frame = _synthetic()
    tables = morphology_by_condition(frame, ["Length", "Volume", "Area"], min_neurons=10, min_spines=20, n_permutations=4000, log=lambda *_: None)
    binary = tables["cell_type_binary"].set_index("metric")
    assert binary.loc["Length", "significant"] and binary.loc["Length", "rel_diff_a_vs_b"] > 0.15
    assert not binary.loc["Area", "significant"]
    comp = tables["compartment"]
    apical_basal = comp[(comp.a == "apical") & (comp.b == "basal")].set_index("metric")
    assert apical_basal.loc["Volume", "significant"] and apical_basal.loc["Volume", "median_paired_rel_diff"] > 0.15
    assert not apical_basal.loc["Area", "significant"]
    # detailed types: 23P vs 4P (both excitatory) do not differ in Length; 23P vs BC do
    pairs = tables["cell_type"]
    same = pairs[(pairs.a.isin(["23P", "4P"])) & (pairs.b.isin(["23P", "4P"])) & (pairs.metric == "Length")]
    assert not same["significant"].any()
    omni = tables["cell_type_omnibus"].set_index("metric")
    assert omni.loc["Length", "significant"] and not omni.loc["Area", "significant"]
    within = tables["cell_type_binary_within_compartment"]
    basal = within[(within.compartment == "basal")].set_index("metric")
    assert basal.loc["Length", "significant"]


def test_neuron_style_alone_is_not_significant_although_spines_would_be():
    frame = _synthetic(seed=3)
    frame["Length"] = frame["Area"] / 2e3  # no type effect, only the neuron style
    tables = morphology_by_condition(frame, ["Length"], min_neurons=10, min_spines=20, n_permutations=2000, log=lambda *_: None)
    assert not tables["cell_type_binary"]["significant"].any()


def test_bh_and_cliffs_delta():
    q = bh_qvalues([0.01, 0.04, 0.03, np.nan, 0.5])
    assert np.isnan(q[3]) and np.allclose(q[[0, 2, 1, 4]], [0.04, 0.0533333, 0.0533333, 0.5])
    assert cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
