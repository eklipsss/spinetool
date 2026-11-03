"""Global report engine on synthetic three-dataset data with built-in effects (src/neuron_model/global_report)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.neuron_model.global_report.datasets import dataset_inventory, tidy_spines
from src.neuron_model.global_report.descriptive import icc1, summarize_metrics
from src.neuron_model.global_report.factors import availability_matrix, valid_comparisons
from src.neuron_model.global_report.significance import run_comparisons, save_tables, to_records

METRICS = ["M_cell", "M_comp", "M_species", "M_path", "M_null"]


def _raw(units, rng, n_spines=25, limbs=True):
    rows = []
    for u in units:
        style = rng.normal(0, 0.15)
        for i in range(n_spines):
            row = {"neuron_id": u, "spine_id": f"s{i}", "branch_id": 0}
            if limbs:
                row["limb_id"] = i % 2          # limb 0 apical, limb 1 basal
            for m in METRICS:
                row[m] = float(np.exp(rng.normal(0, 0.3) + style))
            rows.append(row)
    return pd.DataFrame(rows)


def _build(seed=0):
    rng = np.random.default_rng(seed)
    meta = {"cell_types": {}, "compartments": {}}
    # Minnie65: 50 excitatory + 20 inhibitory; effect of cell type on M_cell, apical vs basal on M_comp
    ids = [f"m{i}" for i in range(70)]
    types = {i: ("23P" if k < 50 else "BC") for k, i in enumerate(ids)}
    meta["cell_types"]["minnie65"] = types
    meta["compartments"]["minnie65"] = {(i, 0): "apical" for i in ids} | {(i, 1): "basal" for i in ids}
    raw = _raw(ids, rng)
    raw.loc[raw["neuron_id"].map(types) == "BC", "M_cell"] *= 1.5
    raw.loc[raw["limb_id"] == 0, "M_comp"] *= 1.3
    minnie = tidy_spines(raw, "minnie65", meta, METRICS)
    # H01: 30 pyramidal + 12 interneuron, same cell-type effect, M_species shifted (+60 %), no compartments
    ids = [f"neuron_{i}" for i in range(42)]
    types = {i: ("PYRAMIDAL" if k < 30 else "INTERNEURON") for k, i in enumerate(ids)}
    meta["cell_types"]["h01"] = types
    raw = _raw(ids, rng)
    raw.loc[raw["neuron_id"].map(types) == "INTERNEURON", "M_cell"] *= 1.5
    raw["M_species"] *= 1.6
    h01 = tidy_spines(raw, "h01", meta, METRICS)
    # LabID: units = branch folders, no limbs, no cell types; disease effect on M_path
    units = ([f"Wt_b{i}" for i in range(14)] + [f"Ab_b{i}" for i in range(14)] + [f"Wt+9009_b{i}" for i in range(10)]
             + [f"Ab+9009_b{i}" for i in range(10)] + [f"Wt_old_b{i}" for i in range(6)])
    raw = _raw(units, rng, n_spines=12, limbs=False)
    raw.loc[raw["neuron_id"].str.startswith("Ab_"), "M_path"] *= 1.5
    lab = tidy_spines(raw, "labid", meta, METRICS)
    return pd.concat([minnie, h01, lab], ignore_index=True)


def _run(frame, **kw):
    return run_comparisons(frame, METRICS, min_neurons=10, min_spines=5, n_permutations=1500, n_boot=100, log=lambda *_: None, **kw)


def test_tidy_labid_has_no_neuron_identity_and_drops_wt_old():
    frame = _build()
    lab = frame[frame["dataset"] == "labid"]
    assert not (lab["group"] == "Wt_old").any()
    assert lab["cell_type"].isna().all() and lab["compartment"].isna().all()
    assert set(lab["health"]) == {"healthy", "disease"} and set(lab["treatment"]) == {"untreated", "9009"}
    assert lab["unit_id"].nunique() == 48 and (lab["unit_id"] == lab["neuron_id"]).all()


def test_availability_matrix_and_guards():
    frame = _build()
    comps = valid_comparisons(frame, min_neurons=10, min_spines=5)
    labid_cell = [c for c in comps if c["dataset"] == "labid" and c["factor"] == "cell_type_binary"]
    assert labid_cell and not labid_cell[0]["valid"]
    h01_comp = [c for c in comps if c["dataset"] == "h01" and c["factor"] == "compartment"]
    assert h01_comp and not any(c["valid"] for c in h01_comp)           # no compartment metadata for H01 here
    assert any(c["valid"] and c["factor"] == "compartment" and c["dataset"] == "minnie65" for c in comps)
    species = [c for c in comps if c["factor"] == "species"]
    assert species and species[0]["valid"] and not species[0]["causal"] and species[0]["confounded_with"]
    comps = valid_comparisons(frame, min_neurons=25, min_spines=5)      # Minnie has 20 inhibitory units
    assert not any(c["valid"] and c["factor"] == "cell_type_binary" and c["dataset"] == "minnie65" for c in comps)
    mat = availability_matrix(frame, min_neurons=10, min_spines=5)
    assert mat.loc["species"].eq("descriptive_only").any()


def test_engine_finds_only_built_in_effects():
    frame = _build()
    res = _run(frame)["results"]
    sig = res[res["significant"]]
    got = set(zip(sig["dataset"], sig["factor"], sig["metric"]))
    assert ("minnie65", "cell_type_binary", "M_cell") in got and ("h01", "cell_type_binary", "M_cell") in got
    assert ("minnie65", "compartment", "M_comp") in got
    assert ("labid", "health | treatment", "M_path") in got
    # species (confounded with dataset): found as DESCRIPTIVE and flagged
    sp = res[(res["factor"] == "species") & (res["metric"] == "M_species")].iloc[0]
    assert sp["significant"] and not sp["causal"] and "descriptive_only" in sp["interpretation"] and sp["confounded_with"]
    # within-dataset contrasts: nothing outside the built-in effects (detailed cell types / strata repeat the same effects)
    within = sig[sig["causal"]]
    allowed_metric = {"cell_type_binary": "M_cell", "cell_type_binary | compartment": "M_cell", "cell_type": "M_cell",
                      "compartment": "M_comp", "health | treatment": "M_path", "treatment | health": None}
    for _, r in within.iterrows():
        assert r["metric"] == allowed_metric[r["factor"]], (r["dataset"], r["factor"], r["metric"])
    null = res[(res["metric"] == "M_null") & res["causal"]]
    assert not null["significant"].any()
    assert set(res["label"]) <= {"significant", "not_significant", "underpowered", "not_tested"}
    assert res.loc[res["significant"], "label"].eq("significant").all()


def test_labid_confounded_treatment_flagged_and_unit_is_branch():
    res = _run(_build())["results"]
    tr = res[res["factor"] == "treatment | health"]
    assert len(tr) and not tr["causal"].any() and tr["warnings"].str.contains("confounded").all()
    health = res[(res["factor"] == "health | treatment") & (res["dataset"] == "labid")]
    assert (health["n_neurons_a"] >= 10).all()               # units = branch folders
    assert set(health["stratum"]) == {"untreated", "9009"}


def test_underpowered_when_few_units():
    frame = _build()
    drop = (frame["dataset"] == "minnie65") & (frame["cell_type_binary"] == "inhibitory") & ~frame["unit_id"].isin([f"minnie65:m{i}" for i in range(50, 60)])
    out = run_comparisons(frame[~drop], ["M_null"], min_neurons=10, min_spines=5, n_permutations=500, n_boot=50, margin_rel=0.02, log=lambda *_: None)
    r = out["results"]
    r = r[(r["dataset"] == "minnie65") & (r["factor"] == "cell_type_binary")]
    assert len(r) and (r["label"] != "significant").all() and (r["label"] == "underpowered").all()


def test_descriptive_inventory_and_serialisation(tmp_path):
    frame = _build()
    d = summarize_metrics(frame, METRICS, by=["dataset"])
    assert {"q50", "icc", "hist_counts", "n_units"} <= set(d.columns) and len(d) == 15
    assert d["icc"].max() > 0.2                                # the built-in neuron style
    assert icc1(np.array([1.0, 1.0, 2.0, 2.0, 2.1]), np.array(["a", "a", "b", "b", "b"])) > 0.9
    inv = dataset_inventory(frame)
    assert inv["by_dataset"].set_index("dataset").loc["labid", "n_units"] == 48
    assert inv["by_dataset_compartment"]["n_rows"].sum() == len(frame)
    out = _run(frame)
    save_tables({**out, "descriptive": d, **{f"inventory_{k}": v for k, v in inv.items()}}, tmp_path)
    recs = json.loads((tmp_path / "results.json").read_text())
    assert recs and "q_bh" in recs[0]
    json.dumps(to_records(d), allow_nan=False)


def test_branch_groups_labid_like_unit_is_branch_folder():
    from src.neuron_model.global_report.datasets import tidy_branch_groups

    rng = np.random.default_rng(1)
    units = [f"Wt_f{i}" for i in range(12)] + [f"Ab_f{i}" for i in range(12)]
    b, s = [], []
    for u in units:
        n = 8
        b.append({"neuron_id": u, "limb_id": 0, "merged_branch_id": 0, "status": "ok", "length_nm": 20000.0,
                  "radius_median_nm": 800.0 * (1.4 if u.startswith("Ab") else 1.0) * np.exp(rng.normal(0, 0.05)), "radius_proximal_nm": 800.0,
                  "radius_distal_nm": 780.0, "ellipse_ratio_median": 0.8, "tortuosity": 1.05, "kappa_rms": 1e-4, "n_spines": n,
                  "train_eligible_T": True, "train_eligible_P": True})
        for k in range(n):
            s.append({"neuron_id": u, "limb_id": 0, "merged_branch_id": 0, "s": float(rng.uniform(0, 20000)), "u": 0.5})
    g = tidy_branch_groups(pd.DataFrame(b), pd.DataFrame(s), "labid", {}, min_branches=1)
    assert len(g) == 24 and g["compartment"].isna().all() and set(g["health"]) == {"healthy", "disease"}
    res = run_comparisons(g, ["radius_median_nm", "density_per_um"], min_neurons=10, min_spines=1, n_permutations=1000, n_boot=50,
                          log=lambda *_: None)["results"]
    r = res[(res["factor"] == "health | treatment") & (res["metric"] == "radius_median_nm")]
    assert len(r) == 1 and bool(r["significant"].iloc[0])


def test_min_spines_per_dataset():
    frame = _build()
    # LabID units carry 12 rows: a per-dataset floor of 20 removes them, 5 keeps them
    res = run_comparisons(frame, ["M_path"], min_neurons=10, min_spines={"default": 5, "labid": 20}, n_permutations=300, n_boot=20, log=lambda *_: None)
    assert not (res["results"]["dataset"] == "labid").any() if len(res["results"]) else True
    res = run_comparisons(frame, ["M_path"], min_neurons=10, min_spines={"default": 5, "labid": 5}, n_permutations=300, n_boot=20, log=lambda *_: None)
    assert (res["results"]["dataset"] == "labid").any()
