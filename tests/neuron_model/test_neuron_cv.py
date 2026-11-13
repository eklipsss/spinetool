import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.neuron_model.neuron_cv import compare_descriptors, flagged_vs_rest, neuron_descriptors, neuron_folds
from src.neuron_model.placement_generation.covariates import BranchCovariates

REPO = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("cv_tp_neurons", REPO / "scripts/neuron_model/cv_tp_neurons.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_folds_are_stratified_and_cover_every_neuron_once():
    neurons = pd.DataFrame({"neuron_id": [f"n{i}" for i in range(103)],
                            "cell_type_binary": ["excitatory"] * 90 + ["inhibitory"] * 11 + [None] * 2})
    folds = neuron_folds(neurons, 5, seed=1, stratum="cell_type_binary")
    assert set(folds.index) == set(neurons["neuron_id"]) and set(folds.unique()) <= set(range(5))
    by = neurons.assign(fold=neurons["neuron_id"].map(folds)).groupby(["cell_type_binary", "fold"]).size().unstack(fill_value=0)
    assert by.loc["inhibitory"].max() - by.loc["inhibitory"].min() <= 1  # 11 inhibitory -> 2 or 3 per fold
    assert by.loc["excitatory"].tolist() == [18] * 5
    other = neuron_folds(neurons, 5, seed=2, stratum="cell_type_binary")
    assert (other != folds).any()  # another repetition, another assignment
    assert neuron_folds(neurons, 5, seed=1, stratum="cell_type_binary").equals(folds)  # deterministic


def _calibrated(rng, n, factor, flag, group="inhibitory"):
    # rows shaped like calibration.neuron_calibration: expected count e_marginal, observed = Poisson(factor * e * neuron effect)
    e = rng.uniform(20, 60, n)
    obs = rng.poisson(factor * e * np.exp(rng.normal(0, 0.3, n)))
    return pd.DataFrame({"neuron": [f"{group[0]}{flag}{i}{rng.integers(1e9)}" for i in range(n)], "cell_type_binary": group, "obs": obs,
                         "e_marginal": e, "e0": e, "log_ratio": np.log((obs + 0.5) / (e + 0.5)), "flag": flag})


def test_flagged_vs_rest_finds_poor_neurons_only_when_they_are_poor():
    rng = np.random.default_rng(0)
    poor = pd.concat([_calibrated(rng, 50, 1.0, False), _calibrated(rng, 8, 0.4, True)], ignore_index=True)
    r = flagged_vs_rest(poor, "flag", n_perm=2000).iloc[0]
    assert r["flagged_obs_over_model"] < 0.6 and r["p_perm"] < 0.01 and r["rarity_share_as_extreme"] < 0.01
    same = pd.concat([_calibrated(rng, 50, 1.0, False), _calibrated(rng, 8, 1.0, True)], ignore_index=True)
    r = flagged_vs_rest(same, "flag", n_perm=2000).iloc[0]
    assert r["p_perm"] > 0.05 and r["rarity_share_as_extreme"] > 0.05


def test_neuron_descriptors_and_comparison():
    t = pd.DataFrame({
        "neuron_id": ["a", "a", "b", "b", "c"], "cell_type_binary": ["inhibitory"] * 5, "cell_type": ["BC", "BC", "MC", "MC", "BC"],
        "status": ["ok", "failed", "ok", "ok", "ok"], "length_nm": [10000.0, 5000.0, 20000.0, 20000.0, 10000.0],
        "n_spines": [4, 0, 10, 2, 0], "n_spines_in_range": [4, 0, 8, 2, 0], "n_unknown": [2, 0, 0, 0, 0], "n_with_position": [3, 0, 8, 1, 0],
        "radius_median_nm": [200.0, 100.0, 300.0, 250.0, 150.0], "train_eligible_T": [True, False, True, True, True],
        "train_eligible_P": [True, False, True, False, True]})
    d = neuron_descriptors(t).set_index("neuron_id")
    assert d.loc["a", "n_branches"] == 2 and d.loc["a", "share_eligible_P"] == 0.5 and d.loc["a", "share_status_ok"] == 0.5
    assert np.isclose(d.loc["a", "share_unknown_spines"], 0.5) and np.isclose(d.loc["b", "share_with_position"], 0.9)
    assert np.isclose(d.loc["b", "density_all_per_um"], 10 / 40.0) and np.isclose(d.loc["b", "density_eligible_P_per_um"], 8 / 20.0)
    assert d.loc["c", "share_empty_eligible_P"] == 1.0 and np.isnan(d.loc["c", "share_with_position"])  # no spines: undefined, not 0
    d["flag"] = [True, False, False]
    c = compare_descriptors(d.reset_index(), "flag")
    assert set(c["descriptor"]) >= {"share_eligible_P", "density_all_per_um"} and (c["n_flagged"] == 1).all()


def test_p0_part_of_the_script_on_synthetic_neurons(tmp_path, monkeypatch):
    """run_p0 end to end with synthetic branches: original val inhibitory neurons are poorer -> found as poorer."""
    mod = _load_script()
    rng = np.random.default_rng(3)
    cov, rows = [], []
    for n in range(60):
        typ = "inhibitory" if n < 20 else "excitatory"
        split = "val" if n % 5 == 0 else "train"
        dens = (0.1 if typ == "inhibitory" else 0.5) * (0.35 if (typ == "inhibitory" and split == "val") else 1.0)
        eff = rng.normal(0, 0.3)
        for _ in range(8):
            length = float(np.exp(rng.uniform(np.log(5000), np.log(60000))))
            radius = float(rng.uniform(200, 400))
            y = int(rng.poisson(rng.gamma(5.0, length / 1e3 * dens * np.exp(eff) / 5.0)))
            cov.append(BranchCovariates(f"n{n}", y, length, 2 * np.pi * radius * length, radius, 50000.0, 1.1, False, "mouse", typ, "basal"))
            rows.append({"neuron_id": f"n{n}", "split": split, "status": "ok", "train_eligible_P": True, "compartment": "basal", "cell_type_binary": typ})
    table = pd.DataFrame(rows)
    monkeypatch.setattr(mod, "count_covariates", lambda r, pack: [cov[i] for i in r.index])
    args = argparse.Namespace(cv_splits=["train", "val"], p0_variants=["nb:length:re:sg2", "nb:length:re:sg1"], k=4, n_sim=20, n_perm=500)
    cfg = {"data": {"exclude_compartments": None}, "model": {"n_basis": 5, "smooth_penalty": 1.0, "max_iter": 100}}
    neurons = table[["neuron_id", "cell_type_binary"]].drop_duplicates()
    folds = {r: neuron_folds(neurons, 4, seed=r, stratum="cell_type_binary") for r in range(2)}
    mod.run_p0(args, cfg, table, None, folds, None, tmp_path, lambda *a: None)
    nll = pd.read_csv(tmp_path / "p0_folds.csv")
    assert len(nll) == 2 * 4 * 2 and nll["heldout_nll_per_branch"].notna().all()
    held = pd.read_csv(tmp_path / "p0_neurons.csv")
    assert held.groupby("repeat")["neuron"].nunique().tolist() == [60, 60] and not held.duplicated(["repeat", "neuron"]).any()
    flagged = pd.read_csv(tmp_path / "p0_v1val_vs_rest.csv").set_index(["repeat", "cell_type_binary"])
    for rep in (0, 1):
        assert flagged.loc[(rep, "inhibitory"), "flagged_obs_over_model"] < 0.7 * flagged.loc[(rep, "inhibitory"), "rest_obs_over_model"]
        assert flagged.loc[(rep, "excitatory"), "p_perm"] > 0.01
    assert (tmp_path / "p0_groups_cell_type.csv").exists() and (tmp_path / "p0_groups_by_v1_split.csv").exists()


def test_t0_part_of_the_script_on_synthetic_branches(tmp_path, monkeypatch):
    """run_t0 end to end: a correct T0 on every fold, one row per fold, original val neurons counted."""
    import sys

    sys.path.insert(0, str(REPO / "tests/neuron_model"))
    from test_t0 import _simulate

    mod = _load_script()
    rng = np.random.default_rng(5)
    branches, _ = _simulate(rng, 240)
    rows = pd.DataFrame({"dataset": "minnie65", "neuron_id": [f"n{i // 8}" for i in range(240)], "limb_id": 0, "merged_branch_id": np.arange(240),
                         "split": ["val" if (i // 8) % 5 == 0 else "train" for i in range(240)], "status": "ok", "train_eligible_T": True,
                         "compartment": "apical", "species": "mouse", "cell_type_binary": "excitatory", "length_nm": [b.length for b in branches],
                         "soma_distance_start_nm": 1.0e4, "is_terminal": False})
    monkeypatch.setattr(mod, "t0_training_branches", lambda r, pack, enc: [branches[i] for i in r["merged_branch_id"]])

    class AnyPack:
        def __contains__(self, key):
            return True

    folds = neuron_folds(rows[["neuron_id"]].drop_duplicates(), 3, seed=0)
    args = argparse.Namespace(cv_splits=["train", "val"], k=3, t0_config=REPO / "configs/neuron-model/trunk/t0.yaml", branches_root=tmp_path, t0_mean="parabola",
                              t0_overrides=["model.use_soma_distance=false", "model.direction=kappa", "model.max_iter=100", "data.max_val_branches=60"])
    mod.run_t0(args, rows, AnyPack(), folds, tmp_path, lambda *a: None)
    t = pd.read_csv(tmp_path / "t0_folds.csv")
    assert t["fold"].tolist() == [0, 1, 2] and (t["acceptance"] > 0.9).all()
    assert t["n_v1_val_neurons"].sum() == 6  # neurons 0, 5, ..., 25 of the 30 are original val
    assert (t["mean_w1_rel"] < 3 * t["noise_floor_mean_w1_rel"] + 0.3).all()
    s = pd.read_csv(tmp_path / "t0_summary.csv")
    from src.neuron_model.trunk_generation.fit import FUNCTIONALS

    assert list(s["functional"]) == list(FUNCTIONALS) and (s["w1_rel_min"] <= s["w1_rel_max"]).all()
