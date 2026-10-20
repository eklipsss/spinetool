"""Loader of the branch preprocessing outputs (geometry pack) and the T0 / P0 fitting procedures."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.neuron_model import branch_data as bd
from src.neuron_model.branch_preprocessing import run
from src.neuron_model.trunk_generation.conditions import DesignEncoder

from test_branch_preprocessing import mini  # noqa: F401  (fixture: synthetic mini dataset)


def test_pack_objects_and_relocation(mini, tmp_path, monkeypatch):
    cfg = mini["cfg"]
    run(cfg)
    # the output folder is moved (copied from the workstation / off the NAS): absolute geometry_path is stale
    moved = tmp_path / "moved"
    shutil.copytree(cfg.out_root, moved)
    shutil.rmtree(cfg.out_root)
    table, spines, pack = bd.prepare_training_data(moved, tmp_path / "pack", io_threads=2)
    ok = bd.select(table, eligible="train_eligible_T")
    assert len(ok) == 1 and bd.branch_key(ok.iloc[0]) in pack
    g = pack.get(bd.branch_key(ok.iloc[0]))
    assert g["gamma"].shape == (len(g["s"]), 3) and np.isnan(g["kappa1"][-1])

    enc = DesignEncoder.fit(ok["length_nm"].tolist())
    t0 = bd.t0_training_branches(ok, pack, enc)[0]
    assert len(t0.kappa1) == len(t0.log_radius) - 1 and np.isfinite(t0.kappa1).all()
    assert t0.length == pytest.approx(float(g["s"][-1]))

    cov = bd.count_covariates(ok, pack)[0]
    assert cov.n_spines == int(ok.iloc[0]["n_spines_in_range"]) and 250.0 < cov.mean_radius_nm < 350.0
    assert cov.soma_distance_nm > cov.length_nm / 2  # midpoint distance = start distance + L / 2

    pos = bd.position_branches(ok, spines, pack)[0]
    assert pos.positions is not None and len(pos.positions) == int(ok.iloc[0]["n_with_position"])
    assert np.isfinite(pos.kappa).all()

    # the pack is reused (no file is read again) unless the branch set changes
    def boom(*a, **k):
        raise AssertionError("re-read")

    monkeypatch.setattr(bd, "geometry_path", boom)
    bd.prepare_training_data(moved, tmp_path / "pack", io_threads=2)
    monkeypatch.undo()
    info = json.loads((tmp_path / "pack" / "pack_info.json").read_text())
    assert info["n_packed"] == 1 and info["n_failed"] == 0


def test_fit_p0_selects_nb_with_neuron_effect_and_estimates_dmin():
    from src.neuron_model.placement_generation.covariates import BranchCovariates
    from src.neuron_model.placement_generation.fit import estimate_d_min, fit_p0
    from src.neuron_model.placement_generation.models.position import PositionBranch

    rng = np.random.default_rng(0)

    def make(n_neurons):
        cov, pos = [], []
        for n in range(n_neurons):
            eff = rng.normal(0, 0.5)
            for _ in range(6):
                L = float(np.exp(rng.uniform(np.log(5000), np.log(60000))))
                mu = L / 1000 * 0.3 * np.exp(eff)
                y = int(rng.poisson(rng.gamma(4.0, mu / 4.0)))
                s = np.linspace(0, L, int(L / 500) + 1)
                cov.append(BranchCovariates(f"n{n}", y, L, 2 * np.pi * 250 * L, 250.0, 20000.0, 1.1, bool(rng.random() < 0.4), "mouse", "excitatory", "basal"))
                pos.append(PositionBranch(s, np.full(len(s), 250.0), np.full(len(s), 1e-4), 20000.0, np.sort(rng.uniform(0, L, y))))
        return cov, pos

    tc, tp = make(80)
    vc, vp = make(30)
    res = fit_p0(tc, vc, tp, vp, families=("poisson", "nb"), exposures=("length",), random_effects=(False, True), position_use=("radius", "end"), max_iter=150, log=lambda *a: None)
    chosen = res["report"]["count_chosen"]
    assert chosen["family"] == "nb" and chosen["random_effect"] is True
    calib = res["report"]["count_calibration_val"]
    assert all(abs(c["mean_model"] - c["mean_data"]) < 0.5 * c["mean_data"] + 2 for c in calib)
    # uniform positions: dropping a covariate costs (almost) nothing
    assert all(abs(r.get("gain_of_dropped", 0.0)) < 0.05 for r in res["report"]["position_variants"])

    spines = pd.DataFrame({"dataset": "m", "neuron_id": "a", "limb_id": 0, "merged_branch_id": 0, "status": "valid", "has_position": True,
                           "s": [0.0, 1000.0, 1100.0], "theta": [0.0, 0.0, 0.0], "radius_at_s": 300.0, "attachment_cap_area": np.pi * 200.0 ** 2})
    d = estimate_d_min(spines, 0.0)
    assert d["d_min_nm"] == pytest.approx(100.0) and d["attachment_radius_median_nm"] == pytest.approx(200.0)


def test_fit_t0_reports_functionals_against_noise_floor():
    from test_t0 import _simulate

    from src.neuron_model.trunk_generation.fit import fit_t0

    rng = np.random.default_rng(3)
    train, enc = _simulate(rng, 200)
    val, _ = _simulate(rng, 60)
    conds = [{"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical"}] * len(val)
    neurons = [f"n{i // 3}" for i in range(len(val))]
    res = fit_t0(train, val, conds, neurons, encoder=enc, mean_models=("parabola",), max_iter=100, log=lambda *a: None, direction="kappa")  # data simulated by the kappa-OU
    rep = res["report"]
    assert rep["real_val_rejection_rate"] <= 0.05  # the calibrated limits let the real branches through
    v = rep["variants"]["parabola"]
    assert v["generation"]["acceptance_rate"] > 0.9
    # a correct model is within a few noise floors on every functional
    assert v["mean_w1_rel"] < 3 * rep["noise_floor_mean_w1_rel"] + 0.3
