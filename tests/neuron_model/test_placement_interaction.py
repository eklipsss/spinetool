"""P1: renewal (P1a), Strauss (P1b) and two-radius Strauss (P1b2) placement models on synthetic branches (models/interaction.py)."""

from __future__ import annotations

import json

import numpy as np

from src.neuron_model.placement_generation.diagnostics import configuration_statistics, simulation_comparison
from src.neuron_model.placement_generation.models.interaction import MultiStraussModel, RenewalModel, StraussModel, interaction_from_dict
from src.neuron_model.placement_generation.models.position import PositionBranch, PositionModel


def _uniform_base() -> PositionModel:
    m = PositionModel(n_basis=4, use=())
    m.beta = np.zeros(3)
    return m


def _branches(n_branches=150, length=20000.0, radius=300.0, step=250.0):
    s = np.arange(0.0, length + step / 2, step)
    return [PositionBranch(s, np.full(len(s), radius), np.zeros(len(s))) for _ in range(n_branches)]


def _with(branches, configs):
    return [PositionBranch(b.s, b.radius, b.kappa, None, np.asarray(c[0]), np.asarray(c[1])) for b, c in zip(branches, configs)]


def test_renewal_recovers_k_and_reduces_to_p0_at_one():
    base, rng = _uniform_base(), np.random.default_rng(0)
    br = _branches()
    for k_true in (3.0, 1.0):
        truth = RenewalModel(base, k_true)
        data = _with(br, [truth.sample_configuration(b, 25, rng) for b in br])
        fit = RenewalModel(base).fit(data)
        assert abs(fit.k - k_true) < 0.25 * k_true, (k_true, fit.k)
    poisson = _with(br, [(base.sample_positions(b, 25, rng), np.zeros(25)) for b in br])
    assert abs(RenewalModel(base).fit(poisson).k - 1.0) < 0.15
    regular = RenewalModel(base, 3.0)
    gaps_r = np.concatenate([np.diff(regular.sample_positions(b, 25, rng)) for b in br[:50]])
    gaps_p = np.concatenate([np.diff(base.sample_positions(b, 25, rng)) for b in br[:50]])
    assert gaps_r.std() / gaps_r.mean() < 0.6 < gaps_p.std() / gaps_p.mean()   # regular vs Poisson-like intervals


def test_strauss_recovers_repulsion_and_not_where_there_is_none():
    base, rng = _uniform_base(), np.random.default_rng(1)
    br = _branches(120)
    truth = StraussModel(base, radius_nm=800.0, log_gamma=np.log(0.05), burn_factor=40)
    data = _with(br, [truth.sample_configuration(b, 30, rng) for b in br])
    radii = [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1600.0]
    fit = StraussModel(base).fit(data, radii=radii, n_angles=16)
    assert fit.radius_nm in (600.0, 800.0, 1000.0) and fit.gamma < 0.3, (fit.radius_nm, fit.gamma)
    poisson = _with(br, [(base.sample_positions(b, 30, rng), np.pi - rng.random(30) * 2 * np.pi) for b in br])
    assert StraussModel(base).fit(poisson, radii=radii, n_angles=16).gamma > 0.7
    # the Monte Carlo check prefers the true model, and the repulsion is visible in the surface nearest-neighbour distance
    check = simulation_comparison({"P0": base, "P1b": fit}, data[:40], n_sim=9, seed=2)
    assert check["models"]["P1b"]["score_mean_abs_z"] < check["models"]["P0"]["score_mean_abs_z"]
    obs = configuration_statistics([(b.positions, b.angles) for b in data[:40]], data[:40])
    ref = configuration_statistics([(b.positions, b.angles) for b in poisson[:40]], poisson[:40])
    assert obs["surface_nn_q10_nm"] > 1.5 * ref["surface_nn_q10_nm"] and obs["pcf_0_500"] < ref["pcf_0_500"]


def test_bundle_round_trip_and_generation_uses_the_interaction(tmp_path):
    from src.neuron_model.placement_generation.fit import load_p0_bundle
    from src.neuron_model.placement_generation.generate import generate_placements
    from src.neuron_model.placement_generation.covariates import CountDesign
    from src.neuron_model.placement_generation.models.count import CountModel

    base = _uniform_base()
    strauss = StraussModel(base, radius_nm=900.0, log_gamma=-20.0, burn_factor=30)
    renewal = RenewalModel(base, 2.5)
    for m in (strauss, renewal, MultiStraussModel(base, [300.0, 900.0], [-20.0, -1.0], burn_factor=30)):
        back = interaction_from_dict(json.loads(json.dumps(m.to_dict())), base)
        assert type(back) is type(m) and back.to_dict()["kind"] == m.kind
    # a P1 bundle: the P0 bundle + "interaction" (count model / design are only carried through here)
    path = tmp_path / "p1b_bundle.json"
    payload = {"design": None, "count_model": None, "position_model": base.to_dict(), "d_min_nm": 100.0, "interaction": strauss.to_dict()}
    path.write_text(json.dumps(payload))
    import pytest

    monkey_design = pytest.MonkeyPatch()
    monkey_design.setattr(CountDesign, "from_dict", classmethod(lambda cls, d: None))
    monkey_design.setattr(CountModel, "from_dict", classmethod(lambda cls, d: None))
    try:
        loaded = load_p0_bundle(path)
    finally:
        monkey_design.undo()
    assert isinstance(loaded["position_model"], StraussModel)
    s = np.arange(0.0, 20001.0, 250.0)
    e1, e2 = np.tile([1.0, 0.0, 0.0], (len(s), 1)), np.tile([0.0, 1.0, 0.0], (len(s), 1))
    trunk = {"s": s, "radius": np.full(len(s), 300.0), "kappa1": np.zeros(len(s)), "kappa2": np.zeros(len(s)), "length": 20000.0,
             "gamma": np.c_[np.zeros(len(s)), np.zeros(len(s)), s], "normal": e1, "binormal": e2}
    out = generate_placements([trunk] * 5, [20] * 5, loaded["position_model"], d_min_nm=0.0, seed=3)
    assert all(p["min_distance_nm"] >= 900.0 for p in out)   # hard core of the Strauss model respected


def test_multi_strauss_one_shell_matches_strauss():
    base, rng = _uniform_base(), np.random.default_rng(4)
    br = _branches(60)
    data = _with(br, [StraussModel(base, radius_nm=700.0, log_gamma=np.log(0.1), burn_factor=30).sample_configuration(b, 25, rng) for b in br])
    radii = [300.0, 500.0, 700.0, 900.0]
    one = StraussModel(base).fit(data, radii=radii, n_angles=16)
    multi = MultiStraussModel(base).fit(data, radii=radii, n_shells=1, n_angles=16)
    assert multi.radii_nm == [one.radius_nm]
    assert abs(multi.log_gammas[0] - one.log_gamma) < 1e-3 and abs(multi.alpha - one.alpha) < 1e-3


def test_multi_strauss_recovers_two_shells_and_collapses_on_one():
    base, rng = _uniform_base(), np.random.default_rng(5)
    br = _branches(150)
    radii = [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0, 1800.0]
    # hard core 400 nm + soft repulsion to 1200 nm
    truth = MultiStraussModel(base, [400.0, 1200.0], [-20.0, np.log(0.4)], burn_factor=40)
    data = _with(br, [truth.sample_configuration(b, 20, rng) for b in br])
    fit = MultiStraussModel(base).fit(data, radii=radii, n_shells=2, n_angles=16)
    r1, r2 = fit.radii_nm
    g1, g2 = fit.gammas
    assert r1 in (200.0, 400.0, 600.0) and r2 in (1000.0, 1200.0, 1500.0), fit.radii_nm
    assert g1 < 0.1 and 0.15 < g2 < 0.7, fit.gammas
    assert fit.fit_info["log_pl_gain_over_one_radius"] > 5.0
    # data from a single Strauss: the second shell adds (almost) nothing
    single = _with(br, [StraussModel(base, radius_nm=800.0, log_gamma=np.log(0.05), burn_factor=40).sample_configuration(b, 20, rng) for b in br])
    fit1 = MultiStraussModel(base).fit(single, radii=radii, n_shells=2, n_angles=16)
    one_best = fit1.fit_info["best_by_n_shells"]["1"]
    assert one_best["radii_nm"][0] in (600.0, 800.0, 1000.0)
    assert fit1.fit_info["log_pl_gain_over_one_radius"] < 0.1 * one_best["log_pl_gain_over_poisson"]
    # the hard core of the true model is respected by the simulation
    s, th = truth.sample_configuration(br[0], 20, rng)
    from src.neuron_model.placement_generation.models.interaction import surface_distances
    r = np.full(len(s), 300.0)
    d = surface_distances(s, th, r, s, th, r)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 400.0
