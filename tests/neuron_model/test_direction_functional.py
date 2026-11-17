"""Fit of the T0 direction model to per-branch functional distributions (direction_functional.py)."""

import numpy as np
import pytest

from src.neuron_model.trunk_generation.fit import functionals
from src.neuron_model.trunk_generation.models.direction import Amplitude
from src.neuron_model.trunk_generation.models.direction_functional import (FunctionalSimulator, _bending_loop, bending, fit_to_functionals,
                                                                            functionals_from_tangents, quantile_loss, quantile_targets, real_functionals)
from src.neuron_model.trunk_generation.models.direction_ms import DirectionMultiScale


def _truth():
    return DirectionMultiScale(sd_fast=0.12, length_fast_nm=1500.0, noise_sd=0.004, drive_std=2e-5, drive_length_nm=1500.0, restore_nm=6000.0,
                               amplitude=Amplitude(0.0, -1.0, 0.25, {}))


class _B:
    def __init__(self, k1, k2, r, step, neuron):
        self.kappa1, self.kappa2, self.step, self.group, self.neuron = k1, k2, step, "basal", neuron
        self.log_radius = np.full(len(k1) + 1, np.log(r))

    @property
    def n_nodes(self):
        return len(self.log_radius)

    @property
    def length(self):
        return (self.n_nodes - 1) * self.step


def _branches(model, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        step = float(rng.choice([250.0, 500.0]))
        nodes = int(rng.integers(20, 160))
        r = float(np.exp(rng.normal(np.log(300), 0.3)))
        k1, k2 = model.sample_bending(nodes, step, "basal", rng, radius_nm=r)
        out.append(_B(k1, k2, r, step, f"n{i // 10}"))
    return out


def test_fast_bending_matches_the_reference_loop_and_fit_functionals():
    rng = np.random.default_rng(0)
    model = _truth()
    k1, k2 = model.sample_bending(80, 500.0, "basal", rng, radius_nm=300.0)
    from src.neuron_model.trunk_generation.models.direction import node_tangents

    _, t = node_tangents(k1, k2, 500.0)
    a, b = bending(t, 500.0)
    c, d = _bending_loop(t, 500.0)
    assert np.allclose(a, c, atol=1e-12) and np.allclose(b, d, atol=1e-12)
    assert np.allclose(np.hypot(a, b), np.hypot(k1, k2), rtol=1e-6)       # same bending up to the choice of the initial frame
    ref = functionals(k1, k2, np.zeros(len(k1) + 1), 500.0)
    lk, lt, ac = real_functionals(k1, k2, 500.0)
    assert np.exp(lk) == pytest.approx(ref["kappa_rms_per_nm"], rel=1e-9)
    assert np.exp(lt) + 1 == pytest.approx(ref["tortuosity"], rel=1e-6)
    assert ac == pytest.approx(ref["kappa_autocorr_1um"], rel=1e-9)
    gk, gt, _ = functionals_from_tangents(t, 500.0)                       # the simulator path gives the same rms and tortuosity
    assert gk == pytest.approx(lk, abs=1e-6) and gt == pytest.approx(lt, abs=1e-4)


def test_simulator_reproduces_the_law_of_sample_bending():
    model = _truth()
    real = _branches(model, 300, 1)
    req = [{"n_nodes": b.n_nodes, "step": b.step, "group": b.group, "radius_nm": float(np.exp(b.log_radius[0]))} for b in real]
    sim = FunctionalSimulator(req, seed=3).simulate(model.theta, model.amplitude)
    obs = np.array([real_functionals(b.kappa1, b.kappa2, b.step) for b in real])
    for k in range(3):   # same law: medians agree within sampling noise
        assert np.nanmedian(sim[:, k]) == pytest.approx(np.nanmedian(obs[:, k]), abs=0.25 * np.nanstd(obs[:, k]))


def test_functional_fit_reduces_the_loss_from_a_wrong_start():
    truth = _truth()
    real = _branches(truth, 240, 2)
    start = _truth()
    start.sd_fast, start.restore_nm = 0.3, 2.0e4          # too much wiggle, too little restoring
    start.amplitude = Amplitude(0.3, 0.0, 0.1, {})
    info = fit_to_functionals(start, real, n_branches=240, maxfev=150, seed=0)
    assert info["loss_end"] < 0.5 * info["loss_start"]


def _radius_truth(gamma):
    m = _truth()
    m.length_fast_radius_exponent = gamma
    return m


def _wide_radius_branches(model, n, seed):
    """Branches with radii 150-600 nm (both radius strata well filled)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        step = float(rng.choice([250.0, 500.0]))
        nodes = int(rng.integers(40, 160))
        r = float(np.exp(rng.uniform(np.log(150.0), np.log(600.0))))
        k1, k2 = model.sample_bending(nodes, step, "basal", rng, radius_nm=r)
        out.append(_B(k1, k2, r, step, f"n{i // 10}"))
    return out


def test_radius_dependent_wiggle_scale_changes_the_autocorrelation_and_round_trips():
    model = _radius_truth(1.5)
    assert model.length_fast_at(300.0) == pytest.approx(1500.0)
    assert model.length_fast_at(600.0) == pytest.approx(1500.0 * 2**1.5)
    assert model.length_fast_at(None) == pytest.approx(1500.0)
    real = _wide_radius_branches(model, 300, 4)
    obs = np.array([real_functionals(b.kappa1, b.kappa2, b.step) for b in real])
    r = np.array([np.exp(b.log_radius[0]) for b in real])
    assert np.nanmedian(obs[r >= 300, 2]) > np.nanmedian(obs[r < 300, 2]) + 0.1   # thick branches bend more persistently
    req = [{"n_nodes": b.n_nodes, "step": b.step, "group": b.group, "radius_nm": float(rr)} for b, rr in zip(real, r)]
    sim = FunctionalSimulator(req, seed=3).simulate(model.theta, model.amplitude, model.length_fast_radius_exponent)
    for mask in (r < 300, r >= 300):   # the simulator follows the same law per radius class
        assert np.nanmedian(sim[mask, 2]) == pytest.approx(np.nanmedian(obs[mask, 2]), abs=0.08)
    loaded = DirectionMultiScale.from_dict(model.to_dict())
    assert loaded.length_fast_radius_exponent == pytest.approx(1.5)
    old = model.to_dict()
    old.pop("length_fast_radius_exponent")
    assert DirectionMultiScale.from_dict(old).length_fast_radius_exponent == 0.0   # files written before 2026-11-17


def test_functional_fit_recovers_the_radius_exponent():
    truth = _radius_truth(1.5)
    real = _wide_radius_branches(truth, 300, 5)
    shared = _radius_truth(0.0)
    with_gamma = _radius_truth(0.0)
    a = fit_to_functionals(shared, real, n_branches=300, maxfev=200, seed=0)
    b = fit_to_functionals(with_gamma, real, n_branches=300, maxfev=200, seed=0, fit_length_radius=True)
    assert shared.length_fast_radius_exponent == 0.0
    assert b["loss_end"] < a["loss_end"]
    assert with_gamma.length_fast_radius_exponent > 0.7
