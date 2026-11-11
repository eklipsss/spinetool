"""Neuron-effect posterior of the count model and the branch-level calibration (2026-11-11), synthetic data."""

import numpy as np

from src.neuron_model.placement_generation.calibration import branch_calibration, spine_weighted_density
from src.neuron_model.placement_generation.covariates import BranchCovariates, CountDesign
from src.neuron_model.placement_generation.models.count import CountModel, build_count_data


def _branches(rng, prefix, n_neurons, dens=0.3, sigma_b=0.4, per=30, shift=0.0, hot_share=0.0, hot_factor=1.0, k=None):
    out, effects = [], []
    for n in range(n_neurons):
        eff = rng.normal(shift, sigma_b)
        effects.append(eff)
        for _ in range(per):
            length = float(np.exp(rng.uniform(np.log(5000), np.log(60000))))
            radius = rng.uniform(200, 400)
            mu = length / 1e3 * dens * np.exp(eff) * (hot_factor if rng.random() < hot_share else 1.0)
            y = int(rng.poisson(mu if k is None else rng.gamma(k, mu / k)))
            out.append(BranchCovariates(f"{prefix}{n}", y, length, 2 * np.pi * radius * length, radius, 50000.0, 1.1, False, "mouse", "excitatory", None))
    return out, np.array(effects)


def test_spine_weighted_density():
    assert np.isclose(spine_weighted_density(np.array([2, 0, 4]), np.array([1.0, 5.0, 2.0])), (4 / 1 + 16 / 2) / 6)
    assert np.isnan(spine_weighted_density(np.zeros(3), np.ones(3)))


def test_posterior_recovers_neuron_effects():
    rng = np.random.default_rng(1)
    train, effects = _branches(rng, "n", 60, sigma_b=0.5, k=8.0)
    design = CountDesign().fit(train)
    model = CountModel("nb", random_effect=True).fit(build_count_data(design, train))
    data = build_count_data(design, train)
    mean = model.posterior_neuron_effects(data)
    centred = effects - effects.mean()
    assert np.corrcoef(mean, centred)[0, 1] > 0.9
    draws = np.vstack([model.posterior_neuron_effects(data, np.random.default_rng(s)) for s in range(40)])
    assert np.abs(draws.mean(0) - mean).max() < 0.1  # draws centred on the posterior mean
    assert 0.02 < draws.std(0).mean() < 0.3  # 30 branches per neuron: a narrow but non-degenerate posterior
    effects_grid, log_post = model.neuron_log_posterior(data)
    assert np.allclose(np.exp(log_post).sum(1), 1.0)


def test_branch_calibration_separates_neuron_level_from_branch_level():
    rng = np.random.default_rng(2)
    train, _ = _branches(rng, "t", 60, k=8.0)
    design = CountDesign().fit(train)
    model = CountModel("nb", random_effect=True).fit(build_count_data(design, train))
    # held-out neurons poorer than the population: off with fresh effects, right with the posterior ones
    # (80 branches per neuron: the shrinkage of the posterior towards the population is ~ 1 / (1 + sigma_b^2 * sum mu)
    # of the neuron's own log-ratio, small only when the neuron has many spines)
    poor, _ = _branches(rng, "v", 15, shift=-0.7, k=8.0, per=80)
    row = branch_calibration(model, design, poor, split="val", n_sim=60).set_index(["factor", "level"]).loc[("all", "all")]
    assert row["obs_density"] < row["fresh_density_p05"]
    assert row["fresh_density"] / row["obs_density"] > 1.5
    assert abs(row["posterior_density"] / row["obs_density"] - 1) < 0.05  # small shrinkage towards the population only
    # branch-level defect: a Poisson model (no within-neuron overdispersion) on data where 20 % of branches are 4x denser
    hot, _ = _branches(rng, "h", 60, hot_share=0.2, hot_factor=4.0)
    pois = CountModel("poisson", random_effect=True).fit(build_count_data(design, hot))
    table = branch_calibration(pois, design, hot, split="train", n_sim=60).set_index(["factor", "level"])
    r = table.loc[("all", "all")]
    assert r["obs_sw_density"] > r["posterior_sw_density_p95"]  # the heavy upper tail is missed even given the neuron
    assert set(table.index.get_level_values("factor")) >= {"radius", "length", "soma_distance", "terminal", "compartment", "cell_type_binary"}
