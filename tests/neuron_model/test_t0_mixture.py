"""T0 mean profile with a mixture of experts for the pair (log r_L, log r_0/r_L): ParabolicMixtureMean."""

from __future__ import annotations

import math

import numpy as np

from src.neuron_model.trunk_generation.models.t0_statistical import MEAN_MODELS, ParabolicMean, ParabolicMixtureMean, T0Branch


def _synthetic(n, rng):
    """Design [1, z_soma, terminal]; a thin population (r_L ~ 110 nm) whose probability grows with the soma distance and
    the terminal flag, and a typical one (r_L ~ 300 nm, slightly thinner distally)."""
    branches, thin = [], []
    for _ in range(n):
        z, term = rng.standard_normal(), float(rng.random() < 0.4)
        p_thin = 1.0 / (1.0 + math.exp(-(-2.0 + 1.8 * z + 1.0 * term)))
        is_thin = rng.random() < p_thin
        if is_thin:
            theta = rng.multivariate_normal([math.log(110.0), 0.25], np.diag([0.12, 0.05]) ** 2)
        else:
            theta = rng.multivariate_normal([math.log(300.0) - 0.08 * z, 0.35], np.diag([0.18, 0.1]) ** 2)
        n_nodes = 40
        w2 = ((n_nodes - 1 - np.arange(n_nodes)) / (n_nodes - 1)) ** 2
        r = math.exp(theta[0]) * (1 - w2) + math.exp(theta[0] + theta[1]) * w2
        log_r = np.log(r) + rng.normal(0, 0.005, n_nodes)
        branches.append(T0Branch(np.zeros(n_nodes - 1), np.zeros(n_nodes - 1), log_r, 500.0, np.array([1.0, z, term]), "g"))
        thin.append(is_thin)
    return branches, np.array(thin)


def _sampled_r_l(model, design, n, rng):
    return np.array([math.exp(model.sample_log_mean(design, 40, rng)[-1]) for _ in range(n)])


def test_mixture_recovers_the_thin_distal_population_that_one_gaussian_misses():
    rng = np.random.default_rng(0)
    branches, _ = _synthetic(3000, rng)
    mix = ParabolicMixtureMean(n_components="auto", max_components=3).fit(branches)
    one = ParabolicMean().fit(branches)
    assert mix.fit_info["chosen_k"] >= 2
    assert mix.fit_info["per_k"]["2"]["bic"] < mix.fit_info["per_k"]["1"]["bic"] - 100
    # share of thin (r_L <= 150 nm) distal terminal branches (true p_thin = sigmoid(-2 + 1.8*1.5 + 1) = 0.85)
    distal = np.array([1.0, 1.5, 1.0])
    proximal = np.array([1.0, -1.0, 0.0])  # true p_thin = sigmoid(-3.8) = 0.02
    gen = np.random.default_rng(1)
    thin_mix_distal = np.mean(_sampled_r_l(mix, distal, 2000, gen) <= 150.0)
    thin_one_distal = np.mean(_sampled_r_l(one, distal, 2000, gen) <= 150.0)
    thin_mix_proximal = np.mean(_sampled_r_l(mix, proximal, 2000, gen) <= 150.0)
    assert abs(thin_mix_distal - 0.85) < 0.08
    assert thin_mix_proximal < 0.06
    assert thin_one_distal < thin_mix_distal - 0.2  # one Gaussian cannot put the mass there


def test_mixture_with_one_component_is_the_gaussian_and_round_trips():
    rng = np.random.default_rng(2)
    branches, _ = _synthetic(800, rng)
    one = ParabolicMean().fit(branches)
    mix1 = ParabolicMixtureMean(n_components=1).fit(branches)
    assert np.allclose(mix1.coefs[0], one.coef, atol=1e-6)
    assert np.allclose(mix1.covs[0], one.cov, atol=2e-3)  # ML vs unbiased covariance + floor
    mix = ParabolicMixtureMean(n_components=2).fit(branches)
    again = MEAN_MODELS["parabola_mix"].from_dict(mix.to_dict())
    d = np.array([1.0, 0.5, 1.0])
    assert np.allclose(again.component_probabilities(d), mix.component_probabilities(d))
    a = again.sample_log_mean(d, 20, np.random.default_rng(5))
    b = mix.sample_log_mean(d, 20, np.random.default_rng(5))
    assert np.allclose(a, b)


def test_t0_model_with_the_mixture_mean_saves_and_loads(tmp_path):
    from src.neuron_model.trunk_generation.conditions import DesignEncoder
    from src.neuron_model.trunk_generation.models.t0_statistical import T0Model

    rng = np.random.default_rng(4)
    rows, branches = [], []
    for i in range(150):
        n_nodes = 30
        log_r = np.full(n_nodes, math.log(250.0 if i % 3 else 110.0)) + rng.normal(0, 0.01, n_nodes)
        k = rng.normal(0, 1e-4, (2, n_nodes - 1))
        rows.append(({"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical"}, 15000.0, log_r, k))
    enc = DesignEncoder.fit([r[1] for r in rows])
    for cond, length, log_r, k in rows:
        branches.append(T0Branch(k[0], k[1], log_r, length / (len(log_r) - 1), enc.row(cond["species"], cond["cell_type_binary"], cond["compartment"], length), "g"))
    model = T0Model("parabola_mix", direction="kappa", radius_residual="pooled").fit(branches, encoder=enc, max_iter=20)
    model.save(tmp_path / "t0.json")
    loaded = T0Model.load(tmp_path / "t0.json")
    assert loaded.mean.kind == "parabola_mix"
    req = [{"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical", "length_nm": 15000.0}]
    assert np.allclose(model.sample(req, np.random.default_rng(1))[0]["log_radius"], loaded.sample(req, np.random.default_rng(1))[0]["log_radius"])
