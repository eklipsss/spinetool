"""T0 mean radius profile with a random effect of the neuron (``parabola_mix_neuron``, 2026-11-24).

Synthetic neurons: a thin distal component (r_L ~ 110 nm, probability growing with the soma distance) and a thick one
(~ 320 nm). A neuron shifts the log r_L of all its branches by a_n ~ N(0, 0.25^2) (level effect) and / or the logit of
the thin component by b_n ~ N(0, 1.5^2) (gate effect: neurons with many / few thin branches). The fit must recover the
spread of the effect that is there (and not invent the other), keep the marginal distribution when the effects are drawn
per branch, and, given the OTHER branches of a held-out neuron, predict the thickness of a branch better than the
conditions alone (the pairing defect of chain AK0)."""

from __future__ import annotations

import math

import numpy as np

from src.neuron_model.trunk_generation.conditions import DesignEncoder
from src.neuron_model.trunk_generation.models.t0_statistical import (ParabolicMixtureMean, ParabolicMixtureNeuronMean, T0Branch,
                                                                       T0Model)

from test_branch_preprocessing import mini  # noqa: F401  (fixture: synthetic mini dataset of the branch preprocessing)

SD_NEURON = 0.25
N_NODES, STEP, LENGTH = 31, 500.0, 15000.0


def _neurons(n_neurons: int, per_neuron: int, seed: int, enc: DesignEncoder = None, level_sd: float = SD_NEURON, gate_sd: float = 0.0,
             species_of=lambda i: "mouse"):
    rng = np.random.default_rng(seed)
    soma_all = rng.uniform(5000.0, 400000.0, n_neurons * per_neuron)
    enc = enc or DesignEncoder.fit([10000.0, 30000.0], soma_all)
    branches, true_log_rl = [], []
    w2 = ((N_NODES - 1 - np.arange(N_NODES)) / (N_NODES - 1)) ** 2
    for i in range(n_neurons):
        a = rng.normal(0.0, level_sd) if level_sd else 0.0
        bg = rng.normal(0.0, gate_sd) if gate_sd else 0.0
        for j in range(per_neuron):
            d = soma_all[i * per_neuron + j]
            p_thin = 1.0 / (1.0 + math.exp(-(math.log1p(d / 1000.0) - 5.0) * 2.0 - bg))
            base = math.log(110.0) if rng.random() < p_thin else math.log(320.0)
            log_rl = base + a + rng.normal(0.0, 0.12)
            ratio = math.exp(0.15 + rng.normal(0.0, 0.05))
            r = math.exp(log_rl) * (1.0 - w2) + math.exp(log_rl) * ratio * w2
            branches.append(T0Branch(rng.normal(0, 1e-4, N_NODES - 1), rng.normal(0, 1e-4, N_NODES - 1), np.log(r) + rng.normal(0, 0.005, N_NODES),
                                     STEP, enc.row(species_of(i), "excitatory", "apical", LENGTH, d, True), f"{species_of(i)}|excitatory|apical", f"n{seed}_{i}"))
            branches[-1].soma_mid = float(d)  # for the requests of the tests (the design uses the midpoint)
            true_log_rl.append(log_rl)
    return branches, np.asarray(true_log_rl), enc


def test_neuron_effect_spread_is_recovered_and_the_marginal_is_kept():
    branches, _, _ = _neurons(80, 25, seed=1)
    with_effect = ParabolicMixtureNeuronMean(n_components=2).fit(branches)
    plain = ParabolicMixtureMean(n_components=2).fit(branches)
    info = with_effect.fit_info["neuron_effect"]
    assert 0.17 < info["sd_log_r_L"] < 0.33, info
    assert info["sd_log_ratio"] < 0.1  # no effect on the ratio was put in
    assert info["tau_gate"] < 0.6, info  # nor on the gate
    # the within-neuron spread of the components shrinks once the neuron level is taken out
    assert max(info["sd_log_r_L_within_after"]) < max(info["sd_log_r_L_within_before"])
    # independent draws per branch: the same marginal distribution of log r_L as the model without the effect
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(4)
    xs = [b.design for b in branches]
    a = np.array([with_effect.sample_theta(x, rng_a)[0] for x in xs])
    b = np.array([plain.sample_theta(x, rng_b)[0] for x in xs])
    real = np.array([with_effect._fit_branch(br)[0] for br in branches])
    assert abs(a.std() - real.std()) < 0.06 and abs(b.std() - real.std()) < 0.06
    assert abs(np.mean(a < math.log(200.0)) - np.mean(real < math.log(200.0))) < 0.04


def test_posterior_of_the_other_branches_predicts_the_thickness_of_a_held_out_branch():
    train, _, enc = _neurons(80, 25, seed=1)
    model = ParabolicMixtureNeuronMean(n_components=2).fit(train)
    held, true_log_rl, _ = _neurons(30, 20, seed=2, enc=enc)
    by_neuron = {}
    for k, b in enumerate(held):
        by_neuron.setdefault(b.neuron, []).append(k)
    rng = np.random.default_rng(5)
    post, indep = [], []
    for k, b in enumerate(held):
        others = [held[j] for j in by_neuron[b.neuron] if j != k]  # leave the branch itself out
        post.append([model.sample_theta(b.design, rng, model.sample_posterior_effect(others, rng))[0] for _ in range(20)])
        indep.append([model.sample_theta(b.design, rng)[0] for _ in range(20)])
    post, indep = np.asarray(post), np.asarray(indep)  # [branch, draw]
    # conditional expectation (mean of 20 draws): the posterior adds the neuron level the conditions do not carry
    c_post, c_indep = np.corrcoef(post.mean(1), true_log_rl)[0, 1], np.corrcoef(indep.mean(1), true_log_rl)[0, 1]
    assert c_post > c_indep + 0.04, (c_post, c_indep)
    # thin real branches stay thin more often with the posterior than with independent draws


def test_gate_effect_is_recovered_and_puts_the_thin_branches_into_the_right_neurons():
    train, _, enc = _neurons(100, 25, seed=6, level_sd=0.0, gate_sd=1.5)
    model = ParabolicMixtureNeuronMean(n_components=2).fit(train)
    info = model.fit_info["neuron_effect"]
    assert 1.0 < info["tau_gate"] < 2.2, info
    assert info["sd_log_r_L"] < 0.12, info  # no level effect was put in
    held, true_log_rl, _ = _neurons(40, 20, seed=7, enc=enc, level_sd=0.0, gate_sd=1.5)
    by_neuron = {}
    for k, b in enumerate(held):
        by_neuron.setdefault(b.neuron, []).append(k)
    rng = np.random.default_rng(8)
    thin = true_log_rl < math.log(200.0)
    post = np.array([[model.sample_theta(b.design, rng, model.sample_posterior_effect([held[j] for j in by_neuron[b.neuron] if j != k], rng))[0]
                      for _ in range(10)] for k, b in enumerate(held)]) < math.log(200.0)
    indep = np.array([[model.sample_theta(b.design, rng)[0] for _ in range(10)] for b in held]) < math.log(200.0)
    # P(generated thin | real thin) up, P(generated thin | real thick) down
    assert post[thin].mean() > indep[thin].mean() + 0.015, (post[thin].mean(), indep[thin].mean())
    assert post[~thin].mean() < indep[~thin].mean() - 0.02, (post[~thin].mean(), indep[~thin].mean())


def test_t0_model_with_the_neuron_effect_samples_saves_and_loads(tmp_path):
    branches, _, enc = _neurons(40, 15, seed=3)
    model = T0Model("parabola_mix_neuron", direction="kappa", radius_residual="pooled").fit(branches, encoder=enc, max_iter=15)
    assert model.has_neuron_effect
    rng = np.random.default_rng(0)
    effect = model.posterior_neuron_effect([b for b in branches if b.neuron == branches[0].neuron][1:], rng)
    assert effect.shape == (3,)
    req = {"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical", "length_nm": LENGTH, "step_nm": STEP,
           "soma_distance_start_nm": 50000.0, "is_terminal": True}
    thin = model.sample([dict(req, t0_neuron_effect=[-1.0, 0.0, 0.0])] * 40, np.random.default_rng(1))
    thick = model.sample([dict(req, t0_neuron_effect=[+1.0, 0.0, 0.0])] * 40, np.random.default_rng(1))
    assert np.median([np.exp(r["log_radius"]).mean() for r in thin]) < 0.3 * np.median([np.exp(r["log_radius"]).mean() for r in thick])
    model.save(tmp_path / "t0.json")
    back = T0Model.load(tmp_path / "t0.json")
    assert back.has_neuron_effect and np.allclose(back.mean.neuron_cov, model.mean.neuron_cov) and back.mean.neuron_gate_sd == model.mean.neuron_gate_sd
    a = back.sample([dict(req, t0_neuron_effect=[0.3, 0.0])], np.random.default_rng(2))[0]["log_radius"]
    b = model.sample([dict(req, t0_neuron_effect=[0.3, 0.0])], np.random.default_rng(2))[0]["log_radius"]
    assert np.allclose(a, b)
    # models without the effect ignore it (old files keep working)
    plain = T0Model("parabola_mix", direction="kappa", radius_residual="pooled").fit(branches, encoder=enc, max_iter=15)
    assert not plain.has_neuron_effect and plain.fresh_neuron_effect(rng) is None
    assert len(plain.sample([dict(req, t0_neuron_effect=[1.0, 0.0])], rng)) == 1


def test_fit_report_neuron_thin_diagnostic_shows_the_pairing_gain():
    from src.neuron_model.trunk_generation.fit import _neuron_thin

    # the gate effect makes the thin share differ between neurons (the level effect alone barely does)
    train, _, enc = _neurons(60, 20, seed=4, gate_sd=1.5)
    val, _, _ = _neurons(25, 20, seed=5, enc=enc, gate_sd=1.5)
    req = [{"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical", "length_nm": b.length, "step_nm": b.step,
            "soma_distance_start_nm": b.soma_mid - 0.5 * b.length, "is_terminal": True} for b in val]
    assert np.allclose(enc.row("mouse", "excitatory", "apical", val[0].length, req[0]["soma_distance_start_nm"] + 0.5 * val[0].length, True), val[0].design)
    model = T0Model("parabola_mix_neuron", direction="kappa", radius_residual="pooled").fit(train, encoder=enc, max_iter=10)
    rows = _neuron_thin(model, req, val, [b.neuron for b in val], seed=0)
    assert set(rows["per_neuron"]) == {"real", "independent", "per_neuron"} and set(rows["pairing"]) == {"independent", "posterior"}
    # one effect per neuron spreads the thin share over the neurons more than independent draws
    assert rows["per_neuron"]["per_neuron"]["sd"] > rows["per_neuron"]["independent"]["sd"]
    assert rows["pairing"]["posterior"]["p_thin_given_real_thin"] > rows["pairing"]["independent"]["p_thin_given_real_thin"]


def test_generate_tp_neuron_effects_per_request(mini, tmp_path):
    """``generate_tp.py --t0-neuron``: one shared fresh effect per source neuron, or the posterior from the OTHER branches."""
    import importlib.util
    from pathlib import Path

    import pandas as pd

    from src.neuron_model import branch_data as bd
    from src.neuron_model.branch_preprocessing import run

    spec = importlib.util.spec_from_file_location("generate_tp_script", Path(__file__).resolve().parents[2] / "scripts/neuron_model/generate_tp.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)

    cfg = mini["cfg"]
    run(cfg)
    table, _, pack = bd.prepare_training_data(cfg.out_root, tmp_path / "pack", io_threads=1)
    table["split"] = "val"
    key = bd.branch_key(bd.select(table, eligible="train_eligible_T").iloc[0])
    branches, _, enc = _neurons(30, 10, seed=9)
    model = T0Model("parabola_mix_neuron", direction="kappa", radius_residual="pooled").fit(branches, encoder=enc, max_iter=5)
    requests = [{"source_branch": key}, {"source_branch": key}]
    rng = np.random.default_rng(0)
    per = gt.neuron_effects_for_requests(model, requests, "per_neuron", table, pack, "val", None, rng)
    assert len(per["effects"]) == 2 and per["effects"][0] == per["effects"][1] and len(per["effects"][0]) == 3
    post = gt.neuron_effects_for_requests(model, requests, "posterior", table, pack, "val", None, rng)
    assert post["n_with_other_branches"] == 0 and len(post["effects"][0]) == 3  # the only branch is the source itself: prior draw
    assert isinstance(pd.Series(post["effects"]).iloc[0], list)


def test_independent_draws_keep_the_mean_even_from_a_shifted_start():
    """Chain AN (2026-11-24): on Minnie65 + H01 the mouse part of the fixed effects ended ~0.25 too high in log r_L with
    the effects of the mouse neurons ~0.25 too low (independent draws +100 nm for Minnie65, -8 nm for H01, the posterior
    draws exact). The PX-EM centring keeps the effects orthogonal to the neuron-level columns of the design (intercept,
    species, ...): started with the intercept shifted by +0.3 and the species column by -0.3, the fit must come back, and
    independent draws must reproduce the median of log r_L in BOTH species."""
    branches, _, enc = _neurons(80, 40, seed=11, level_sd=0.26, gate_sd=2.0, species_of=lambda i: "human" if i % 4 == 0 else "mouse")
    human_col = 2  # 'species=human' (after the intercept and z_log_length)
    assert all(b.design[human_col] == (1.0 if b.group.startswith("human") else 0.0) for b in branches)
    model = ParabolicMixtureNeuronMean(n_components=2)
    ParabolicMixtureMean.fit(model, branches)          # the fit without the effects
    model.coefs[:, 0, 0] += 0.3                          # a deliberately wrong start of the fixed part
    model.coefs[:, human_col, 0] -= 0.3
    theta = np.vstack([model._fit_branch(b) for b in branches])
    x = np.vstack([b.design for b in branches])
    model._fit_neuron_effect(theta, x, [b.neuron for b in branches], {"k": 2})
    rng = np.random.default_rng(0)
    gen = np.array([model.sample_theta(b.design, rng)[0] for b in branches])
    for sp in ("mouse", "human"):
        sel = np.array([b.group.startswith(sp) for b in branches])
        assert abs(np.median(gen[sel]) - np.median(theta[sel, 0])) < 0.06, sp
        names = sorted({b.neuron for b in branches if b.group.startswith(sp)})
        post = [model.neuron_posterior([b for b in branches if b.neuron == n])["a_mean"][0] for n in names]
        assert abs(np.mean(post)) < 0.04, sp
