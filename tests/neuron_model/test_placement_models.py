import numpy as np
import pytest
from scipy import stats

from src.neuron_model.placement_generation.covariates import BranchCovariates, CountDesign
from src.neuron_model.placement_generation.models.count import CountModel, build_count_data
from src.neuron_model.placement_generation.models.position import PositionBranch, PositionModel, sample_angles


def _branches(rng, n_neurons=120, per=8, sigma_b=0.0):
    out, truth = [], []
    for n in range(n_neurons):
        eff = rng.normal(0, sigma_b) if sigma_b else 0.0
        for _ in range(per):
            length = float(np.exp(rng.uniform(np.log(5000), np.log(80000))))
            radius = rng.uniform(150, 400)
            term = bool(rng.random() < 0.4)
            comp = "apical" if rng.random() < 0.5 else "basal"
            out.append(BranchCovariates(f"n{n}", -1, length, 2 * np.pi * radius * length, radius, float(rng.uniform(5000, 150000)), 1.1, term, "mouse", "excitatory", comp))
            truth.append(eff)
    return out, np.array(truth)


def _simulate(rng, family, sigma_b, k=5.0, pi=0.0, dens=0.2):
    cov, eff = _branches(rng, sigma_b=sigma_b)
    design = CountDesign().fit(cov)
    basal = np.array([c.compartment == "basal" for c in cov], dtype=float)
    mu = np.array([c.length_nm / 1e3 for c in cov]) * dens * np.exp(0.3 * basal + eff)
    y = rng.poisson(rng.gamma(k, mu / k)) if family != "poisson" else rng.poisson(mu)
    if pi:
        y = np.where(rng.random(len(y)) < pi, 0, y)
    for c, v in zip(cov, y):
        c.n_spines = int(v)
    return cov, design, basal


def test_nb_recovers_density_dispersion_and_covariate_effect():
    rng = np.random.default_rng(0)
    cov, design, basal = _simulate(rng, "nb", sigma_b=0.0, k=5.0)
    data = build_count_data(design, cov)
    m = CountModel("nb", random_effect=False).fit(data)
    mean = m.mean(data)
    assert mean.sum() == pytest.approx(data.y.sum(), rel=0.03)
    ratio = mean[basal == 1].sum() / mean[basal == 0].sum() / (data.y[basal == 1].sum() / data.y[basal == 0].sum())
    assert ratio == pytest.approx(1.0, rel=0.1)
    assert 3.0 < m.dispersion < 8.0


def test_random_effect_variance_is_recovered_and_improves_likelihood():
    rng = np.random.default_rng(1)
    cov, design, _ = _simulate(rng, "nb", sigma_b=0.5, k=20.0)
    data = build_count_data(design, cov)
    with_re = CountModel("nb", random_effect=True).fit(data)
    without = CountModel("nb", random_effect=False).fit(data)
    assert with_re.sigma_b == pytest.approx(0.5, abs=0.15)
    assert with_re.fit_info["nll"] < without.fit_info["nll"] - 5


def test_quadrature_converges_and_nb_tends_to_poisson():
    rng = np.random.default_rng(2)
    cov, design, _ = _simulate(rng, "poisson", sigma_b=0.0)
    data = build_count_data(design, cov)
    m = CountModel("nb", random_effect=True, n_quad=15).fit(data, max_iter=100)
    nll15 = m.neg_loglik(data)
    m.n_quad = 40
    assert m.neg_loglik(data) == pytest.approx(nll15, rel=1e-6)
    assert m.dispersion > 100  # Poisson data: k -> infinity (the optimiser pushes it up)


def test_zinb_recovers_excess_zeros_and_hurdle_samples_are_consistent():
    rng = np.random.default_rng(3)
    cov, design, _ = _simulate(rng, "nb", sigma_b=0.0, k=8.0, pi=0.3)
    data = build_count_data(design, cov)
    z = CountModel("zinb", random_effect=False).fit(data)
    nb = CountModel("nb", random_effect=False).fit(data)
    assert z.fit_info["nll"] < nb.fit_info["nll"] - 20
    drawn = z.sample(data, np.random.default_rng(0))
    assert (drawn == 0).mean() == pytest.approx((data.y == 0).mean(), abs=0.04)
    h = CountModel("hurdle", random_effect=False).fit(data)
    hs = h.sample(data, np.random.default_rng(0))
    assert (hs == 0).mean() == pytest.approx((data.y == 0).mean(), abs=0.05)
    assert hs.mean() == pytest.approx(data.y.mean(), rel=0.15)


def test_count_persistence_and_residuals(tmp_path):
    rng = np.random.default_rng(4)
    cov, design, _ = _simulate(rng, "nb", sigma_b=0.0)
    data = build_count_data(design, cov)
    m = CountModel("nb", random_effect=False).fit(data)
    m.save(tmp_path / "c.json")
    m2 = CountModel.load(tmp_path / "c.json")
    assert np.allclose(m.mean(data), m2.mean(data))
    r = m.randomized_quantile_residuals(data, np.random.default_rng(0))
    assert abs(r.mean()) < 0.1 and r.std() == pytest.approx(1.0, abs=0.1)


def _pos_branch(rng, length=30000.0, soma=20000.0):
    s = np.linspace(0, length, int(length / 500) + 1)
    return PositionBranch(s, 250 + 80 * np.sin(s / 6000), np.abs(rng.normal(1e-4, 5e-5, len(s))), soma)


def test_position_model_recovers_intensity_and_samples_match_density():
    rng = np.random.default_rng(5)
    branches = [_pos_branch(rng, rng.uniform(15000, 60000), rng.uniform(5000, 100000)) for _ in range(150)]
    true = lambda b, s: 1.5 * np.interp(s, b.s, (b.radius - 250) / 80) + 2.0 * ((b.soma_start_nm + s) < 40000)
    for b in branches:
        grid = np.linspace(0, b.length, 4000)
        lam = np.exp(true(b, grid))
        cdf = np.r_[0, np.cumsum(0.5 * (lam[1:] + lam[:-1]))]
        cdf /= cdf[-1]
        b.positions = np.sort(np.interp(rng.random(60), cdf, grid))
    m = PositionModel(use=("radius", "soma")).fit(branches)
    assert m.beta[m.names.index("radius")] > 0.5
    flat = PositionModel(use=()).fit(branches)
    assert m.loglik(branches) > flat.loglik(branches) + 20
    b = branches[0]
    cdf = m.cdf_on_nodes(b)
    draws = m.sample_positions(b, 4000, np.random.default_rng(0))
    assert stats.kstest(np.interp(draws, b.s, cdf), "uniform").pvalue > 0.01


def test_constant_intensity_gives_uniform_positions_and_uniform_angles():
    rng = np.random.default_rng(6)
    branches = []
    for _ in range(80):
        b = _pos_branch(rng)
        b.radius = np.full_like(b.radius, 250.0)
        b.kappa = np.full_like(b.kappa, 1e-4)
        b.positions = rng.uniform(0, b.length, 40)
        branches.append(b)
    m = PositionModel(use=()).fit(branches)
    draws = np.concatenate([m.sample_positions(branches[0], 40, np.random.default_rng(i)) for i in range(50)])
    assert stats.kstest(draws / branches[0].length, "uniform").pvalue > 0.01
    th = sample_angles(5000, np.random.default_rng(0))
    assert th.min() > -np.pi - 1e-12 and th.max() <= np.pi
    assert abs(np.exp(1j * th).mean()) < 0.05


def _toy_trunk(length=20000.0, radius=300.0):
    from src.neuron_model.trunk_generation.representation import geometry_from_features

    m = int(length / 500) + 1
    geom = geometry_from_features(np.full(m - 1, 2e-5), np.zeros(m - 1), np.full(m, np.log(radius)), length / (m - 1))
    return {"s": geom["s"], "gamma": geom["gamma"], "radius": geom["radius"], "normal": geom["e1"], "binormal": geom["e2"],
            "kappa1": geom["kappa1"], "kappa2": geom["kappa2"], "length": geom["length"]}


def test_generated_anchors_lie_on_the_tube_and_respect_dmin():
    from src.neuron_model.placement_generation.generate import generate_placements

    rng = np.random.default_rng(7)
    branches = [_pos_branch(rng) for _ in range(20)]
    for b in branches:
        b.positions = rng.uniform(0, b.length, 20)
    pm = PositionModel(use=()).fit(branches)
    trunk = _toy_trunk()
    rec = generate_placements([trunk], [25], pm, d_min_nm=300.0, seed=3)[0]
    assert rec["n_spines"] == 25 and rec["d_min_satisfied"] and rec["min_distance_nm"] >= 300.0
    # distance from each anchor to the centreline equals the local radius
    dense = np.linspace(0, trunk["length"], 2000)
    gamma = np.stack([np.interp(dense, trunk["s"], trunk["gamma"][:, k]) for k in range(3)], axis=1)
    d = np.linalg.norm(rec["anchor"][:, None, :] - gamma[None], axis=2).min(axis=1)
    assert np.allclose(d, 300.0, atol=8.0)
    assert np.allclose(np.linalg.norm(rec["normal"], axis=1), 1.0)
    assert np.all(np.diff(rec["s"]) >= 0) and rec["u"].max() <= 1.0
