import math

import numpy as np
import pytest

from src.neuron_model.trunk_generation.conditions import DesignEncoder, group_key
from src.neuron_model.trunk_generation.constraints import TrunkLimits
from src.neuron_model.trunk_generation.generate import generate_trunks
from src.neuron_model.trunk_generation.models.t0_statistical import GamMean, ParabolicMean, T0Branch, T0Model

TRUE_L = np.array([3000.0, 1500.0, 6000.0])  # correlation lengths of (kappa1, kappa2, rho), nm
TRUE_MU = np.array([2.0e-5, -1.0e-5])
TRUE_SIGMA = np.array(
    [[4.0e-9, 0.5e-9, 0.6e-9 * 0.0],
     [0.5e-9, 3.0e-9, -0.4e-9 * 0.0],
     [0.0, 0.0, 0.0]]
)
# stationary covariance (kappa in nm^-1, rho dimensionless); the last block couples bending and thickness
SIGMA = np.array([[4.0e-9, 0.5e-9, 1.0e-6], [0.5e-9, 3.0e-9, -0.6e-6], [1.0e-6, -0.6e-6, 0.04]])


def _simulate(rng, n_branches, step_choices=(500.0,), lengths=(8000.0, 70000.0), groups=(("mouse", "excitatory", "apical"),), L=TRUE_L, mu=TRUE_MU, sigma=SIGMA, mean="parabola"):
    enc = DesignEncoder.fit(list(lengths))
    out = []
    chol = np.linalg.cholesky(sigma)
    for i in range(n_branches):
        species, ctb, comp = groups[i % len(groups)]
        length = float(np.exp(rng.uniform(math.log(lengths[0]), math.log(lengths[1]))))
        n_nodes = max(int(math.ceil(length / rng.choice(step_choices))) + 1, 6)
        step = length / (n_nodes - 1)
        phi = np.exp(-step / L)
        q = sigma * (1.0 - np.outer(phi, phi))
        cq = np.linalg.cholesky(q)
        z = np.zeros((n_nodes, 3))
        z[0] = chol @ rng.standard_normal(3)
        for j in range(n_nodes - 1):
            z[j + 1] = phi * z[j] + cq @ rng.standard_normal(3)
        u = np.arange(n_nodes) / (n_nodes - 1)
        r_l = rng.uniform(150.0, 300.0)
        ratio = rng.uniform(1.0, 1.8)
        m = np.log(r_l * (1 - (1 - u) ** 2) + r_l * ratio * (1 - u) ** 2)
        out.append(T0Branch(mu[0] + z[:-1, 0], mu[1] + z[:-1, 1], m + z[:, 2], step, enc.row(species, ctb, comp, length), group_key(species, ctb, comp)))
    return out, enc


def test_ou_parameters_are_recovered():
    rng = np.random.default_rng(0)
    branches, enc = _simulate(rng, 600)
    model = T0Model("parabola").fit(branches, encoder=enc, max_iter=300)
    p = model.params_for(branches[0].group)
    # bending is not touched by the mean-profile fit: its memory, variance and mean are recovered
    assert np.allclose(p.length_scales[:2], TRUE_L[:2], rtol=0.15)
    assert np.allclose(np.diag(p.sigma)[:2], np.diag(SIGMA)[:2], rtol=0.15)
    assert np.allclose(p.mu, TRUE_MU, atol=2.5e-6)
    assert p.sigma[0, 1] == pytest.approx(SIGMA[0, 1], abs=0.2e-9)
    # rho is the residual about EACH branch's own parabola (two free parameters per branch absorb the slow part of
    # the radius variation, which the sampled pair (r_L, r_0 / r_L) carries instead): shorter memory, smaller variance,
    # same sign of the coupling with the bending
    assert 0.3 * TRUE_L[2] < p.length_scales[2] < 1.1 * TRUE_L[2]
    assert p.sigma[2, 2] < SIGMA[2, 2] * 1.1
    assert np.sign(p.sigma[0, 2]) == np.sign(SIGMA[0, 2])


def test_length_scales_do_not_depend_on_the_discretisation_step():
    rng = np.random.default_rng(1)
    fine, enc = _simulate(rng, 300, step_choices=(250.0,))
    coarse, enc2 = _simulate(rng, 300, step_choices=(1000.0,))
    ls = [T0Model("parabola").fit(b, encoder=e, max_iter=300).params_for(b[0].group).length_scales for b, e in ((fine, enc), (coarse, enc2))]
    # the bending memory is a property of the physical process, not of the sampling step (the radius residual is not:
    # it depends on how much of the slow variation the per-branch profile absorbs, which depends on the step)
    assert np.allclose(ls[0][:2], TRUE_L[:2], rtol=0.2) and np.allclose(ls[1][:2], TRUE_L[:2], rtol=0.2)


def test_group_shrinkage_separates_large_groups_and_keeps_small_close_to_pooled():
    rng = np.random.default_rng(2)
    a, enc = _simulate(rng, 400, groups=(("mouse", "excitatory", "apical"),), L=np.array([2500.0, 1500.0, 6000.0]))
    b, _ = _simulate(rng, 400, groups=(("mouse", "inhibitory", "basal"),), L=np.array([6000.0, 1500.0, 6000.0]))
    small, _ = _simulate(rng, 20, groups=(("mouse", "excitatory", "uncertain"),), L=np.array([6000.0, 1500.0, 6000.0]))
    model = T0Model("parabola").fit(a + b + small, encoder=enc, max_iter=300, shrinkage_std=0.3, min_group=10)
    la, lb = (model.params_for(g).length_scales[0] for g in ("mouse|excitatory|apical", "mouse|inhibitory|basal"))
    assert la < 3500 and lb > 5000  # the two big groups keep their own bending length (true: 2500 and 6000 nm)
    pooled = math.exp(model.pooled[0])
    ls = model.params_for("mouse|excitatory|uncertain").length_scales[0]
    assert abs(math.log(ls / pooled)) < abs(math.log(6000.0 / pooled)) + 0.05  # the small group is not pushed away from pooled
    assert model.params_for("mouse|human|unseen").length_scales[0] == pytest.approx(pooled)  # unknown group -> pooled


@pytest.mark.parametrize("mean", ["parabola", "gam"])
def test_sampling_reproduces_statistics_and_geometry_checks(mean):
    rng = np.random.default_rng(3)
    branches, enc = _simulate(rng, 400)
    model = T0Model(mean).fit(branches, encoder=enc, max_iter=200)
    reqs = [{"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical", "length_nm": 30000.0}] * 300
    raws = model.sample(reqs, np.random.default_rng(0))
    assert len(raws) == 300 and all(len(r["log_radius"]) == len(r["kappa1"]) + 1 for r in raws)
    assert raws[0]["length"] == pytest.approx(30000.0)
    k1 = np.concatenate([r["kappa1"] for r in raws])
    assert abs(float(k1.mean()) - TRUE_MU[0]) < 3e-6 and float(k1.std()) == pytest.approx(math.sqrt(SIGMA[0, 0]), rel=0.15)
    # one-step autocorrelation of the sampled bending matches exp(-step / l)
    step = raws[0]["step"]
    a = np.concatenate([r["kappa1"][:-1] - TRUE_MU[0] for r in raws])
    b = np.concatenate([r["kappa1"][1:] - TRUE_MU[0] for r in raws])
    assert float(np.corrcoef(a, b)[0, 1]) == pytest.approx(math.exp(-step / TRUE_L[0]), abs=0.05)
    # mean log-radius level and its within-branch spread (the quantity the model must reproduce, however it splits
    # the variation into profile and residual) match the training data
    train = np.mean([b.log_radius.mean() for b in branches])
    assert abs(np.mean([r["log_radius"].mean() for r in raws]) - train) < 0.15
    spread_train = np.mean([b.log_radius.std() for b in branches])
    spread_gen = np.mean([r["log_radius"].std() for r in raws])
    assert spread_gen == pytest.approx(spread_train, rel=0.3)
    # the full procedure: geometry + checks + rejection
    limits = TrunkLimits.from_data([np.hypot(b.kappa1, b.kappa2) for b in branches], [1.5] * len(branches), check_contact=False)
    records, summary = generate_trunks(model, reqs[:40], limits, seed=1)
    assert summary["n_accepted"] >= 38 and summary["acceptance_rate"] > 0.9
    rec = next(r for r in records if r is not None)
    assert rec["gamma"].shape == (rec["n_nodes"], 3) and rec["length"] == pytest.approx(30000.0)


def test_automatic_length_follows_the_training_distribution_and_persistence(tmp_path):
    rng = np.random.default_rng(4)
    branches, enc = _simulate(rng, 500, lengths=(6000.0, 120000.0))
    model = T0Model("parabola", length_bounds=(6000.0, 120000.0)).fit(branches, encoder=enc, max_iter=100)
    reqs = [{"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical"}] * 400
    lengths = np.array([r["length"] for r in model.sample(reqs, np.random.default_rng(0))])
    train = np.array([b.length for b in branches])
    assert 6000.0 <= lengths.min() and lengths.max() <= 120000.0
    assert abs(np.median(np.log(lengths)) - np.median(np.log(train))) < 0.2
    path = tmp_path / "t0.json"
    model.save(path)
    loaded = T0Model.load(path)
    a = model.sample(reqs[:5], np.random.default_rng(7))
    b = loaded.sample(reqs[:5], np.random.default_rng(7))
    assert all(np.allclose(x["kappa1"], y["kappa1"]) and np.allclose(x["log_radius"], y["log_radius"]) for x, y in zip(a, b))


def test_mean_models_fit_known_profiles():
    rng = np.random.default_rng(5)
    branches, enc = _simulate(rng, 300)
    pm = ParabolicMean().fit(branches)
    gm = GamMean().fit(branches)
    b = branches[0]
    # parabola: the per-branch pair reproduces the noise-free generating profile within the (small) residual process
    assert abs(float(np.mean(b.log_radius - pm.branch_log_mean(b)))) < 0.25
    assert gm.sigma_b >= 0.0 and gm.beta is not None
    sample = gm.sample_log_mean(b.design, b.n_nodes, np.random.default_rng(0))
    assert sample.shape == (b.n_nodes,) and np.isfinite(sample).all()
