import numpy as np

from src.neuron_model.placement_generation.covariates import BranchCovariates, CountDesign
from src.neuron_model.placement_generation.models.count import CountModel, build_count_data


def _data(rng, n_exc=150, n_inh=60, per=10, sigma=(0.3, 0.8)):
    out = []
    for n in range(n_exc + n_inh):
        inh = n >= n_exc
        eff = rng.normal(0, sigma[int(inh)])
        dens = 0.1 if inh else 0.4
        for _ in range(per):
            length = float(np.exp(rng.uniform(np.log(5000), np.log(60000))))
            mu = length / 1e3 * dens * np.exp(eff)
            y = int(rng.poisson(rng.gamma(8.0, mu / 8.0)))
            out.append(BranchCovariates(f"n{n}", y, length, 2 * np.pi * 300 * length, 300.0, 50000.0, 1.1, False, "mouse",
                                        "inhibitory" if inh else "excitatory", "basal" if not inh else "inhibitory_dendrite"))
    return out


def test_own_sigma_for_inhibitory_is_recovered_and_used_in_sampling():
    rng = np.random.default_rng(1)
    train, val = _data(rng), _data(rng)
    design = CountDesign().fit(train)
    dtr, dva = build_count_data(design, train), build_count_data(design, val)
    one = CountModel("nb", random_effect=True).fit(dtr)
    two = CountModel("nb", random_effect=True, sigma_groups=2).fit(dtr)
    s_exc, s_inh = np.exp(two.log_sigma)
    assert abs(s_exc - 0.3) < 0.1 and abs(s_inh - 0.8) < 0.2
    assert two.neg_loglik(dva) < one.neg_loglik(dva)
    # sampling: spread of the neuron mean (log total / expected) larger for inhibitory neurons
    sig = two.neuron_sigmas(dva)
    groups = dva.neuron_group()
    assert np.allclose(sig[groups == 0], s_exc) and np.allclose(sig[groups == 1], s_inh)
    # persistence, and old one-group files
    back = CountModel.from_dict(two.to_dict())
    assert np.allclose(back.log_sigma, two.log_sigma) and back.sigma_groups == 2
    old = one.to_dict()
    old.pop("log_sigma"), old.pop("sigma_groups")
    legacy = CountModel.from_dict(old)
    assert legacy.sigma_groups == 1 and abs(legacy.sigma_b - one.sigma_b) < 1e-12
    assert np.allclose(legacy.sample(dva, np.random.default_rng(0)), one.sample(dva, np.random.default_rng(0)))


def test_group_without_neurons_keeps_its_start_value():
    rng = np.random.default_rng(2)
    train = _data(rng, n_exc=60, n_inh=0)
    design = CountDesign().fit(train)
    m = CountModel("nb", random_effect=True, sigma_groups=2).fit(build_count_data(design, train))
    assert abs(np.exp(m.log_sigma[1]) - 0.3) < 1e-12
