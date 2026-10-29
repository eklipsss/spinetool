import numpy as np

from src.neuron_model.significance import cluster_bootstrap_diff, ks_pvalue


def _sample(rng, n_neurons, per_neuron, shift, style_sd=1.0):
    values, groups = [], []
    for k in range(n_neurons):
        style = rng.normal(0, style_sd)
        values.append(rng.normal(style + shift, 1.0, per_neuron))
        groups += [f"n{k}"] * per_neuron
    return np.concatenate(values), np.array(groups)


def test_cluster_bootstrap_finds_a_real_shift_and_not_a_neuron_style():
    rng = np.random.default_rng(0)
    # a real shift of 1 sd, every neuron in both sets
    a, ga = _sample(rng, 40, 25, 1.5, style_sd=0.5)
    b, gb = _sample(rng, 40, 25, 0.0, style_sd=0.5)
    res = cluster_bootstrap_diff(a, ga, b, gb, n_boot=400, seed=1)
    assert res["significant"] and res["lo"] > 0.5 and res["diff"] > 0.9
    # two samples from ONE distribution of neurons: with 1000 values the plain KS is often "significant" because of the
    # shared neuron styles, the cluster bootstrap says "no difference" for (nearly) every seed
    flagged = 0
    for seed in range(8):
        r = np.random.default_rng(100 + seed)
        a, ga = _sample(r, 12, 40, 0.0, style_sd=1.5)
        b, gb = _sample(r, 12, 40, 0.0, style_sd=1.5)
        gb = np.array([f"m{x[1:]}" for x in gb])  # different neurons in the two sets
        flagged += cluster_bootstrap_diff(a, ga, b, gb, n_boot=300, seed=seed)["significant"]
    assert flagged <= 2
    assert ks_pvalue([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) > 0.9


def test_cluster_bootstrap_handles_too_little_data():
    res = cluster_bootstrap_diff([1.0], ["a"], [2.0, 3.0], ["a", "a"])
    assert np.isnan(res["diff"]) and not res["significant"]
