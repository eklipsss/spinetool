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


def _twins(rng, n_neurons, per_neuron, shift, style_sd=1.0, scale=1.0):
    """Real branches of neurons with a style, and one generated twin per branch from a NEW neuron-style draw (a correct
    generator of new neurons when shift = 0 and scale = 1); twins are labelled with the neuron of their prototype."""
    real, gen, groups = [], [], []
    for k in range(n_neurons):
        real.append(rng.normal(rng.normal(0, style_sd), 1.0, per_neuron))
        gen.append(rng.normal(rng.normal(0, style_sd) + shift, scale, per_neuron))
        groups += [f"n{k}"] * per_neuron
    return np.concatenate(gen), np.concatenate(real), np.array(groups)


def test_ks_by_neuron_keeps_a_correct_generator_and_finds_a_wrong_one():
    from src.neuron_model.significance import ks_neuron_floor_test, ks_statistic

    assert abs(ks_statistic(np.array([0.0, 1.0]), np.array([0.0, 1.0]))) < 1e-12
    flagged, naive = 0, 0
    for seed in range(8):
        rng = np.random.default_rng(200 + seed)
        g, r, n = _twins(rng, 20, 40, 0.0, style_sd=1.0)
        flagged += ks_neuron_floor_test(g, n, r, n, n_splits=300, seed=seed)["significant"]
        naive += ks_pvalue(g, r) < 0.05
    assert flagged <= 1  # the neuron-level test keeps a correct generator
    assert naive >= 4    # the plain KS over 800 branches rejects it often (neuron styles)
    # a shift of 0.8 sd of the within-neuron noise: found (10 of 10 seeds in a check)
    g, r, n = _twins(np.random.default_rng(50), 40, 30, 0.8, style_sd=0.7)
    res = ks_neuron_floor_test(g, n, r, n, n_splits=300, seed=1)
    assert res["significant"] and res["d_model"] > res["d_null_p95"]
    # same median, wrong spread: KS sees the shape; the test is conservative (spread x2: 4 of 10 seeds, x3: 10 of 10)
    g, r, n = _twins(np.random.default_rng(51), 40, 30, 0.0, style_sd=0.7, scale=3.0)
    assert ks_neuron_floor_test(g, n, r, n, n_splits=300, seed=2)["significant"]
    assert not ks_neuron_floor_test([1.0, 2.0], ["a", "a"], [1.0, 2.0], ["a", "a"])["significant"]


def _clustered_reference(rng, n_neurons=60, per_neuron=300, style_sd=0.5):
    styles = rng.normal(0.0, style_sd, n_neurons)
    values = np.concatenate([rng.normal(m, 1.0, per_neuron) for m in styles])
    neurons = np.repeat([f"n{k}" for k in range(n_neurons)], per_neuron)
    return values, neurons, styles


def test_ks_of_a_small_sample_against_clustered_neurons():
    from src.neuron_model.significance import ks_pvalue, ks_sample_vs_neurons

    rejected_ok, rejected_shift, naive_ok = 0, 0, 0
    for rep in range(8):
        rng = np.random.default_rng(100 + rep)
        ref, ref_n, _ = _clustered_reference(rng)
        # a correct generator: i.i.d. spines of the POPULATION (new neurons with their own style), 500 samples
        new_styles = rng.normal(0.0, 0.5, 500)
        good = rng.normal(new_styles, 1.0)
        bad = rng.normal(rng.normal(0.0, 0.5, 500), 1.0) + 0.5 * np.sqrt(1.25)  # +0.5 sd of the population (0.3 sd: power ~0.6 at this neuron style)
        r_ok = ks_sample_vs_neurons(good, None, ref, ref_n, n_splits=150, seed=rep)
        r_bad = ks_sample_vs_neurons(bad, None, ref, ref_n, n_splits=150, seed=rep)
        rejected_ok += r_ok["significant"]
        rejected_shift += r_bad["significant"]
        naive_ok += ks_pvalue(good, ref) < 0.05
        assert r_ok["n_compared"] == 500
    assert rejected_ok <= 1
    assert rejected_shift >= 6
    # processed real spines WITH their neurons: compared only with other neurons, a correct chain is kept
    rng = np.random.default_rng(7)
    ref, ref_n, _ = _clustered_reference(rng)
    pick = rng.choice(len(ref), 400, replace=False)
    floor = ref[pick] + rng.normal(0.0, 0.05, 400)
    r = ks_sample_vs_neurons(floor, ref_n[pick], ref, ref_n, n_splits=150, seed=1)
    assert not r["significant"] and r["n_compared"] >= 100
