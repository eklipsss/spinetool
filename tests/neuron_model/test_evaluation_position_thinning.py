import numpy as np

from src.neuron_model.branch_assembly.evaluate import position_keep_fraction, position_keep_mask, spine_geometry_stats


def test_keep_fraction_edge_cases():
    assert position_keep_fraction(20, 18) == 0.9
    assert position_keep_fraction(0, 0) == 1.0
    assert position_keep_fraction(None, 3) == 1.0
    assert position_keep_fraction(float("nan"), 3) == 1.0
    assert position_keep_fraction(5, 7) == 1.0  # clipped


def test_thinning_reproduces_the_bias_of_missing_positions():
    """Same process, real side thinned by 7 %: comparing with ALL generated spines shows a spurious shorter neighbour
    distance; thinning the generated side the same way removes it."""
    rng = np.random.default_rng(0)
    length, radius, keep = 40000.0, 300.0, 0.93
    nn_real, nn_all, nn_thin = [], [], []
    for _ in range(3000):
        n = rng.poisson(16)
        real = (rng.uniform(0, length, n), rng.uniform(-np.pi, np.pi, n))
        gen = (rng.uniform(0, length, n), rng.uniform(-np.pi, np.pi, n))
        m = position_keep_mask(n, keep, rng)
        nn_real.append(spine_geometry_stats(real[0][m], real[1][m], np.full(m.sum(), radius), length)["nn_surface_nm"])
        nn_all.append(spine_geometry_stats(gen[0], gen[1], np.full(n, radius), length)["nn_surface_nm"])
        g = position_keep_mask(n, keep, rng)
        nn_thin.append(spine_geometry_stats(gen[0][g], gen[1][g], np.full(g.sum(), radius), length)["nn_surface_nm"])
    med = lambda xs: np.nanmedian(np.concatenate(xs))
    assert med(nn_all) < med(nn_real) * 0.98          # unthinned comparison: generated look too close
    assert abs(med(nn_thin) / med(nn_real) - 1) < 0.02  # thinned: no bias
