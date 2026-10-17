import numpy as np

from src.neuron_model.placement_generation import diagnostics as dg
from src.neuron_model.placement_generation.models.position import PositionBranch, PositionModel


def _branches(rng, n=60, length=20000.0, repulsive=False):
    out = []
    for _ in range(n):
        s = np.linspace(0, length, int(length / 500) + 1)
        if repulsive:  # hard-core 400 nm by sequential inhibition
            pos = []
            while len(pos) < 25:
                x = rng.uniform(0, length)
                if all(abs(x - p) > 400 for p in pos):
                    pos.append(x)
            pos = np.sort(pos)
        else:
            pos = np.sort(rng.uniform(0, length, 25))
        out.append(PositionBranch(s, np.full(len(s), 250.0), np.full(len(s), 1e-4), None, pos))
    return out


def test_k_function_poisson_and_envelopes():
    rng = np.random.default_rng(0)
    br = _branches(rng)
    h = np.linspace(100, 3000, 15)
    k = dg.k_function([b.positions for b in br], [b.length for b in br], h)
    assert np.allclose(k, 2 * h, rtol=0.1)
    m = PositionModel(use=()).fit(br)
    env = dg.envelopes(m, br, h, n_sim=39, seed=1)
    assert env["K_outside_fraction"] <= 0.2 and not env["short_interval_deficit"]


def test_repulsion_is_detected():
    rng = np.random.default_rng(1)
    br = _branches(rng, repulsive=True)
    m = PositionModel(use=()).fit(br)
    env = dg.envelopes(m, br, np.linspace(50, 3000, 20), n_sim=39, seed=2)
    assert env["short_interval_deficit"] and env["G_outside_fraction"] > 0.2


def test_circular_tests_and_shift_invariance():
    rng = np.random.default_rng(2)
    uni = rng.uniform(-np.pi, np.pi, 800)
    vm = rng.vonmises(0.7, 1.0, 800)
    bimodal = np.r_[rng.vonmises(0.0, 4.0, 400), rng.vonmises(np.pi, 4.0, 400)]  # Rayleigh misses it, Kuiper/Watson do not
    su, sv, sb = dg.circular_summary(uni), dg.circular_summary(vm), dg.circular_summary(bimodal)
    assert su["rayleigh"]["p"] > 0.01 and su["kuiper"]["p"] > 0.01 and su["watson_u2"]["p"] > 0.01
    assert sv["rayleigh"]["p"] < 1e-3 and sv["kuiper"]["p"] < 1e-3 and sv["watson_u2"]["p"] < 1e-3
    assert sb["rayleigh"]["p"] > 0.01 and sb["kuiper"]["p"] < 1e-3 and sb["watson_u2"]["p"] < 1e-3 and sb["abs_c2"] > 0.5
    # a shift of the zero direction moves mu by the same angle and leaves neighbour differences unchanged
    s = [np.sort(rng.uniform(0, 1e4, 30)) for _ in range(5)]
    th = [rng.vonmises(1.0, 2.0, 30) for _ in range(5)]
    d0 = dg.nearest_angle_differences(th, s)
    d1 = dg.nearest_angle_differences([t + 0.9 for t in th], s)
    assert np.allclose(d0, d1)
    mu0 = np.angle(np.exp(1j * np.concatenate(th)).mean())
    mu1 = np.angle(np.exp(1j * (np.concatenate(th) + 0.9)).mean())
    assert np.isclose(np.angle(np.exp(1j * (mu1 - mu0))), 0.9)
