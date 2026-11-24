from __future__ import annotations

import numpy as np
import pandas as pd

from src.neuron_model.spine_generation.evaluation.density_ratio import DensityRatioSelector, per_metric_shift

METRICS = ("Length", "Area", "Volume")


def frame(n, shift, rng, keys=False):
    x = np.exp(rng.normal(0, 0.5, (n, 3)) + np.array(shift))
    f = pd.DataFrame(x, columns=METRICS)
    if keys:
        f["spine_key"] = [f"minnie65/{i % 40}/000/000/{i:05d}" for i in range(n)]
    return f


def test_selection_removes_a_size_shift_on_held_out_data():
    rng = np.random.default_rng(0)
    real_fit, real_eval = frame(4000, [0, 0, 0], rng, True), frame(4000, [0, 0, 0], rng, True)
    gen_fit, gen_eval = frame(1500, [0, 0.25, 0.4], rng), frame(1500, [0, 0.25, 0.4], rng)
    sel = DensityRatioSelector(metrics=METRICS, model="logistic").fit(real_fit, gen_fit)
    keep = sel.accept(gen_eval, rng)
    assert 0.1 < keep.mean() < 0.9
    t = per_metric_shift(real_eval, {"raw": gen_eval, "sel": gen_eval[keep]}, metrics=METRICS, n_boot=200).set_index(["set", "metric"])
    assert t.loc[("raw", "Volume"), "significant"] and abs(t.loc[("sel", "Volume"), "d"]) < 0.1
    assert abs(t.loc[("sel", "Area"), "d"]) < 0.08


def test_incomplete_rows_get_the_mean_acceptance():
    rng = np.random.default_rng(1)
    sel = DensityRatioSelector(metrics=METRICS, model="logistic").fit(frame(800, [0, 0, 0], rng, True), frame(800, [0, 0, 0.3], rng))
    g = frame(10, [0, 0, 0.3], rng)
    g.loc[0, "Area"] = np.nan
    s = sel.score(g)
    assert np.isnan(s[0]) and np.all(np.isfinite(s[1:]))
