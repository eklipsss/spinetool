"""scripts/neuron_model/compare_conditional_generation.py: the comparison of generated vs real spines per condition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("ccg", Path(__file__).resolve().parents[2] / "scripts/neuron_model/compare_conditional_generation.py")
ccg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ccg)


def test_follows_and_ignores_the_condition():
    rng = np.random.default_rng(0)
    real = {"inhibitory": pd.DataFrame({"Volume": rng.lognormal(np.log(0.72), 0.3, 3000)}),
            "excitatory": pd.DataFrame({"Volume": rng.lognormal(0.0, 0.3, 3000)})}
    sd = {"Volume": float(pd.concat(real.values())["Volume"].std())}
    good = {("inhibitory", "g"): pd.DataFrame({"Volume": rng.lognormal(np.log(0.72), 0.3, 400)}),
            ("excitatory", "g"): pd.DataFrame({"Volume": rng.lognormal(0.0, 0.3, 400)})}
    blind = {("inhibitory", "b"): pd.DataFrame({"Volume": rng.lognormal(0.0, 0.3, 400)}),
             ("excitatory", "b"): pd.DataFrame({"Volume": rng.lognormal(0.0, 0.3, 400)})}
    per_set, pairs = ccg.compare({**good, **blind}, real, ["Volume"], sd)
    g = pairs[pairs["label"] == "g"].iloc[0]
    b = pairs[pairs["label"] == "b"].iloc[0]
    assert g["a"] == "excitatory" and g["ratio_real"] == pytest.approx(1 / 0.72, rel=0.05)
    assert g["share_of_real_difference"] == pytest.approx(1.0, abs=0.25) and g["p_gen_a_vs_b"] < 1e-6
    assert abs(b["share_of_real_difference"]) < 0.3
    own = per_set.set_index(["value", "label"])["closer_to_own"]
    assert own[("inhibitory", "g")] and own[("excitatory", "g")] and not own[("inhibitory", "b")]
