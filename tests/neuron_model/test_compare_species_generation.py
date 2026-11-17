"""scripts/neuron_model/compare_species_generation.py: groups species/type and the species effect through ccg.compare."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

root = Path(__file__).resolve().parents[2] / "scripts/neuron_model"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csg = _load("compare_species_generation")
ccg = _load("compare_conditional_generation")


def _frame(rng, n_exc, n_inh, volume_exc, volume_inh):
    return pd.DataFrame({
        "neuron_id": [f"n{i % 7}" for i in range(n_exc + n_inh)],
        "cell_type_binary": ["excitatory"] * n_exc + ["inhibitory"] * n_inh,
        "Volume": np.concatenate([rng.lognormal(np.log(volume_exc), 0.3, n_exc), rng.lognormal(np.log(volume_inh), 0.3, n_inh)])})


def test_groups_and_species_effect_are_reproduced_by_a_follower_only():
    rng = np.random.default_rng(0)
    real = csg.real_groups({"mouse": _frame(rng, 3000, 400, 1.0, 0.7), "human": _frame(rng, 3000, 0, 1.6, 1.0)}, 2500, rng)
    assert set(real) == {"mouse/excitatory", "mouse/inhibitory", "human/excitatory"}      # no human inhibitory spines -> no group
    assert len(real["mouse/excitatory"]) == 2500                                          # random subset of the cap
    sd = {"Volume": float(pd.concat(real.values())["Volume"].std())}
    follower = {("mouse/excitatory", "f"): pd.DataFrame({"Volume": rng.lognormal(0.0, 0.3, 400)}),
                ("human/excitatory", "f"): pd.DataFrame({"Volume": rng.lognormal(np.log(1.6), 0.3, 400)})}
    blind = {("mouse/excitatory", "b"): pd.DataFrame({"Volume": rng.lognormal(0.0, 0.3, 400)}),
             ("human/excitatory", "b"): pd.DataFrame({"Volume": rng.lognormal(0.0, 0.3, 400)})}
    _, pairs = ccg.compare({**follower, **blind}, real, ["Volume"], sd)
    f = pairs[pairs["label"] == "f"].iloc[0]
    b = pairs[pairs["label"] == "b"].iloc[0]
    assert f["share_of_real_difference"] == pytest.approx(1.0, abs=0.25)
    assert abs(b["share_of_real_difference"]) < 0.3


def test_spine_key_matches_the_split_tables():
    frame = pd.DataFrame({"dataset": ["H01_Seg"], "neuron_id": ["neuron_5"], "limb_id": ["limb_000"], "branch_id": ["branch_001"], "spine_id": ["spine_002"]})
    key = csg.spine_key(frame).iloc[0]
    assert key.split("/")[1] == "neuron_5" and key.count("/") == 4
