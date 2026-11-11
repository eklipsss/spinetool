"""``generate_tp.py --counts real|real_positioned``: N is taken from the source branch, not from the count model."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/neuron_model/generate_tp.py"


def _module():
    spec = importlib.util.spec_from_file_location("generate_tp_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_counts_columns_and_missing():
    mod = _module()
    rows = pd.DataFrame({"n_spines_in_range": [5, 0, 12], "n_with_position": [4, 0, 11]}).iloc[[2, 0]]
    assert mod.real_counts(rows, "real").tolist() == [12, 5]
    assert mod.real_counts(rows, "real_positioned").tolist() == [11, 4]
    assert mod.real_counts(rows, "real").dtype == np.int64
    with pytest.raises(ValueError):
        mod.real_counts(pd.DataFrame({"n_spines_in_range": [3, None], "n_with_position": [3, 1]}), "real")


def test_posterior_counts_follow_the_observed_neuron():
    """``--counts model_neuron``: a poor neuron gets few spines, a neuron without observed branches a fresh effect."""
    from src.neuron_model.placement_generation.covariates import BranchCovariates, CountDesign
    from src.neuron_model.placement_generation.models.count import CountModel, build_count_data

    mod = _module()
    rng = np.random.default_rng(0)

    def branch(neuron, y, length=20000.0):
        return BranchCovariates(neuron, y, length, 2 * np.pi * 300 * length, 300.0, 50000.0, 1.1, False, "mouse", "excitatory", None)

    effects = rng.normal(0, 0.4, 40)
    train = [branch(f"t{n}", int(rng.poisson(6 * np.exp(effects[n])))) for n in range(40) for _ in range(20)]
    design = CountDesign().fit(train)
    model = CountModel("nb", random_effect=True).fit(build_count_data(design, train))
    observed = [branch("poor", int(rng.poisson(1.5))) for _ in range(60)] + [branch("rich", int(rng.poisson(15))) for _ in range(60)]
    requests = [branch(n, -1) for n in ["poor"] * 300 + ["rich"] * 300 + ["unseen"] * 300]
    counts, info = mod.posterior_counts(model, design, requests, observed, rng)
    assert info["n_posterior"] == 2 and info["n_fresh"] == 1
    assert counts[:300].mean() < 2.5 and counts[300:600].mean() > 11
    assert 3 < counts[600:].mean() < 10  # the population level
