import dataclasses

import numpy as np
import pandas as pd

from src.neuron_model.placement_generation.calibration import group_calibration, held_out_rarity, neuron_calibration
from src.neuron_model.placement_generation.covariates import BranchCovariates, CountDesign
from src.neuron_model.placement_generation.models.count import CountModel, build_count_data


def _neurons(rng, prefix, n_neurons, dens, sigma_b=0.4, per=10):
    out = []
    for n in range(n_neurons):
        eff = rng.normal(0, sigma_b)
        for _ in range(per):
            length = float(np.exp(rng.uniform(np.log(5000), np.log(60000))))
            radius = rng.uniform(200, 400)
            mu = length / 1e3 * dens * np.exp(eff)
            y = int(rng.poisson(rng.gamma(5.0, mu / 5.0)))
            out.append(BranchCovariates(f"{prefix}{n}", y, length, 2 * np.pi * radius * length, radius, 50000.0, 1.1, False, "mouse", "inhibitory", None))
    return out


def test_calibration_separates_model_defect_from_held_out_neurons():
    rng = np.random.default_rng(0)
    train = _neurons(rng, "t", 80, dens=0.1)
    design = CountDesign().fit(train)
    model = CountModel("nb", random_effect=True).fit(build_count_data(design, train))
    sparse_val = _neurons(rng, "v", 20, dens=0.03)  # held-out neurons with 0.3 of the training density
    neurons = pd.concat([neuron_calibration(model, design, train, split="train", n_sim=50),
                                           neuron_calibration(model, design, sparse_val, split="val", n_sim=50)], ignore_index=True)
    groups = group_calibration(neurons).set_index("split")
    assert abs(groups.loc["train", "obs_over_model"] - 1) < 0.15
    assert abs(groups.loc["val", "obs_over_model"] - 0.3) < 0.1
    assert abs(groups.loc["train", "obs_empty"] - groups.loc["train", "model_empty"]) < 0.05
    assert groups.loc["val", "obs_empty"] > groups.loc["val", "model_empty"] + 0.1
    assert 0.2 < groups.loc["train", "sd_log_ratio"] < 0.7  # neuron effect spread ~ sigma_b 0.4 (+ count noise)
    rare = held_out_rarity(neurons, n_draws=2000).set_index("split")
    assert rare.loc["val", "share_as_extreme"] < 0.01  # 20 neurons at 0.3 of the density: far outside the subsets
    same = pd.concat([neurons[neurons.split == "train"], neurons[neurons.split == "train"].iloc[:20].assign(split="val")])
    assert held_out_rarity(same, n_draws=2000).iloc[0]["share_as_extreme"] > 0.05  # training neurons themselves: typical
    assert set(neurons.columns) >= {"neuron", "split", "cell_type", "obs", "e0", "e_marginal", "log_ratio"}
    # the detailed cell type is mapped per neuron
    typed = neuron_calibration(model, design, [dataclasses.replace(b) for b in train[:20]], split="train", cell_type={"t0": "BC", "t1": "MC"}, n_sim=5)
    assert typed.set_index("neuron").loc["t1", "cell_type"] == "MC"
