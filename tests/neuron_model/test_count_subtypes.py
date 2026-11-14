"""P0 with detailed cell type columns (CountDesign(subtypes=True), CountModel(subtype_prior_sd)) on synthetic counts."""

import numpy as np

from src.neuron_model.placement_generation.covariates import BranchCovariates, CountDesign
from src.neuron_model.placement_generation.fit import fit_p0
from src.neuron_model.placement_generation.models.count import CountModel, build_count_data

# excitatory subtypes: log density deviation from the excitatory mean; "rare" has only 2 neurons (no column)
SUBTYPES = {"23P": (0.3, 40), "5P-NP": (-1.0, 8), "6P-CT": (0.0, 40), "rare": (1.5, 2), "Unsure": (0.0, 10)}


def _data(rng, per=8, sigma=0.3, scale=1):
    out = []
    n = 0
    for sub, (dev, n_neurons) in SUBTYPES.items():
        for _ in range(n_neurons * scale):
            eff = rng.normal(0, sigma)
            for _ in range(per):
                length = float(np.exp(rng.uniform(np.log(5000), np.log(60000))))
                mu = length / 1e3 * 0.3 * np.exp(dev + eff)
                y = int(rng.poisson(rng.gamma(8.0, mu / 8.0)))
                out.append(BranchCovariates(f"n{n}", y, length, 2 * np.pi * 300 * length, 300.0, 50000.0, 1.1, False, "mouse",
                                            "excitatory", "basal", sub))
            n += 1
    return out


def test_subtype_columns_recover_the_deviations_and_shrink_with_tau():
    rng = np.random.default_rng(3)
    train, val = _data(rng), _data(rng)
    design = CountDesign(subtypes=True).fit(train)
    assert design.subtypes == ["23P", "5P-NP", "6P-CT"]  # 'rare' < 3 neurons, Unsure never
    assert design.names[-3:] == ["subtype_23P", "subtype_5P-NP", "subtype_6P-CT"]
    dtr, dva = build_count_data(design, train), build_count_data(design, val)
    assert dtr.n_subtype == 3 and dtr.X.shape[1] == len(design.names)
    wide = CountModel("nb", random_effect=True, subtype_prior_sd=1.0).fit(dtr)
    coef = dict(zip(design.subtypes, wide.fit_info["subtype_coefficients"]))
    # deviations are identified up to the shared binary-type mean: compare differences between subtypes
    assert abs((coef["23P"] - coef["6P-CT"]) - 0.3) < 0.15
    assert abs((coef["5P-NP"] - coef["6P-CT"]) - (-1.0)) < 0.25
    tight = CountModel("nb", random_effect=True, subtype_prior_sd=0.02).fit(dtr)
    assert np.abs(tight.fit_info["subtype_coefficients"]).max() < 0.05  # the prior pulls everything to the binary mean
    base = CountModel("nb", random_effect=True).fit(build_count_data(CountDesign().fit(train), train))
    assert wide.neg_loglik(dva) < base.neg_loglik(build_count_data(CountDesign().fit(train), val))
    assert wide.sigma_b < base.sigma_b  # the subtype explains part of the neuron spread
    # persistence (design + model) and old designs without subtypes
    d2, m2 = CountDesign.from_dict(design.to_dict()), CountModel.from_dict(wide.to_dict())
    assert d2.subtypes == design.subtypes and m2.n_subtype == 3 and m2.subtype_prior_sd == 1.0
    assert np.allclose(m2.mean(build_count_data(d2, val)), wide.mean(dva))
    old = CountDesign().fit(train).to_dict()
    old.pop("subtypes"), old.pop("subtype_min_neurons")
    assert CountDesign.from_dict(old).n_subtype == 0


def test_fit_p0_keeps_subtypes_only_when_they_help():
    from src.neuron_model.placement_generation.models.position import PositionBranch

    rng = np.random.default_rng(4)
    train, val = _data(rng), _data(rng)
    pos = [PositionBranch(np.linspace(0, 10000, 21), np.full(21, 300.0), np.zeros(21), 0.0, np.sort(rng.uniform(0, 10000, 5)))]
    res = fit_p0(train, val, pos, pos, families=["nb"], exposures=["length"], random_effects=[True], subtype_prior_sds=[0.5],
                 position_use=(), max_iter=100, log=lambda *_: None)
    assert res["report"]["count_chosen"]["subtype_prior_sd"] == 0.5 and res["design"].n_subtype == 3
    assert res["report"]["subtype_variants"][0]["val_nll_gain"] > 0
    # no real subtype effect: the subtypes are not kept (or kept with a negligible gain)
    global SUBTYPES
    saved = SUBTYPES
    SUBTYPES = {k: (0.0, n) for k, (_, n) in saved.items()}
    try:
        tr0, va0 = _data(rng), _data(rng)
    finally:
        SUBTYPES = saved
    res0 = fit_p0(tr0, va0, pos, pos, families=["nb"], exposures=["length"], random_effects=[True], subtype_prior_sds=[0.5],
                  position_use=(), max_iter=100, log=lambda *_: None)
    gain = res0["report"]["subtype_variants"][0]["val_nll_gain"]
    assert gain < 0.01
