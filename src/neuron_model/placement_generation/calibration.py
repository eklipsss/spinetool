"""Calibration of the P0 count model by neuron and cell type (diagnostic, no fitting).

For every neuron of the given branches: the observed number of spines against the model mean with the neuron effect
set to 0 (``E0``, the typical neuron) and integrated out (``E0 * exp(sigma_b^2 / 2)``, the mean over neurons), the
share of branches without spines against the share simulated with fresh neuron effects. ``log_ratio`` =
log((obs + 0.5) / (E0 + 0.5)) is a rough estimate of the neuron effect; its spread over the neurons of a group is
compared with the model ``sigma_b`` of the neurons (one value for all neurons, or one per
``SIGMA_GROUPS`` group of ``CountModel(sigma_groups=2)``). The group table answers whether the model is off
for a group on the TRAINING neurons too (model defect) or only on held-out neurons (they differ from the training
ones; with few neurons a sampling effect).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .models.count import CountModel, build_count_data


def neuron_calibration(model: CountModel, design: Any, cov: Sequence[Any], *, split: str,
                       cell_type: Optional[Mapping[str, Any]] = None, n_sim: int = 200, seed: int = 0) -> pd.DataFrame:
    """One row per neuron of ``cov`` (``BranchCovariates`` with the observed counts)."""
    if not cov:
        return pd.DataFrame()
    data = build_count_data(design, cov, model.exposure)
    e0 = model.mean(data)
    rng = np.random.default_rng(seed)
    sim_zero = np.zeros(len(cov))
    for _ in range(n_sim):
        sim_zero += model.sample(data, rng) == 0
    sim_zero /= n_sim
    frame = pd.DataFrame({"neuron": [b.neuron for b in cov], "cell_type_binary": [b.cell_type_binary for b in cov],
                          "obs": [b.n_spines for b in cov], "length_um": [b.length_nm / 1e3 for b in cov], "e0": e0, "sim_zero": sim_zero})
    frame["obs_zero"] = (frame["obs"] == 0).astype(float)
    g = frame.groupby("neuron", sort=False).agg(cell_type_binary=("cell_type_binary", "first"), n_branches=("obs", "size"), obs=("obs", "sum"),
                                                 length_um=("length_um", "sum"), e0=("e0", "sum"), obs_zero=("obs_zero", "mean"), sim_zero=("sim_zero", "mean")).reset_index()
    sigma = dict(zip(pd.unique(frame["neuron"]), model.neuron_sigmas(data)))  # neuron index order = first appearance
    g["sigma_b"] = g["neuron"].map(sigma)
    g["e_marginal"] = g["e0"] * np.exp(g["sigma_b"] ** 2 / 2)
    g["obs_density"] = g["obs"] / g["length_um"]
    g["e_marginal_density"] = g["e_marginal"] / g["length_um"]
    g["log_ratio"] = np.log((g["obs"] + 0.5) / (g["e0"] + 0.5))
    g.insert(1, "split", split)
    g.insert(3, "cell_type", g["neuron"].map(dict(cell_type)) if cell_type else None)
    return g


def group_calibration(neurons: pd.DataFrame, by: Sequence[str] = ("split", "cell_type_binary")) -> pd.DataFrame:
    """Per group: total observed / total marginal mean, empty shares, typical neuron effect and its spread vs sigma_b."""
    rows = []
    for key, x in neurons.groupby(list(by), dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "n_neurons": len(x), "n_branches": int(x["n_branches"].sum()),
                     "obs_density": x["obs"].sum() / x["length_um"].sum(), "model_density": x["e_marginal"].sum() / x["length_um"].sum(),
                     "obs_over_model": x["obs"].sum() / max(x["e_marginal"].sum(), 1e-12),
                     "obs_empty": float((x["obs_zero"] * x["n_branches"]).sum() / x["n_branches"].sum()),
                     "model_empty": float((x["sim_zero"] * x["n_branches"]).sum() / x["n_branches"].sum()),
                     "median_neuron_density": float(x["obs_density"].median()), "median_log_ratio": float(x["log_ratio"].median()),
                     "sd_log_ratio": float(x["log_ratio"].std(ddof=1)) if len(x) > 1 else float("nan"), "sigma_b": float(x["sigma_b"].mean())})
    return pd.DataFrame(rows)


def held_out_rarity(neurons: pd.DataFrame, *, reference_split: str = "train", by: str = "cell_type_binary",
                    n_draws: int = 20000, seed: int = 0) -> pd.DataFrame:
    """How unusual is a held-out group given the training neurons of the same group?

    For every (split != reference, group): ratio of totals obs / model mean of its k neurons against the same ratio of
    random k-neuron subsets of the reference neurons of the group (two-sided share of subsets at least as far from the
    reference median on the log scale). A small share = the held-out neurons differ from the training ones beyond
    neuron-to-neuron variation, a large share = a sampling effect of few neurons."""
    rng = np.random.default_rng(seed)
    rows = []
    ref_all = neurons[neurons["split"] == reference_split]
    for (split, group), x in neurons[neurons["split"] != reference_split].groupby(["split", by], dropna=True):
        ref = ref_all[ref_all[by] == group]
        k = len(x)
        if len(ref) < k or k == 0:
            continue
        obs, exp = ref["obs"].to_numpy(float), ref["e_marginal"].to_numpy(float)
        idx = np.array([rng.choice(len(ref), k, replace=False) for _ in range(n_draws)])
        sub = np.log(obs[idx].sum(1) / exp[idx].sum(1))
        value = float(np.log(x["obs"].sum() / x["e_marginal"].sum()))
        centre = float(np.median(sub))
        rows.append({"split": split, by: group, "k_neurons": k, "n_reference_neurons": len(ref), "ratio": float(np.exp(value)),
                     "reference_subsets_ratio_p05": float(np.exp(np.quantile(sub, 0.05))), "reference_subsets_ratio_p95": float(np.exp(np.quantile(sub, 0.95))),
                     "share_as_extreme": float(np.mean(np.abs(sub - centre) >= abs(value - centre)))})
    return pd.DataFrame(rows)
