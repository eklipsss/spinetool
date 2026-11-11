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


# ---- branch level: where does the count model put too many / too few spines? (2026-11-11) --------------------------
def _branch_classes(cov: Sequence[Any]) -> pd.DataFrame:
    """Covariate classes of each branch (the factors of ``branch_calibration``)."""
    r = np.array([b.mean_radius_nm for b in cov], dtype=float)
    length = np.array([b.length_nm for b in cov], dtype=float) / 1e3
    soma = np.array([np.nan if b.soma_distance_nm is None else b.soma_distance_nm for b in cov], dtype=float) / 1e3
    cut = lambda v, edges, labels: pd.Series(pd.cut(v, edges, labels=labels, right=False)).astype(object).where(~np.isnan(v), "unknown")
    return pd.DataFrame({
        "all": "all",
        "radius": cut(r, [0, 200, 300, 400, np.inf], ["< 200 nm", "200-300 nm", "300-400 nm", ">= 400 nm"]),
        "length": cut(length, [0, 20, 50, np.inf], ["< 20 um", "20-50 um", ">= 50 um"]),
        "soma_distance": cut(soma, [0, 100, 200, 400, np.inf], ["< 100 um", "100-200 um", "200-400 um", ">= 400 um"]),
        "terminal": ["terminal" if b.is_terminal else "internal" for b in cov],
        "compartment": [str(b.compartment) for b in cov],
        "cell_type_binary": [str(b.cell_type_binary) for b in cov],
    })


def spine_weighted_density(counts: np.ndarray, length_um: np.ndarray) -> float:
    """Sum N_b^2 / L_b over sum N_b: the density seen by a random SPINE (per-spine statistics such as the nearest
    neighbour weight branches by their number of spines, so a heavier upper tail of N lowers them even when the mean
    density per branch is right)."""
    n = np.asarray(counts, dtype=float)
    total = n.sum()
    return float((n ** 2 / np.asarray(length_um, dtype=float)).sum() / total) if total > 0 else float("nan")


def branch_calibration(model: CountModel, design: Any, cov: Sequence[Any], *, split: str, n_sim: int = 200, seed: int = 0) -> pd.DataFrame:
    """Observed against simulated counts per covariate class, two kinds of simulation.

    ``fresh``: a new effect for every neuron (what generation does); ``posterior``: the effect of each neuron drawn from
    its posterior given its own observed counts (removes "these neurons are richer / poorer than the population").
    A class that is off under ``posterior`` points to the branch-level part of the model (covariate shape, dispersion);
    off only under ``fresh`` - the neuron level. Statistics: density (sum N / sum L), share of empty branches and the
    spine-weighted density (``spine_weighted_density``), simulated mean and 5 / 95 % quantiles over ``n_sim`` draws.
    The posterior shrinks a neuron's effect towards the population by roughly 1 / (1 + sigma^2 * expected spines of the
    neuron): negligible for excitatory neurons (hundreds of spines), noticeable for sparse (inhibitory) ones, where the
    ``posterior`` column stays biased towards the population."""
    if not cov:
        return pd.DataFrame()
    data = build_count_data(design, cov, model.exposure)
    classes = _branch_classes(cov)
    y = np.array([b.n_spines for b in cov], dtype=float)
    length = np.array([b.length_nm for b in cov], dtype=float) / 1e3
    rng = np.random.default_rng(seed)
    posterior = model.neuron_log_posterior(data)
    sims = {"fresh": [], "posterior": []}
    for _ in range(n_sim):
        sims["fresh"].append(model.sample(data, rng))
        sims["posterior"].append(model.sample(data, rng, neuron_effects=model.posterior_neuron_effects(data, rng, posterior=posterior)))
    sims = {k: np.vstack(v).astype(float) for k, v in sims.items()}
    rows = []
    for factor in classes.columns:
        for level, idx in classes.groupby(factor, sort=True).groups.items():
            idx = np.asarray(list(idx))
            row = {"split": split, "factor": factor, "level": level, "n_branches": len(idx),
                   "n_neurons": len({cov[i].neuron for i in idx}),
                   "obs_density": y[idx].sum() / length[idx].sum(), "obs_empty": float((y[idx] == 0).mean()),
                   "obs_sw_density": spine_weighted_density(y[idx], length[idx])}
            for kind, s in sims.items():
                dens = s[:, idx].sum(1) / length[idx].sum()
                empty = (s[:, idx] == 0).mean(1)
                sw = np.array([spine_weighted_density(r, length[idx]) for r in s[:, idx]])
                for name, v in (("density", dens), ("empty", empty), ("sw_density", sw)):
                    v = v[np.isfinite(v)]
                    row[f"{kind}_{name}"] = float(v.mean()) if len(v) else float("nan")
                    row[f"{kind}_{name}_p05"] = float(np.quantile(v, 0.05)) if len(v) else float("nan")
                    row[f"{kind}_{name}_p95"] = float(np.quantile(v, 0.95)) if len(v) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)
