"""Cross-validation of the T0 / P0 models by NEURON (2026-11-14).

Why: in ``splits_v1`` the 8 validation inhibitory neurons are much poorer in spines than the training ones (P0 obs /
model 0.44, rarity 0.001 against random 8-neuron subsets of the 47 training inhibitory neurons). One fixed validation
set cannot tell whether that is a property of those neurons or a sampling accident of the split, and with 8 neurons
it gives no interval. Here every train + val neuron of the split is held out exactly once (K folds, stratified by a
neuron-level label, e.g. ``cell_type_binary``), possibly over several repetitions with different fold assignments; the
test part of the split is never used (it stays untouched for the final evaluation).

Functions:
- ``neuron_folds``: stratified assignment neuron -> fold.
- ``flagged_vs_rest``: within each group, do the neurons with a flag (e.g. the validation neurons of ``splits_v1``)
  differ from the other held-out neurons of the group? Both sides are held out, so the comparison is fair (in the
  plain split the val neurons were held out and the train neurons were not).
- ``neuron_descriptors``: per neuron counts from the branch table (all branches, not only the eligible ones): how
  many branches, share eligible for T / P, share of spines without an S verdict, share with a position, radius,
  density; to see what makes the flagged neurons different.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .placement_generation.calibration import held_out_rarity


def neuron_folds(neurons: pd.DataFrame, k: int, *, seed: int = 0, stratum: Optional[str] = None, neuron_col: str = "neuron_id") -> pd.Series:
    """Fold index (0 .. k-1) of every neuron, as a Series indexed by neuron id.

    Within each stratum (missing values are a stratum of their own) the neurons are shuffled and dealt round robin
    starting from a random fold, so every fold gets floor or ceil of n_stratum / k neurons of each stratum and small
    strata do not always land in fold 0."""
    if k < 2:
        raise ValueError("k >= 2")
    rng = np.random.default_rng(seed)
    frame = neurons.drop_duplicates(neuron_col)
    groups = [frame] if stratum is None else [g for _, g in frame.groupby(frame[stratum].astype(object).where(frame[stratum].notna(), "__missing__"), sort=True)]
    out = {}
    for g in groups:
        ids = rng.permutation(g[neuron_col].astype(str).to_numpy())
        start = int(rng.integers(k))
        for i, n in enumerate(ids):
            out[n] = (start + i) % k
    return pd.Series(out, name="fold").sort_index()


def flagged_vs_rest(neurons: pd.DataFrame, flag: str, *, by: str = "cell_type_binary", n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Held-out neurons with ``flag`` True against the other held-out neurons of the same group.

    ``neurons``: rows of ``calibration.neuron_calibration`` over the held-out folds (one row per neuron and repetition;
    columns ``obs``, ``e_marginal``, ``log_ratio``, ``by``, ``flag``, optional ``repeat``). Per group (and repetition):
    the ratio of totals obs / model of the flagged neurons and the share of random subsets of as many neurons of the
    group that are as far from the median (``held_out_rarity`` with the rest as reference), and the difference of
    the mean neuron effect ``log_ratio`` (flagged - rest) with a permutation p (labels permuted among the neurons of
    the group)."""
    rng = np.random.default_rng(seed)
    rows = []
    reps = neurons["repeat"].unique() if "repeat" in neurons.columns else [None]
    for rep in reps:
        x = neurons if rep is None else neurons[neurons["repeat"] == rep]
        frame = x.assign(split=np.where(x[flag].astype(bool), "flagged", "rest"))
        rare = held_out_rarity(frame, reference_split="rest", by=by, seed=seed)
        rare = rare.set_index(by) if len(rare) else rare
        for group, g in frame.groupby(by, dropna=True):
            f, r = g[g["split"] == "flagged"], g[g["split"] == "rest"]
            if not len(f) or not len(r):
                continue
            lr = g["log_ratio"].to_numpy(float)
            is_f = (g["split"] == "flagged").to_numpy()
            diff = float(lr[is_f].mean() - lr[~is_f].mean())
            perm = np.array([lr[p[:is_f.sum()]].mean() - lr[p[is_f.sum():]].mean() for p in (rng.permutation(len(lr)) for _ in range(n_perm))])
            row = {"repeat": rep, by: group, "n_flagged": len(f), "n_rest": len(r),
                   "flagged_obs_over_model": f["obs"].sum() / max(f["e_marginal"].sum(), 1e-12),
                   "rest_obs_over_model": r["obs"].sum() / max(r["e_marginal"].sum(), 1e-12),
                   "flagged_median_log_ratio": float(np.median(lr[is_f])), "rest_median_log_ratio": float(np.median(lr[~is_f])),
                   "diff_mean_log_ratio": diff, "p_perm": float((np.sum(np.abs(perm) >= abs(diff) - 1e-12) + 1) / (n_perm + 1))}
            if len(rare) and group in rare.index:
                row["rarity_share_as_extreme"] = float(rare.loc[group, "share_as_extreme"])
            rows.append(row)
    return pd.DataFrame(rows)


def neuron_descriptors(table: pd.DataFrame) -> pd.DataFrame:
    """Per neuron summary of the branch table (all branches of the neuron)."""
    t = table.copy()
    num = lambda c, default=np.nan: pd.to_numeric(t[c], errors="coerce") if c in t.columns else pd.Series(default, index=t.index)
    t["_len_um"] = num("length_nm") / 1e3
    t["_in_range"] = num("n_spines_in_range")
    t["_spines"] = num("n_spines")
    t["_unknown"] = num("n_unknown")
    t["_with_pos"] = num("n_with_position")
    t["_radius"] = num("radius_median_nm")
    t["_elig_P"] = t["train_eligible_P"].fillna(False).astype(bool) if "train_eligible_P" in t.columns else False
    t["_elig_T"] = t["train_eligible_T"].fillna(False).astype(bool) if "train_eligible_T" in t.columns else False
    t["_ok"] = (t["status"].astype(str) == "ok") if "status" in t.columns else True
    g = t.groupby("neuron_id", sort=True)
    out = pd.DataFrame({
        "cell_type_binary": g["cell_type_binary"].first() if "cell_type_binary" in t.columns else None,
        "cell_type": g["cell_type"].first() if "cell_type" in t.columns else None,
        "n_branches": g.size(),
        "share_status_ok": g["_ok"].mean(),
        "share_eligible_T": g["_elig_T"].mean(),
        "share_eligible_P": g["_elig_P"].mean(),
        "cable_um": g["_len_um"].sum(),
        "median_radius_nm": g["_radius"].median(),
        "spines_in_range": g["_in_range"].sum(),
        "density_all_per_um": g["_in_range"].sum() / g["_len_um"].sum(),
        "share_unknown_spines": g["_unknown"].sum() / g["_spines"].sum().where(g["_spines"].sum() > 0),
        "share_with_position": g["_with_pos"].sum() / g["_in_range"].sum().where(g["_in_range"].sum() > 0),
    })
    elig = t[t["_elig_P"]]
    ge = elig.groupby("neuron_id")
    out["cable_eligible_P_um"] = ge["_len_um"].sum()
    out["density_eligible_P_per_um"] = ge["_in_range"].sum() / ge["_len_um"].sum()
    out["share_empty_eligible_P"] = ge["_in_range"].apply(lambda s: float((s == 0).mean()))
    return out.reset_index()


def compare_descriptors(desc: pd.DataFrame, flag: str, *, by: str = "cell_type_binary",
                        columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Median of every descriptor for flagged / other neurons of each group, with a rank-sum p (descriptive)."""
    from scipy.stats import mannwhitneyu

    columns = columns or [c for c in desc.columns if c not in ("neuron_id", "cell_type_binary", "cell_type", flag) and pd.api.types.is_numeric_dtype(desc[c])]
    rows = []
    for group, g in desc.groupby(by, dropna=True):
        f, r = g[g[flag].astype(bool)], g[~g[flag].astype(bool)]
        if not len(f) or not len(r):
            continue
        for c in columns:
            a, b = f[c].dropna().to_numpy(float), r[c].dropna().to_numpy(float)
            p = float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else float("nan")
            rows.append({by: group, "descriptor": c, "n_flagged": len(a), "n_rest": len(b),
                         "median_flagged": float(np.median(a)) if len(a) else np.nan, "median_rest": float(np.median(b)) if len(b) else np.nan, "p_ranksum": p})
    return pd.DataFrame(rows)
