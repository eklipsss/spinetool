"""Do real spines differ in each morphometric between cell types and between compartments?

The unit of inference is the NEURON, not the spine: spines of one neuron share its "style" (segmentation, size of the
cell, local conditions), so treating ~1000 spines of a neuron as independent samples would call almost any difference
significant (the same trap as the naive KS test of the assembled branches, significance.py).

- **Cell type** (a property of the neuron; binary excitatory / inhibitory and the detailed types with at least
  ``min_neurons`` neurons): per neuron the median of a metric over its spines (neurons with >= ``min_spines`` spines);
  statistic = difference of the medians of the neuron medians of the two groups; null = the labels permuted between
  neurons (``n_permutations``); the omnibus test over all detailed types = permutation of the between-group spread of
  the neuron medians (a Kruskal-like statistic on ranks).
- **Compartment** (a property of the limb; one neuron often has apical and basal limbs): WITHIN-neuron comparison -
  for every neuron with both compartments (>= ``min_spines`` spines in each) the difference of the two medians;
  statistic = median of the per-neuron differences; null = random sign flips (the neuron style cancels out).
- **Cell type within a compartment**: the binary cell-type test repeated on the spines of one compartment (inhibitory
  cells have no apical dendrite, so a pooled cell-type difference can be a compartment difference).

Per test: number of neurons / spines, the group medians, the relative difference, Cliff's delta between the neuron
medians (or the share of positive per-neuron differences), the permutation p and the Benjamini-Hochberg q over all
tests of the same factor (metrics x pairs). Values: the scalar morphometrics of stage 8.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

KEYS = ["dataset", "neuron_id", "limb_id", "branch_id", "spine_id"]


def bh_qvalues(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) over all pairs (-1..1; 0 = no shift)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def neuron_medians(frame: pd.DataFrame, metrics: Sequence[str], by: Sequence[str], min_spines: int) -> pd.DataFrame:
    """Median of every metric per group ``by`` (e.g. neuron, or neuron + compartment) + ``n_spines``; groups with fewer
    than ``min_spines`` spines dropped."""
    grouped = frame.groupby(list(by), observed=True)
    med = grouped[list(metrics)].median()
    med["n_spines"] = grouped.size()
    return med[med["n_spines"] >= min_spines].reset_index()


def _permutation_index(n: int, n_permutations: int, rng: np.random.Generator) -> np.ndarray:
    """``[n_permutations, n]`` random permutations of range(n) (vectorised permutation tests)."""
    return np.argsort(rng.random((n_permutations, n)), axis=1)


def _perm_two_groups(x: np.ndarray, labels: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p of median(x[labels]) - median(x[~labels]) (labels permuted between neurons)."""
    obs = abs(np.median(x[labels]) - np.median(x[~labels]))
    na = int(labels.sum())
    idx = _permutation_index(len(x), n_permutations, rng)
    xs = x[idx]
    null = np.abs(np.median(xs[:, :na], axis=1) - np.median(xs[:, na:], axis=1))
    return (1 + int((null >= obs - 1e-15).sum())) / (1 + n_permutations)


def _perm_sign_flip(d: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p of median(d) (paired within-neuron differences)."""
    obs = abs(np.median(d))
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(d)))
    null = np.abs(np.median(signs * d[None, :], axis=1))
    return (1 + int((null >= obs - 1e-15).sum())) / (1 + n_permutations)


def _perm_omnibus(x: np.ndarray, groups: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    """Permutation p of the between-group spread of the mean RANKS of ``x`` (Kruskal-Wallis statistic)."""
    from scipy.stats import rankdata

    r = rankdata(x) - (len(x) + 1) / 2.0
    _, codes = np.unique(groups, return_inverse=True)
    k = codes.max() + 1
    onehot = np.eye(k)[codes]                      # [n, k]
    sizes = onehot.sum(0)

    def stat(rank_rows):                           # [P, n] -> [P]
        sums = rank_rows @ onehot                  # [P, k]
        return (sums ** 2 / sizes).sum(1)

    obs = stat(r[None, :])[0]
    null = stat(r[_permutation_index(len(r), n_permutations, rng)])
    return (1 + int((null >= obs - 1e-9).sum())) / (1 + n_permutations)


def compare_neuron_groups(med: pd.DataFrame, column: str, a: str, b: str, metrics: Sequence[str], *, n_permutations: int,
                          seed: int, spines: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """Rows (one per metric) of the neuron-level test of value ``a`` vs ``b`` of ``column``."""
    rng = np.random.default_rng(seed)
    sub = med[med[column].isin([a, b])]
    labels = (sub[column] == a).to_numpy()
    rows = []
    for m in metrics:
        x = sub[m].to_numpy(float)
        ok = np.isfinite(x)
        xa, xb = x[ok & labels], x[ok & ~labels]
        row = {"factor": column, "a": a, "b": b, "metric": m, "n_neurons_a": int(len(xa)), "n_neurons_b": int(len(xb)),
               "n_spines_a": int(sub.loc[ok & labels, "n_spines"].sum()), "n_spines_b": int(sub.loc[ok & ~labels, "n_spines"].sum())}
        if len(xa) >= 3 and len(xb) >= 3:
            ma, mb = float(np.median(xa)), float(np.median(xb))
            row.update({"median_a": ma, "median_b": mb, "rel_diff_a_vs_b": ma / mb - 1.0 if mb != 0 else float("nan"),
                        "cliffs_delta": cliffs_delta(xa, xb), "p": _perm_two_groups(x[ok], labels[ok], n_permutations, rng)})
        rows.append(row)
    return rows


def compare_within_neuron(frame: pd.DataFrame, column: str, a: str, b: str, metrics: Sequence[str], *, min_spines: int,
                          n_permutations: int, seed: int) -> List[Dict[str, Any]]:
    """Paired within-neuron test of ``a`` vs ``b`` of a LIMB-level ``column`` (compartment)."""
    rng = np.random.default_rng(seed)
    med = neuron_medians(frame[frame[column].isin([a, b])], metrics, ["neuron_id", column], min_spines)
    wide = med.pivot(index="neuron_id", columns=column)
    rows = []
    for m in metrics:
        row = {"factor": column, "a": a, "b": b, "metric": m}
        if (m, a) not in wide.columns or (m, b) not in wide.columns:
            rows.append(row)
            continue
        pair = wide[[(m, a), (m, b)]].dropna()
        n_sp = wide[[("n_spines", a), ("n_spines", b)]].loc[pair.index]
        row.update({"n_neurons_paired": int(len(pair)), "n_spines_a": int(n_sp[("n_spines", a)].sum()), "n_spines_b": int(n_sp[("n_spines", b)].sum())})
        if len(pair) >= 5:
            va, vb = pair[(m, a)].to_numpy(float), pair[(m, b)].to_numpy(float)
            d = va - vb
            row.update({"median_a": float(np.median(va)), "median_b": float(np.median(vb)),
                        "median_paired_rel_diff": float(np.median(np.where(vb != 0, va / vb - 1.0, np.nan))),
                        "share_neurons_a_gt_b": float(np.mean(d > 0)), "p": _perm_sign_flip(d, n_permutations, rng)})
        rows.append(row)
    return rows


def morphology_by_condition(frame: pd.DataFrame, metrics: Sequence[str], *, min_neurons: int = 10, min_spines: int = 20,
                            n_permutations: int = 10000, seed: int = 0, exclude_types: Sequence[str] = ("Unsure", "Unsure E", "Unsure I"),
                            compartments: Sequence[str] = ("apical", "basal", "uncertain"), log=print) -> Dict[str, pd.DataFrame]:
    """``frame``: one row per spine with ``neuron_id``, ``cell_type``, ``cell_type_binary``, ``compartment`` and the
    metrics. Returns tables ``cell_type_binary``, ``cell_type`` (pairs), ``cell_type_omnibus``, ``compartment``,
    ``cell_type_binary_within_compartment``; q = BH within each table."""
    out: Dict[str, pd.DataFrame] = {}
    neuron_info = frame.groupby("neuron_id")[["cell_type", "cell_type_binary"]].first()

    # binary cell type, all spines
    med = neuron_medians(frame, metrics, ["neuron_id"], min_spines).merge(neuron_info, on="neuron_id")
    rows = compare_neuron_groups(med, "cell_type_binary", "inhibitory", "excitatory", metrics, n_permutations=n_permutations, seed=seed)
    out["cell_type_binary"] = _with_q(rows)
    log(f"[morph] cell_type_binary: {int((med.cell_type_binary == 'inhibitory').sum())} inhibitory / {int((med.cell_type_binary == 'excitatory').sum())} excitatory neurons")

    # detailed types with enough neurons
    counts = med["cell_type"].value_counts()
    types = [t for t, c in counts.items() if c >= min_neurons and isinstance(t, str) and t not in set(exclude_types)]
    log(f"[morph] detailed types with >= {min_neurons} neurons: {dict(counts[types])}; left out: {dict(counts.drop(types, errors='ignore'))}")
    rows, omni = [], []
    sub = med[med["cell_type"].isin(types)]
    rng = np.random.default_rng(seed + 1)
    for m in metrics:
        x = sub[m].to_numpy(float)
        ok = np.isfinite(x)
        if len(types) >= 2 and ok.sum() >= 10:
            omni.append({"factor": "cell_type", "metric": m, "n_types": len(types), "n_neurons": int(ok.sum()),
                         "p": _perm_omnibus(x[ok], sub["cell_type"].to_numpy()[ok], n_permutations, rng)})
    for i, (a, b) in enumerate(combinations(types, 2)):
        rows += compare_neuron_groups(med, "cell_type", a, b, metrics, n_permutations=n_permutations, seed=seed + 10 + i)
    out["cell_type_omnibus"] = _with_q(omni)
    out["cell_type"] = _with_q(rows)

    # compartment, within neuron
    present = [c for c in compartments if (frame["compartment"] == c).any()]
    rows = []
    for i, (a, b) in enumerate(combinations(present, 2)):
        rows += compare_within_neuron(frame, "compartment", a, b, metrics, min_spines=min_spines, n_permutations=n_permutations, seed=seed + 100 + i)
    out["compartment"] = _with_q(rows)

    # binary cell type inside each compartment
    rows = []
    for i, c in enumerate(present):
        med_c = neuron_medians(frame[frame["compartment"] == c], metrics, ["neuron_id"], min_spines).merge(neuron_info, on="neuron_id")
        for r in compare_neuron_groups(med_c, "cell_type_binary", "inhibitory", "excitatory", metrics, n_permutations=n_permutations, seed=seed + 200 + i):
            rows.append({**r, "compartment": c})
    out["cell_type_binary_within_compartment"] = _with_q(rows)
    return out


def _with_q(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if len(table) and "p" in table.columns:
        table["q_bh"] = bh_qvalues(table["p"].to_numpy(float))
        table["significant"] = table["q_bh"] < 0.05
    return table
