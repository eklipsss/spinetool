"""Do real spines differ between the values of a condition block? (s-module-conditional-design.md, section 6, check 0)

A block is worth conditioning on only if its values change the spine morphology more than the variation between
neurons does. Conditions are properties of the NEURON (cell type) or the limb (compartment), and a rare value is
carried by a few neurons, so a naive test (spines as independent samples) would call any neuron-to-neuron style a
"condition effect". Here:

- statistic for a pair of values (a, b): mean over the scalar morphometrics of W1 / sigma between the spines of a
  and of b (features standardised over both groups; spines subsampled to at most ``max_per_neuron`` per neuron, so
  that a few very dense neurons do not dominate);
- null distribution: the value labels are PERMUTED BETWEEN NEURONS (each neuron keeps all its spines and gets the
  label of another neuron of the pair), ``n_permutations`` times; p = (1 + #{perm >= observed}) / (1 + n);
  the noise floor = 95th percentile of the permuted statistic;
- secondary: balanced accuracy of a logistic classifier of the two values with GroupKFold BY NEURON (chance 0.5);
- a block is enabled when at least one of its pairs is significant after Bonferroni over the pairs of the block
  (alpha 0.05) and both values have >= ``min_neurons`` neurons; a block with one value in the data, or without a
  testable pair, is disabled (it carries no information for this training set).

Values: the scalar morphometrics of preprocessing stage 8 (``spine_morphometrics.SCALAR_METRICS``).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _w1_rel(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(a, b))


def pair_statistic(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean over features of W1 between label 1 and label 0 rows of standardised ``x`` [N, F] (NaN-safe per feature)."""
    vals = []
    for f in range(x.shape[1]):
        col = x[:, f]
        ok = np.isfinite(col)
        a, b = col[ok & labels], col[ok & ~labels]
        if len(a) >= 5 and len(b) >= 5:
            vals.append(_w1_rel(a, b))
    return float(np.mean(vals)) if vals else float("nan")


def per_feature_w1(x: np.ndarray, labels: np.ndarray, names: Sequence[str]) -> Dict[str, float]:
    out = {}
    for f, name in enumerate(names):
        col = x[:, f]
        ok = np.isfinite(col)
        a, b = col[ok & labels], col[ok & ~labels]
        out[name] = _w1_rel(a, b) if len(a) >= 5 and len(b) >= 5 else float("nan")
    return out


def _standardise(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    return (x - mu) / sd


def _classifier_accuracy(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    ok = np.isfinite(x).all(axis=1)
    x, y, g = x[ok], labels[ok].astype(int), groups[ok]
    n_groups = len(np.unique(g))
    if n_groups < 4 or len(np.unique(y)) < 2:
        return float("nan")
    folds = GroupKFold(n_splits=min(n_splits, n_groups))
    pred = np.empty(len(y), dtype=int)
    for tr, te in folds.split(x, y, g):
        if len(np.unique(y[tr])) < 2:
            pred[te] = int(np.round(y[tr].mean()))
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(x[tr], y[tr])
        pred[te] = clf.predict(x[te])
    return float(balanced_accuracy_score(y, pred))


def test_pair(features: pd.DataFrame, neurons: pd.Series, values: pd.Series, a: str, b: str, feature_names: Sequence[str], *,
              n_permutations: int = 200, max_per_neuron: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Neuron-level permutation test of value ``a`` vs ``b`` (rows = spines)."""
    rng = np.random.default_rng(seed)
    mask = values.isin([a, b]).to_numpy()
    frame = pd.DataFrame({"neuron": neurons.to_numpy()[mask], "value": values.to_numpy()[mask]})
    x_all = features.to_numpy(dtype=float)[mask]
    # at most max_per_neuron spines per neuron
    keep = np.zeros(len(frame), dtype=bool)
    for _, idx in frame.groupby("neuron").indices.items():
        idx = np.asarray(idx)
        keep[idx if len(idx) <= max_per_neuron else rng.choice(idx, max_per_neuron, replace=False)] = True
    frame, x = frame[keep].reset_index(drop=True), _standardise(x_all[keep])
    neuron_value = frame.groupby("neuron")["value"].agg(lambda s: s.mode().iloc[0])
    n_a, n_b = int((neuron_value == a).sum()), int((neuron_value == b).sum())
    labels = (frame["value"] == a).to_numpy()
    observed = pair_statistic(x, labels)
    neuron_ids = neuron_value.index.to_numpy()
    neuron_labels = (neuron_value == a).to_numpy()
    row_neuron = frame["neuron"].to_numpy()
    pos = {n: k for k, n in enumerate(neuron_ids)}
    row_pos = np.fromiter((pos[n] for n in row_neuron), dtype=int, count=len(row_neuron))
    perm = np.empty(n_permutations)
    for r in range(n_permutations):
        shuffled = rng.permutation(neuron_labels)
        perm[r] = pair_statistic(x, shuffled[row_pos])
    perm = perm[np.isfinite(perm)]
    p_value = float((1 + np.sum(perm >= observed)) / (1 + len(perm))) if len(perm) else float("nan")
    return {
        "a": a, "b": b, "n_neurons_a": n_a, "n_neurons_b": n_b, "n_spines_a": int(labels.sum()), "n_spines_b": int((~labels).sum()),
        "mean_w1_rel": observed, "noise_floor_p95": float(np.percentile(perm, 95)) if len(perm) else float("nan"),
        "permuted_median": float(np.median(perm)) if len(perm) else float("nan"), "p_value": p_value,
        "per_feature_w1_rel": per_feature_w1(x, labels, feature_names),
        "classifier_balanced_accuracy": _classifier_accuracy(x, labels, row_neuron, seed=seed),
    }


def condition_effects(features: pd.DataFrame, conditions: pd.DataFrame, neurons: pd.Series, blocks: Sequence[tuple], *,
                      min_neurons: int = 5, alpha: float = 0.05, n_permutations: int = 200, max_per_neuron: int = 200,
                      seed: int = 0, log=print) -> Dict[str, Any]:
    """Report per block: value counts, every testable pair, ``enabled`` + ``reason``."""
    feature_names = list(features.columns)
    out: Dict[str, Any] = {"blocks": {}, "settings": {"min_neurons": min_neurons, "alpha": alpha, "n_permutations": n_permutations,
                                                     "max_per_neuron": max_per_neuron, "features": feature_names}}
    for name, values in blocks:
        if name not in conditions.columns:
            out["blocks"][name] = {"enabled": False, "reason": "no column"}
            continue
        col = conditions[name].astype(object)
        neuron_counts = pd.DataFrame({"n": neurons.to_numpy(), "v": col.to_numpy()}).dropna().drop_duplicates()
        counts = {str(v): {"spines": int((col == v).sum()), "neurons": int((neuron_counts["v"] == v).sum())} for v in values}
        counts["unknown"] = {"spines": int(col.isna().sum())}
        present = [v for v in values if counts[str(v)]["spines"] > 0]
        testable = [v for v in present if counts[str(v)]["neurons"] >= min_neurons]
        block: Dict[str, Any] = {"counts": counts, "pairs": []}
        if len(present) < 2:
            block.update(enabled=False, reason="one value in the data" if present else "no value in the data")
        elif len(testable) < 2:
            block.update(enabled=False, reason=f"fewer than 2 values with >= {min_neurons} neurons")
        else:
            pairs = list(combinations(testable, 2))
            threshold = alpha / len(pairs)  # Bonferroni within the block
            for a, b in pairs:
                res = test_pair(features, neurons, col, a, b, feature_names, n_permutations=n_permutations, max_per_neuron=max_per_neuron, seed=seed)
                res["significant"] = bool(res["p_value"] < threshold)
                block["pairs"].append(res)
                if log:
                    log(f"[effects] {name}: {a} vs {b} ({res['n_neurons_a']} / {res['n_neurons_b']} neurons): mean W1/sigma {res['mean_w1_rel']:.3f}, "
                        f"floor (neuron permutation, p95) {res['noise_floor_p95']:.3f}, p {res['p_value']:.3f}, classifier {res['classifier_balanced_accuracy']:.3f}")
            sig = [p for p in block["pairs"] if p["significant"]]
            block.update(enabled=bool(sig), reason=(f"{len(sig)} of {len(pairs)} pairs above the neuron-level noise (Bonferroni alpha {alpha})"
                                                     if sig else f"no pair above the neuron-level noise (Bonferroni alpha {alpha})"))
        out["blocks"][name] = block
    return out
