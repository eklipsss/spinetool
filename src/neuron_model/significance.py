"""Is a difference between generated and real samples larger than chance? (T and P evaluation)

Branches and spines of one neuron are not independent (neurons differ in style), so a plain two-sample KS p-value is far too
optimistic with thousands of branches: it tests "are these two piles of numbers from one distribution" as if every number were
an independent draw. Here the unit of resampling is the NEURON (cluster bootstrap): neurons are drawn with replacement, all
generated and real values of a drawn neuron come along, and the difference of the medians is recomputed. The 95 % interval of
this difference tells the size and the sign of the discrepancy; "significant" = the interval excludes 0. The KS p-value of the
plain test is kept next to it for reference only.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def ks_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy import stats

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(stats.ks_2samp(a, b).pvalue) if len(a) >= 2 and len(b) >= 2 else float("nan")


def cluster_bootstrap_diff(a: Sequence[float], groups_a: Sequence[str], b: Sequence[float], groups_b: Sequence[str], *,
                           n_boot: int = 1000, seed: int = 0, statistic=np.median) -> Dict[str, float]:
    """Difference ``statistic(a) - statistic(b)`` with a 95 % interval from resampling clusters (neurons).

    ``groups_*``: the cluster (neuron) of every value. Returns diff, lo, hi, p (two-sided bootstrap), significant, n_clusters.
    NaN values are ignored."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ga, gb = np.asarray(groups_a).astype(str), np.asarray(groups_b).astype(str)
    ok_a, ok_b = np.isfinite(a), np.isfinite(b)
    a, ga, b, gb = a[ok_a], ga[ok_a], b[ok_b], gb[ok_b]
    names = np.array(sorted(set(ga) | set(gb)))
    nan = float("nan")
    if len(a) < 2 or len(b) < 2 or len(names) < 2:
        return {"diff": nan, "lo": nan, "hi": nan, "p": nan, "significant": False, "n_clusters": int(len(names))}
    pos = {n: i for i, n in enumerate(names)}
    ia = [[] for _ in names]
    ib = [[] for _ in names]
    for k, n in enumerate(ga):
        ia[pos[n]].append(k)
    for k, n in enumerate(gb):
        ib[pos[n]].append(k)
    ia = [np.asarray(x, dtype=int) for x in ia]
    ib = [np.asarray(x, dtype=int) for x in ib]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        draw = rng.integers(0, len(names), len(names))
        va = np.concatenate([a[ia[i]] for i in draw]) if any(len(ia[i]) for i in draw) else np.zeros(0)
        vb = np.concatenate([b[ib[i]] for i in draw]) if any(len(ib[i]) for i in draw) else np.zeros(0)
        diffs[r] = statistic(va) - statistic(vb) if len(va) and len(vb) else np.nan
    diffs = diffs[np.isfinite(diffs)]
    if not len(diffs):
        return {"diff": nan, "lo": nan, "hi": nan, "p": nan, "significant": False, "n_clusters": int(len(names))}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(min(1.0, 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return {"diff": float(statistic(a) - statistic(b)), "lo": float(lo), "hi": float(hi), "p": p, "significant": bool(lo > 0 or hi < 0), "n_clusters": int(len(names))}


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D = sup |F_a - F_b| (NaN-free 1-D arrays)."""
    a, b = np.sort(a), np.sort(b)
    if not len(a) or not len(b):
        return float("nan")
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(b, x, side="right") / len(b))))


def ks_neuron_floor_test(gen: Sequence[float], gen_neurons: Sequence[str], real: Sequence[float], real_neurons: Sequence[str], *,
                         n_splits: int = 1000, seed: int = 0, min_neurons: int = 4) -> Dict[str, float]:
    """Kolmogorov-Smirnov with the NEURON as the unit: is the generated distribution farther from the real one than one
    half of the real neurons is from the other half?

    Every generated value belongs to the neuron of its real prototype. For each of ``n_splits`` random splits of the real
    neurons into halves A / B:
      null  D_null = D(real of A, real of B)                       two sets of real neurons, sizes of a half
      model D_mod  = (D(gen of A, real of B) + D(gen of B, real of A)) / 2   the same sizes and the same split
    The null distribution of D is the one of D_null over the splits (the spread of two independent groups of real neurons,
    neuron style included); the model statistic is the mean of D_mod over the splits. p = (1 + #{D_null >= mean D_mod}) /
    (1 + n_splits). The mean of D_mod has a smaller spread than a single D_null, so the test is conservative: under a
    correct generator p is around 0.5, it becomes small only when the generated values are typically farther from the real
    ones than a typical half of the real neurons is from the other half. The plain KS p-value treats thousands of branches
    of ~80 neurons as independent and rejects even a perfect generator (see the module docstring).
    """
    a, b = np.asarray(gen, dtype=float), np.asarray(real, dtype=float)
    ga, gb = np.asarray(gen_neurons).astype(str), np.asarray(real_neurons).astype(str)
    ok_a, ok_b = np.isfinite(a), np.isfinite(b)
    a, ga, b, gb = a[ok_a], ga[ok_a], b[ok_b], gb[ok_b]
    names = np.array(sorted(set(gb)))
    nan = float("nan")
    out = {"d": ks_statistic(a, b) if len(a) and len(b) else nan, "d_model": nan, "d_null_median": nan, "d_null_p95": nan,
           "p": nan, "significant": False, "n_neurons": int(len(names)), "n_splits": 0}
    if len(names) < min_neurons or len(a) < 2 or len(b) < 2:
        return out
    rng = np.random.default_rng(seed)
    d_null, d_mod = [], []
    for _ in range(n_splits):
        half = set(rng.permutation(names)[: len(names) // 2])
        ra, rb = np.isin(gb, list(half)), ~np.isin(gb, list(half))
        qa, qb = np.isin(ga, list(half)), ~np.isin(ga, list(half))
        if min(ra.sum(), rb.sum(), qa.sum(), qb.sum()) < 2:
            continue
        d_null.append(ks_statistic(b[ra], b[rb]))
        d_mod.append(0.5 * (ks_statistic(a[qa], b[rb]) + ks_statistic(a[qb], b[ra])))
    if not d_null:
        return out
    d_null, d_mod = np.asarray(d_null), np.asarray(d_mod)
    m = float(d_mod.mean())
    p = float((1 + np.sum(d_null >= m)) / (1 + len(d_null)))
    out.update(d_model=m, d_null_median=float(np.median(d_null)), d_null_p95=float(np.percentile(d_null, 95)), p=p,
               significant=bool(p < 0.05), n_splits=int(len(d_null)))
    return out
