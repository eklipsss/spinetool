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
