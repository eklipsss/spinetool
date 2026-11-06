"""Diversity: is the generated set as varied as the real one? (spec, "diversity")

Compares the distribution of pairwise distances WITHIN the real set (real-real) with the
one WITHIN the generated set (generated-generated), both by Chamfer distance on point
clouds and by Euclidean distance in standardised morphometric space, plus the per-feature
variance ratio. A generated-generated distribution much narrower / closer to 0 than
real-real is the signature of mode collapse.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist


MAX_REAL_PAIRS = 5_000


def upper_triangle(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return d[np.triu_indices(d.shape[0], k=1)]


def compare_pairwise(real_pairwise: np.ndarray, generated_pairwise: np.ndarray) -> Dict[str, float]:
    """``mean_ratio`` = mean(gen-gen) / mean(real-real): ~1 is the target, << 1 means collapse."""
    r, g = np.asarray(real_pairwise, dtype=float), np.asarray(generated_pairwise, dtype=float)
    return {
        "mean_real": float(r.mean()),
        "mean_generated": float(g.mean()),
        "mean_ratio": float(g.mean() / r.mean()) if r.mean() > 0 else np.nan,
        "median_ratio": float(np.median(g) / np.median(r)) if np.median(r) > 0 else np.nan,
        "std_ratio": float(g.std() / r.std()) if r.std() > 0 else np.nan,
        "w1": float(stats.wasserstein_distance(r, g)),
        "ks_statistic": float(stats.ks_2samp(r, g).statistic),
    }


def feature_variance_ratio(real_x: np.ndarray, gen_x: np.ndarray, features: Sequence[str]) -> Dict[str, float]:
    """var_gen / var_real per (standardised) feature + their geometric mean."""
    var_r, var_g = np.var(real_x, axis=0), np.var(gen_x, axis=0)
    ratios = var_g / np.where(var_r > 0, var_r, np.nan)
    result = {f"variance_ratio_{name}": float(v) for name, v in zip(features, ratios)}
    finite = ratios[np.isfinite(ratios) & (ratios > 0)]
    result["variance_ratio_geometric_mean"] = float(np.exp(np.log(finite).mean())) if finite.size else np.nan
    return result


def diversity_report(
    *,
    real_real_chamfer: np.ndarray,
    gen_gen_chamfer: np.ndarray,
    real_x: np.ndarray,
    gen_x: np.ndarray,
    features: Sequence[str],
    max_real_pairs: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Pairwise Chamfer, pairwise morphometric distance and feature variances, real vs generated. ``max_real_pairs``:
    at most this many real spines enter the real-real pairwise distances (default ``MAX_REAL_PAIRS`` = 5 000)."""
    max_real_pairs = MAX_REAL_PAIRS if max_real_pairs is None else int(max_real_pairs)
    chamfer = compare_pairwise(upper_triangle(real_real_chamfer), upper_triangle(gen_gen_chamfer))
    # all real-real pairs of a large reference do not fit in memory (69 520 untouched test spines -> 2.4e9 pairs, 18 GB):
    # a seeded subset (12.5 M pairs from 5 000 spines estimate the distribution of pairwise distances well; 15 000 spines
    # = 112 M pairs needed ~9 GB in the Wasserstein / KS sorting)
    rx = np.asarray(real_x)
    if len(rx) > max_real_pairs:
        rx = rx[np.sort(np.random.default_rng(seed).choice(len(rx), max_real_pairs, replace=False))]
    morph = compare_pairwise(pdist(rx), pdist(gen_x))
    return {
        **{f"chamfer_{k}": v for k, v in chamfer.items()},
        **{f"morphometric_{k}": v for k, v in morph.items()},
        **feature_variance_ratio(real_x, gen_x, features),
    }
