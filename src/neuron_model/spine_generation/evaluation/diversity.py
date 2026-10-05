"""Diversity: is the generated set as varied as the real one? (spec, "diversity")

Compares the distribution of pairwise distances WITHIN the real set (real-real) with the
one WITHIN the generated set (generated-generated), both by Chamfer distance on point
clouds and by Euclidean distance in standardised morphometric space, plus the per-feature
variance ratio. A generated-generated distribution much narrower / closer to 0 than
real-real is the signature of mode collapse.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist


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
) -> Dict[str, float]:
    """Pairwise Chamfer, pairwise morphometric distance and feature variances, real vs generated."""
    chamfer = compare_pairwise(upper_triangle(real_real_chamfer), upper_triangle(gen_gen_chamfer))
    morph = compare_pairwise(pdist(real_x), pdist(gen_x))
    return {
        **{f"chamfer_{k}": v for k, v in chamfer.items()},
        **{f"morphometric_{k}": v for k, v in morph.items()},
        **feature_variance_ratio(real_x, gen_x, features),
    }
