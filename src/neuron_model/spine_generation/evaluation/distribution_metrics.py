"""Distribution comparison of morphometrics: real vs generated (spec, "Тестирование").

- per scalar metric: Wasserstein-1, Kolmogorov-Smirnov, Energy distance;
- OldChordDistribution (a 100-bin density per spine): W1 between the mean densities +
  MMD between the per-spine histogram vectors;
- joint: the 10-feature vector standardised with TRAIN statistics -> RBF-kernel MMD,
  and the difference of correlation matrices ``||R_real - R_gen||_F`` (a model can match
  every marginal and still break the relations between features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

# spec's joint morphometric vector (OldChordDistribution is a histogram, JunctionArea is
# left out of the joint vector by the spec)
JOINT_FEATURES = (
    "OpenAngle",
    "CVD",
    "AverageDistance",
    "LengthVolumeRatio",
    "LengthAreaRatio",
    "Length",
    "Area",
    "Volume",
    "ConvexHullVolume",
    "ConvexHullRatio",
)
UNIVARIATE_FEATURES = JOINT_FEATURES + ("JunctionArea",)


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def univariate_distances(real, generated, *, scale: Optional[float] = None) -> Dict[str, float]:
    """W1 / KS / Energy between two 1-D samples (non-finite values dropped and counted).

    ``scale`` (default: std of ``real``) gives ``w1_normalized = w1 / scale`` - W1 is in
    the metric's own units, so the normalised value is what is comparable across metrics.
    """
    r, g = _finite(real), _finite(generated)
    result: Dict[str, float] = {
        "n_real": int(r.size),
        "n_generated": int(g.size),
        "n_real_missing": int(np.asarray(real, dtype=float).size - r.size),
        "n_generated_missing": int(np.asarray(generated, dtype=float).size - g.size),
    }
    if r.size < 2 or g.size < 2:
        return {**result, "w1": np.nan, "w1_normalized": np.nan, "ks_statistic": np.nan, "ks_pvalue": np.nan, "energy": np.nan}
    scale = float(np.std(r)) if scale is None else float(scale)
    ks = stats.ks_2samp(r, g)
    w1 = float(stats.wasserstein_distance(r, g))
    result.update(
        {
            "w1": w1,
            "w1_normalized": w1 / scale if scale > 0 else np.nan,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "energy": float(stats.energy_distance(r, g)),
            "mean_real": float(r.mean()),
            "mean_generated": float(g.mean()),
            "std_real": float(r.std()),
            "std_generated": float(g.std()),
        }
    )
    return result


def univariate_table(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    features: Sequence[str] = UNIVARIATE_FEATURES,
    *,
    scales: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """One row per feature (features missing from either table are skipped).

    ``scales`` - per-feature normaliser for ``w1_normalized`` (pass the TRAIN std so every
    model is normalised identically; default: std of ``real``).
    """
    rows = []
    for feature in features:
        if feature not in real.columns or feature not in generated.columns:
            continue
        scale = None if scales is None else scales.get(feature)
        rows.append({"feature": feature, **univariate_distances(real[feature], generated[feature], scale=scale)})
    return pd.DataFrame(rows)


def _stack_histograms(column) -> np.ndarray:
    rows = [np.asarray(v, dtype=float) for v in column if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not rows:
        return np.empty((0, 0))
    return np.vstack(rows)


def histogram_distances(real_hists, generated_hists, *, value_range: Tuple[float, float] = (0.0, 1.0)) -> Dict[str, float]:
    """Compare per-spine histogram metrics (OldChordDistribution: density over ``value_range``).

    ``w1_mean_density``: W1 between the two average densities (a distribution over the bin
    centres); ``mmd2``: RBF MMD^2 between the per-spine histogram vectors.
    """
    r, g = _stack_histograms(real_hists), _stack_histograms(generated_hists)
    if r.shape[0] < 2 or g.shape[0] < 2:
        return {"n_real": int(r.shape[0]), "n_generated": int(g.shape[0]), "w1_mean_density": np.nan, "mmd2": np.nan}
    if r.shape[1] != g.shape[1]:
        raise ValueError(f"Histogram bin counts differ: {r.shape[1]} vs {g.shape[1]}")
    n_bins = r.shape[1]
    lo, hi = value_range
    centers = lo + (np.arange(n_bins) + 0.5) * (hi - lo) / n_bins
    mean_r, mean_g = r.mean(axis=0), g.mean(axis=0)
    w1 = stats.wasserstein_distance(centers, centers, u_weights=np.clip(mean_r, 0, None) + 1e-12, v_weights=np.clip(mean_g, 0, None) + 1e-12)
    mmd2, sigma = rbf_mmd2(r, g)
    return {"n_real": int(r.shape[0]), "n_generated": int(g.shape[0]), "w1_mean_density": float(w1), "mmd2": float(mmd2), "mmd_sigma": float(sigma)}


@dataclass(frozen=True)
class Standardizer:
    """``(x - mean_train) / std_train`` per feature - fitted on the TRAIN split only (spec)."""

    features: Tuple[str, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]

    @classmethod
    def fit(cls, train: pd.DataFrame, features: Sequence[str] = JOINT_FEATURES) -> "Standardizer":
        values = train[list(features)].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        if np.any(~np.isfinite(std)) or np.any(std == 0):
            bad = [f for f, s in zip(features, std) if not np.isfinite(s) or s == 0]
            raise ValueError(f"Cannot standardise constant/empty features: {bad}")
        return cls(tuple(features), tuple(mean.tolist()), tuple(std.tolist()))

    def transform(self, frame: pd.DataFrame, *, drop_incomplete: bool = True) -> np.ndarray:
        values = frame[list(self.features)].to_numpy(dtype=float)
        values = (values - np.asarray(self.mean)) / np.asarray(self.std)
        if drop_incomplete:
            values = values[np.isfinite(values).all(axis=1)]
        return values

    def to_dict(self) -> Dict[str, list]:
        return {"features": list(self.features), "mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence]) -> "Standardizer":
        return cls(tuple(data["features"]), tuple(float(v) for v in data["mean"]), tuple(float(v) for v in data["std"]))


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # cdist computes each difference directly (numerically stable, unlike sq_a + sq_b - 2ab -
    # see pointinfinity.knn_indices) with O(n*m) memory, no [n, m, d] intermediate
    return cdist(x, y, metric="sqeuclidean")


def median_heuristic_sigma(x: np.ndarray, y: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:
    """Median pairwise distance of the pooled sample (standard RBF bandwidth choice)."""
    pooled = np.vstack([x, y])
    if len(pooled) > max_points:
        pooled = pooled[np.random.default_rng(seed).choice(len(pooled), max_points, replace=False)]
    d = np.sqrt(_squared_distances(pooled, pooled))
    upper = d[np.triu_indices(len(pooled), k=1)]
    sigma = float(np.median(upper))
    return sigma if sigma > 0 else 1.0


def rbf_mmd2(x: np.ndarray, y: np.ndarray, *, sigma: Optional[float] = None, unbiased: bool = True) -> Tuple[float, float]:
    """MMD^2 with a Gaussian kernel ``exp(-||a-b||^2 / (2 sigma^2))``; returns ``(mmd2, sigma)``.

    Unbiased U-statistic by default (can be slightly negative for equal distributions);
    ``sigma=None`` -> median heuristic on the pooled sample. Compare models with ONE fixed
    sigma (pass it explicitly), otherwise the scale of the kernel differs per model.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sigma = median_heuristic_sigma(x, y) if sigma is None else float(sigma)
    gamma = 1.0 / (2.0 * sigma**2)
    k_xx = np.exp(-gamma * _squared_distances(x, x))
    k_yy = np.exp(-gamma * _squared_distances(y, y))
    k_xy = np.exp(-gamma * _squared_distances(x, y))
    n, m = len(x), len(y)
    if unbiased:
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD needs at least 2 samples per set")
        term_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        term_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    else:
        term_xx, term_yy = k_xx.mean(), k_yy.mean()
    return float(term_xx + term_yy - 2.0 * k_xy.mean()), sigma


def mmd_permutation_pvalue(x: np.ndarray, y: np.ndarray, *, sigma: float, n_permutations: int = 200, seed: int = 0) -> float:
    """p-value of H0 "same distribution" by permuting the pooled sample's labels."""
    pooled = np.vstack([x, y])
    n = len(x)
    observed, _ = rbf_mmd2(x, y, sigma=sigma)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(pooled))
        value, _ = rbf_mmd2(pooled[perm[:n]], pooled[perm[n:]], sigma=sigma)
        exceed += value >= observed
    return float((exceed + 1) / (n_permutations + 1))


def correlation_difference(real_x: np.ndarray, gen_x: np.ndarray, *, method: str = "pearson") -> Dict[str, object]:
    """``||Corr(real) - Corr(gen)||_F`` (+ max absolute entry difference and both matrices)."""
    if method == "pearson":
        r_real, r_gen = np.corrcoef(real_x, rowvar=False), np.corrcoef(gen_x, rowvar=False)
    elif method == "spearman":
        r_real, r_gen = stats.spearmanr(real_x).statistic, stats.spearmanr(gen_x).statistic
    else:
        raise ValueError(f"method must be pearson or spearman, got {method!r}")
    diff = np.nan_to_num(r_real - r_gen)
    return {
        "frobenius": float(np.linalg.norm(diff, ord="fro")),
        "max_abs": float(np.abs(diff).max()),
        "R_real": r_real,
        "R_generated": r_gen,
    }
