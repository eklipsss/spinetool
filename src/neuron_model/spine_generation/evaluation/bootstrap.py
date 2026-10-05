"""Uncertainty of the comparison metrics.

- :func:`bootstrap_ci`: percentile confidence interval of a two-sample statistic by
  resampling real and generated rows independently with replacement - tells whether a
  difference between two models is larger than the noise of a finite sample;
- :func:`aggregate_over_seeds`: mean / std / min / max of one metric across the
  generation seeds (spec protocol: 1000 samples x 5 generation seeds).
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

import numpy as np


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    real: np.ndarray,
    generated: np.ndarray,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    real, generated = np.asarray(real), np.asarray(generated)
    rng = np.random.default_rng(seed)
    point = float(statistic(real, generated))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        samples[b] = statistic(real[rng.integers(0, len(real), len(real))], generated[rng.integers(0, len(generated), len(generated))])
    alpha = (1.0 - confidence) / 2.0
    finite = samples[np.isfinite(samples)]
    return {
        "value": point,
        "ci_low": float(np.quantile(finite, alpha)) if finite.size else np.nan,
        "ci_high": float(np.quantile(finite, 1.0 - alpha)) if finite.size else np.nan,
        "boot_std": float(finite.std()) if finite.size else np.nan,
        "confidence": confidence,
        "n_boot": n_boot,
    }


def aggregate_over_seeds(per_seed: Mapping[int, Mapping[str, float]], keys: Sequence[str] = ()) -> Dict[str, Dict[str, float]]:
    """``{seed: {metric: value}}`` -> ``{metric: {mean, std, min, max, n_seeds}}`` (numeric metrics only)."""
    if not per_seed:
        return {}
    keys = keys or sorted({k for values in per_seed.values() for k, v in values.items() if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)})
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(per_seed[s][key]) for s in per_seed if key in per_seed[s]], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            result[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_seeds": int(values.size),
            }
    return result
