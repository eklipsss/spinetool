"""Empirical semivariogram of log radius along a branch: real vs generated (diagnostic of the radius residual).

gamma(h) = 0.5 E[(log r(s + h) - log r(s))^2], pooled over the branches of a radius class, lag in nodes = round(h / step).
It shows WHICH scale of the radius variation a model loses: a missing fast component (section noise, local swellings, a
fixed size in nm that is large relative to a thin radius) makes the generated gamma too small at short lags, a missing
slow component at long lags. sqrt(2 gamma) is the typical difference of log r (= relative radius change). Units: nm.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

LAGS_NM = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
RADIUS_CLASSES = (("r < 300 nm", 0.0, 300.0), ("300-500 nm", 300.0, 500.0), ("r >= 500 nm", 500.0, 1e12))


def semivariogram(series: Sequence[np.ndarray], steps: Sequence[float], lags_nm: Sequence[float] = LAGS_NM) -> Dict[float, Optional[float]]:
    """Pooled semivariance of log r series (one array per branch, nodes ``step`` apart) at each lag (None if no pair)."""
    out: Dict[float, Optional[float]] = {}
    for h in lags_nm:
        total, count = 0.0, 0
        for x, step in zip(series, steps):
            lag = int(round(h / step))
            if lag < 1 or len(x) <= lag + 1 or abs(lag * step - h) > 0.5 * step:
                continue  # the lag is not resolved by this step (or the branch is shorter)
            d = x[lag:] - x[:-lag]
            total += float(np.sum(d**2))
            count += len(d)
        out[h] = 0.5 * total / count if count >= 30 else None
    return out


def by_radius_class(log_radius: Sequence[np.ndarray], steps: Sequence[float], radii: Sequence[float]) -> Dict[str, Dict[float, Optional[float]]]:
    """Semivariogram per class of the branch median radius (``radii``, nm)."""
    radii = np.asarray(radii, dtype=float)
    out = {}
    for name, lo, hi in RADIUS_CLASSES:
        idx = np.flatnonzero((radii >= lo) & (radii < hi))
        if len(idx) >= 3:
            out[name] = {"n": len(idx), **semivariogram([log_radius[i] for i in idx], [steps[i] for i in idx])}
    return out
