"""Choice of the pool spine for a placement: at random, or matched to the local trunk radius ("context").

Real data (``branch_spines.parquet`` of the branch preprocessing: attachment loop area of every spine and the trunk
radius at its attachment point) give the conditional distribution of the base size given the trunk radius::

    log b = a + c log(r / 300 nm) + N(0, s^2),   b = sqrt(loop area / pi)   (equivalent base radius)

At a placement with trunk radius r the pool spine k is drawn with probability proportional to
N(log b_k; a + c log(r/300), s^2) / g(log b_k), g = kernel density of log b over the pool: the draws then follow the
real conditional distribution (importance weights), not the pool's own size distribution. The pool base size comes
from ``DownFacingCapFinder`` (``loop_area`` = projection of the cap, validated against the real loop at ~12 %, +6 %
bias) - the real one from the seal of the S preprocessing. A weight floor keeps every spine possible. Units: nm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

R_REF_NM = 300.0


def base_radius(loop_area_nm2: Optional[float], cap_area_nm2: Optional[float] = None) -> float:
    """Equivalent base radius sqrt(area / pi) (nm); the loop (projected) area, else the cap area."""
    for a in (loop_area_nm2, cap_area_nm2):
        if a is not None and np.isfinite(a) and a > 0:
            return math.sqrt(float(a) / math.pi)
    return float("nan")


@dataclass
class ContextSelector:
    a: float                 # log base radius at r = 300 nm
    c: float                 # slope on log(r / 300 nm)
    s: float                 # residual spread
    info: Dict[str, Any] = field(default_factory=dict)
    weight_floor: float = 1e-3  # relative to the largest weight of a placement

    @classmethod
    def fit(cls, spines, *, min_rows: int = 30) -> "ContextSelector":
        """``spines``: DataFrame with ``attachment_loop_area`` (or ``attachment_cap_area``) and ``radius_at_s`` (real data)."""
        loop = spines["attachment_loop_area"] if "attachment_loop_area" in spines else None
        cap = spines["attachment_cap_area"] if "attachment_cap_area" in spines else None
        b = np.array([base_radius(None if loop is None else loop.iloc[i], None if cap is None else cap.iloc[i]) for i in range(len(spines))])
        r = spines["radius_at_s"].to_numpy(dtype=float)
        ok = np.isfinite(b) & np.isfinite(r) & (r > 0) & (b > 0)
        if "has_position" in spines:
            ok &= spines["has_position"].fillna(False).to_numpy(dtype=bool)
        if ok.sum() < min_rows:
            raise ValueError(f"only {int(ok.sum())} real spines with a base area and a trunk radius (need {min_rows})")
        y, x = np.log(b[ok]), np.log(r[ok] / R_REF_NM)
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        s = float(np.sqrt(np.mean(resid**2)))
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        return cls(float(coef[0]), float(coef[1]), max(s, 0.05), {"n": int(ok.sum()), "corr_log_r_log_b": corr,
                                                                  "base_radius_median_nm": float(np.exp(np.median(y))),
                                                                  "trunk_radius_median_nm": float(np.median(r[ok]))})

    def probabilities(self, pool_log_b: np.ndarray, radius_nm: float, pool_density: np.ndarray) -> np.ndarray:
        mu = self.a + self.c * math.log(max(float(radius_nm), 1.0) / R_REF_NM)
        logw = -0.5 * ((pool_log_b - mu) / self.s) ** 2 - np.log(np.maximum(pool_density, 1e-12))
        logw[~np.isfinite(pool_log_b)] = -np.inf
        w = np.exp(logw - np.max(logw[np.isfinite(logw)]))
        w = np.where(np.isfinite(pool_log_b), np.maximum(w, self.weight_floor), 0.0)
        return w / w.sum()

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "c": self.c, "s": self.s, "weight_floor": self.weight_floor, "info": dict(self.info)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextSelector":
        return cls(float(d["a"]), float(d["c"]), float(d["s"]), dict(d.get("info", {})), float(d.get("weight_floor", 1e-3)))


def pool_log_base(pool) -> np.ndarray:
    return np.log(np.array([sp.base_radius_nm for sp in pool.spines], dtype=float))


def kde_density(values: np.ndarray) -> np.ndarray:
    """Gaussian kernel density of ``values`` at themselves (Silverman bandwidth); nan stays nan."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    out = np.full(len(v), np.nan)
    x = v[ok]
    if len(x) == 0:
        return out
    h = 1.06 * max(float(np.std(x)), 1e-3) * len(x) ** (-0.2)
    d = (x[:, None] - x[None, :]) / h
    out[ok] = np.exp(-0.5 * d**2).sum(1) / (len(x) * h * math.sqrt(2 * math.pi))
    return out


class SpineChooser:
    """Draws a pool index for a placement: uniform (``selector=None``) or by the trunk radius."""

    def __init__(self, pool, selector: Optional[ContextSelector] = None) -> None:
        self.pool = pool
        self.selector = selector
        if selector is not None:
            self.log_b = pool_log_base(pool)
            if not np.isfinite(self.log_b).any():
                raise ValueError("no pool spine has a base area: context selection impossible")
            self.density = kde_density(self.log_b)

    def choose(self, rng: np.random.Generator, radius_nm: float) -> int:
        if self.selector is None:
            return int(rng.integers(len(self.pool)))
        return int(rng.choice(len(self.pool), p=self.selector.probabilities(self.log_b, radius_nm, self.density)))
