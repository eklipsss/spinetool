"""Radius residual of T0 with a per-branch amplitude (random effect).

Why: the joint OU of T0 has ONE stationary variance of rho = log r - m(s) for all branches (per group). Its maximum
likelihood value is the node-weighted mean square, i.e. it is set by the long branches; if the relative thickness
fluctuations differ between branches (thin / short branches fluctuate more in log r: the section noise and the
wiggles are of fixed size in nm), every generated branch gets the same typical amplitude and the spread of the
radius along a branch is off (10 Minnie neurons, dendrites: radius CV along a branch 0.07 generated vs 0.11-0.13 real).

Model (rho_j at nodes ``step`` apart, exact AR(1) discretisation of an OU with the shared memory ``l_rho``)::

    rho_{j+1} = phi rho_j + N(0, s_b^2 (1 - phi^2)),   phi = exp(-step / l_rho),   rho_0 ~ N(0, s_b^2)
    log s_b ~ N(alpha + alpha_group + beta_r log(r_b / 300 nm) + beta_l log(L_b / 20 um), tau^2)

``l_rho`` is fitted with the per-branch variance profiled out (closed form, as for the turning amplitude of
``direction.DirectionModel``); then the log of the per-branch estimates is regressed on the covariates (weights =
number of values) and tau^2 = residual spread minus the sampling noise of the estimates (var log s_hat ~ 1 / (2 n)).
r_b = median radius, L_b = length. Generation: s_b drawn for the branch from its (generated) median radius and
length, stationary start. Units: nm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .direction import R_REF_NM

L_REF_NM = 20000.0
MIN_VALUES = 8  # branches with fewer residual values do not inform the per-branch amplitude
TAU_FLOOR = 0.05


def _branch_sums(x: np.ndarray, mask: np.ndarray, step: np.ndarray, length_nm: float):
    """Per branch: S_b (whitened sum of squares), n_b (values), q_b = 1 - phi^2 for 1-D chains x [B,n]."""
    phi = np.exp(-step / length_nm)
    q = np.maximum(1.0 - phi**2, 1e-12)
    first = x[:, 0] ** 2 * mask[:, 0]
    resid = x[:, 1:] - phi[:, None] * x[:, :-1]
    m = mask[:, 1:] * mask[:, :-1]
    s = first + (resid**2 * m).sum(-1) / q
    return s, mask.sum(-1), q


def profile_nll(length_nm: float, x: np.ndarray, mask: np.ndarray, step: np.ndarray) -> float:
    """Exact AR(1) NLL of 1-D chains with the per-branch variance maximised out (constants dropped)."""
    s, n, q = _branch_sums(x, mask, step, length_nm)
    s = np.maximum(s, 1e-300)
    return 0.5 * float((n + n * np.log(s / n) + (n - 1.0) * np.log(q)).sum())


@dataclass
class RadiusResidualModel:
    length_nm: float = 3000.0
    alpha: float = math.log(0.05)
    beta_r: float = 0.0
    beta_l: float = 0.0
    tau: float = TAU_FLOOR
    groups: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def mean_log(self, group: str, radius_nm: Optional[float], length_nm: Optional[float]) -> float:
        r = R_REF_NM if radius_nm is None or not np.isfinite(radius_nm) or radius_nm <= 0 else float(radius_nm)
        L = L_REF_NM if length_nm is None or not np.isfinite(length_nm) or length_nm <= 0 else float(length_nm)
        return self.alpha + self.groups.get(group, 0.0) + self.beta_r * math.log(r / R_REF_NM) + self.beta_l * math.log(L / L_REF_NM)

    def fit(self, rho: Sequence[np.ndarray], steps: Sequence[float], radii: Sequence[float], lengths: Sequence[float],
            groups: Sequence[str], *, min_group: int = 50) -> "RadiusResidualModel":
        """``rho``: per-branch residual series (M nodes each), ``radii``: median radius, ``lengths``: branch length."""
        from scipy.optimize import minimize_scalar

        keep = [i for i, r in enumerate(rho) if len(r) >= MIN_VALUES]
        if not keep:
            raise ValueError(f"no branch with at least {MIN_VALUES} nodes")
        n_max = max(len(rho[i]) for i in keep)
        x = np.zeros((len(keep), n_max))
        mask = np.zeros((len(keep), n_max))
        for k, i in enumerate(keep):
            x[k, : len(rho[i])] = rho[i]
            mask[k, : len(rho[i])] = 1.0
        step = np.asarray([steps[i] for i in keep], dtype=float)
        res = minimize_scalar(lambda v: profile_nll(math.exp(v), x, mask, step), bounds=(math.log(100.0), math.log(1e6)), method="bounded")
        self.length_nm = math.exp(float(res.x))
        s, n, _ = _branch_sums(x, mask, step, self.length_nm)
        y = 0.5 * np.log(np.maximum(s, 1e-300) / n)  # log of the per-branch amplitude
        xr = np.log(np.asarray([radii[i] for i in keep], dtype=float) / R_REF_NM)
        xl = np.log(np.asarray([lengths[i] for i in keep], dtype=float) / L_REF_NM)
        w = n / n.sum()
        design = np.column_stack([np.ones_like(xr), xr, xl])
        coef = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        resid = y - design @ coef
        g = np.asarray([groups[i] for i in keep])
        self.groups = {}
        for name in np.unique(g):
            m = g == name
            if m.sum() >= min_group:
                self.groups[str(name)] = float(np.average(resid[m], weights=n[m]))
                resid[m] -= self.groups[str(name)]
        tau2 = float(np.average(resid**2, weights=n)) - float(np.average(1.0 / (2.0 * n), weights=n))
        self.alpha, self.beta_r, self.beta_l = (float(c) for c in coef)
        self.tau = math.sqrt(max(tau2, TAU_FLOOR**2))
        # the pooled single-amplitude value for comparison: sqrt of the node-weighted mean square
        self.info = {"n_branches": len(keep), "pooled_amplitude": float(math.sqrt(s.sum() / n.sum())),
                     "median_branch_amplitude": float(np.exp(np.median(y)))}
        return self

    def sample(self, n_nodes: int, step: float, group: str, rng: np.random.Generator, radius_nm: Optional[float] = None,
               length_nm: Optional[float] = None) -> np.ndarray:
        """[n_nodes] residual rho of one branch (stationary start)."""
        s_b = math.exp(self.mean_log(group, radius_nm, length_nm) + self.tau * rng.standard_normal())
        phi = math.exp(-step / self.length_nm)
        sd = s_b * math.sqrt(max(1.0 - phi**2, 0.0))
        out = np.empty(n_nodes)
        out[0] = rng.normal(0.0, s_b)
        eps = rng.normal(0.0, sd, n_nodes)
        for j in range(n_nodes - 1):
            out[j + 1] = phi * out[j] + eps[j]
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"length_nm": self.length_nm, "alpha": self.alpha, "beta_r": self.beta_r, "beta_l": self.beta_l, "tau": self.tau,
                "groups": dict(self.groups), "info": dict(self.info)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RadiusResidualModel":
        return cls(float(d["length_nm"]), float(d["alpha"]), float(d.get("beta_r", 0.0)), float(d.get("beta_l", 0.0)), float(d.get("tau", TAU_FLOOR)),
                   {k: float(v) for k, v in d.get("groups", {}).items()}, dict(d.get("info", {})))
