"""P0-position: longitudinal positions of N spines on a branch (p-module-technical-spec.md, section 5).

Given N the positions are i.i.d. with density ``f(s) ~ exp(beta . phi(s))`` on [0, L] (binomial point process).
phi(s): B-splines of u = s / L (the first one dropped, the constant cancels in the normalisation), standardised
radius, bending ``kappa``, distance to the soma (zero + mask where undefined), distance to the nearest end.
The conditional log-likelihood is concave; the integral is taken over the node grid (trapezoid). Angles are uniform
(``sample_angles``). Units: nm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from ..covariates import _bspline


@dataclass
class PositionBranch:
    """Trunk quantities on the node grid plus (for training) the observed spine positions s_i, nm."""

    s: np.ndarray  # [M] node arc length, nm
    radius: np.ndarray  # [M]
    kappa: np.ndarray  # [M] (|bending|; the last entry may be NaN and is replaced by its neighbour)
    soma_start_nm: Optional[float] = None  # distance to the soma at s = 0, along the dendritic path
    positions: Optional[np.ndarray] = None  # observed s_i (those entering the positional likelihood)

    @property
    def length(self) -> float:
        return float(self.s[-1])


class PositionModel:
    def __init__(self, n_basis: int = 6, smooth_penalty: float = 1.0, ridge: float = 1e-3, use: Sequence[str] = ("radius", "kappa", "soma", "end")) -> None:
        self.n_basis, self.smooth_penalty, self.ridge, self.use = n_basis, smooth_penalty, ridge, tuple(use)
        self.beta: Optional[np.ndarray] = None
        self.scale: Dict[str, Any] = {}
        self.fit_info: Dict[str, Any] = {}

    # ---- features -------------------------------------------------------------------------------------------------
    def _basis(self, branch: PositionBranch, s: np.ndarray) -> np.ndarray:
        length = branch.length
        kappa = np.nan_to_num(branch.kappa, nan=0.0)
        if len(kappa) == len(branch.s) - 1:
            kappa = np.r_[kappa, kappa[-1] if len(kappa) else 0.0]
        cols = [_bspline(s / length, self.n_basis, 0.0, 1.0)[:, 1:]]
        sc = self.scale
        if "radius" in self.use:
            cols.append(((np.interp(s, branch.s, branch.radius) - sc["r_mean"]) / sc["r_std"])[:, None])
        if "kappa" in self.use:
            cols.append(((np.interp(s, branch.s, kappa) - sc["k_mean"]) / sc["k_std"])[:, None])
        if "soma" in self.use:
            if branch.soma_start_nm is None:
                cols.append(np.zeros((len(s), 1)))
            else:
                cols.append((((branch.soma_start_nm + s) - sc["d_mean"]) / sc["d_std"])[:, None])
        if "end" in self.use:
            cols.append((np.minimum(s, length - s) / sc["end_scale"])[:, None])
        return np.hstack(cols)

    @property
    def names(self):
        out = [f"u_spline{i}" for i in range(1, self.n_basis)]
        out += [{"radius": "radius", "kappa": "kappa", "soma": "d_soma", "end": "end_distance"}[k] for k in self.use]
        return out

    def _fit_scales(self, branches: Sequence[PositionBranch]) -> None:
        r = np.concatenate([b.radius for b in branches])
        k = np.concatenate([np.nan_to_num(b.kappa, nan=0.0) for b in branches])
        d = np.concatenate([b.soma_start_nm + b.s for b in branches if b.soma_start_nm is not None] or [np.array([0.0, 1.0])])
        self.scale = {"r_mean": float(r.mean()), "r_std": float(r.std() + 1e-9), "k_mean": float(k.mean()), "k_std": float(k.std() + 1e-12),
                      "d_mean": float(d.mean()), "d_std": float(d.std() + 1e-9), "end_scale": float(np.median([b.length for b in branches]) / 2.0)}

    # ---- fit ----------------------------------------------------------------------------------------------------------
    def fit(self, branches: Sequence[PositionBranch], max_iter: int = 200) -> "PositionModel":
        self._fit_scales(branches)
        grid, grid_w, grid_b, pos, n_obs = [], [], [], [], []
        for i, b in enumerate(branches):
            grid.append(self._basis(b, b.s))
            ds = np.diff(b.s)
            w = np.zeros(len(b.s))
            w[:-1] += ds / 2.0
            w[1:] += ds / 2.0
            grid_w.append(w / b.length)  # normalised to a unit interval: only affects the constant
            grid_b.append(np.full(len(b.s), i))
            if b.positions is not None and len(b.positions):
                pos.append((i, self._basis(b, np.asarray(b.positions, dtype=float))))
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
        G, W = f(np.vstack(grid)), f(np.concatenate(grid_w))
        gb = torch.as_tensor(np.concatenate(grid_b), dtype=torch.long)
        Pm = f(np.vstack([p for _, p in pos])) if pos else f(np.zeros((0, G.shape[1])))
        pb = torch.as_tensor(np.concatenate([np.full(len(p), i) for i, p in pos]) if pos else [], dtype=torch.long)
        nb = len(branches)
        p = G.shape[1]
        beta = torch.zeros(p, dtype=torch.float64, requires_grad=True)
        n_total = max(int(Pm.shape[0]), 1)

        def nll():
            eta = (G @ beta).clamp(max=30.0)
            integral = torch.zeros(nb, dtype=torch.float64).index_add(0, gb, torch.exp(eta) * W)
            counts = torch.zeros(nb, dtype=torch.float64).index_add(0, pb, torch.ones(len(pb), dtype=torch.float64))
            loglik = (Pm @ beta).sum() - (counts * torch.log(integral)).sum()
            blk = beta[: self.n_basis - 1]
            d2 = blk[2:] - 2 * blk[1:-1] + blk[:-2]
            pen = self.smooth_penalty * (d2 ** 2).sum() + self.ridge * (beta ** 2).sum()
            return -(loglik) / n_total + pen / n_total

        opt = torch.optim.LBFGS([beta], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-13)

        def closure():
            opt.zero_grad()
            loss = nll()
            loss.backward()
            return loss

        opt.step(closure)
        self.beta = beta.detach().numpy().copy()
        self.fit_info = {"n_branches_with_spines": len(pos), "n_positions": int(Pm.shape[0]), "nll_per_spine": float(nll().detach())}
        return self

    # ---- density / generation -------------------------------------------------------------------------------------
    def log_intensity(self, branch: PositionBranch, s: Optional[np.ndarray] = None) -> np.ndarray:
        s = branch.s if s is None else np.asarray(s, dtype=float)
        return self._basis(branch, s) @ self.beta

    def cdf_on_nodes(self, branch: PositionBranch) -> np.ndarray:
        """F(s_j) at the nodes (trapezoid of exp(beta.phi), normalised to F(L) = 1)."""
        lam = np.exp(self.log_intensity(branch) - self.log_intensity(branch).max())
        seg = 0.5 * (lam[1:] + lam[:-1]) * np.diff(branch.s)
        cum = np.r_[0.0, np.cumsum(seg)]
        return cum / cum[-1]

    def sample_positions(self, branch: PositionBranch, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sorted s_i by the inverse of F (linear interpolation between nodes)."""
        cdf = self.cdf_on_nodes(branch)
        keep = np.r_[True, np.diff(cdf) > 0]  # strictly increasing for the interpolation
        return np.sort(np.interp(rng.random(n), cdf[keep], branch.s[keep]))

    def loglik(self, branches: Sequence[PositionBranch]) -> float:
        """Conditional log-likelihood sum_b [sum_i log f(s_bi)] (held-out evaluation)."""
        total = 0.0
        for b in branches:
            if b.positions is None or not len(b.positions):
                continue
            lam = self.log_intensity(b)
            shift = lam.max()
            integral = np.trapezoid(np.exp(lam - shift), b.s)
            total += float((self.log_intensity(b, b.positions) - shift).sum() - len(b.positions) * np.log(integral))
        return total

    # ---- persistence ----------------------------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"n_basis": self.n_basis, "smooth_penalty": self.smooth_penalty, "ridge": self.ridge, "use": list(self.use),
                "beta": self.beta.tolist(), "scale": self.scale, "fit_info": self.fit_info}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PositionModel":
        m = cls(d["n_basis"], d["smooth_penalty"], d["ridge"], d["use"])
        m.beta, m.scale, m.fit_info = np.array(d["beta"]), d["scale"], d.get("fit_info", {})
        return m

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "PositionModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def sample_angles(n: int, rng: np.random.Generator) -> np.ndarray:
    """Base angular model: uniform on (-pi, pi], independent of s (section 5, 'Угловая координата')."""
    return np.pi - rng.random(n) * 2.0 * np.pi
