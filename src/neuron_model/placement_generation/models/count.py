"""P0-count: number of spines on a branch (p-module-technical-spec.md, section 4).

Families: Poisson, NB2, zero-inflated NB, hurdle NB. Mean: ``log mu_b = log E_b + X_b beta + b_neuron`` with the exposure
``E`` (length or surface area, in micrometres) and a Gaussian neuron effect integrated out by Gauss-Hermite quadrature
(non-adaptive: the nodes are centred at zero, which is sufficient while sigma_b is moderate). Dispersion:
``log k_b = kappa0 + kappa1 z(log L_b)``. The penalised likelihood is maximised with L-BFGS (torch autograd).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

FAMILIES = ("poisson", "nb", "zinb", "hurdle")
EXPOSURES = ("length", "area")


@dataclass
class CountData:
    X: np.ndarray  # [B, p] mean design
    Z: np.ndarray  # [B, q] inflation / hurdle design (first column 1)
    log_exposure: np.ndarray  # [B] log of the exposure in micrometres (or um^2)
    y: np.ndarray  # [B] counts
    neuron: np.ndarray  # [B] integer neuron index 0..n_neurons-1
    z_log_length: np.ndarray  # [B] standardised log-length (dispersion covariate)

    @property
    def n_neurons(self) -> int:
        return int(self.neuron.max()) + 1 if len(self.neuron) else 0


def build_count_data(design, branches: Sequence[Any], exposure: str = "length") -> CountData:
    if exposure not in EXPOSURES:
        raise ValueError(exposure)
    scale = 1e3 if exposure == "length" else 1e6
    key = (lambda b: b.length_nm) if exposure == "length" else (lambda b: b.area_nm2)
    names = {}
    neuron = np.array([names.setdefault(b.neuron, len(names)) for b in branches], dtype=int)
    return CountData(
        X=design.matrix(branches), Z=design.inflation_matrix(branches),
        log_exposure=np.log(np.array([key(b) for b in branches], dtype=float) / scale),
        y=np.array([b.n_spines for b in branches], dtype=float), neuron=neuron,
        z_log_length=np.array([design.encoder.z_length(b.length_nm) for b in branches], dtype=float),
    )


def _nb_logpmf(y, mu, k):
    return (torch.lgamma(y + k) - torch.lgamma(k) - torch.lgamma(y + 1.0) + k * (torch.log(k) - torch.log(k + mu)) + y * (torch.log(mu) - torch.log(k + mu)))


def _log_pmf(family: str, y, mu, k, logit):
    if family == "poisson":
        return y * torch.log(mu) - mu - torch.lgamma(y + 1.0)
    nb = _nb_logpmf(y, mu, k)
    if family == "nb":
        return nb
    zero = y == 0
    nb0 = k * (torch.log(k) - torch.log(k + mu))  # log NB(0)
    if family == "zinb":
        log_pi, log_1m_pi = torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit)
        return torch.where(zero, torch.logaddexp(log_pi, log_1m_pi + nb0), log_1m_pi + nb)
    # hurdle: logit = P(N > 0)
    log_q, log_1m_q = torch.nn.functional.logsigmoid(logit), torch.nn.functional.logsigmoid(-logit)
    log_pos = nb - torch.log(-torch.expm1(nb0).clamp(max=-1e-12))  # truncated at zero
    return torch.where(zero, log_1m_q, log_q + log_pos)


class CountModel:
    def __init__(self, family: str = "nb", exposure: str = "length", random_effect: bool = True, n_quad: int = 15,
                 n_basis: int = 5, smooth_penalty: float = 1.0, ridge: float = 1e-3) -> None:
        if family not in FAMILIES:
            raise ValueError(family)
        self.family, self.exposure, self.random_effect, self.n_quad = family, exposure, random_effect, n_quad
        self.n_basis, self.smooth_penalty, self.ridge = n_basis, smooth_penalty, ridge
        self.beta: Optional[np.ndarray] = None
        self.gamma: Optional[np.ndarray] = None
        self.kappa = np.array([math.log(10.0), 0.0])
        self.log_sigma_b = math.log(0.3)
        self.fit_info: Dict[str, Any] = {}

    # ---- parameters <-> torch ---------------------------------------------------------------------------------
    def _unpack(self, theta, p, q):
        beta = theta[:p]
        gamma = theta[p:p + q]
        kappa = theta[p + q:p + q + 2]
        log_sigma = theta[p + q + 2]
        return beta, gamma, kappa, log_sigma

    def _nll(self, theta, t, p, q, penalty=True):
        beta, gamma, kappa, log_sigma = self._unpack(theta, p, q)
        eta = t["X"] @ beta + t["off"]
        k = torch.exp(kappa[0] + kappa[1] * t["zl"]).clamp(1e-3, 1e6)
        logit = t["Z"] @ gamma if self.family in ("zinb", "hurdle") else None
        if self.random_effect:
            nodes, weights = t["nodes"], t["lw"]
            sigma = torch.exp(log_sigma)
            mu = torch.exp(eta[:, None] + math.sqrt(2.0) * sigma * nodes[None, :]).clamp(1e-10, 1e8)  # [B, Q]
            lp = _log_pmf(self.family, t["y"][:, None], mu, k[:, None], None if logit is None else logit[:, None])
            per_neuron = torch.zeros(t["n_neurons"], lp.shape[1], dtype=lp.dtype).index_add(0, t["neuron"], lp)
            total = torch.logsumexp(per_neuron + weights[None, :], dim=1).sum()
        else:
            mu = torch.exp(eta).clamp(1e-10, 1e8)
            total = _log_pmf(self.family, t["y"], mu, k, logit).sum()
        nll = -total
        if penalty:
            nll = nll + self._penalty(beta, gamma, p)
        return nll

    def _penalty(self, beta, gamma, p):
        pen = self.ridge * (beta ** 2).sum() + self.ridge * (gamma[1:] ** 2).sum()
        for start in (0, self.n_basis):  # radius block, soma block
            blk = beta[start:start + self.n_basis]
            d2 = blk[2:] - 2 * blk[1:-1] + blk[:-2]
            pen = pen + self.smooth_penalty * (d2 ** 2).sum()
        return pen

    def _tensors(self, data: CountData):
        nodes, w = np.polynomial.hermite.hermgauss(self.n_quad)
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
        return {"X": f(data.X), "Z": f(data.Z), "off": f(data.log_exposure), "y": f(data.y), "zl": f(data.z_log_length),
                "neuron": torch.as_tensor(data.neuron, dtype=torch.long), "n_neurons": data.n_neurons,
                "nodes": f(nodes), "lw": f(np.log(w / math.sqrt(math.pi)))}

    # ---- fit / predict ------------------------------------------------------------------------------------------
    def fit(self, data: CountData, max_iter: int = 300) -> "CountModel":
        p, q = data.X.shape[1], data.Z.shape[1]
        t = self._tensors(data)
        beta0 = np.zeros(p)
        beta0[:self.n_basis] = math.log(max(data.y.sum() / np.exp(data.log_exposure).sum(), 1e-3)) / 1.0
        gamma0 = np.zeros(q)
        if self.family in ("zinb", "hurdle"):
            frac = float(np.clip((data.y > 0).mean() if self.family == "hurdle" else (data.y == 0).mean(), 0.02, 0.98))
            gamma0[0] = math.log(frac / (1 - frac))
        theta = torch.tensor(np.r_[beta0, gamma0, self.kappa, self.log_sigma_b], dtype=torch.float64, requires_grad=True)
        free = torch.ones_like(theta, dtype=torch.bool)
        if self.family == "poisson":
            free[p + q:p + q + 2] = False
        if self.family not in ("zinb", "hurdle"):
            free[p:p + q] = False
        if not self.random_effect:
            free[p + q + 2] = False
        mask = free.to(torch.float64)
        opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            loss = self._nll(theta, t, p, q)
            loss.backward()
            theta.grad.mul_(mask)
            return loss

        opt.step(closure)
        th = theta.detach()
        self.beta, self.gamma = th[:p].numpy().copy(), th[p:p + q].numpy().copy()
        self.kappa = th[p + q:p + q + 2].numpy().copy()
        self.log_sigma_b = float(th[p + q + 2])
        self.fit_info = {"nll": float(self._nll(th, t, p, q, penalty=False)), "n_branches": len(data.y), "n_neurons": data.n_neurons}
        return self

    @property
    def sigma_b(self) -> float:
        return math.exp(self.log_sigma_b) if self.random_effect else 0.0

    @property
    def dispersion(self) -> float:
        return math.exp(self.kappa[0])

    def neg_loglik(self, data: CountData) -> float:
        """Held-out negative log-likelihood (neuron effect integrated out), without penalty."""
        theta = torch.tensor(np.r_[self.beta, self.gamma, self.kappa, self.log_sigma_b], dtype=torch.float64)
        return float(self._nll(theta, self._tensors(data), len(self.beta), len(self.gamma), penalty=False))

    def mean(self, data: CountData, neuron_effect: float = 0.0) -> np.ndarray:
        """E[N] given the neuron effect (marginal mean when it is 0 and ``sigma_b`` is ignored); the zero part is included."""
        mu = np.exp(data.X @ self.beta + data.log_exposure + neuron_effect)
        if self.family in ("zinb", "hurdle"):
            k = np.exp(self.kappa[0] + self.kappa[1] * data.z_log_length)
            s = 1.0 / (1.0 + np.exp(-(data.Z @ self.gamma)))
            if self.family == "zinb":
                return (1 - s) * mu
            return s * mu / (1.0 - (k / (k + mu)) ** k)
        return mu

    # ---- generation -------------------------------------------------------------------------------------------------
    def sample(self, data: CountData, rng: np.random.Generator, neuron_effects: Optional[np.ndarray] = None) -> np.ndarray:
        """One count per row. Rows with the same ``data.neuron`` share one drawn effect."""
        n_neurons = data.n_neurons
        if neuron_effects is None:
            neuron_effects = rng.normal(0.0, self.sigma_b, size=n_neurons) if self.random_effect else np.zeros(n_neurons)
        mu = np.exp(data.X @ self.beta + data.log_exposure + neuron_effects[data.neuron])
        k = np.exp(self.kappa[0] + self.kappa[1] * data.z_log_length)
        if self.family == "poisson":
            return rng.poisson(mu).astype(int)
        nb = lambda size_mask=None: rng.poisson(rng.gamma(k, mu / k))
        if self.family == "nb":
            return nb().astype(int)
        s = 1.0 / (1.0 + np.exp(-(data.Z @ self.gamma)))
        if self.family == "zinb":
            return np.where(rng.random(len(mu)) < s, 0, nb()).astype(int)
        out = np.zeros(len(mu), dtype=int)
        positive = rng.random(len(mu)) < s
        draw = nb()
        for _ in range(200):  # truncated at zero: redraw the zeros
            bad = positive & (draw == 0)
            if not bad.any():
                break
            draw[bad] = rng.poisson(rng.gamma(k[bad], mu[bad] / k[bad]))
        out[positive] = np.maximum(draw[positive], 1)
        return out

    def randomized_quantile_residuals(self, data: CountData, rng: np.random.Generator) -> np.ndarray:
        """Dunn-Smyth residuals under the fitted marginal (no neuron effect for the mean); ~N(0,1) when calibrated."""
        from scipy import stats

        mu = np.exp(data.X @ self.beta + data.log_exposure)
        k = np.exp(self.kappa[0] + self.kappa[1] * data.z_log_length)
        if self.family == "poisson":
            lo, hi = stats.poisson.cdf(data.y - 1, mu), stats.poisson.cdf(data.y, mu)
        else:
            pr = k / (k + mu)
            lo, hi = stats.nbinom.cdf(data.y - 1, k, pr), stats.nbinom.cdf(data.y, k, pr)
        return stats.norm.ppf(np.clip(lo + rng.random(len(mu)) * (hi - lo), 1e-9, 1 - 1e-9))

    # ---- persistence ----------------------------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "exposure": self.exposure, "random_effect": self.random_effect, "n_quad": self.n_quad,
                "n_basis": self.n_basis, "smooth_penalty": self.smooth_penalty, "ridge": self.ridge,
                "beta": self.beta.tolist(), "gamma": self.gamma.tolist(), "kappa": self.kappa.tolist(),
                "log_sigma_b": self.log_sigma_b, "fit_info": self.fit_info}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CountModel":
        m = cls(d["family"], d["exposure"], d["random_effect"], d["n_quad"], d["n_basis"], d["smooth_penalty"], d["ridge"])
        m.beta, m.gamma = np.array(d["beta"]), np.array(d["gamma"])
        m.kappa, m.log_sigma_b, m.fit_info = np.array(d["kappa"]), d["log_sigma_b"], d.get("fit_info", {})
        return m

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "CountModel":
        return cls.from_dict(json.loads(Path(path).read_text()))
