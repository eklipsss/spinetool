"""T0: statistical model of a branch trunk (t-module-technical-spec.md, section 4).

Residual process (multivariate stationary Ornstein-Uhlenbeck, discretised exactly on the physical step)::

    z_j = (kappa1_j - mu1, kappa2_j - mu2, rho_j),   rho_j = log r_j - m(s_j)
    z_{j+1} = Phi_b z_j + eta_j,   Phi_b = diag(exp(-ds_b / l_k)),   eta_j ~ N(0, Sigma - Phi Sigma Phi)
    z_0 ~ N(0, Sigma)

Parameters: three correlation lengths ``l_k`` (nm), the stationary covariance ``Sigma`` (cross terms = joint
statistics of bending and thickness), the bending means ``mu``. They are estimated by the exact Gaussian
likelihood of all training chains (L-BFGS, autograd) and may differ per group (species | cell type |
compartment) through a Gaussian prior on the deviation from the pooled value (hierarchical shrinkage: a
small group stays close to the pooled parameters, a large one is dominated by its own data).

Mean log-radius profile ``m(s)``, two competing descriptions:

- ``parabola`` (T0-a): r(s) = r_L + alpha (L - s)^2 (Bird, Cuntz 2016), per-branch pair (log r_L, log r_0/r_L)
  distributed N(B^T x, S) with a linear dependence on log L and the conditions;
- ``gam`` (T0-b): m(s) = f(u, log L, C) + b_branch, penalised B-spline model plus a random branch shift.

Branch length for fully automatic generation: log-normal mixture (<= 3 components) per group.

Direction (``direction="multiscale"``, default since 2026-10-28: ``direction_ms.DirectionMultiScale`` = the model below
+ a smooth Gaussian-kernel wiggle of the heading fitted to the pooled semivariograms of kappa and of the tilt; ``"tilt"``:
the model below alone; the text below describes ``"tilt"``): the bending of a generated trunk comes from ``direction.DirectionModel``
(the heading relaxes to the branch axis, bounded tortuosity) instead of the kappa components of the OU above, which
integrate to a free random walk of the heading (tortuosity growing with the length). The radius residual rho keeps its
OU marginal (diagonal Phi: the rho component alone is an exact AR(1)); the bending-thickness cross terms are not used
in this mode. ``direction="kappa"``: the original model.

Radius residual: ``radius_residual="multiscale"`` (default): rho = two AR(1) components (short and long memory, fitted to
the pooled semivariogram of rho) + white noise, with the amplitude of the branch as a random effect (radius, length):
``radius.RadiusMultiScale``. ``"branch"``: ONE AR(1) with the branch amplitude (``RadiusResidualModel``; on the full
Minnie65 data it lost the variation at 0.5-8 um: memory 54 um). ``"pooled"``: the joint OU component (old behaviour).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..conditions import DesignEncoder, group_key
from .direction import DirectionModel
from .direction_ms import DirectionMultiScale, direction_model_from_dict
from .radius import RadiusMultiScale, RadiusResidualModel, radius_model_from_dict


@dataclass
class T0Branch:
    """One training chain. ``kappa*``: M-1 steps, ``log_radius``: M nodes."""

    kappa1: np.ndarray
    kappa2: np.ndarray
    log_radius: np.ndarray
    step: float
    design: np.ndarray  # DesignEncoder.row(...)
    group: str

    @property
    def n_nodes(self) -> int:
        return len(self.log_radius)

    @property
    def length(self) -> float:
        return (self.n_nodes - 1) * self.step

    @property
    def u(self) -> np.ndarray:
        return np.arange(self.n_nodes) / (self.n_nodes - 1)


# ---------------------------------------------------------------------------
# mean log-radius profile
# ---------------------------------------------------------------------------


def _bspline_basis(u: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    from scipy.interpolate import BSpline

    interior = np.linspace(0.0, 1.0, n_basis - degree + 1)[1:-1]
    knots = np.r_[[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    return BSpline.design_matrix(np.clip(u, 0.0, 1.0), knots, degree).toarray()


class ParabolicMean:
    """T0-a: r(s) = r_L + (r_0 - r_L) w^2, w = (L - s) / L; (log r_L, log r_0/r_L) ~ N(B^T x, S)."""

    kind = "parabola"

    def __init__(self) -> None:
        self.coef: Optional[np.ndarray] = None  # [d,2]
        self.cov: Optional[np.ndarray] = None  # [2,2]

    @staticmethod
    def _fit_branch(b: T0Branch) -> np.ndarray:
        r = np.exp(b.log_radius)
        w2 = ((b.n_nodes - 1 - np.arange(b.n_nodes)) / (b.n_nodes - 1)) ** 2
        basis = np.column_stack([1.0 - w2, w2])
        (r_l, r_0), *_ = np.linalg.lstsq(basis, r, rcond=None)
        floor = 0.25 * float(np.median(r))
        r_l, r_0 = max(r_l, floor), max(r_0, floor)
        return np.array([math.log(r_l), math.log(r_0 / r_l)])

    def fit(self, branches: Sequence[T0Branch], ridge: float = 1.0) -> "ParabolicMean":
        theta = np.vstack([self._fit_branch(b) for b in branches])
        x = np.vstack([b.design for b in branches])
        penalty = ridge * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(x.T @ x + penalty, x.T @ theta)
        resid = theta - x @ self.coef
        self.cov = np.atleast_2d(np.cov(resid.T)) + 1e-9 * np.eye(2)
        return self

    @staticmethod
    def _profile(theta: np.ndarray, n_nodes: int) -> np.ndarray:
        r_l, ratio = math.exp(theta[0]), math.exp(theta[1])
        w2 = ((n_nodes - 1 - np.arange(n_nodes)) / (n_nodes - 1)) ** 2
        return np.log(r_l * (1.0 - w2) + r_l * ratio * w2)

    def branch_log_mean(self, b: T0Branch) -> np.ndarray:
        """m(s_j) of a TRAINING branch from its own fitted pair."""
        return self._profile(self._fit_branch(b), b.n_nodes)

    def sample_log_mean(self, design: np.ndarray, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
        theta = rng.multivariate_normal(design @ self.coef, self.cov)
        return self._profile(theta, n_nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "coef": self.coef.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParabolicMean":
        obj = cls()
        obj.coef, obj.cov = np.asarray(d["coef"]), np.asarray(d["cov"])
        return obj


class GamMean:
    """T0-b: m = f(u, z log L, conditions) + b_branch, b ~ N(0, sigma_b^2)."""

    kind = "gam"

    def __init__(self, n_basis: int = 8) -> None:
        self.n_basis = int(n_basis)
        self.beta: Optional[np.ndarray] = None
        self.sigma_b: float = 0.0

    def _design(self, u: np.ndarray, design: np.ndarray) -> np.ndarray:
        spline = _bspline_basis(u, self.n_basis)
        extra = np.tile(design[1:], (len(u), 1))  # drop the intercept (the spline basis sums to one)
        return np.hstack([spline, extra, spline * design[1]])  # design[1] = z(log L)

    def _penalty(self, n_extra: int, smooth: float, ridge: float) -> np.ndarray:
        d2 = np.diff(np.eye(self.n_basis), n=2, axis=0)
        block = d2.T @ d2
        k = self.n_basis
        pen = np.zeros((2 * k + n_extra, 2 * k + n_extra))
        pen[:k, :k] = smooth * block
        pen[k + n_extra:, k + n_extra:] = smooth * block
        pen[k:k + n_extra, k:k + n_extra] = ridge * np.eye(n_extra)
        return pen

    def fit(self, branches: Sequence[T0Branch], smooth: float = 1.0, ridge: float = 1.0) -> "GamMean":
        n_extra = len(branches[0].design) - 1
        dim = 2 * self.n_basis + n_extra
        xtx, xty = np.zeros((dim, dim)), np.zeros(dim)
        for b in branches:
            x = self._design(b.u, b.design)
            xtx += x.T @ x
            xty += x.T @ b.log_radius
        self.beta = np.linalg.solve(xtx + self._penalty(n_extra, smooth, ridge) * len(branches) / 100.0 + 1e-9 * np.eye(dim), xty)
        shifts = np.array([float(np.mean(b.log_radius - self._design(b.u, b.design) @ self.beta)) for b in branches])
        self.sigma_b = float(np.std(shifts))
        return self

    def branch_log_mean(self, b: T0Branch) -> np.ndarray:
        f = self._design(b.u, b.design) @ self.beta
        return f + float(np.mean(b.log_radius - f))

    def sample_log_mean(self, design: np.ndarray, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
        u = np.arange(n_nodes) / (n_nodes - 1)
        return self._design(u, design) @ self.beta + rng.normal(0.0, self.sigma_b)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "n_basis": self.n_basis, "beta": self.beta.tolist(), "sigma_b": self.sigma_b}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GamMean":
        obj = cls(d["n_basis"])
        obj.beta, obj.sigma_b = np.asarray(d["beta"]), float(d["sigma_b"])
        return obj


MEAN_MODELS = {"parabola": ParabolicMean, "gam": GamMean}


# ---------------------------------------------------------------------------
# length model
# ---------------------------------------------------------------------------


class LengthModel:
    """Log-normal mixture (<= 3 components, BIC) of log L per group, pooled fallback for small groups."""

    def __init__(self, min_group: int = 200, max_components: int = 3) -> None:
        self.min_group = min_group
        self.max_components = max_components
        self.mixtures: Dict[str, Dict[str, list]] = {}

    @staticmethod
    def _fit_mixture(log_l: np.ndarray, max_components: int, seed: int = 0) -> Dict[str, list]:
        from sklearn.mixture import GaussianMixture

        best = None
        for k in range(1, max_components + 1):
            if len(log_l) < 20 * k:
                break
            with warnings.catch_warnings():  # "KMeans is known to have a memory leak on Windows with MKL": harmless here
                warnings.simplefilter("ignore", UserWarning)
                gm = GaussianMixture(k, random_state=seed, n_init=2).fit(log_l.reshape(-1, 1))
            bic = gm.bic(log_l.reshape(-1, 1))
            if best is None or bic < best[0]:
                best = (bic, gm)
        gm = best[1]
        return {"weights": gm.weights_.tolist(), "means": gm.means_.ravel().tolist(), "stds": np.sqrt(gm.covariances_.ravel()).tolist()}

    def fit(self, lengths: Sequence[float], groups: Sequence[str]) -> "LengthModel":
        log_l = np.log(np.asarray(lengths, dtype=float))
        groups = np.asarray(groups)
        self.mixtures = {"pooled": self._fit_mixture(log_l, self.max_components)}
        for g in np.unique(groups):
            if (groups == g).sum() >= self.min_group:
                self.mixtures[str(g)] = self._fit_mixture(log_l[groups == g], self.max_components)
        return self

    def sample(self, group: str, bounds: Tuple[float, float], rng: np.random.Generator, size: int = 1) -> np.ndarray:
        mix = self.mixtures.get(group, self.mixtures["pooled"])
        out = np.empty(0)
        while len(out) < size:
            comp = rng.choice(len(mix["weights"]), size=2 * size, p=mix["weights"])
            draw = np.exp(rng.normal(np.asarray(mix["means"])[comp], np.asarray(mix["stds"])[comp]))
            out = np.r_[out, draw[(draw >= bounds[0]) & (draw <= bounds[1])]]
        return out[:size]

    def to_dict(self) -> Dict[str, Any]:
        return {"min_group": self.min_group, "max_components": self.max_components, "mixtures": self.mixtures}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LengthModel":
        obj = cls(d["min_group"], d["max_components"])
        obj.mixtures = d["mixtures"]
        return obj


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck parameters
# ---------------------------------------------------------------------------

N_PARAMS = 11  # 3 log-lengths, 6 Cholesky entries (3 log-diagonal + 3 below), 2 bending means


def _unpack(theta):
    """theta [...,11] -> (length scales [...,3], Cholesky [...,3,3], mu [...,2]) (torch)."""
    import torch

    log_l = theta[..., :3]
    d = torch.exp(theta[..., 3:6])
    chol = torch.zeros(*theta.shape[:-1], 3, 3, dtype=theta.dtype)
    chol[..., 0, 0], chol[..., 1, 1], chol[..., 2, 2] = d[..., 0], d[..., 1], d[..., 2]
    chol[..., 1, 0], chol[..., 2, 0], chol[..., 2, 1] = theta[..., 6], theta[..., 7], theta[..., 8]
    return torch.exp(log_l), chol, theta[..., 9:11]


def _chain_loglik(theta, k1, k2, rho, mask, step):
    """Exact log-likelihood of each chain. theta [B,11]; k1,k2,rho,mask [B,n]; step [B] -> [B]."""
    import torch

    length, chol, mu = _unpack(theta)
    z = torch.stack([k1 - mu[:, 0:1], k2 - mu[:, 1:2], rho], dim=-1)  # [B,n,3]
    z = z * mask.unsqueeze(-1)
    sigma = chol @ chol.transpose(-1, -2)
    # initial state
    z0 = z[:, 0:1, :].transpose(1, 2)  # [B,3,1]
    sol = torch.linalg.solve_triangular(chol, z0, upper=False)
    ll0 = -0.5 * (sol**2).sum(dim=(1, 2)) - torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1) - 1.5 * math.log(2 * math.pi)
    # transitions with an exact discretisation on the physical step
    phi = torch.exp(-step.unsqueeze(-1) / length)  # [B,3]
    q = sigma * (1.0 - phi.unsqueeze(-1) * phi.unsqueeze(-2))
    evals, evecs = torch.linalg.eigh(q)
    evals = evals.clamp_min(1e-10 * evals.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30))
    resid = z[:, 1:, :] - phi.unsqueeze(1) * z[:, :-1, :]  # [B,n-1,3]
    proj = resid @ evecs  # [B,n-1,3]
    quad = (proj**2 / evals.unsqueeze(1)).sum(-1)
    logdet = torch.log(evals).sum(-1, keepdim=True)
    step_ll = -0.5 * (quad + logdet + 3 * math.log(2 * math.pi))
    step_mask = mask[:, 1:] * mask[:, :-1]
    return ll0 + (step_ll * step_mask).sum(-1)


@dataclass
class OUParams:
    theta: np.ndarray  # [11]

    @property
    def length_scales(self) -> np.ndarray:
        return np.exp(self.theta[:3])

    @property
    def chol(self) -> np.ndarray:
        c = np.zeros((3, 3))
        c[np.arange(3), np.arange(3)] = np.exp(self.theta[3:6])
        c[1, 0], c[2, 0], c[2, 1] = self.theta[6:9]
        return c

    @property
    def sigma(self) -> np.ndarray:
        c = self.chol
        return c @ c.T

    @property
    def mu(self) -> np.ndarray:
        return self.theta[9:11]


class T0Model:
    """Statistical trunk model (see module docstring)."""

    def __init__(self, mean_model: str = "parabola", length_bounds: Tuple[float, float] = (5000.0, 255500.0), direction: str = "multiscale",
                 radius_residual: str = "multiscale") -> None:
        if mean_model not in MEAN_MODELS:
            raise ValueError(f"mean_model must be one of {tuple(MEAN_MODELS)}")
        if direction not in ("multiscale", "tilt", "kappa"):
            raise ValueError("direction must be 'multiscale', 'tilt' or 'kappa'")
        if radius_residual not in ("multiscale", "branch", "pooled"):
            raise ValueError("radius_residual must be 'multiscale', 'branch' or 'pooled'")
        self.direction_mode = direction
        self.radius_mode = radius_residual
        self.radius = None  # RadiusMultiScale | RadiusResidualModel | None (pooled)
        self.direction: Optional[DirectionModel] = None
        self.mean = MEAN_MODELS[mean_model]()
        self.length_model = LengthModel()
        self.length_bounds = tuple(float(b) for b in length_bounds)
        self.pooled = np.zeros(N_PARAMS)
        self.group_theta: Dict[str, np.ndarray] = {}
        self.encoder: Optional[DesignEncoder] = None
        self.fit_info: Dict[str, Any] = {}

    # ---------------------------------------------------------------- fitting
    def _residual_arrays(self, branches: Sequence[T0Branch]):
        n = max(b.n_nodes - 1 for b in branches)
        count = len(branches)
        k1, k2, rho = np.zeros((count, n)), np.zeros((count, n)), np.zeros((count, n))
        mask = np.zeros((count, n))
        step = np.zeros(count)
        for i, b in enumerate(branches):
            m = b.n_nodes - 1
            k1[i, :m], k2[i, :m] = b.kappa1[:m], b.kappa2[:m]
            rho[i, :m] = (b.log_radius - self.mean.branch_log_mean(b))[:m]
            mask[i, :m] = 1.0
            step[i] = b.step
        return k1, k2, rho, mask, step

    def fit(self, branches: Sequence[T0Branch], *, encoder: DesignEncoder, shrinkage_std: float = 0.3, max_iter: int = 200, min_group: int = 50, verbose: bool = False) -> "T0Model":
        import torch

        self.encoder = encoder
        self.mean.fit(branches)
        self.length_model.fit([b.length for b in branches], [b.group for b in branches])
        k1, k2, rho, mask, step = (torch.tensor(a, dtype=torch.float64) for a in self._residual_arrays(branches))
        groups = np.array([b.group for b in branches])
        names = [g for g in np.unique(groups) if (groups == g).sum() >= min_group]
        index = {g: i for i, g in enumerate(names)}
        group_of = torch.tensor([index.get(g, -1) for g in groups])

        valid = mask.bool()
        flat_k1, flat_k2, flat_rho = k1[valid], k2[valid], rho[valid]
        z = torch.stack([flat_k1, flat_k2, flat_rho], dim=-1)
        cov = torch.cov((z - z.mean(0)).T) + 1e-12 * torch.eye(3, dtype=torch.float64)
        chol0 = torch.linalg.cholesky(cov)
        theta0 = torch.zeros(N_PARAMS, dtype=torch.float64)
        theta0[:3] = math.log(3000.0)
        theta0[3:6] = torch.log(torch.diagonal(chol0))
        theta0[6], theta0[7], theta0[8] = chol0[1, 0], chol0[2, 0], chol0[2, 1]
        theta0[9], theta0[10] = z[:, 0].mean(), z[:, 1].mean()
        pooled = theta0.clone().requires_grad_(True)
        delta = torch.zeros(len(names), N_PARAMS, dtype=torch.float64, requires_grad=True)
        n_total = float(mask.sum())

        def branch_theta():
            extra = torch.where(group_of.unsqueeze(-1) >= 0, delta[group_of.clamp_min(0)], torch.zeros(1, N_PARAMS, dtype=torch.float64))
            return pooled + extra

        def closure():
            optimizer.zero_grad()
            ll = _chain_loglik(branch_theta(), k1, k2, rho, mask, step)
            loss = -ll.sum() / n_total + (delta**2).sum() / (2.0 * shrinkage_std**2) / n_total
            loss.backward()
            return loss

        params = [pooled] + ([delta] if len(names) else [])
        optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
        optimizer.step(closure)
        final = float(closure())
        self.pooled = pooled.detach().numpy().copy()
        self.group_theta = {g: (pooled + delta[i]).detach().numpy().copy() for g, i in index.items()}
        self.fit_info = {"nll_per_node": final, "n_branches": len(branches), "n_nodes": int(n_total), "groups": names, "shrinkage_std": shrinkage_std}
        if self.direction_mode in ("tilt", "multiscale"):
            self.direction = (DirectionMultiScale() if self.direction_mode == "multiscale" else DirectionModel()).fit(branches, min_group=min_group)
            self.fit_info["direction"] = self.direction.to_dict()
        if self.radius_mode in ("branch", "multiscale"):
            rho_series = [b.log_radius - self.mean.branch_log_mean(b) for b in branches]
            self.radius = (RadiusMultiScale() if self.radius_mode == "multiscale" else RadiusResidualModel()).fit(rho_series, [b.step for b in branches], [float(np.median(np.exp(b.log_radius))) for b in branches],
                                                    [b.length for b in branches], [b.group for b in branches], min_group=min_group)
            self.fit_info["radius_residual"] = self.radius.to_dict()
        if verbose:
            print(self.fit_info)
        return self

    def params_for(self, group: str) -> OUParams:
        return OUParams(self.group_theta.get(group, self.pooled))

    # -------------------------------------------------------------- generation
    def sample(self, conditions: Sequence[Dict[str, Any]], rng: np.random.Generator) -> List[Dict[str, Any]]:
        """One raw feature sample per request. A request: ``species``, ``cell_type_binary``, ``compartment`` and
        optionally ``length_nm`` (otherwise drawn from the length model) and ``step_nm`` (node step of the request, e.g. of
        the real branch it copies: the preprocessing step is 250-500 nm with the adaptive smoothing; default 500). Returns dicts with ``kappa1``, ``kappa2``
        (M-1 steps), ``log_radius`` (M nodes), ``step``, ``length``, ``group``."""
        n = len(conditions)
        meta = []
        for c in conditions:
            group = group_key(c.get("species"), c.get("cell_type_binary"), c.get("compartment"))
            length = c.get("length_nm")
            if length is None:
                length = float(self.length_model.sample(group, self.length_bounds, rng)[0])
            step_req = c.get("step_nm")
            step_req = float(step_req) if step_req is not None and np.isfinite(step_req) and step_req > 0 else 500.0
            n_nodes = max(int(math.ceil(length / step_req)) + 1, 3)
            meta.append({"group": group, "length": float(length), "n_nodes": n_nodes, "step": float(length) / (n_nodes - 1),
                         "design": self.encoder.row(c.get("species"), c.get("cell_type_binary"), c.get("compartment"), float(length))})
        n_max = max(m["n_nodes"] for m in meta)
        z = np.zeros((n, n_max, 3))
        phi = np.zeros((n, 3))
        chol_q = np.zeros((n, 3, 3))
        mu = np.zeros((n, 2))
        for i, m in enumerate(meta):
            p = self.params_for(m["group"])
            sigma = p.sigma
            phi[i] = np.exp(-m["step"] / p.length_scales)
            q = sigma * (1.0 - np.outer(phi[i], phi[i]))
            w, v = np.linalg.eigh(q)
            chol_q[i] = v * np.sqrt(np.clip(w, 0.0, None))  # q = chol chol^T with an eigen square root (robust to PSD)
            z[i, 0] = np.linalg.cholesky(sigma) @ rng.standard_normal(3)
            mu[i] = p.mu
        eps = rng.standard_normal((n_max - 1, n, 3))
        for j in range(n_max - 1):
            z[:, j + 1] = phi * z[:, j] + np.einsum("nab,nb->na", chol_q, eps[j])
        out = []
        for i, m in enumerate(meta):
            log_mean = self.mean.sample_log_mean(m["design"], m["n_nodes"], rng)
            k = m["n_nodes"] - 1
            if self.direction is not None:
                kappa1, kappa2 = self.direction.sample_bending(m["n_nodes"], m["step"], m["group"], rng, radius_nm=float(np.exp(np.median(log_mean))))
            else:
                kappa1, kappa2 = mu[i, 0] + z[i, :k, 0], mu[i, 1] + z[i, :k, 1]
            if self.radius is not None:
                rho = self.radius.sample(m["n_nodes"], m["step"], m["group"], rng, radius_nm=float(np.exp(np.median(log_mean))), length_nm=m["length"])
            else:
                rho = z[i, : m["n_nodes"], 2]
            out.append({
                "kappa1": kappa1,
                "kappa2": kappa2,
                "log_radius": log_mean + rho,
                "step": m["step"], "length": m["length"], "group": m["group"],
            })
        return out

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "t0", "mean": self.mean.to_dict(), "length_model": self.length_model.to_dict(), "length_bounds": list(self.length_bounds),
            "pooled": self.pooled.tolist(), "group_theta": {g: t.tolist() for g, t in self.group_theta.items()},
            "encoder": self.encoder.to_dict(), "fit_info": self.fit_info,
            "direction_mode": self.direction_mode, "direction": self.direction.to_dict() if self.direction is not None else None,
            "radius_mode": self.radius_mode, "radius_residual": self.radius.to_dict() if self.radius is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "T0Model":
        mean = d["mean"]["kind"]
        obj = cls(mean, tuple(d["length_bounds"]), direction=d.get("direction_mode", "kappa"), radius_residual=d.get("radius_mode", "pooled"))
        obj.direction = direction_model_from_dict(d["direction"]) if d.get("direction") else None
        obj.radius = radius_model_from_dict(d["radius_residual"]) if d.get("radius_residual") else None
        obj.mean = MEAN_MODELS[mean].from_dict(d["mean"])
        obj.length_model = LengthModel.from_dict(d["length_model"])
        obj.pooled = np.asarray(d["pooled"])
        obj.group_theta = {g: np.asarray(t) for g, t in d["group_theta"].items()}
        obj.encoder = DesignEncoder.from_dict(d["encoder"])
        obj.fit_info = d.get("fit_info", {})
        return obj

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "T0Model":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
