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
- ``gam`` (T0-b): m(s) = f(u, log L, C) + b_branch, penalised B-spline model plus a random branch shift;
- ``parabola_mix``: T0-a with a mixture of experts for the pair (``ParabolicMixtureMean``: components with their own
  regression and covariance, the component probabilities a softmax regression on the design; the thin distal population);
- ``parabola_mix_neuron`` (2026-11-24): the same with a random effect of the neuron on the pair
  (``ParabolicMixtureNeuronMean``); a request may carry ``t0_neuron_effect`` (the shared / posterior effect of its neuron),
  without it the effect is drawn independently per branch.

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
import time
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
    neuron: Optional[str] = None  # within-neuron estimate of the bending amplitude (direction_ms)

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

    def sample_log_mean(self, design: np.ndarray, n_nodes: int, rng: np.random.Generator, neuron_effect=None) -> np.ndarray:
        theta = rng.multivariate_normal(design @ self.coef, self.cov)  # no neuron effect in this model: ``neuron_effect`` ignored
        return self._profile(theta, n_nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "coef": self.coef.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParabolicMean":
        obj = cls()
        obj.coef, obj.cov = np.asarray(d["coef"]), np.asarray(d["cov"])
        return obj


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _gauss_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Row-wise log N(x_i; mean_i, cov) for 2-D (or d-D) rows."""
    chol = np.linalg.cholesky(cov)
    diff = np.linalg.solve(chol, (x - mean).T)
    return -0.5 * ((diff ** 2).sum(0) + 2.0 * np.log(np.diag(chol)).sum() + x.shape[1] * math.log(2.0 * math.pi))


class ParabolicMixtureMean(ParabolicMean):
    """T0-a with a mixture of experts for the per-branch pair theta = (log r_L, log r_0/r_L):

        p(theta | x) = sum_k pi_k(x) N(theta; B_k^T x, S_k),   pi(x) = softmax(G^T x)

    Why (2026-10-29): with one Gaussian the generated radii miss the thin tail of the real ones (5th percentile 141 vs
    107 nm, share <= 150 nm 7.5 % vs 18.5 %), and thin real branches are distal and spine-free; a Gaussian whose mean
    is linear in the soma distance shifts the whole distribution instead of adding a separate thin population. Here a
    component can be "thin" and the design (soma distance, terminal flag, cell type, compartment, log L) sets how
    likely a branch falls into it. ``n_components = 1`` is exactly ``ParabolicMean``.

    Fitting: EM. M-step = weighted ridge regressions (the intercept unpenalised) and weighted covariances (floor
    ``cov_floor`` on the diagonal) per component, the gating = multinomial logistic regression on the soft labels with a
    ridge penalty (L-BFGS); start = k-means of the residuals of the one-Gaussian fit. ``n_components="auto"``: the K
    of 1..``max_components`` with the lowest BIC (written to ``fit_info``)."""

    kind = "parabola_mix"

    def __init__(self, n_components: Any = "auto", max_components: int = 3, ridge: float = 1.0, gate_ridge: float = 1.0,
                 cov_floor: float = 1e-4, max_iter: int = 300, tol: float = 1e-7, seed: int = 0,
                 neuron_effect: bool = False, neuron_level: bool = True, neuron_gate: bool = True, gate_grid: int = 21,
                 neuron_iter: int = 40, neuron_tol: float = 1e-3) -> None:
        super().__init__()
        self.n_components = n_components
        self.max_components = int(max_components)
        self.ridge, self.gate_ridge, self.cov_floor = float(ridge), float(gate_ridge), float(cov_floor)
        self.max_iter, self.tol, self.seed = int(max_iter), float(tol), int(seed)
        self.neuron_effect, self.neuron_level, self.neuron_gate = bool(neuron_effect), bool(neuron_level), bool(neuron_gate)
        self.gate_grid, self.neuron_iter, self.neuron_tol = int(gate_grid), int(neuron_iter), float(neuron_tol)
        self.coefs: Optional[np.ndarray] = None  # [K, d, 2]
        self.covs: Optional[np.ndarray] = None   # [K, 2, 2]  (with the neuron effect: WITHIN-neuron covariance)
        self.gate: Optional[np.ndarray] = None   # [d, K] (last column 0)
        self.neuron_cov: Optional[np.ndarray] = None       # T [2, 2] of the level effect a_n (None: none)
        self.neuron_gate_sd: Optional[float] = None        # tau of the gate effect b_n (None: none)
        self.gate_direction: Optional[np.ndarray] = None   # s [K]: b_n s_k is added to the logit of component k (0 thickest, 1 thinnest)
        self.fit_info: Dict[str, Any] = {}

    # ------------------------------------------------------------ EM
    def _penalty(self, d: int) -> np.ndarray:
        pen = self.ridge * np.eye(d)
        pen[0, 0] = 0.0
        return pen

    def _fit_gate(self, x: np.ndarray, resp: np.ndarray, start: np.ndarray, offset: Optional[np.ndarray] = None,
                  weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Multinomial logistic regression of the soft labels ``resp`` on ``x`` (ridge, intercept free); ``offset`` [n, K] is
        added to the logits (the gate effect of the neuron), ``weight`` [n] weights the rows (grid of the gate effect)."""
        from scipy.optimize import minimize

        n, d = x.shape
        k = resp.shape[1]
        if k == 1:
            return np.zeros((d, 1))
        pen = np.ones(d)
        pen[0] = 0.0
        wr = resp if weight is None else resp * weight[:, None]

        def fun(flat):
            g = np.zeros((d, k))
            g[:, :-1] = flat.reshape(d, k - 1)
            logits = x @ g if offset is None else x @ g + offset
            logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
            loss = -(wr * logp).sum() + 0.5 * self.gate_ridge * (pen[:, None] * g[:, :-1] ** 2).sum()
            grad = x.T @ (np.exp(logp) * wr.sum(1, keepdims=True) - wr)
            grad = grad[:, :-1] + self.gate_ridge * pen[:, None] * g[:, :-1]
            return loss, grad.ravel()

        res = minimize(fun, start[:, :-1].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 200})
        g = np.zeros((d, k))
        g[:, :-1] = res.x.reshape(d, k - 1)
        return g

    def _log_pi(self, x: np.ndarray, b: Optional[np.ndarray] = None, gate: Optional[np.ndarray] = None) -> np.ndarray:
        """log component probabilities [n, K]; ``b`` [n]: gate effect of the neuron of every row (logit_k += b s_k)."""
        logits = x @ (self.gate if gate is None else gate)
        if b is not None and self.gate_direction is not None:
            logits = logits + np.asarray(b, dtype=float).reshape(-1, 1) * self.gate_direction[None, :]
        return logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)

    def _comp_loglik(self, theta: np.ndarray, x: np.ndarray, coefs=None, covs=None) -> np.ndarray:
        coefs = self.coefs if coefs is None else coefs
        covs = self.covs if covs is None else covs
        return np.column_stack([_gauss_logpdf(theta, x @ coefs[j], covs[j]) for j in range(len(coefs))])

    def _responsibilities(self, theta: np.ndarray, x: np.ndarray, coefs=None, covs=None, gate=None, b=None) -> np.ndarray:
        comp = self._log_pi(x, b, gate) + self._comp_loglik(theta, x, coefs, covs)
        return np.exp(comp - np.logaddexp.reduce(comp, axis=1)[:, None])

    def _em(self, theta: np.ndarray, x: np.ndarray, k: int, init: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """EM for K components; ``init`` (coefs, covs, gate): warm start from these parameters (their E-step) instead of k-means."""
        n, d = x.shape
        pen = self._penalty(d)
        if init is not None:
            resp = self._responsibilities(theta, x, init["coefs"], init["covs"], init["gate"])
        elif k == 1:
            resp = np.ones((n, 1))
        else:
            from sklearn.cluster import KMeans

            base = np.linalg.solve(x.T @ x + pen, x.T @ theta)
            labels = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit_predict(theta - x @ base)
            resp = np.eye(k)[labels] * 0.9 + 0.1 / k
        gate = np.zeros((d, k)) if init is None else np.array(init["gate"], dtype=float)
        loglik_old = -np.inf
        coefs, covs = np.zeros((k, d, 2)), np.zeros((k, 2, 2))
        for it in range(self.max_iter):
            for j in range(k):
                w = resp[:, j]
                coefs[j] = np.linalg.solve(x.T @ (x * w[:, None]) + pen, x.T @ (theta * w[:, None]))
                res = theta - x @ coefs[j]
                covs[j] = (res * w[:, None]).T @ res / max(w.sum(), 1e-9) + self.cov_floor * np.eye(2)
            gate = self._fit_gate(x, resp, gate)
            log_pi = np.log(np.clip(_softmax(x @ gate), 1e-300, None))
            comp = np.column_stack([log_pi[:, j] + _gauss_logpdf(theta, x @ coefs[j], covs[j]) for j in range(k)])
            total = np.logaddexp.reduce(comp, axis=1)
            resp = np.exp(comp - total[:, None])
            loglik = float(total.sum())
            if loglik - loglik_old < self.tol * n:
                break
            loglik_old = loglik
        n_params = k * (2 * d + 3) + (k - 1) * d
        return {"k": k, "coefs": coefs.copy(), "covs": covs.copy(), "gate": gate, "loglik": loglik, "iterations": it + 1,
                "bic": -2.0 * loglik + n_params * math.log(n), "weights": resp.mean(axis=0).tolist()}

    def fit(self, branches: Sequence[T0Branch], ridge: Optional[float] = None) -> "ParabolicMixtureMean":
        if ridge is not None:
            self.ridge = float(ridge)
        theta = np.vstack([self._fit_branch(b) for b in branches])
        x = np.vstack([b.design for b in branches])
        ks = range(1, self.max_components + 1) if self.n_components == "auto" else [int(self.n_components)]
        fits = [self._em(theta, x, k) for k in ks]
        best = min(fits, key=lambda f: f["bic"])
        self.coefs, self.covs, self.gate = best["coefs"], best["covs"], best["gate"]
        # the one-Gaussian fields of the parent: the first component (keeps ParabolicMean users working)
        self.coef, self.cov = self.coefs[0], self.covs[0]
        self.fit_info = {
            "chosen_k": best["k"],
            "per_k": {str(f["k"]): {"bic": f["bic"], "loglik": f["loglik"], "iterations": f["iterations"], "weights": f["weights"]} for f in fits},
            "components": [{"weight": w, "log_r_L_intercept_nm": float(math.exp(c[0, 0])), "sd_log_r_L": float(math.sqrt(v[0, 0]))}
                           for w, c, v in zip(best["weights"], best["coefs"], best["covs"])],
        }
        if self.neuron_effect:
            neurons = [b.neuron for b in branches]
            if any(n is None for n in neurons):
                raise ValueError("neuron_effect needs T0Branch.neuron for every training branch")
            self._fit_neuron_effect(theta, x, neurons, best)
        return self

    # ------------------------------------------------------------ neuron effect
    @property
    def has_neuron_effect(self) -> bool:
        return self.neuron_cov is not None or self.neuron_gate_sd is not None

    def _grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Standard-normal quadrature nodes z_j and log weights for the gate effect b = tau z."""
        z = np.linspace(-3.5, 3.5, max(self.gate_grid, 3))
        logw = -0.5 * z ** 2
        return z, logw - np.logaddexp.reduce(logw)

    def _neuron_estep(self, theta: np.ndarray, x: np.ndarray, g: np.ndarray, n_g: int, a: np.ndarray):
        """E-step of the neuron effects at the current level effects ``a`` [n_g, 2] (plugged in): responsibilities [B, K]
        averaged over the posterior of the gate effect, the grid posterior w [n_g, J] (None without the gate effect) and the
        per-grid responsibilities [B, J, K] (for the gate refit)."""
        comp = self._comp_loglik(theta - a[g], x)
        if self.neuron_gate_sd is None:
            lp = self._log_pi(x)
            joint = lp + comp
            return np.exp(joint - np.logaddexp.reduce(joint, axis=1, keepdims=True)), None, None
        z, logw = self._grid()
        b = self.neuron_gate_sd * z
        logits = (x @ self.gate)[:, None, :] + b[None, :, None] * self.gate_direction[None, None, :]
        lp = logits - np.logaddexp.reduce(logits, axis=2, keepdims=True)          # [B, J, K]
        joint = lp + comp[:, None, :]
        lb = np.logaddexp.reduce(joint, axis=2)                                    # [B, J]: log p(theta_b | b_j)
        ln = np.zeros((n_g, len(z)))
        np.add.at(ln, g, lb)
        ln += logw[None, :]
        w = np.exp(ln - np.logaddexp.reduce(ln, axis=1, keepdims=True))            # [n_g, J]
        resp_j = np.exp(joint - lb[..., None])                                     # [B, J, K]
        return np.einsum("bj,bjk->bk", w[g], resp_j), w, resp_j

    def _level_update(self, theta: np.ndarray, x: np.ndarray, g: np.ndarray, n_g: int, resp: np.ndarray):
        """Gaussian posterior N(a_n, V_n) of the level effects given the responsibilities: P_b = sum_k resp_bk S_k^-1,
        r_b = theta_b - sum_k resp_bk B_k^T x_b, V_n = (T^-1 + sum_b P_b)^-1, a_n = V_n sum_b P_b r_b."""
        means = np.einsum("bk,kbj->bj", resp, np.stack([x @ c for c in self.coefs]))
        prec = np.einsum("bk,kij->bij", resp, np.linalg.inv(self.covs))
        r = theta - means
        sp = np.zeros((n_g, 2, 2))
        np.add.at(sp, g, prec)
        spr = np.zeros((n_g, 2))
        np.add.at(spr, g, np.einsum("bij,bj->bi", prec, r))
        v = np.linalg.inv(np.linalg.inv(self.neuron_cov)[None] + sp)
        return np.einsum("gij,gj->gi", v, spr), v

    def _fit_neuron_effect(self, theta: np.ndarray, x: np.ndarray, neurons: Sequence[str], best: Dict[str, Any]) -> None:
        """Two random effects of the neuron n, K fixed at the BIC choice without them:

            level:  theta_b = B_k^T x_b + a_n + e_b,          a_n ~ N(0, T)
            gate:   P(k | x_b, n) = softmax_k(G^T x_b + b_n s_k),  b_n ~ N(0, tau^2)

        s_k = (max c - c_k) / (max c - min c), c_k = mean log r_L of component k (0 for the thickest, 1 for the thinnest):
        b_n is the shift of the log odds of the thinnest against the thickest component, b_n > 0 moves the branches of the
        neuron towards the thin components (a neuron with many thin branches); a_n shifts the thickness of all its
        branches. ECM, each round: (i) E-step over b_n on a quadrature grid with a_n plugged in, (ii) Gaussian posterior of
        a_n given the responsibilities, (iii) T = mean (a a^T + V), tau^2 = mean E[b^2] (EM steps of the random-effect
        variances), (iv) M-step of the components (weighted ridge on theta - a_n) and of the gate (logistic regression on
        the grid-expanded rows with the offset b_j s and weights w_nj). Stops when T and tau change by < ``neuron_tol``."""
        names, g = np.unique(np.asarray(neurons, dtype=object).astype(str), return_inverse=True)
        n_g, (n, d), k = len(names), x.shape, best["k"]
        cov_before = self.covs.copy()
        pen = self._penalty(d)
        c = np.array([float(np.mean((x @ cf)[:, 0])) for cf in self.coefs])
        gate_on = self.neuron_gate and k > 1 and c.std() > 0
        self.gate_direction = (c.max() - c) / (c.max() - c.min()) if gate_on else None
        self.neuron_gate_sd = 1.0 if gate_on else None
        a = np.zeros((n_g, 2))
        v = np.zeros((n_g, 2, 2))
        if self.neuron_level:
            resp0 = self._responsibilities(theta, x)
            r0 = theta - np.einsum("bk,kbj->bj", resp0, np.stack([x @ cf for cf in self.coefs]))
            m = np.zeros((n_g, 2))
            np.add.at(m, g, r0)
            m /= np.bincount(g, minlength=n_g)[:, None]
            self.neuron_cov = 0.5 * np.cov(m.T) + 1e-4 * np.eye(2)  # start: half the spread of the neuron means of the residuals
        z, _ = self._grid()
        it, history = 0, []
        for it in range(1, self.neuron_iter + 1):
            resp, w, resp_j = self._neuron_estep(theta, x, g, n_g, a)
            change = 0.0
            if self.neuron_level:
                a, v = self._level_update(theta, x, g, n_g, resp)
                t_new = (np.einsum("gi,gj->ij", a, a) + v.sum(0)) / n_g + 1e-8 * np.eye(2)
                change = max(change, float(np.abs(t_new - self.neuron_cov).max() / max(np.abs(self.neuron_cov).max(), 1e-12)))
                self.neuron_cov = t_new
                resp, w, resp_j = self._neuron_estep(theta, x, g, n_g, a)
            if w is not None:
                tau_new = self.neuron_gate_sd * math.sqrt(float(np.mean(w @ z ** 2)))
                # M-step of the gate at the CURRENT grid (before tau moves), rows (branch, grid node) weighted by w_nj
                offset = np.repeat((self.neuron_gate_sd * z)[None, :, None] * self.gate_direction[None, None, :], n, axis=0).reshape(-1, k)
                self.gate = self._fit_gate(np.repeat(x, len(z), axis=0), resp_j.reshape(-1, k), self.gate, offset=offset, weight=w[g].reshape(-1))
                change = max(change, abs(tau_new - self.neuron_gate_sd) / max(self.neuron_gate_sd, 1e-6))
                self.neuron_gate_sd = max(tau_new, 1e-3)
            shifted = theta - a[g]
            for j in range(k):
                wj = resp[:, j]
                self.coefs[j] = np.linalg.solve(x.T @ (x * wj[:, None]) + pen, x.T @ (shifted * wj[:, None]))
                res = shifted - x @ self.coefs[j]
                self.covs[j] = (res * wj[:, None]).T @ res / max(wj.sum(), 1e-9) + self.cov_floor * np.eye(2)
            if w is None:
                self.gate = self._fit_gate(x, resp, self.gate)
            history.append({"sd_level": None if self.neuron_cov is None else float(math.sqrt(self.neuron_cov[0, 0])), "tau_gate": self.neuron_gate_sd})
            if change < self.neuron_tol:
                break
        self.coef, self.cov = self.coefs[0], self.covs[0]
        resp, w, _ = self._neuron_estep(theta, x, g, n_g, a)
        weights = resp.mean(0)
        within = float(np.sum(weights * self.covs[:, 0, 0]))
        info: Dict[str, Any] = {"n_neurons": int(n_g), "iterations": it, "history": history[-5:], "weights": weights.tolist(),
                                "sd_log_r_L_within_before": [float(math.sqrt(cv[0, 0])) for cv in cov_before],
                                "sd_log_r_L_within_after": [float(math.sqrt(cv[0, 0])) for cv in self.covs],
                                "component_mean_log_r_L": c.tolist()}
        if self.neuron_cov is not None:
            t_cov = self.neuron_cov
            info.update(cov=t_cov.tolist(), sd_log_r_L=float(math.sqrt(t_cov[0, 0])), sd_log_ratio=float(math.sqrt(t_cov[1, 1])),
                        corr=float(t_cov[0, 1] / math.sqrt(t_cov[0, 0] * t_cov[1, 1])),
                        # share of the between-neuron variance in the (mixture-weighted) variance of log r_L around the regression
                        share_log_r_L=float(t_cov[0, 0] / (t_cov[0, 0] + within)))
        if self.neuron_gate_sd is not None:
            info.update(tau_gate=float(self.neuron_gate_sd), gate_direction=self.gate_direction.tolist())
        self.fit_info["neuron_effect"] = info

    def neuron_posterior(self, branches: Sequence["T0Branch"], n_iter: int = 6) -> Dict[str, np.ndarray]:
        """Posterior of the effects of ONE neuron given its observed branches (none: the prior): level N(a_mean, a_cov)
        and the grid posterior of the gate effect (``b_values``, ``b_weights``)."""
        if not self.has_neuron_effect:
            raise ValueError("this mean model has no neuron effect")
        z, logw = self._grid()
        b_vals = (self.neuron_gate_sd or 0.0) * z
        out = {"a_mean": np.zeros(2), "a_cov": self.neuron_cov.copy() if self.neuron_cov is not None else np.zeros((2, 2)),
               "b_values": b_vals, "b_weights": np.exp(logw)}
        if not len(branches):
            return out
        theta = np.vstack([self._fit_branch(b) for b in branches])
        x = np.vstack([b.design for b in branches])
        g = np.zeros(len(theta), dtype=int)
        a = np.zeros((1, 2))
        for _ in range(n_iter):
            resp, w, _ = self._neuron_estep(theta, x, g, 1, a)
            if self.neuron_cov is None:
                break
            a, v = self._level_update(theta, x, g, 1, resp)
            out["a_cov"] = v[0]
        out["a_mean"] = a[0]
        if w is not None:
            out["b_weights"] = w[0]
        return out

    def sample_posterior_effect(self, branches: Sequence["T0Branch"], rng: np.random.Generator) -> np.ndarray:
        """A draw [a_log_r_L, a_log_ratio, b_gate] of the effect of the neuron whose observed branches are ``branches``."""
        post = self.neuron_posterior(branches)
        a = rng.multivariate_normal(post["a_mean"], post["a_cov"]) if self.neuron_cov is not None else np.zeros(2)
        b = 0.0
        if self.neuron_gate_sd is not None:
            j = int(rng.choice(len(post["b_values"]), p=post["b_weights"] / post["b_weights"].sum()))
            step = post["b_values"][1] - post["b_values"][0]
            b = float(post["b_values"][j] + rng.uniform(-0.5, 0.5) * step)
        return np.r_[a, b]

    def sample_neuron_effect(self, rng: np.random.Generator) -> np.ndarray:
        """A fresh effect [a_log_r_L, a_log_ratio, b_gate] of a NEW neuron from the prior (zeros where an effect is off)."""
        a = rng.multivariate_normal(np.zeros(2), self.neuron_cov) if self.neuron_cov is not None else np.zeros(2)
        b = rng.normal(0.0, self.neuron_gate_sd) if self.neuron_gate_sd is not None else 0.0
        return np.r_[a, b]

    # ------------------------------------------------------------ sampling / io
    def component_probabilities(self, design: np.ndarray, gate_effect: float = 0.0) -> np.ndarray:
        return np.exp(self._log_pi(np.atleast_2d(design), np.array([gate_effect]) if gate_effect else None))

    def sample_theta(self, design: np.ndarray, rng: np.random.Generator, neuron_effect=None) -> np.ndarray:
        """``neuron_effect``: [a_log_r_L, a_log_ratio(, b_gate)] of the neuron the branch belongs to (shared by its
        branches, or a posterior draw); None = an independent fresh draw for this branch, which gives the same MARGINAL
        distribution of the pair as the fitted hierarchical model averaged over neurons."""
        e = np.zeros(3)
        if self.has_neuron_effect:
            src = self.sample_neuron_effect(rng) if neuron_effect is None else np.asarray(neuron_effect, dtype=float).ravel()
            e[: len(src)] = src[:3]
        pi = self.component_probabilities(design, float(e[2]))[0]
        k = int(rng.choice(len(pi), p=pi / pi.sum()))
        return rng.multivariate_normal(design @ self.coefs[k], self.covs[k]) + e[:2]

    def sample_log_mean(self, design: np.ndarray, n_nodes: int, rng: np.random.Generator, neuron_effect=None) -> np.ndarray:
        return self._profile(self.sample_theta(design, rng, neuron_effect), n_nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "coefs": self.coefs.tolist(), "covs": self.covs.tolist(), "gate": self.gate.tolist(),
                "neuron_cov": None if self.neuron_cov is None else self.neuron_cov.tolist(),
                "neuron_gate_sd": self.neuron_gate_sd,
                "gate_direction": None if self.gate_direction is None else self.gate_direction.tolist(),
                "settings": {"n_components": self.n_components, "max_components": self.max_components, "ridge": self.ridge,
                             "gate_ridge": self.gate_ridge, "cov_floor": self.cov_floor, "neuron_effect": self.neuron_effect,
                             "neuron_level": self.neuron_level, "neuron_gate": self.neuron_gate, "gate_grid": self.gate_grid,
                             "neuron_iter": self.neuron_iter, "neuron_tol": self.neuron_tol},
                "fit_info": self.fit_info}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParabolicMixtureMean":
        obj = cls(**d.get("settings", {}))
        obj.coefs, obj.covs, obj.gate = np.asarray(d["coefs"]), np.asarray(d["covs"]), np.asarray(d["gate"])
        obj.neuron_cov = None if d.get("neuron_cov") is None else np.asarray(d["neuron_cov"])
        obj.neuron_gate_sd = None if d.get("neuron_gate_sd") is None else float(d["neuron_gate_sd"])
        obj.gate_direction = None if d.get("gate_direction") is None else np.asarray(d["gate_direction"])
        obj.coef, obj.cov = obj.coefs[0], obj.covs[0]
        obj.fit_info = d.get("fit_info", {})
        return obj


class ParabolicMixtureNeuronMean(ParabolicMixtureMean):
    """``parabola_mix`` with random effects of the NEURON (2026-11-24, ``_fit_neuron_effect``): a level effect a_n on the
    pair (all branches of the neuron thicker / thinner) and a gate effect b_n on the component probabilities (more or
    fewer branches in the thin components).

    Why: in the full generation the spine excess came from the trunk (chain AK0): real thin branches got a thick trunk
    (only 26-30 % of the real thin branches stayed thin, three times the chance rate), because the share of thin
    branches differs strongly between neurons (half the neurons <= 2 %, a tenth >= 20 %; 22 % of the variance of the
    log radius is between neurons) and the conditions do not carry that. With the effects, the branches of one neuron
    share their thickness: a generated neuron draws its effects once, a request copying a real branch can use the
    posterior of that neuron. Drawing the effects independently per branch keeps the marginal distribution."""

    kind = "parabola_mix_neuron"

    def __init__(self, **kw: Any) -> None:
        kw.setdefault("neuron_effect", True)
        super().__init__(**kw)


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

    def sample_log_mean(self, design: np.ndarray, n_nodes: int, rng: np.random.Generator, neuron_effect=None) -> np.ndarray:
        u = np.arange(n_nodes) / (n_nodes - 1)
        return self._design(u, design) @ self.beta + rng.normal(0.0, self.sigma_b)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "n_basis": self.n_basis, "beta": self.beta.tolist(), "sigma_b": self.sigma_b}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GamMean":
        obj = cls(d["n_basis"])
        obj.beta, obj.sigma_b = np.asarray(d["beta"]), float(d["sigma_b"])
        return obj


MEAN_MODELS = {"parabola": ParabolicMean, "parabola_mix": ParabolicMixtureMean, "parabola_mix_neuron": ParabolicMixtureNeuronMean, "gam": GamMean}


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
                 radius_residual: str = "multiscale", direction_options: Optional[Dict[str, Any]] = None) -> None:
        if mean_model not in MEAN_MODELS:
            raise ValueError(f"mean_model must be one of {tuple(MEAN_MODELS)}")
        if direction not in ("multiscale", "tilt", "kappa"):
            raise ValueError("direction must be 'multiscale', 'tilt' or 'kappa'")
        if radius_residual not in ("multiscale", "branch", "pooled"):
            raise ValueError("radius_residual must be 'multiscale', 'branch' or 'pooled'")
        self.direction_mode = direction
        self.direction_options = dict(direction_options or {})  # DirectionMultiScale fit settings (tilt_min_lag_nm, simulation_corrections, ...)
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

    def fit(self, branches: Sequence[T0Branch], *, encoder: DesignEncoder, shrinkage_std: float = 0.3, max_iter: int = 200, min_group: int = 50, verbose: bool = False, progress=None) -> "T0Model":
        import torch

        say = progress or (lambda *_: None)  # progress(text): stage messages with elapsed time (fit_t0 passes its log)
        t_start = time.time()
        say(f"[t0-fit] {len(branches)} branches: mean radius profile and length model ...")
        self.encoder = encoder
        self.mean.fit(branches)
        self.length_model.fit([b.length for b in branches], [b.group for b in branches])
        say(f"[t0-fit] profile and length done ({time.time() - t_start:.0f} s); OU likelihood (L-BFGS, up to {max_iter} iterations) ...")
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
        # at least one row: with no group of >= min_group branches delta[group_of.clamp_min(0)] still indexes row 0 (masked
        # out by torch.where, it stays 0 under the shrinkage prior); an empty tensor raised IndexError
        delta = torch.zeros(max(len(names), 1), N_PARAMS, dtype=torch.float64, requires_grad=True)
        n_total = float(mask.sum())

        def branch_theta():
            extra = torch.where(group_of.unsqueeze(-1) >= 0, delta[group_of.clamp_min(0)], torch.zeros(1, N_PARAMS, dtype=torch.float64))
            return pooled + extra

        evals = [0]

        def closure():
            evals[0] += 1
            if evals[0] % 25 == 0:
                say(f"[t0-fit]   OU likelihood: {evals[0]} evaluations, {time.time() - t_start:.0f} s")
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
        say(f"[t0-fit] OU done ({time.time() - t_start:.0f} s, {evals[0]} evaluations); direction model ({self.direction_mode}) ...")
        if self.direction_mode in ("tilt", "multiscale"):
            self.direction = (DirectionMultiScale(**self.direction_options) if self.direction_mode == "multiscale" else DirectionModel()).fit(branches, min_group=min_group)
            self.fit_info["direction"] = self.direction.to_dict()
        say(f"[t0-fit] direction done ({time.time() - t_start:.0f} s); radius residual ({self.radius_mode}) ...")
        if self.radius_mode in ("branch", "multiscale"):
            rho_series = [b.log_radius - self.mean.branch_log_mean(b) for b in branches]
            self.radius = (RadiusMultiScale() if self.radius_mode == "multiscale" else RadiusResidualModel()).fit(rho_series, [b.step for b in branches], [float(np.median(np.exp(b.log_radius))) for b in branches],
                                                    [b.length for b in branches], [b.group for b in branches], min_group=min_group)
            self.fit_info["radius_residual"] = self.radius.to_dict()
        say(f"[t0-fit] done ({time.time() - t_start:.0f} s)")
        if verbose:
            print(self.fit_info)
        return self

    @property
    def has_neuron_effect(self) -> bool:
        return bool(getattr(self.mean, "has_neuron_effect", False))

    def fresh_neuron_effect(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        """Effect of a NEW neuron (prior draw), to be shared by all its branches; None without a neuron effect."""
        return self.mean.sample_neuron_effect(rng) if self.has_neuron_effect else None

    def posterior_neuron_effect(self, branches: Sequence[T0Branch], rng: np.random.Generator) -> Optional[np.ndarray]:
        """A draw of the effect of the neuron whose OTHER branches are ``branches`` (T0Branch with this model's design);
        no branch: a prior draw. None without a neuron effect."""
        if not self.has_neuron_effect:
            return None
        return self.mean.sample_posterior_effect(branches, rng)

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
                         "design": self.encoder.row(c.get("species"), c.get("cell_type_binary"), c.get("compartment"), float(length),
                                                    None if c.get("soma_distance_start_nm") is None else float(c["soma_distance_start_nm"]) + 0.5 * float(length),
                                                    c.get("is_terminal"))})
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
            log_mean = self.mean.sample_log_mean(m["design"], m["n_nodes"], rng, neuron_effect=conditions[i].get("t0_neuron_effect"))
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
