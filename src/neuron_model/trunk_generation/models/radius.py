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


# ---------------------------------------------------------------------------
# multi-scale residual (2026-10-28)
# ---------------------------------------------------------------------------
#
# The AR(1) residual above has ONE correlation length. On the full Minnie65 data (diagnose_radius.py, semivariogram of log r
# along the branch) the generated radius matched the data at the fastest lag (0.25 um) but was 1.3-2.3 times too weak at
# 0.5-8 um (ratio down to 0.43 at 2-4 um) of branches thinner than 500 nm - with or without the branches of unknown
# compartment: the real variation saturates at ~4 um (about 13 % around the local mean), the single exponential fitted a
# memory of 54 um (a slow drift) and could not describe the swellings of a few micrometres as well.
#
# SECOND finding (full-data run 2026-10-28, first version of this model with two exponential components): the pooled
# semivariogram of log r is NOT that of an Ornstein-Uhlenbeck process. sqrt(2 gamma) grows LINEARLY with the lag up to
# ~2 um and saturates at ~4 um (0.020, 0.038, 0.070, 0.125, 0.182, 0.187, 0.191 at 0.25 ... 16 um): gamma ~ h^2 at short
# lags = a SMOOTH (differentiable) process - the radius profile is smoothed by the preprocessing (radius_smoothing_nm
# 1000), while an exponential kernel gives sqrt(2 gamma) ~ sqrt(h) (0.075 at 0.25 um). A Gaussian kernel
# gamma = sd^2 (1 - exp(-(h/l)^2)) with sd 0.134 and l 2.66 um reproduces all seven points to 0.002. The first component
# is therefore Gaussian; the second (exponential, slow drift) and the white noise stay as options of the fit.
#
# Model: rho = sum_k comp_k(sd_k, l_k) + white noise (sd_0), comp_k Gaussian-kernel ("gauss") or AR(1) ("exp"),
# all multiplied by the amplitude m_b of the branch
#   log m_b ~ N(alpha + alpha_group + beta_r log(r_b / 300 nm) + beta_l log(L_b / 20 um), tau^2).
# (sd_k, l_k, sd_0) are fitted to the pooled empirical semivariogram of rho (relative least squares over log-spaced lag
# bins); m_b comes from the ratio of the semivariogram of each branch to the model; tau^2 = spread of log m_b minus the
# sampling noise of that estimate, found by a PARAMETRIC BOOTSTRAP (series simulated from the fitted model with m = 1 on
# the same grids): robust to the strong autocorrelation, which makes a closed-form variance of the estimate unreliable.

LAG_NODES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
BIN_EDGES_NM = tuple(200.0 * 1.45**k for k in range(0, 17))  # 200 nm ... ~ 85 um


def _branch_variogram(x: np.ndarray, step: float):
    """(h [nm], sum of squared differences, count) for the lags of LAG_NODES that fit into the branch (<= n / 2)."""
    n = len(x)
    h, ss, cnt = [], [], []
    for lag in LAG_NODES:
        if lag > n // 2:
            break
        d = x[lag:] - x[:-lag]
        h.append(lag * step)
        ss.append(float(np.sum(d * d)))
        cnt.append(len(d))
    return np.asarray(h), np.asarray(ss), np.asarray(cnt, dtype=float)


def pooled_variogram(series: Sequence[np.ndarray], steps: Sequence[float]):
    """Pooled semivariance in log-spaced lag bins: (h_mean, gamma, pair count) of the bins with >= 200 pairs."""
    edges = np.asarray(BIN_EDGES_NM)
    sh, sg, sc = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for x, step in zip(series, steps):
        h, ss, cnt = _branch_variogram(np.asarray(x, dtype=float), float(step))
        if not len(h):
            continue
        b = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, len(edges) - 2)
        np.add.at(sh, b, h * cnt)
        np.add.at(sg, b, ss)
        np.add.at(sc, b, cnt)
    keep = sc >= 200
    return sh[keep] / sc[keep], 0.5 * sg[keep] / sc[keep], sc[keep]


def _gamma_model(h: np.ndarray, sd: Sequence[float], length: Sequence[float], nugget_sd: float, kernels: Sequence[str] = ("exp", "exp")) -> np.ndarray:
    g = nugget_sd**2 * np.ones_like(h, dtype=float)
    for s, l, k in zip(sd, length, kernels):
        g = g + s**2 * (1.0 - (np.exp(-((h / l) ** 2)) if k == "gauss" else np.exp(-h / l)))
    return g


def _gauss_process(n: int, step: float, sd: float, length_nm: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary Gaussian process with covariance sd^2 exp(-(h/l)^2) on n nodes: circulant embedding (exact up to the
    clipping of tiny negative eigenvalues of the embedding)."""
    m = 2 * (2 * n)  # padded circulant: the Gaussian kernel is positive definite on the padded grid
    j = np.arange(m)
    lag = np.minimum(j, m - j) * step
    cov = sd**2 * np.exp(-((lag / length_nm) ** 2))
    lam = np.clip(np.fft.fft(cov).real, 0.0, None)
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    x = np.fft.fft(np.sqrt(lam / m) * w)  # real part and imaginary part are two independent samples
    return x.real[:n]


def _ar1_stationary(n: int, step: float, sd: float, length_nm: float, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import lfilter

    phi = math.exp(-step / length_nm)
    eps = rng.normal(0.0, sd * math.sqrt(max(1.0 - phi**2, 0.0)), n)
    eps[0] = rng.normal(0.0, sd)
    return lfilter([1.0], [1.0, -phi], eps)


@dataclass
class RadiusMultiScale:
    sd: List[float] = field(default_factory=lambda: [0.1, 0.1])
    length_nm: List[float] = field(default_factory=lambda: [3000.0, 40000.0])
    nugget_sd: float = 0.0
    alpha: float = 0.0
    beta_r: float = 0.0
    beta_l: float = 0.0
    tau: float = TAU_FLOOR
    groups: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    kernels: List[str] = field(default_factory=lambda: ["exp", "exp"])  # per component: "gauss" | "exp"
    kind: str = "multiscale"

    def mean_log(self, group: str, radius_nm: Optional[float], length_nm: Optional[float]) -> float:
        r = R_REF_NM if radius_nm is None or not np.isfinite(radius_nm) or radius_nm <= 0 else float(radius_nm)
        L = L_REF_NM if length_nm is None or not np.isfinite(length_nm) or length_nm <= 0 else float(length_nm)
        return self.alpha + self.groups.get(group, 0.0) + self.beta_r * math.log(r / R_REF_NM) + self.beta_l * math.log(L / L_REF_NM)

    # -------------------------------------------------------------- fitting
    def _fit_global(self, series: Sequence[np.ndarray], steps: Sequence[float], kernels: Sequence[str]) -> None:
        from scipy.optimize import least_squares

        h, gamma, count = pooled_variogram(series, steps)
        if len(h) < 4:
            raise ValueError("too few lag bins with enough pairs to fit the variogram")
        kernels = list(kernels)
        sill = float(gamma[-3:].mean())
        wts = np.sqrt(count / count.max())

        def unpack(p):
            return [math.exp(p[0]), math.exp(p[2])], [math.exp(p[1]), math.exp(p[3])], math.exp(p[4])

        # the residual is taken in the scale of sqrt(2 gamma) (the typical change of the radius): short lags, where gamma
        # is tiny, are not over-weighted and a component that is not needed does not blow the fit up
        def resid(p):
            sd, ln, n0 = unpack(p)
            return (np.sqrt(2 * _gamma_model(h, sd, ln, n0, kernels)) - np.sqrt(2 * gamma)) / np.sqrt(2 * gamma) * wts

        lo = np.log([1e-4, 200.0, 1e-4, 4000.0, 1e-5])
        hi = np.log([2.0, 20000.0, 2.0, 400000.0, 0.5])
        best = None
        for l1, l2, f1 in ((2500.0, 20000.0, 0.8), (1500.0, 40000.0, 0.6), (4000.0, 12000.0, 0.9)):
            start = np.log([math.sqrt(sill) * f1, l1, math.sqrt(sill) * 0.3, l2, max(math.sqrt(gamma[0]) * 0.2, 1e-4)])
            res = least_squares(resid, np.clip(start, lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
            if best is None or res.cost < best.cost:
                best = res
        sd, ln, n0 = unpack(best.x)
        self.sd, self.length_nm, self.nugget_sd, self.kernels = sd, ln, n0, kernels
        model = _gamma_model(h, sd, ln, n0, kernels)
        self.info["variogram_fit"] = {"h_nm": h.tolist(), "empirical": gamma.tolist(), "model": model.tolist(),
                                      "typical_change_empirical": np.sqrt(2 * gamma).tolist(), "typical_change_model": np.sqrt(2 * model).tolist(),
                                      "relative_rms_error": float(np.sqrt(np.mean(((np.sqrt(2 * model) - np.sqrt(2 * gamma)) / np.sqrt(2 * gamma)) ** 2)))}

    def _log_scale(self, x: np.ndarray, step: float) -> Optional[tuple]:
        """(log m_hat, weight) of one branch: ratio of its semivariogram to the model's, weighted by the pairs."""
        h, ss, cnt = _branch_variogram(np.asarray(x, dtype=float), float(step))
        if len(h) < 3:
            return None
        model = _gamma_model(h, self.sd, self.length_nm, self.nugget_sd, self.kernels)
        ratio = (0.5 * ss.sum() / cnt.sum()) / float(np.sum(model * cnt) / cnt.sum())
        return 0.5 * math.log(max(ratio, 1e-6)), float(cnt.sum())

    def fit(self, rho: Sequence[np.ndarray], steps: Sequence[float], radii: Sequence[float], lengths: Sequence[float],
            groups: Sequence[str], *, min_group: int = 50, n_bootstrap: int = 300, seed: int = 0,
            kernels: Sequence[str] = ("gauss", "exp")) -> "RadiusMultiScale":
        keep = [i for i, r in enumerate(rho) if len(r) >= MIN_VALUES]
        if not keep:
            raise ValueError(f"no branch with at least {MIN_VALUES} nodes")
        series = [np.asarray(rho[i], dtype=float) for i in keep]
        st = [float(steps[i]) for i in keep]
        self._fit_global(series, st, kernels)
        est = [self._log_scale(x, h) for x, h in zip(series, st)]
        ok = [k for k, e in enumerate(est) if e is not None]
        y = np.array([est[k][0] for k in ok])
        w = np.array([est[k][1] for k in ok])
        xr = np.log(np.array([radii[keep[k]] for k in ok], dtype=float) / R_REF_NM)
        xl = np.log(np.array([lengths[keep[k]] for k in ok], dtype=float) / L_REF_NM)
        design = np.column_stack([np.ones_like(xr), xr, xl])
        sw = np.sqrt(w / w.sum())
        coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
        resid = y - design @ coef
        g = np.array([groups[keep[k]] for k in ok])
        self.groups = {}
        for name in np.unique(g):
            m = g == name
            if m.sum() >= min_group:
                self.groups[str(name)] = float(np.average(resid[m], weights=w[m]))
                resid[m] -= self.groups[str(name)]
        self.alpha, self.beta_r, self.beta_l = (float(c) for c in coef)
        # parametric bootstrap: the spread of log m_hat of branches whose TRUE amplitude is 1 is the sampling noise
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(ok), size=min(n_bootstrap, len(ok)), replace=False)
        saved = (self.alpha, self.beta_r, self.beta_l, self.tau, dict(self.groups))
        self.alpha, self.beta_r, self.beta_l, self.tau, self.groups = 0.0, 0.0, 0.0, 0.0, {}
        noise = []
        for k in pick:
            n_nodes, step = len(series[ok[k]]), st[ok[k]]
            sim = self.sample(n_nodes, step, "", rng, radius_nm=R_REF_NM, length_nm=L_REF_NM)
            e = self._log_scale(sim, step)
            if e is not None:
                noise.append(e[0])
        self.alpha, self.beta_r, self.beta_l, self.tau, self.groups = saved
        noise_var = float(np.var(noise)) if len(noise) > 5 else 0.0
        spread = float(np.average(resid**2, weights=w))
        self.tau = math.sqrt(max(spread - noise_var, TAU_FLOOR**2))
        self.info.update({"n_branches": len(ok), "log_scale_spread": math.sqrt(spread), "sampling_noise_sd": math.sqrt(noise_var)})
        return self

    # ----------------------------------------------------------- generation
    def sample(self, n_nodes: int, step: float, group: str, rng: np.random.Generator, radius_nm: Optional[float] = None,
               length_nm: Optional[float] = None) -> np.ndarray:
        m = math.exp(self.mean_log(group, radius_nm, length_nm) + self.tau * rng.standard_normal())
        out = self.nugget_sd * m * rng.standard_normal(n_nodes)
        for sd, ln, k in zip(self.sd, self.length_nm, self.kernels):
            out = out + (_gauss_process if k == "gauss" else _ar1_stationary)(n_nodes, step, sd * m, ln, rng)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "multiscale", "sd": list(self.sd), "length_nm": list(self.length_nm), "nugget_sd": self.nugget_sd, "alpha": self.alpha,
                "beta_r": self.beta_r, "beta_l": self.beta_l, "tau": self.tau, "groups": dict(self.groups), "info": dict(self.info), "kernels": list(self.kernels)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RadiusMultiScale":
        return cls([float(v) for v in d["sd"]], [float(v) for v in d["length_nm"]], float(d["nugget_sd"]), float(d["alpha"]), float(d.get("beta_r", 0.0)),
                   float(d.get("beta_l", 0.0)), float(d.get("tau", TAU_FLOOR)), {k: float(v) for k, v in d.get("groups", {}).items()}, dict(d.get("info", {})),
                   list(d.get("kernels", ["exp", "exp"])))


def radius_model_from_dict(d: Dict[str, Any]):
    """Dispatch on ``kind`` (dicts of the single-AR(1) model have none)."""
    return RadiusMultiScale.from_dict(d) if d.get("kind") == "multiscale" else RadiusResidualModel.from_dict(d)
