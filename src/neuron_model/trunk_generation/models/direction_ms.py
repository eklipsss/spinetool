"""Two-scale direction model of T0: a smooth fast wiggle of the heading + a slow turning that relaxes to the axis.

Why (full Minnie65 data, dendrites, 2026-10-28): ``direction.DirectionModel`` (one turning rate OU, memory 3.4 um)
gives the bending kappa1 only 0.58-0.64 of the real semivariance at 0.5-2 um (thin branches), too little overall
(rms 1.65e-4 vs 2.2e-4 1/nm), too persistent (autocorrelation 0.54 vs 0.37) and a too tortuous heavy tail (p90 2.2
vs 1.27 at >= 50 um). The real semivariogram of kappa1 saturates already at ~1 um and overshoots slightly at 2 um
(thin branches sqrt(2 gamma): 1.2e-4 / 2.0e-4 / 3.0e-4 / 3.4e-4 / 3.1e-4 / 3.1e-4 / 3.0e-4 1/nm at 0.25 ... 16 um): the
signature of the bending of a SMOOTH, STATIONARY transverse displacement of the axis (the centre line is a smoothing
spline), not of a random walk of the heading. With a Gaussian kernel of scale ~0.6 um the three first points are
reproduced (the same lesson as for the radius, ``radius.RadiusMultiScale``).

Model (per transverse component, isotropic; ``tilt`` = tan of the angle between the node tangent and the axis)::

    tilt = tilt_slow + w
    tilt_slow:  tilt' = -tilt / L_d + g,   g = OU(s, l_g)             (the old model: restoring to the axis, bounded tortuosity)
    w:          stationary Gaussian process, covariance sd_w^2 exp(-(h / l_w)^2)   (smooth wiggle: bending sd_w sqrt(2) / l_w)
    + e_j:      white noise of the tilt of each node, sd sd_n (the centre line is noisy at the node scale: the bending is
                its second difference, variance 2 sd_n^2 / step^2, flat semivariance from the lag of 2 nodes)

Both are multiplied by the amplitude of the branch m_b (random effect): log m_b ~ N(alpha + group + beta log(r_b/300 nm), tau^2).

Fit: the six shared parameters (sd_w, l_w, s, l_g, L_d, sd_n) from TWO pooled semivariograms at once, in the sqrt(2 gamma)
scale (relative error): of kappa (7 lags, 0.25-16 um) and of the tilt (lags up to 16 um, only branches longer than 3 lag:
the tilt is measured against the chord of the branch). The model semivariograms are computed from the spectra (the
discrete bending of nodes ``step`` apart is the spectrum times (2 sin(w step / 2) / step)^2). The amplitudes of the
branches: the ratio of the branch mean square bending to the model one, regression on log r / group, spread tau by a
parametric bootstrap (the noise of a short branch is subtracted); the intercept is normalised to E m^2 = 1 (the pooled
fit already carries the mean square amplitude). Generation: tangents = normalize((tilt, 1)), kappa1, kappa2 by
``direction.bending_from_tangents`` - the same convention as the other direction model, the rest of T0 is unchanged.
Units: nm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..variogram import LAGS_NM, semivariogram
from .direction import (R_REF_NM, Amplitude, MIN_DRIVES, bending_from_tangents, node_tangents, tilt_series)
from .radius import _gauss_process

OMEGA = np.logspace(-7.0, -1.0, 3000)  # angular frequency grid, 1/nm
_LN_OMEGA = np.log(OMEGA)
_DLN = np.diff(_LN_OMEGA)
MIN_RESTORE_NM, MAX_RESTORE_NM = 1000.0, 1.0e5
TILT_LENGTH_FACTOR = 3.0  # a lag h uses only branches of length >= 3 h


def _integral(f: np.ndarray) -> np.ndarray:
    """(1/pi) int f(w) dw over the grid (trapezoid in ln w); ``f`` [..., len(OMEGA)]."""
    y = f * OMEGA
    return (0.5 * (y[..., 1:] + y[..., :-1]) * _DLN).sum(-1) / math.pi


def spectrum(theta: Sequence[float]) -> np.ndarray:
    """Spectral density of ONE transverse tilt component for theta = (sd_w, l_w, s, l_g, L_d) (nm, tilt units)."""
    sd_w, l_w, s, l_g, restore = (float(v) for v in theta[:5])
    fast = sd_w**2 * l_w * math.sqrt(math.pi) * np.exp(-((OMEGA * l_w) ** 2) / 4.0)
    inv = 0.0 if math.isinf(restore) else 1.0 / restore
    slow = 2.0 * s**2 * l_g / (1.0 + (OMEGA * l_g) ** 2) / (OMEGA**2 + inv**2)
    return fast + slow


def bending_filter(steps: Sequence[float]) -> np.ndarray:
    """Mean over the node steps of |discrete derivative|^2 = (2 sin(w step / 2) / step)^2, averaged over up to 5 quantiles."""
    steps = np.asarray(steps, dtype=float)
    q = np.unique(np.quantile(steps, [0.1, 0.3, 0.5, 0.7, 0.9]))
    return np.mean([(2.0 * np.sin(OMEGA * h / 2.0) / h) ** 2 for h in q], axis=0)


def semivariance(lags_nm: Sequence[float], spec: np.ndarray) -> np.ndarray:
    """gamma(h) = (1/pi) int S (1 - cos w h) dw for each lag."""
    h = np.asarray(lags_nm, dtype=float)[:, None]
    return _integral(spec[None, :] * (1.0 - np.cos(OMEGA[None, :] * h)))


def noise_kappa_semivariance(lags_nm: Sequence[float], steps: Sequence[float], sd_n: float) -> np.ndarray:
    """Semivariance of the bending of white tilt noise: 3 sd^2 / step^2 at the lag of 1 node, 2 sd^2 / step^2 beyond;
    averaged over the node steps (only the steps that resolve the lag; if none does, all)."""
    steps = np.unique(np.asarray(steps, dtype=float))
    out = []
    for h in lags_nm:
        ok = [st for st in steps if int(round(h / st)) >= 1 and abs(int(round(h / st)) * st - h) <= 0.5 * st]
        use = ok if ok else list(steps)
        out.append(float(np.mean([(3.0 if int(round(h / st)) <= 1 else 2.0) * sd_n**2 / st**2 for st in use])))
    return np.array(out)


def kappa_variance(theta: Sequence[float], step: float) -> float:
    """Variance of one bending component of a branch with the node step ``step`` (unit amplitude)."""
    f = (2.0 * np.sin(OMEGA * step / 2.0) / step) ** 2
    return float(_integral(f * spectrum(theta[:5]))) + 2.0 * float(theta[5]) ** 2 / step**2


def tilt_variogram(tilts: Sequence[np.ndarray], steps: Sequence[float], lengths: Sequence[float], lags_nm: Sequence[float] = LAGS_NM) -> Dict[float, Optional[float]]:
    """Pooled semivariance of the tilt components (one [M,2] array per branch) at each lag; branches shorter than 3 lag skipped."""
    out: Dict[float, Optional[float]] = {}
    for h in lags_nm:
        total, count = 0.0, 0
        for x, step, length in zip(tilts, steps, lengths):
            lag = int(round(h / step))
            if lag < 1 or len(x) <= lag + 1 or abs(lag * step - h) > 0.5 * step or length < TILT_LENGTH_FACTOR * h:
                continue
            d = x[lag:] - x[:-lag]
            total += float((d**2).sum())
            count += d.size
        out[h] = 0.5 * total / count if count >= 60 else None
    return out


def kappa_variogram(kappa1: Sequence[np.ndarray], kappa2: Sequence[np.ndarray], steps: Sequence[float], lags_nm: Sequence[float] = LAGS_NM) -> Dict[float, Optional[float]]:
    a, b = semivariogram(kappa1, steps, lags_nm), semivariogram(kappa2, steps, lags_nm)
    return {h: (None if a[h] is None or b[h] is None else 0.5 * (a[h] + b[h])) for h in lags_nm}


@dataclass
class DirectionMultiScale:
    sd_fast: float = 0.0           # sd of the fast tilt wiggle of a branch with m = 1 (tilt, per component)
    length_fast_nm: float = 600.0  # Gaussian kernel scale l_w
    noise_sd: float = 0.0          # sd_n: white tilt noise of a node (m = 1)
    drive_std: float = 0.0         # s: sd of the turning rate of the slow part, 1/nm
    drive_length_nm: float = 3000.0  # l_g
    restore_nm: float = 5000.0     # L_d
    amplitude: Optional[Amplitude] = None
    info: Dict[str, Any] = field(default_factory=dict)
    kind: str = "multiscale"

    # ------------------------------------------------------------------ pieces
    @property
    def theta(self) -> List[float]:
        return [self.sd_fast, self.length_fast_nm, self.drive_std, self.drive_length_nm, self.restore_nm, self.noise_sd]

    @property
    def groups(self) -> Dict[str, float]:
        return self.amplitude.groups if self.amplitude is not None else {}

    def _fit_global(self, kappa_data: Dict[float, Optional[float]], tilt_data: Dict[float, Optional[float]], steps: Sequence[float]) -> None:
        from scipy.optimize import least_squares

        lags_k = [h for h in LAGS_NM if kappa_data.get(h)]
        lags_t = [h for h in LAGS_NM if tilt_data.get(h)]
        if len(lags_k) < 4:
            raise ValueError("too few resolved lags of the bending semivariogram")
        gk = np.array([kappa_data[h] for h in lags_k])
        gt = np.array([tilt_data[h] for h in lags_t])
        filt = bending_filter(steps)

        def curves(x):
            theta = np.exp(x)
            spec = spectrum(theta)
            return (semivariance(lags_k, spec * filt) + noise_kappa_semivariance(lags_k, steps, theta[5]),
                    (semivariance(lags_t, spec) + theta[5] ** 2 if lags_t else np.zeros(0)))

        def residual(x):
            mk, mt = curves(x)
            r = [0.5 * (np.log(np.maximum(mk, 1e-300)) - np.log(gk))]
            if lags_t:
                r.append(0.5 * (np.log(np.maximum(mt, 1e-300)) - np.log(gt)))
            return np.concatenate(r)

        sill = float(np.max(gk))
        lo = np.log([1e-4, 100.0, 1e-9, 300.0, MIN_RESTORE_NM, 1e-5])
        hi = np.log([2.0, 5000.0, 1e-2, 5e4, MAX_RESTORE_NM, 1.0])
        step_med = float(np.median(steps))
        best = None
        for l_w in (500.0, 1000.0):
            for l_g in (1000.0, 6000.0):
                for restore in (3000.0, 15000.0, 60000.0):
                    sd_w0 = math.sqrt(sill) * l_w / math.sqrt(2.0) * 0.7  # sill of kappa = 2 sd_w^2 / l_w^2
                    x0 = np.clip(np.log([sd_w0, l_w, 0.3 * math.sqrt(sill), l_g, restore, 0.3 * math.sqrt(sill / 2.0) * step_med]), lo + 1e-6, hi - 1e-6)
                    res = least_squares(residual, x0, bounds=(lo, hi))
                    if best is None or res.cost < best.cost:
                        best = res
        self.sd_fast, self.length_fast_nm, self.drive_std, self.drive_length_nm, self.restore_nm, self.noise_sd = (float(v) for v in np.exp(best.x))
        mk, mt = curves(best.x)
        self.info["variogram_fit"] = {
            "kappa_lags_nm": lags_k, "kappa_empirical": gk.tolist(), "kappa_model": mk.tolist(),
            "tilt_lags_nm": lags_t, "tilt_empirical": gt.tolist(), "tilt_model": mt.tolist(),
            "relative_rms_error": float(np.sqrt(np.mean(residual(best.x) ** 2) * 2.0)),
        }

    @staticmethod
    def _log_scale(kappa1: np.ndarray, kappa2: np.ndarray, theta: Sequence[float], step: float, cache: Dict[float, float]) -> float:
        key = round(float(step), 1)
        if key not in cache:
            cache[key] = kappa_variance(theta, float(step))
        ms = 0.5 * (float(np.mean(kappa1**2)) + float(np.mean(kappa2**2)))
        return 0.5 * math.log(max(ms, 1e-300) / cache[key])

    # --------------------------------------------------------------------- fit
    def fit(self, branches: Sequence[Any], *, min_group: int = 50, n_bootstrap: int = 400, seed: int = 0) -> "DirectionMultiScale":
        """``branches``: T0Branch-like (kappa1, kappa2, log_radius, step, length, group)."""
        use = [b for b in branches if b.n_nodes >= MIN_DRIVES + 1]
        if not use:
            raise ValueError(f"no branch with at least {MIN_DRIVES + 1} nodes")
        steps = [float(b.step) for b in use]
        lengths = [float(b.length) for b in use]
        tilts = []
        for b in use:
            gamma, t = node_tangents(b.kappa1, b.kappa2, b.step)
            tilts.append(tilt_series(gamma, t))
        kappa_data = kappa_variogram([b.kappa1 for b in use], [b.kappa2 for b in use], steps)
        tilt_data = tilt_variogram(tilts, steps, lengths)
        self.info = {"n_branches": len(use)}
        self._fit_global(kappa_data, tilt_data, steps)
        # amplitude of each branch: ratio of its mean square bending to the model one
        cache: Dict[float, float] = {}
        y = np.array([self._log_scale(b.kappa1, b.kappa2, self.theta, b.step, cache) for b in use])
        n = np.array([len(b.kappa1) for b in use], dtype=float)
        x = np.log(np.array([float(np.median(np.exp(b.log_radius))) for b in use]) / R_REF_NM)
        w = n / n.sum()
        design = np.column_stack([np.ones_like(x), x])
        coef = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        resid = y - design @ coef
        groups_arr = np.asarray([b.group for b in use])
        offsets: Dict[str, float] = {}
        for name in np.unique(groups_arr):
            m = groups_arr == name
            if m.sum() >= min_group:
                offsets[str(name)] = float(np.average(resid[m], weights=n[m]))
                resid[m] -= offsets[str(name)]
        var_real = float(np.average(resid**2, weights=n))
        # sampling noise of the log ratio of a branch (its length, step): parametric bootstrap at m = 1
        rng = np.random.default_rng(seed)
        self.amplitude = Amplitude(0.0, 0.0, 0.0, {})
        idx = rng.choice(len(use), size=min(n_bootstrap, len(use)), replace=False)
        y_sim, n_sim = [], []
        for i in idx:
            b = use[i]
            k1, k2 = self.sample_bending(len(b.kappa1) + 1, float(b.step), b.group, rng, radius_nm=None, amplitude=1.0)
            y_sim.append(self._log_scale(k1, k2, self.theta, b.step, cache))
            n_sim.append(len(b.kappa1))
        y_sim, n_sim = np.array(y_sim), np.array(n_sim, dtype=float)
        var_noise = float(np.average((y_sim - np.average(y_sim, weights=n_sim)) ** 2, weights=n_sim))
        tau = math.sqrt(max(var_real - var_noise, 0.0025))
        # E m^2 = 1: the pooled fit carries the mean square amplitude
        fixed = np.log(np.average(np.exp(2.0 * (coef[1] * x + np.array([offsets.get(g, 0.0) for g in groups_arr]))), weights=n))
        alpha = -tau**2 - 0.5 * float(fixed)
        self.amplitude = Amplitude(float(alpha), float(coef[1]), float(tau), offsets)
        self.info.update({"tau_raw_spread": math.sqrt(var_real), "tau_noise": math.sqrt(var_noise), "alpha_regression": float(coef[0])})
        return self

    # -------------------------------------------------------------- generation
    def sample_tilt(self, n_nodes: int, step: float, m: float, rng: np.random.Generator) -> np.ndarray:
        """[n_nodes, 2] tilt: slow part (burn-in from the stationary regime) + the smooth Gaussian wiggle."""
        a = math.exp(-step / self.restore_nm)
        phi = math.exp(-step / self.drive_length_nm)
        s_b = self.drive_std * m
        sd = s_b * math.sqrt(max(1.0 - phi**2, 0.0))
        burn = min(int(math.ceil(5.0 * max(self.restore_nm, self.drive_length_nm) / step)), 5000)
        total = burn + n_nodes
        tilt = np.zeros((total, 2))
        g = rng.normal(0.0, s_b, 2)
        eps = rng.normal(0.0, sd, (total, 2))
        for j in range(total - 1):
            tilt[j + 1] = a * tilt[j] + step * g
            g = phi * g + eps[j]
        fast = np.column_stack([_gauss_process(n_nodes, step, self.sd_fast * m, self.length_fast_nm, rng) for _ in range(2)])
        return tilt[burn:] + fast + rng.normal(0.0, self.noise_sd * m, (n_nodes, 2))

    def sample_bending(self, n_nodes: int, step: float, group: str, rng: np.random.Generator, radius_nm: Optional[float] = None,
                       amplitude: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(kappa1, kappa2) [n_nodes - 1]; ``amplitude`` fixes m (bootstrap), otherwise drawn from the amplitude law."""
        if amplitude is None:
            amp = self.amplitude or Amplitude(0.0)
            m = math.exp(amp.mean_log(group, radius_nm) + amp.tau * rng.standard_normal())
        else:
            m = float(amplitude)
        tilt = self.sample_tilt(n_nodes, step, m, rng)
        t = np.column_stack([tilt, np.ones(len(tilt))])
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        return bending_from_tangents(t, step)

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "sd_fast": self.sd_fast, "length_fast_nm": self.length_fast_nm, "noise_sd": self.noise_sd, "drive_std": self.drive_std,
                "drive_length_nm": self.drive_length_nm, "restore_nm": self.restore_nm,
                "amplitude": self.amplitude.to_dict() if self.amplitude else None, "groups": dict(self.groups), "info": self.info}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectionMultiScale":
        amp = d.get("amplitude")
        return cls(float(d["sd_fast"]), float(d["length_fast_nm"]), float(d.get("noise_sd", 0.0)), float(d["drive_std"]), float(d["drive_length_nm"]),
                   float(d["restore_nm"]), Amplitude.from_dict(amp) if amp else None, dict(d.get("info", {})))


def direction_model_from_dict(d: Dict[str, Any]):
    """Dispatch on the saved kind (the older ``DirectionModel`` has none)."""
    from .direction import DirectionModel

    return DirectionMultiScale.from_dict(d) if d.get("kind") == "multiscale" else DirectionModel.from_dict(d)
