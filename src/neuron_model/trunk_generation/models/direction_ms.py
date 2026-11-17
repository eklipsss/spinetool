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

Simulation-corrected fit (2026-11-15, ``simulation_corrections``): the model curves of the fit are LINEAR in the tilt
(kappa = tilt'), but the tilt is a tangent (component / cos to the chord) and the bending is computed from the normalised
tangents, so at tilts of 0.5-1 the generated bending is only ~0.77 of the linear curve at >= 1 um and the generated tilt
differs too (simulation of the fitted chain-T model: kappa generated / linear 0.99 / 0.92 / 0.72 / 0.77 / 0.76 / 0.77 /
0.77 at 0.25 ... 16 um). The optimiser therefore fitted curves that are not the generated ones (printed "kappa model 1.32
x empirical" at >= 2 um was in fact ~1.0, while the generated bending at 0.5-1 um was 0.75-0.89 of the real one and the
long-lag tilt 1.08: too tortuous). Now: fit -> amplitudes -> simulate the training branches (their steps, lengths,
radii, groups) -> per-lag ratio generated / linear -> refit to empirical / ratio; the reported curves
(``variogram_generated``) are the simulated ones. ``tilt_min_lag_nm`` (default 2 um): the short tilt lags are left to
the bending curve (the tangent inflates the real short-lag tilt increments by 1 / cos^2, which the model cannot follow;
on the chain-T curves: bending at 0.5-1 um 0.90-0.95 of real, at >= 2 um 1.04-1.08, tilt at >= 8 um 1.02).

Within-neuron amplitude (``amplitude_within_neuron``, needs ``branch.neuron``): the radius slope and the group
(compartment) offsets of log m_b are estimated from branches of the SAME neuron (neuron fixed effects), the level of a
group of groups that never meet in one neuron from all its branches. Reason: chain T printed offsets apical -0.00 /
basal +0.01 from the pooled residuals, while the real apical branches of a neuron bend 21 % less than its basal ones
(section 4 of the global report, paired by neuron) and the generated ones 5 % less.
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
    tilt_min_lag_nm: float = 2000.0       # fit settings (see the module docstring)
    simulation_corrections: int = 3
    amplitude_within_neuron: bool = True
    n_simulated: int = 600
    fit_target: str = "variogram"         # "functionals": then refit to the per-branch functional distributions (direction_functional.py)
    functional_maxfev: int = 400
    functional_n_branches: int = 800
    # Radius dependence of the Gaussian wiggle scale: l_w(r) = l_w (r / 300 nm)^gamma (2026-11-17). Chain AA: the real kappa
    # autocorrelation at 1 um rises with the radius (thin 0.16-0.29, thick 0.52-0.60) while the model's memory scale was
    # shared; the centre line is smoothed with a width ~2 r (branch preprocessing), so the correlation length of the
    # bending is expected to grow with r. gamma = 0 is the previous model (old files); it is fitted only in the functional
    # fit (``functional_fit_length_radius``), the semivariogram stage keeps gamma = 0.
    length_fast_radius_exponent: float = 0.0
    functional_fit_length_radius: bool = False

    # ------------------------------------------------------------------ pieces
    @property
    def theta(self) -> List[float]:
        return [self.sd_fast, self.length_fast_nm, self.drive_std, self.drive_length_nm, self.restore_nm, self.noise_sd]

    def length_fast_at(self, radius_nm: Optional[float]) -> float:
        """Gaussian wiggle scale of a branch of median radius ``radius_nm`` (the shared scale when unknown or gamma = 0)."""
        return length_fast_for(self.length_fast_nm, self.length_fast_radius_exponent, radius_nm)

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
    def _fit_amplitude(self, use: Sequence[Any], cache: Dict[float, float], min_group: int, n_bootstrap: int, seed: int) -> None:
        """log m_b ~ alpha + beta log(r_b / 300 nm) + group offset, spread tau (sampling noise of a branch subtracted)."""
        y = np.array([self._log_scale(b.kappa1, b.kappa2, self.theta, b.step, cache) for b in use])
        n = np.array([len(b.kappa1) for b in use], dtype=float)
        x = np.log(np.array([float(np.median(np.exp(b.log_radius))) for b in use]) / R_REF_NM)
        groups_arr = np.asarray([b.group for b in use])
        names = [str(g) for g in np.unique(groups_arr) if (groups_arr == g).sum() >= min_group]
        pooled_slope, pooled_offsets, pooled_alpha = _pooled_amplitude(y, x, n, groups_arr, names)
        neurons = [getattr(b, "neuron", None) for b in use]
        within = self.amplitude_within_neuron and all(v is not None for v in neurons)
        if within:
            slope, offsets, alpha_reg, w_info = _within_neuron_amplitude(y, x, n, groups_arr, np.asarray(neurons, dtype=object), names)
        else:
            slope, offsets, alpha_reg, w_info = pooled_slope, pooled_offsets, pooled_alpha, {}
        resid = y - alpha_reg - slope * x - np.array([offsets.get(str(g), 0.0) for g in groups_arr])
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
        fixed = np.log(np.average(np.exp(2.0 * (slope * x + np.array([offsets.get(str(g), 0.0) for g in groups_arr]))), weights=n))
        alpha = -tau**2 - 0.5 * float(fixed)
        self.amplitude = Amplitude(float(alpha), float(slope), float(tau), offsets)
        self.info.update({"tau_raw_spread": math.sqrt(var_real), "tau_noise": math.sqrt(var_noise), "alpha_regression": float(alpha_reg),
                          "amplitude_within_neuron": bool(within), "amplitude_pooled": {"slope": float(pooled_slope), "offsets": pooled_offsets},
                          **({"amplitude_within": w_info} if within else {})})

    def _linear_curves(self, steps: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        spec = spectrum(self.theta)
        return (semivariance(LAGS_NM, spec * bending_filter(steps)) + noise_kappa_semivariance(LAGS_NM, steps, self.noise_sd),
                semivariance(LAGS_NM, spec) + self.noise_sd**2)

    def _simulated_curves(self, branches: Sequence[Any], seed: int) -> Tuple[Dict[float, Optional[float]], Dict[float, Optional[float]]]:
        """Pooled semivariograms of bending and tilt of one generated chain per branch (its step, length, radius, group)."""
        rng = np.random.default_rng(seed)
        k1s, k2s, tilts, steps, lengths = [], [], [], [], []
        for b in branches:
            k1, k2 = self.sample_bending(len(b.kappa1) + 1, float(b.step), b.group, rng, radius_nm=float(np.median(np.exp(b.log_radius))))
            gamma, t = node_tangents(k1, k2, b.step)
            k1s.append(k1), k2s.append(k2), tilts.append(tilt_series(gamma, t)), steps.append(float(b.step)), lengths.append(float(b.length))
        return kappa_variogram(k1s, k2s, steps), tilt_variogram(tilts, steps, lengths)

    def fit(self, branches: Sequence[Any], *, min_group: int = 50, n_bootstrap: int = 400, seed: int = 0) -> "DirectionMultiScale":
        """``branches``: T0Branch-like (kappa1, kappa2, log_radius, step, length, group[, neuron])."""
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
        tilt_fit = {h: (v if h >= self.tilt_min_lag_nm else None) for h, v in tilt_data.items()}
        self.length_fast_radius_exponent = 0.0  # the semivariogram stage has a shared scale; the functional fit may free it
        self.info = {"n_branches": len(use), "tilt_min_lag_nm": self.tilt_min_lag_nm, "simulation_corrections": int(self.simulation_corrections)}
        rng = np.random.default_rng(seed + 7)
        sim_set = [use[i] for i in rng.choice(len(use), size=min(self.n_simulated, len(use)), replace=False)]
        ck = {h: 1.0 for h in LAGS_NM}
        ct = {h: 1.0 for h in LAGS_NM}
        history = []
        cache: Dict[float, float] = {}
        for it in range(int(self.simulation_corrections) + 1):
            self._fit_global({h: (None if v is None else v / ck[h]) for h, v in kappa_data.items()},
                             {h: (None if v is None else v / ct[h]) for h, v in tilt_fit.items()}, steps)
            cache = {}
            self._fit_amplitude(use, cache, min_group, n_bootstrap, seed)
            sim_k, sim_t = self._simulated_curves(sim_set, seed + 11)  # common random numbers across the iterations
            lin_k, lin_t = self._linear_curves(steps)
            ratio = lambda d, e: [None if (d.get(h) is None or e.get(h) is None) else float(math.sqrt(d[h] / e[h])) for h in LAGS_NM]  # noqa: E731
            history.append({"iteration": it, "theta": list(self.theta), "linear_fit_error": self.info["variogram_fit"]["relative_rms_error"],
                            "kappa_generated_over_empirical": ratio(sim_k, kappa_data), "tilt_generated_over_empirical": ratio(sim_t, tilt_data)})
            for j, h in enumerate(LAGS_NM):
                if sim_k.get(h) and lin_k[j] > 0:
                    ck[h] = sim_k[h] / lin_k[j]
                if sim_t.get(h) and lin_t[j] > 0:
                    ct[h] = sim_t[h] / lin_t[j]
        fitted = [h for h in LAGS_NM if kappa_data.get(h)] , [h for h in LAGS_NM if tilt_fit.get(h)]
        errs = [math.log(sim_k[h] / kappa_data[h]) / 2 for h in fitted[0] if sim_k.get(h)] + [math.log(sim_t[h] / tilt_data[h]) / 2 for h in fitted[1] if sim_t.get(h)]
        self.info["variogram_generated"] = {
            "lags_nm": list(LAGS_NM), "kappa_empirical": [kappa_data.get(h) for h in LAGS_NM], "kappa_generated": [sim_k.get(h) for h in LAGS_NM],
            "tilt_empirical": [tilt_data.get(h) for h in LAGS_NM], "tilt_generated": [sim_t.get(h) for h in LAGS_NM],
            "relative_rms_error_fitted_lags": float(np.sqrt(np.mean(np.square(errs)))) if errs else None}
        self.info["correction_history"] = history
        if self.fit_target == "functionals":
            from .direction_functional import fit_to_functionals

            self.info["variogram_estimate"] = {"theta": list(self.theta), "amplitude": self.amplitude.to_dict() if self.amplitude else None}
            self.info["functional_fit"] = fit_to_functionals(self, use, n_branches=self.functional_n_branches, maxfev=self.functional_maxfev, seed=seed,
                                                             fit_length_radius=self.functional_fit_length_radius,
                                                             log=lambda text: print(text, flush=True))
        elif self.fit_target != "variogram":
            raise ValueError(f"unknown fit_target {self.fit_target!r} (variogram | functionals)")
        return self

    # -------------------------------------------------------------- generation
    def sample_tilt(self, n_nodes: int, step: float, m: float, rng: np.random.Generator, radius_nm: Optional[float] = None) -> np.ndarray:
        """[n_nodes, 2] tilt: slow part (burn-in from the stationary regime) + the smooth Gaussian wiggle (its scale from the radius)."""
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
        l_w = self.length_fast_at(radius_nm)
        fast = np.column_stack([_gauss_process(n_nodes, step, self.sd_fast * m, l_w, rng) for _ in range(2)])
        return tilt[burn:] + fast + rng.normal(0.0, self.noise_sd * m, (n_nodes, 2))

    def sample_bending(self, n_nodes: int, step: float, group: str, rng: np.random.Generator, radius_nm: Optional[float] = None,
                       amplitude: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(kappa1, kappa2) [n_nodes - 1]; ``amplitude`` fixes m (bootstrap), otherwise drawn from the amplitude law."""
        if amplitude is None:
            amp = self.amplitude or Amplitude(0.0)
            m = math.exp(amp.mean_log(group, radius_nm) + amp.tau * rng.standard_normal())
        else:
            m = float(amplitude)
        tilt = self.sample_tilt(n_nodes, step, m, rng, radius_nm=radius_nm)
        t = np.column_stack([tilt, np.ones(len(tilt))])
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        return bending_from_tangents(t, step)

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "sd_fast": self.sd_fast, "length_fast_nm": self.length_fast_nm, "noise_sd": self.noise_sd, "drive_std": self.drive_std,
                "drive_length_nm": self.drive_length_nm, "restore_nm": self.restore_nm, "tilt_min_lag_nm": self.tilt_min_lag_nm,
                "simulation_corrections": self.simulation_corrections, "amplitude_within_neuron": self.amplitude_within_neuron,
                "fit_target": self.fit_target, "length_fast_radius_exponent": self.length_fast_radius_exponent,
                "functional_fit_length_radius": self.functional_fit_length_radius,
                "amplitude":self.amplitude.to_dict() if self.amplitude else None, "groups": dict(self.groups), "info": self.info}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectionMultiScale":
        amp = d.get("amplitude")
        return cls(float(d["sd_fast"]), float(d["length_fast_nm"]), float(d.get("noise_sd", 0.0)), float(d["drive_std"]), float(d["drive_length_nm"]),
                   float(d["restore_nm"]), Amplitude.from_dict(amp) if amp else None, dict(d.get("info", {})),
                   tilt_min_lag_nm=float(d.get("tilt_min_lag_nm", 0.0)), simulation_corrections=int(d.get("simulation_corrections", 0)),
                   amplitude_within_neuron=bool(d.get("amplitude_within_neuron", False)), fit_target=str(d.get("fit_target", "variogram")),
                   length_fast_radius_exponent=float(d.get("length_fast_radius_exponent", 0.0)),
                   functional_fit_length_radius=bool(d.get("functional_fit_length_radius", False)))


LENGTH_FAST_BOUNDS_NM = (50.0, 20000.0)


def length_fast_for(length_nm: float, exponent: float, radius_nm: Optional[float]) -> float:
    """l_w (r / R_REF)^gamma, clipped to a sane range; the shared scale when the radius is unknown or gamma = 0."""
    if not exponent or radius_nm is None or not np.isfinite(radius_nm) or radius_nm <= 0:
        return float(length_nm)
    return float(np.clip(length_nm * (float(radius_nm) / R_REF_NM) ** float(exponent), *LENGTH_FAST_BOUNDS_NM))


def _pooled_amplitude(y, x, n, groups, names):
    """The original estimate: WLS of y on (1, x), then group offsets = weighted mean residual of the group."""
    design = np.column_stack([np.ones_like(x), x])
    sw = np.sqrt(n / n.sum())
    coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
    resid = y - design @ coef
    offsets = {g: float(np.average(resid[groups == g], weights=n[groups == g])) for g in names}
    return float(coef[1]), offsets, float(coef[0])


def _within_neuron_amplitude(y, x, n, groups, neurons, names):
    """Radius slope and group offsets from the contrasts between branches of one neuron (neuron fixed effects, WLS on the
    neuron-demeaned data). Groups that meet in a neuron form a component: their within differences are kept, the component
    level is the weighted mean of all its branches; a group that never meets another one gets its overall level. Returns
    (slope, offsets relative to the intercept, intercept, info)."""
    gi = {g: k for k, g in enumerate(names)}
    D = np.zeros((len(y), len(names)))
    for b, g in enumerate(groups):
        if str(g) in gi:
            D[b, gi[str(g)]] = 1.0
    A = np.column_stack([x, D])
    codes, uniq = pd_factorize(neurons)
    wsum = np.bincount(codes, weights=n)
    mean_of = lambda v: (np.bincount(codes, weights=n * v) / np.maximum(wsum, 1e-300))[codes]  # noqa: E731
    Ad = np.column_stack([A[:, j] - mean_of(A[:, j]) for j in range(A.shape[1])])
    yd = y - mean_of(y)
    sw = np.sqrt(n / n.sum())
    coef = np.linalg.lstsq(Ad * sw[:, None], yd * sw, rcond=None)[0]
    slope = float(coef[0])
    w_g = coef[1:]
    # components: groups that share a neuron
    parent = list(range(len(names)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for c in range(len(uniq)):
        present = np.flatnonzero(D[codes == c].sum(axis=0) > 0)
        for a in present[1:]:
            parent[find(a)] = find(present[0])
    comp = np.array([find(k) for k in range(len(names))])
    r = y - slope * x
    in_group = D.sum(axis=1) > 0
    level = {}
    for c in np.unique(comp):
        members = np.flatnonzero(comp == c)
        rows = D[:, members].sum(axis=1) > 0
        within_part = D[:, members] @ w_g[members] if len(members) > 1 else np.zeros(len(y))
        base = float(np.average((r - within_part)[rows], weights=n[rows]))
        for k in members:
            level[names[k]] = base + (float(w_g[k]) if len(members) > 1 else 0.0)
    if (~in_group).any():
        intercept = float(np.average(r[~in_group], weights=n[~in_group]))
    else:
        intercept = float(np.average([level[str(g)] for g in groups], weights=n))
    offsets = {g: float(v - intercept) for g, v in level.items()}
    info = {"slope": slope, "offsets": offsets, "n_neurons": int(len(uniq)),
            "components": [[names[k] for k in np.flatnonzero(comp == c)] for c in np.unique(comp)]}
    return slope, offsets, intercept, info


def pd_factorize(values):
    import pandas as pd

    codes, uniq = pd.factorize(pd.Series(values, dtype=object))
    return codes, uniq


def direction_model_from_dict(d: Dict[str, Any]):
    """Dispatch on the saved kind (the older ``DirectionModel`` has none)."""
    from .direction import DirectionModel

    return DirectionMultiScale.from_dict(d) if d.get("kind") == "multiscale" else DirectionModel.from_dict(d)
