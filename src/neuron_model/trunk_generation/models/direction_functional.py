"""Fit of the T0 direction model to the DISTRIBUTIONS of per-branch functionals (method of simulated quantiles).

Why (chain Y, 2026-11-15): after the simulation-corrected fit the GENERATED pooled semivariograms of the training branches
match the empirical ones (bending 0.94-1.05, tilt 0.86-1.08), yet the per-branch functionals stay biased in every CV
fold (kappa rms +2.6e-5 1/nm, tortuosity +0.025, kappa autocorrelation at 1 um -0.11). A pooled semivariogram weights a
branch by its number of node pairs (~ its length); the evaluation compares distributions over branches, each branch with
weight 1. The fit target is therefore changed to what is evaluated.

Targets: for every training branch the three functionals of ``fit.functionals`` that depend on the direction model only:
``log kappa_rms``, ``log(tortuosity - 1)`` and ``kappa_autocorr_1um``. Branches are grouped in strata (length class x
radius class, as in fit.py); per stratum and functional the quantiles 0.1 / 0.25 / 0.5 / 0.75 / 0.9 of the real branches
are the targets, their standard errors from a bootstrap over NEURONS (branches of a neuron are not independent).

Simulator: one chain per training branch (a fixed subset), with its own node step, number of nodes, group and REAL median
radius, from the same law as ``DirectionMultiScale.sample_tilt`` / ``sample_bending`` (slow restoring turning + smooth
Gaussian wiggle + white node noise, times the branch amplitude m_b). Common random numbers: all standard normals of a
branch are drawn once, so the loss is a deterministic, nearly smooth function of the parameters (the burn-in of the slow
part has a FIXED length per branch, enough for the largest restoring length allowed).

Parameters (9, unconstrained): log of (sd_w, l_w, s, l_g, L_d, sd_n), the amplitude intercept alpha, its radius slope and
log tau; the group (compartment) offsets keep their within-neuron estimate. Loss: sum over strata, functionals and
quantiles of ((Q_sim - Q_real) / se)^2. Optimiser: Nelder-Mead from the semivariogram fit (the previous estimate).
Units: nm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .direction import R_REF_NM, Amplitude

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
FUNCTIONAL_NAMES = ("log_kappa_rms", "log_tortuosity_excess", "kappa_autocorr_1um")
LENGTH_EDGES = (20000.0, 50000.0)          # < 20 um, 20-50 um, >= 50 um (fit.LENGTH_CLASSES)
RADIUS_EDGES = (300.0,)                    # < 300 nm, >= 300 nm (thin branches bend most)
MAX_RESTORE_NM = 1.0e5
TORT_EPS = 1e-4


def _bending_loop(t: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """``direction.bending_from_tangents`` without scipy Rotation (Rodrigues by hand): kappa1, kappa2 in the transported frame."""
    m = len(t)
    k1 = np.zeros(m - 1)
    k2 = np.zeros(m - 1)
    t0 = t[0]
    ref = np.array([1.0, 0.0, 0.0]) if abs(t0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(t0, ref)
    e1 /= np.linalg.norm(e1)
    for j in range(m - 1):
        a, b = t[j], t[j + 1]
        axis = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
        sin = math.sqrt(axis @ axis)
        cos = float(a @ b)
        angle = math.atan2(sin, cos)
        e2 = np.array([a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2], a[0] * e1[1] - a[1] * e1[0]])
        if sin > 1e-15:
            k = axis / sin
            w = k * angle
            k1[j] = float(w @ e2) / step
            k2[j] = -float(w @ e1) / step
            c, s = math.cos(angle), math.sin(angle)
            kxe = np.array([k[1] * e1[2] - k[2] * e1[1], k[2] * e1[0] - k[0] * e1[2], k[0] * e1[1] - k[1] * e1[0]])
            e1 = e1 * c + kxe * s + k * float(k @ e1) * (1.0 - c)
        e1 = e1 - float(e1 @ b) * b
        e1 /= np.linalg.norm(e1)
    return k1, k2


try:  # numba makes the per-node loop ~50x faster; the pure-numpy loop above is the reference
    from numba import njit

    @njit(cache=True)
    def _bending_numba(t, step):  # pragma: no cover - same algorithm as _bending_loop
        m = t.shape[0]
        k1 = np.zeros(m - 1)
        k2 = np.zeros(m - 1)
        if abs(t[0, 0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(t[0], ref)
        e1 = e1 / np.sqrt((e1 * e1).sum())
        for j in range(m - 1):
            a = t[j]
            b = t[j + 1]
            axis = np.cross(a, b)
            sin = np.sqrt((axis * axis).sum())
            cos = (a * b).sum()
            angle = math.atan2(sin, cos)
            e2 = np.cross(a, e1)
            if sin > 1e-15:
                k = axis / sin
                k1[j] = (k * e2).sum() * angle / step
                k2[j] = -(k * e1).sum() * angle / step
                c = math.cos(angle)
                s = math.sin(angle)
                e1 = e1 * c + np.cross(k, e1) * s + k * (k * e1).sum() * (1.0 - c)
            e1 = e1 - (e1 * b).sum() * b
            e1 = e1 / np.sqrt((e1 * e1).sum())
        return k1, k2

    def bending(t: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
        return _bending_numba(np.ascontiguousarray(t, dtype=np.float64), float(step))
except Exception:  # noqa: BLE001
    bending = _bending_loop


def functionals_from_tangents(t: np.ndarray, step: float) -> Tuple[float, float, float]:
    """(log kappa_rms, log(tortuosity - 1), kappa autocorrelation at 1 um) of unit node tangents [M,3], exactly as
    ``fit.functionals`` computes them from (kappa1, kappa2) (gamma_{j+1} = gamma_j + step normalize(t_j + t_{j+1}))."""
    k1, k2 = bending(t, step)
    k = np.hypot(k1, k2)
    mid = t[:-1] + t[1:]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    chord = float(np.linalg.norm(mid.sum(axis=0) * step))
    m = len(t)
    tort = (m - 1) * step / max(chord, 1e-9)
    lag = max(1, int(round(1000.0 / step)))
    kc = k1 - k1.mean()
    if len(kc) > lag + 2 and kc.std() > 0:
        ac = float(np.corrcoef(kc[:-lag], kc[lag:])[0, 1])
    else:
        ac = float("nan")
    return float(np.log(max(np.sqrt(np.mean(k**2)), 1e-12))), float(np.log(max(tort - 1.0, TORT_EPS))), ac


def real_functionals(kappa1: np.ndarray, kappa2: np.ndarray, step: float) -> Tuple[float, float, float]:
    """The same three numbers for a real branch, exactly as ``fit.functionals``: kappa rms and the autocorrelation of
    kappa1 from the stored bending (its frame), the tortuosity from the rebuilt centre line."""
    from ..fit import functionals

    k1, k2 = np.asarray(kappa1, float), np.asarray(kappa2, float)
    f = functionals(k1, k2, np.zeros(len(k1) + 1), float(step))
    return (float(np.log(max(f["kappa_rms_per_nm"], 1e-12))), float(np.log(max(f["tortuosity"] - 1.0, TORT_EPS))), float(f["kappa_autocorr_1um"]))


def stratum_of(length_nm: float, radius_nm: float) -> int:
    li = int(np.searchsorted(LENGTH_EDGES, length_nm, side="right"))
    ri = int(np.searchsorted(RADIUS_EDGES, radius_nm, side="right"))
    return li * (len(RADIUS_EDGES) + 1) + ri


def stratum_name(code: int) -> str:
    lnames = ("<20 um", "20-50 um", ">=50 um")
    rnames = ("r<300", "r>=300")
    return f"{lnames[code // 2]} {rnames[code % 2]}"


@dataclass
class _Chain:
    n: int
    step: float
    group: str
    x: float                     # log(r / R_REF)
    burn: int
    g0: np.ndarray               # (2,)
    eps: np.ndarray              # (burn + n, 2)
    gp: np.ndarray               # (2, 4n) complex standard normals of the circulant embedding (one per component)
    white: np.ndarray            # (n, 2)
    z: float                     # amplitude draw


def _gauss_from_normals(w: np.ndarray, n: int, step: float, sd: float, length_nm: float) -> np.ndarray:
    """``radius._gauss_process`` with given complex normals (same embedding size 4n)."""
    m = len(w)
    j = np.arange(m)
    lag = np.minimum(j, m - j) * step
    cov = sd**2 * np.exp(-((lag / length_nm) ** 2))
    lam = np.clip(np.fft.fft(cov).real, 0.0, None)
    return np.fft.fft(np.sqrt(lam / m) * w).real[:n]


class FunctionalSimulator:
    """Common-random-number simulator of the three direction functionals for a fixed set of branch requests."""

    def __init__(self, requests: Sequence[Dict[str, Any]], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.chains: List[_Chain] = []
        for r in requests:
            n, step = int(r["n_nodes"]), float(r["step"])
            burn = min(int(math.ceil(5.0 * MAX_RESTORE_NM / step)), 5000)
            self.chains.append(_Chain(n, step, str(r["group"]), math.log(max(float(r["radius_nm"]), 1e-6) / R_REF_NM), burn,
                                      rng.standard_normal(2), rng.standard_normal((burn + n, 2)),
                                      rng.standard_normal((2, 4 * n)) + 1j * rng.standard_normal((2, 4 * n)),
                                      rng.standard_normal((n, 2)), float(rng.standard_normal())))
        self.strata = np.array([stratum_of((r["n_nodes"] - 1) * r["step"], r["radius_nm"]) for r in requests])

    def tangents(self, c: _Chain, theta: Sequence[float], amp: Amplitude) -> np.ndarray:
        from scipy.signal import lfilter

        sd_w, l_w, s, l_g, restore, sd_n = (float(v) for v in theta)
        m = math.exp(amp.mean_log(c.group, R_REF_NM * math.exp(c.x)) + amp.tau * c.z)
        a = math.exp(-c.step / restore)
        phi = math.exp(-c.step / l_g)
        s_b = s * m
        sd = s_b * math.sqrt(max(1.0 - phi**2, 0.0))
        # g_{j+1} = phi g_j + eps_j, g_0 ~ N(0, s_b^2);  tilt_{j+1} = a tilt_j + step g_j, tilt_0 = 0
        e = c.eps * sd
        e[0] = e[0] + phi * c.g0 * s_b              # g_1 = phi g_0 + eps_0
        g_rest = lfilter([1.0], [1.0, -phi], e, axis=0)  # g_1 .. g_T
        g = np.vstack([c.g0 * s_b, g_rest[:-1]])     # g_0 .. g_{T-1}
        tilt = lfilter([0.0, c.step], [1.0, -a], g, axis=0)  # tilt_0 = 0, tilt_{j+1} = a tilt_j + step g_j
        tilt = tilt[c.burn:]
        fast = np.column_stack([_gauss_from_normals(c.gp[k], c.n, c.step, sd_w * m, l_w) for k in range(2)])
        tilt = tilt + fast + c.white * (sd_n * m)
        t = np.column_stack([tilt, np.ones(c.n)])
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    def simulate(self, theta: Sequence[float], amp: Amplitude) -> np.ndarray:
        """[n_branches, 3] functionals."""
        return np.array([functionals_from_tangents(self.tangents(c, theta, amp), c.step) for c in self.chains])


def quantile_targets(values: np.ndarray, strata: np.ndarray, neurons: np.ndarray, *, n_boot: int = 200, min_branches: int = 30,
                     seed: int = 0) -> Dict[int, Dict[str, np.ndarray]]:
    """Per stratum (>= ``min_branches`` real branches): quantiles of each functional and their SE from a neuron bootstrap."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) < min_branches:
            continue
        v = values[idx]
        q = np.nanquantile(v, QUANTILES, axis=0)                        # [5, 3]
        units = neurons[idx]
        uniq = np.unique(units)
        by = [np.flatnonzero(units == u) for u in uniq]
        boots = []
        for _ in range(n_boot):
            pick = np.concatenate([by[i] for i in rng.integers(0, len(by), len(by))])
            boots.append(np.nanquantile(v[pick], QUANTILES, axis=0))
        se = np.nanstd(np.array(boots), axis=0)
        floor = 0.02 * np.nanstd(v, axis=0)                              # avoid zero SE on tiny strata
        out[int(s)] = {"q": q, "se": np.maximum(se, floor[None, :]), "n": len(idx), "n_neurons": len(uniq)}
    return out


def quantile_loss(sim: np.ndarray, strata: np.ndarray, targets: Dict[int, Dict[str, np.ndarray]]) -> float:
    total = 0.0
    for s, t in targets.items():
        v = sim[strata == s]
        if len(v) < 5:
            continue
        q = np.nanquantile(v, QUANTILES, axis=0)
        total += float(np.nansum(((q - t["q"]) / t["se"]) ** 2))
    return total


def comparison_table(real: np.ndarray, real_strata: np.ndarray, sim: np.ndarray, sim_strata: np.ndarray) -> List[Dict[str, Any]]:
    """Per stratum and functional: medians and p10 / p90 of real vs simulated branches (for the log / report)."""
    rows = []
    for s in np.unique(real_strata):
        a, b = real[real_strata == s], sim[sim_strata == s]
        if len(a) < 5 or len(b) < 5:
            continue
        for k, name in enumerate(FUNCTIONAL_NAMES):
            rows.append({"stratum": stratum_name(int(s)), "functional": name, "n_real": int(len(a)), "n_sim": int(len(b)),
                         "real_p10": float(np.nanquantile(a[:, k], 0.1)), "real_median": float(np.nanmedian(a[:, k])), "real_p90": float(np.nanquantile(a[:, k], 0.9)),
                         "sim_p10": float(np.nanquantile(b[:, k], 0.1)), "sim_median": float(np.nanmedian(b[:, k])), "sim_p90": float(np.nanquantile(b[:, k], 0.9))})
    return rows


def fit_to_functionals(model, branches: Sequence[Any], *, n_branches: int = 800, maxfev: int = 400, seed: int = 0,
                       log=None) -> Dict[str, Any]:
    """Refit ``model`` (a fitted ``DirectionMultiScale``) in place to the per-branch functional distributions of ``branches``
    (T0Branch-like: kappa1, kappa2, log_radius, step, group, neuron). Returns the fit info (loss before / after, tables)."""
    from scipy.optimize import minimize

    log = log or (lambda *_: None)
    use = [b for b in branches if b.n_nodes >= 8]
    rng = np.random.default_rng(seed)
    real = np.array([real_functionals(b.kappa1, b.kappa2, b.step) for b in use])
    radius = np.array([float(np.median(np.exp(b.log_radius))) for b in use])
    strata = np.array([stratum_of(b.length, r) for b, r in zip(use, radius)])
    neurons = np.array([str(getattr(b, "neuron", None) or i) for i, b in enumerate(use)])
    targets = quantile_targets(real, strata, neurons, seed=seed)
    # simulated requests: a stratified subset of the training branches (their step, nodes, group, real radius)
    pick = rng.choice(len(use), size=min(n_branches, len(use)), replace=False)
    requests = [{"n_nodes": use[i].n_nodes, "step": use[i].step, "group": use[i].group, "radius_nm": radius[i]} for i in pick]
    sim = FunctionalSimulator(requests, seed=seed + 1)
    amp0 = model.amplitude or Amplitude(0.0)
    groups = dict(amp0.groups)
    x0 = np.r_[np.log(np.maximum(model.theta, 1e-12)), amp0.alpha, amp0.beta, math.log(max(amp0.tau, 1e-3))]
    lo = np.r_[np.log([1e-4, 100.0, 1e-9, 300.0, 1000.0, 1e-5]), -6.0, -4.0, math.log(1e-3)]
    hi = np.r_[np.log([2.0, 5000.0, 1e-2, 5e4, MAX_RESTORE_NM, 1.0]), 6.0, 4.0, math.log(3.0)]

    def unpack(x):
        x = np.clip(x, lo, hi)
        return np.exp(x[:6]), Amplitude(float(x[6]), float(x[7]), float(math.exp(x[8])), groups)

    calls = {"n": 0}

    def loss(x):
        theta, amp = unpack(x)
        calls["n"] += 1
        value = quantile_loss(sim.simulate(theta, amp), sim.strata, targets)
        if calls["n"] % 25 == 0:
            log(f"[t0-direction] functional fit: evaluation {calls['n']}, loss {value:.1f}")
        return value

    start = loss(x0)
    sim_start = sim.simulate(*unpack(x0))
    log(f"[t0-direction] functional fit: {len(use)} branches ({len(targets)} strata), {len(requests)} simulated; start loss {start:.1f}")
    res = minimize(loss, x0, method="Nelder-Mead", options={"maxfev": int(maxfev), "xatol": 1e-3, "fatol": 1e-2, "adaptive": True})
    theta, amp = unpack(res.x)
    model.sd_fast, model.length_fast_nm, model.drive_std, model.drive_length_nm, model.restore_nm, model.noise_sd = (float(v) for v in theta)
    model.amplitude = amp
    sim_end = sim.simulate(theta, amp)
    info = {"target": "functional_quantiles", "n_branches": len(use), "n_simulated": len(requests), "strata": {stratum_name(s): {"n": t["n"], "n_neurons": t["n_neurons"]} for s, t in targets.items()},
            "loss_start": float(start), "loss_end": float(res.fun), "n_evaluations": int(calls["n"]), "converged": bool(res.success),
            "table_start": comparison_table(real, strata, sim_start, sim.strata), "table_end": comparison_table(real, strata, sim_end, sim.strata)}
    log(f"[t0-direction] functional fit: loss {start:.1f} -> {res.fun:.1f} after {calls['n']} evaluations")
    return info
