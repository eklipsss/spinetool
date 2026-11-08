"""Diagnostics of the spine placement (p-module-technical-spec.md, section 8).

Longitudinal: Ripley's K on the segment [0, L] with the translation edge correction (K(h) = 2h for a homogeneous
Poisson process), pooled over branches weighted by n(n-1); nearest-neighbour interval distribution G; both with
pointwise envelopes from simulations of the FITTED P0-position at the observed N of every branch (so an
inhomogeneous density is not taken for clustering). Angular: Rayleigh, Kuiper and Watson U^2 tests of uniformity and
the Fourier coefficients |c_m|. Units: nm.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np


def k_function(positions: Sequence[np.ndarray], lengths: Sequence[float], h: np.ndarray) -> np.ndarray:
    """Pooled K(h): sum_b sum_{i!=j} 1(|d|<=h) L/(L-|d|) * L_b / (n_b (n_b-1)), averaged with weights n_b (n_b-1)."""
    h = np.asarray(h, dtype=float)
    num = np.zeros(len(h))
    weight = 0.0
    for s, L in zip(positions, lengths):
        s = np.asarray(s, dtype=float)
        n = len(s)
        if n < 2:
            continue
        d = np.abs(s[:, None] - s[None, :])[~np.eye(n, dtype=bool)]
        w = L / np.maximum(L - d, 1e-9)
        d_sorted_idx = np.argsort(d)
        cum = np.cumsum(w[d_sorted_idx])
        cnt = np.searchsorted(d[d_sorted_idx], h, side="right")
        k_b = np.where(cnt > 0, cum[np.maximum(cnt - 1, 0)], 0.0) * L / (n * (n - 1))
        num += n * (n - 1) * k_b
        weight += n * (n - 1)
    return num / weight if weight else np.full(len(h), np.nan)


def nn_intervals(positions: Sequence[np.ndarray]) -> np.ndarray:
    out = []
    for s in positions:
        s = np.sort(np.asarray(s, dtype=float))
        if len(s) < 2:
            continue
        gaps = np.diff(s)
        out.append(np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf]))
    return np.concatenate(out) if out else np.zeros(0)


def envelopes(position_model: Any, branches: Sequence[Any], h: np.ndarray, *, n_sim: int = 99, seed: int = 0) -> Dict[str, Any]:
    """Observed K and G with pointwise 2.5-97.5 % envelopes under P0-position at the observed N per branch."""
    rng = np.random.default_rng(seed)
    obs = [b.positions for b in branches if b.positions is not None and len(b.positions) >= 2]
    used = [b for b in branches if b.positions is not None and len(b.positions) >= 2]
    lengths = [b.length for b in used]
    k_obs = k_function(obs, lengths, h)
    g_obs = _ecdf(nn_intervals(obs), h)
    k_sim, g_sim = [], []
    for _ in range(n_sim):
        sim = [position_model.sample_positions(b, len(b.positions), rng) for b in used]
        k_sim.append(k_function(sim, lengths, h))
        g_sim.append(_ecdf(nn_intervals(sim), h))
    k_sim, g_sim = np.array(k_sim), np.array(g_sim)
    lo_k, hi_k = np.percentile(k_sim, [2.5, 97.5], axis=0)
    lo_g, hi_g = np.percentile(g_sim, [2.5, 97.5], axis=0)
    return {
        "h_nm": h.tolist(), "K_obs": k_obs.tolist(), "K_lo": lo_k.tolist(), "K_hi": hi_k.tolist(), "K_poisson": (2 * h).tolist(),
        "G_obs": g_obs.tolist(), "G_lo": lo_g.tolist(), "G_hi": hi_g.tolist(),
        "K_outside_fraction": float(np.mean((k_obs < lo_k) | (k_obs > hi_k))),
        "G_outside_fraction": float(np.mean((g_obs < lo_g) | (g_obs > hi_g))),
        # deficit of short intervals = repulsion, excess = clustering (relative to P0 at the same density)
        "short_interval_deficit": bool(np.any(g_obs[: max(1, len(h) // 5)] < lo_g[: max(1, len(h) // 5)])),
        "n_branches": len(used), "n_sim": n_sim,
    }


def _ecdf(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    x = np.sort(x[np.isfinite(x)])
    return np.searchsorted(x, h, side="right") / max(len(x), 1)


# ---- circular uniformity ---------------------------------------------------------------------------------------


def rayleigh(theta: np.ndarray) -> Dict[str, float]:
    n = len(theta)
    r = abs(np.exp(1j * theta).mean())
    z = n * r ** 2
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * r) ** 2)) - (1 + 2 * n))  # Zar (2010), eq. 27.4
    return {"R": float(r), "z": float(z), "p": float(min(max(p, 0.0), 1.0))}


def kuiper(theta: np.ndarray) -> Dict[str, float]:
    u = np.sort(np.mod(theta, 2 * np.pi) / (2 * np.pi))
    n = len(u)
    i = np.arange(1, n + 1)
    v = float(np.max(i / n - u) + np.max(u - (i - 1) / n))
    lam = (math.sqrt(n) + 0.155 + 0.24 / math.sqrt(n)) * v
    p = sum(2 * (4 * j * j * lam * lam - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return {"V": v, "p": float(min(max(p, 0.0), 1.0))}


def watson_u2(theta: np.ndarray) -> Dict[str, float]:
    u = np.sort(np.mod(theta, 2 * np.pi) / (2 * np.pi))
    n = len(u)
    i = np.arange(1, n + 1)
    u2 = float(np.sum((u - (2 * i - 1) / (2 * n)) ** 2) - n * (u.mean() - 0.5) ** 2 + 1 / (12 * n))
    u2s = (u2 - 0.1 / n + 0.1 / n ** 2) * (1 + 0.8 / n)
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * math.pi ** 2 * u2s) for j in range(1, 101))
    return {"U2": u2, "p": float(min(max(p, 0.0), 1.0))}


def circular_summary(theta: np.ndarray, m_max: int = 4) -> Dict[str, Any]:
    theta = np.asarray(theta, dtype=float)
    theta = theta[np.isfinite(theta)]
    if len(theta) < 3:
        return {"n": int(len(theta))}
    c = {f"abs_c{m}": float(abs(np.exp(1j * m * theta).mean())) for m in range(1, m_max + 1)}
    return {"n": int(len(theta)), "rayleigh": rayleigh(theta), "kuiper": kuiper(theta), "watson_u2": watson_u2(theta), **c,
            "fourier_noise_level": float(1.0 / math.sqrt(len(theta)))}


def nearest_angle_differences(spines_theta: Sequence[np.ndarray], spines_s: Sequence[np.ndarray]) -> np.ndarray:
    """Angle difference between longitudinal neighbours (independent of the zero direction of theta)."""
    out = []
    for th, s in zip(spines_theta, spines_s):
        if len(s) < 2:
            continue
        o = np.argsort(s)
        d = np.diff(np.asarray(th)[o])
        out.append(np.angle(np.exp(1j * d)))
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------------------------------------------------------
# P0 / P1 comparison by simulation at the observed N of every branch (fit_p1.py)
# ---------------------------------------------------------------------------------------------------------------------

PCF_BINS_NM = ((0.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0))


def configuration_statistics(configs: Sequence[Any], branches: Sequence[Any]) -> Dict[str, float]:
    """Pooled summary of one set of configurations [(s, theta) per branch] on the given branches (same order):
    g(r) along the branch in PCF_BINS_NM (observed pairs / pairs of independent uniform points at the same N and L),
    median distance to the nearest neighbour along the branch, CV of the intervals (each normalised by its branch mean),
    median and 10 % quantile of the nearest-neighbour SURFACE distance sqrt(ds^2 + (r dtheta)^2)."""
    from ..branch_statistics import expected_pairs
    from .models.interaction import surface_distances

    lengths, counts, pair_d, gaps, nn2d, along = [], [], [], [], [], []
    for (s, th), b in zip(configs, branches):
        s, th = np.asarray(s, float), np.asarray(th, float)
        n = len(s)
        lengths.append(b.length)
        counts.append(n)
        if n < 2:
            continue
        order = np.argsort(s)
        s, th = s[order], th[order]
        d = np.abs(s[:, None] - s[None, :])[~np.eye(n, dtype=bool)]
        pair_d.append(d)
        g = np.diff(s)
        if g.mean() > 0:
            gaps.append(g / g.mean())
        along.append(np.minimum(np.r_[np.inf, g], np.r_[g, np.inf]))
        r = np.interp(s, b.s, b.radius)
        dd = surface_distances(s, th, r, s, th, r)
        np.fill_diagonal(dd, np.inf)
        nn2d.append(dd.min(axis=1))
    pair_d = np.concatenate(pair_d) if pair_d else np.zeros(0)
    out = {}
    for a, c in PCF_BINS_NM:
        exp = expected_pairs(np.asarray(lengths), np.asarray(counts), a, c)
        out[f"pcf_{int(a)}_{int(c)}"] = float(((pair_d >= a) & (pair_d < c)).sum() / exp) if exp > 0 else float("nan")
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    along = np.concatenate(along) if along else np.zeros(0)
    nn2d = np.concatenate(nn2d) if nn2d else np.zeros(0)
    out["nn_along_median_nm"] = float(np.median(along)) if len(along) else float("nan")
    out["gap_cv"] = float(gaps.std() / gaps.mean()) if len(gaps) > 1 else float("nan")
    out["surface_nn_median_nm"] = float(np.median(nn2d)) if len(nn2d) else float("nan")
    out["surface_nn_q10_nm"] = float(np.quantile(nn2d, 0.10)) if len(nn2d) else float("nan")
    return out


def simulation_comparison(models: Dict[str, Any], branches: Sequence[Any], *, n_sim: int = 19, seed: int = 0) -> Dict[str, Any]:
    """Observed statistics of ``branches`` (positions AND angles) against ``n_sim`` simulations of every model at the
    observed N of every branch (Monte Carlo goodness of fit). Per model and statistic: mean and sd of the simulations,
    the 2.5-97.5 % envelope, z = (observed - mean) / sd; ``score`` = mean |z| over the statistics."""
    from .models.position import sample_angles

    used = [b for b in branches if b.positions is not None and b.angles is not None and len(b.positions) >= 1]
    observed = configuration_statistics([(b.positions, b.angles) for b in used], used)
    result = {"observed": observed, "n_branches": len(used), "n_points": int(sum(len(b.positions) for b in used)), "n_sim": n_sim, "models": {}}
    for name, model in models.items():
        rng = np.random.default_rng(seed)
        sims = []
        for _ in range(n_sim):
            configs = []
            for b in used:
                n = len(b.positions)
                if hasattr(model, "sample_configuration"):
                    configs.append(model.sample_configuration(b, n, rng))
                else:
                    configs.append((model.sample_positions(b, n, rng), sample_angles(n, rng)))
            sims.append(configuration_statistics(configs, used))
        table = {}
        for key, obs in observed.items():
            v = np.array([s[key] for s in sims], float)
            mean, sd = float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if n_sim > 1 else float("nan")
            lo, hi = (float(x) for x in np.nanpercentile(v, [2.5, 97.5]))
            table[key] = {"observed": obs, "sim_mean": mean, "sim_sd": sd, "sim_lo": lo, "sim_hi": hi,
                          "z": (obs - mean) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan"), "outside": bool(obs < lo or obs > hi)}
        zs = [abs(t["z"]) for t in table.values() if np.isfinite(t["z"])]
        result["models"][name] = {"statistics": table, "score_mean_abs_z": float(np.mean(zs)) if zs else float("nan"),
                                  "n_outside": int(sum(t["outside"] for t in table.values()))}
    return result
