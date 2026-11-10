"""P1: spine positions with interaction between neighbours (p-module-technical-spec.md, section 7).

Both models work at a GIVEN N (from P0-count) and keep the inhomogeneous longitudinal density of P0-position
(``base``, a fitted ``PositionModel``): only the dependence between points is added.

P1a  renewal process along the branch. Time change tau = N F(s), F the P0 cumulative density on [0, L], so the process is
     (approximately) of unit rate on [0, N]. Inter-event intervals ~ Weibull(k) with mean 1 (scale 1 / Gamma(1 + 1/k)):
     k = 1 is P0 (Poisson), k > 1 fewer short intervals (regularity), k < 1 clusters. Likelihood of one branch with
     censored ends (stationary renewal): first point S(tau_1) (equilibrium density S(x) / mean, mean 1), interior
     intervals f(x), after the last point S(N - tau_N). Simulation at given N: N + 1 i.i.d. Weibull intervals normalised to
     sum to 1, v_i = cumulative share, s_i = F^-1(v_i) (exactly the conditional Poisson process at k = 1). One-dimensional.

P1b  Strauss process on the trunk surface W = [0, L] x (-pi, pi] with the metric d = sqrt(ds^2 + (r_mean dtheta)^2)
     (section 6): f(P) ~ prod lambda(p_i) gamma^{n_R(P)}, 0 <= gamma <= 1. Fitted by Besag's pseudo-likelihood (Baddeley &
     Turner 2000) with the conditional intensity lambda_c(p | P) = e^alpha N_b f_s(s) / (2 pi) gamma^{t(p, P)}: f_s the P0
     density (offset), alpha a global level, t the number of points within R; the integral on the node grid x n_angles
     angles. For fixed (R, log gamma) alpha is profiled out in closed form, log gamma <= 0 by a bounded 1-D search, R by the
     profile pseudo-likelihood over a grid. Simulation at given N: Metropolis-Hastings moving one point, proposal from
     f_s x uniform angle (the trend cancels), acceptance min(1, gamma^{t_new - t_old}), burn-in ``burn_factor`` * N moves
     from a P0 configuration.

P1b2 multi-Strauss (``MultiStraussModel``): the same with a piecewise-constant pair potential, K shells with their own
     gamma_k (default K = 2); K = 1 is P1b.

Units: nm, radians.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .position import PositionBranch, PositionModel, sample_angles

LOG_GAMMA_MIN = -20.0  # gamma = e^-20 ~ hard core


# ---------------------------------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------------------------------


def _cdf(base: PositionModel, branch: PositionBranch) -> Tuple[np.ndarray, np.ndarray]:
    """F on the nodes (strictly increasing part) and the node positions, for interpolation in both directions."""
    cdf = base.cdf_on_nodes(branch)
    keep = np.r_[True, np.diff(cdf) > 0]
    return cdf[keep], branch.s[keep]


def _density_on_nodes(base: PositionModel, branch: PositionBranch) -> np.ndarray:
    """f_s(s_j) = lambda~(s_j) / int lambda~ on the nodes (integrates to 1 on [0, L] by the trapezoid rule)."""
    eta = base.log_intensity(branch)
    lam = np.exp(eta - eta.max())
    return lam / np.trapezoid(lam, branch.s)


def surface_distances(s1, th1, r1, s2, th2, r2) -> np.ndarray:
    """Pairwise surface distances [len(s1), len(s2)] with the cylinder metric of section 6."""
    ds = np.asarray(s1)[:, None] - np.asarray(s2)[None, :]
    dth = np.abs(np.asarray(th1)[:, None] - np.asarray(th2)[None, :])
    dth = np.minimum(dth, 2.0 * np.pi - dth)
    rbar = 0.5 * (np.asarray(r1)[:, None] + np.asarray(r2)[None, :])
    return np.sqrt(ds ** 2 + (rbar * dth) ** 2)


def _base_configuration(base: PositionModel, branch: PositionBranch, n: int, rng: np.random.Generator):
    return base.sample_positions(branch, n, rng), sample_angles(n, rng)


# ---------------------------------------------------------------------------------------------------------------------
# P1a: renewal process
# ---------------------------------------------------------------------------------------------------------------------


def _weibull_terms(k: float):
    lam = 1.0 / math.gamma(1.0 + 1.0 / k)

    def log_f(x):
        z = np.maximum(x, 1e-300) / lam
        return math.log(k) - math.log(lam) + (k - 1.0) * np.log(z) - z ** k

    def log_s(x):
        return -(np.maximum(x, 0.0) / lam) ** k

    return log_f, log_s, lam


class RenewalModel:
    kind = "renewal"

    def __init__(self, base: PositionModel, k: float = 1.0) -> None:
        self.base, self.k = base, float(k)
        self.fit_info: Dict[str, Any] = {}

    def transformed(self, branch: PositionBranch) -> np.ndarray:
        """tau_i = N F(s_i) of the observed positions (N = number of positioned spines)."""
        cdf, s = _cdf(self.base, branch)
        pos = np.asarray(branch.positions, dtype=float)
        return len(pos) * np.interp(pos, s, cdf)

    def _branch_terms(self, branches: Sequence[PositionBranch]):
        first, gaps, last = [], [], []
        for b in branches:
            if b.positions is None or len(b.positions) == 0:
                continue
            tau = np.sort(self.transformed(b))
            n = len(tau)
            first.append(tau[0])
            last.append(n - tau[-1])
            if n > 1:
                gaps.append(np.diff(tau))
        return np.asarray(first), (np.concatenate(gaps) if gaps else np.zeros(0)), np.asarray(last)

    def loglik(self, branches: Sequence[PositionBranch], k: Optional[float] = None, *, terms=None) -> float:
        first, gaps, last = terms if terms is not None else self._branch_terms(branches)
        log_f, log_s, _ = _weibull_terms(self.k if k is None else float(k))
        return float(log_s(first).sum() + log_f(gaps).sum() + log_s(last).sum())

    def fit(self, branches: Sequence[PositionBranch], k_bounds: Tuple[float, float] = (0.2, 10.0)) -> "RenewalModel":
        from scipy.optimize import minimize_scalar

        terms = self._branch_terms(branches)
        res = minimize_scalar(lambda lk: -self.loglik(branches, math.exp(lk), terms=terms), bounds=tuple(map(math.log, k_bounds)), method="bounded",
                              options={"xatol": 1e-4})
        self.k = float(math.exp(res.x))
        ll_k, ll_1 = self.loglik(branches, terms=terms), self.loglik(branches, 1.0, terms=terms)
        self.fit_info = {"k": self.k, "loglik": ll_k, "loglik_poisson": ll_1, "lr_statistic": 2.0 * (ll_k - ll_1),
                         "n_branches": int(len(terms[0])), "n_interior_gaps": int(len(terms[1])),
                         "gap_cv_transformed": float(np.std(terms[1]) / np.mean(terms[1])) if len(terms[1]) > 1 else float("nan")}
        return self

    def sample_positions(self, branch: PositionBranch, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.zeros(0)
        _, _, lam = _weibull_terms(self.k)
        g = lam * rng.weibull(self.k, size=n + 1)
        v = np.cumsum(g)[:-1] / g.sum()
        cdf, s = _cdf(self.base, branch)
        return np.sort(np.interp(v, cdf, s))

    def sample_configuration(self, branch: PositionBranch, n: int, rng: np.random.Generator):
        return self.sample_positions(branch, n, rng), sample_angles(n, rng)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "k": self.k, "fit_info": self.fit_info}


# ---------------------------------------------------------------------------------------------------------------------
# P1b: Strauss process on the surface
# ---------------------------------------------------------------------------------------------------------------------


class StraussModel:
    kind = "strauss"

    def __init__(self, base: PositionModel, radius_nm: float = 0.0, log_gamma: float = 0.0, alpha: float = 0.0, burn_factor: int = 50) -> None:
        self.base, self.radius_nm, self.log_gamma, self.alpha, self.burn_factor = base, float(radius_nm), float(log_gamma), float(alpha), int(burn_factor)
        self.fit_info: Dict[str, Any] = {}

    @property
    def gamma(self) -> float:
        return math.exp(self.log_gamma)

    # ---- pseudo-likelihood ----------------------------------------------------------------------------------------------
    def _branch_counts(self, branch: PositionBranch, radii: np.ndarray, n_angles: int):
        """Per R: sum of t_i over the data points, and A[T] = sum of w * N f_s / (2 pi) over grid points with T neighbours.
        Returns (offset sum over data, list of t-sums, list of A arrays)."""
        s, th = np.asarray(branch.positions, float), np.asarray(branch.angles, float)
        n = len(s)
        r_pts = np.interp(s, branch.s, branch.radius)
        f_nodes = _density_on_nodes(self.base, branch)
        offset_pts = np.log(n * np.maximum(np.interp(s, branch.s, f_nodes), 1e-300) / (2.0 * np.pi))
        d_pp = surface_distances(s, th, r_pts, s, th, r_pts)
        np.fill_diagonal(d_pp, np.inf)
        ds = np.diff(branch.s)
        w_s = np.zeros(len(branch.s))
        w_s[:-1] += ds / 2.0
        w_s[1:] += ds / 2.0
        phi = -np.pi + (np.arange(n_angles) + 0.5) * (2.0 * np.pi / n_angles)
        gs = np.repeat(branch.s, n_angles)
        gth = np.tile(phi, len(branch.s))
        gr = np.repeat(branch.radius, n_angles)
        gw = np.repeat(w_s * n * f_nodes / (2.0 * np.pi), n_angles) * (2.0 * np.pi / n_angles)
        d_gp = surface_distances(gs, gth, gr, s, th, r_pts)
        t_sums, a_arrays = [], []
        for R in radii:
            t_sums.append(float((d_pp < R).sum()))  # = 2 n_R(P) = sum_i t_i
            T = (d_gp < R).sum(axis=1)
            a_arrays.append(np.bincount(T, weights=gw))
        return float(offset_pts.sum()), n, t_sums, a_arrays

    @staticmethod
    def _profile(n_total: int, offset_total: float, t_sum: float, a: np.ndarray):
        """max over log gamma in [LOG_GAMMA_MIN, 0] of the pseudo-likelihood with alpha profiled out."""
        from scipy.optimize import minimize_scalar

        T = np.arange(len(a))

        def neg(lg):
            integral = float((a * np.exp(lg * T)).sum())
            alpha = math.log(n_total / integral)
            return -(n_total * alpha + offset_total + lg * t_sum - n_total)

        res = minimize_scalar(neg, bounds=(LOG_GAMMA_MIN, 0.0), method="bounded", options={"xatol": 1e-5})
        lg = float(res.x)
        alpha = math.log(n_total / float((a * np.exp(lg * T)).sum()))
        return -float(res.fun), lg, alpha, -neg(0.0)

    def fit(self, branches: Sequence[PositionBranch], *, radii: Optional[Sequence[float]] = None, d_min_nm: float = 50.0, n_radii: int = 15,
            n_angles: int = 32, max_branches: int = 3000, seed: int = 0) -> "StraussModel":
        use = [b for b in branches if b.positions is not None and b.angles is not None and len(b.positions) >= 1]
        if not use:
            raise ValueError("P1b needs branches with observed positions AND angles")
        rng = np.random.default_rng(seed)
        if len(use) > max_branches:
            use = [use[i] for i in rng.choice(len(use), max_branches, replace=False)]
        if radii is None:
            nn = []
            for b in use:
                if len(b.positions) < 2:
                    continue
                r = np.interp(b.positions, b.s, b.radius)
                d = surface_distances(b.positions, b.angles, r, b.positions, b.angles, r)
                np.fill_diagonal(d, np.inf)
                nn.append(d.min(axis=1))
            r_max = float(np.percentile(np.concatenate(nn), 90)) if nn else 2000.0
            radii = np.linspace(max(d_min_nm, 1.0), max(r_max, d_min_nm + 1.0), n_radii)
        radii = np.asarray(radii, dtype=float)
        n_total, off_total = 0, 0.0
        t_tot = np.zeros(len(radii))
        a_tot = [np.zeros(1) for _ in radii]
        for b in use:
            off, n, t_sums, a_arrays = self._branch_counts(b, radii, n_angles)
            n_total += n
            off_total += off
            t_tot += np.asarray(t_sums)
            for j, a in enumerate(a_arrays):
                if len(a) > len(a_tot[j]):
                    a_tot[j] = np.r_[a_tot[j], np.zeros(len(a) - len(a_tot[j]))]
                a_tot[j][: len(a)] += a
        profile = []
        for j, R in enumerate(radii):
            pl, lg, alpha, pl_poisson = self._profile(n_total, off_total, float(t_tot[j]), a_tot[j])
            profile.append({"radius_nm": float(R), "log_pl": pl, "log_gamma": lg, "gamma": math.exp(lg), "alpha": alpha,
                            "log_pl_gain_over_poisson": pl - pl_poisson, "mean_neighbours_within_R": float(t_tot[j] / n_total)})
        best = max(profile, key=lambda r: r["log_pl"])
        self.radius_nm, self.log_gamma, self.alpha = best["radius_nm"], best["log_gamma"], best["alpha"]
        self.fit_info = {"radius_nm": self.radius_nm, "gamma": self.gamma, "log_gamma": self.log_gamma, "alpha": self.alpha,
                         "n_branches": len(use), "n_points": int(n_total), "n_angles": n_angles, "profile": profile,
                         "radius_at_grid_edge": bool(best["radius_nm"] in (float(radii[0]), float(radii[-1])))}
        return self

    # ---- simulation ------------------------------------------------------------------------------------------------------
    def sample_configuration(self, branch: PositionBranch, n: int, rng: np.random.Generator):
        s, th = _base_configuration(self.base, branch, n, rng)
        if n < 2 or self.radius_nm <= 0.0 or self.log_gamma >= 0.0:
            return s, th
        cdf, nodes = _cdf(self.base, branch)
        r = np.interp(s, branch.s, branch.radius)
        R, lg = self.radius_nm, self.log_gamma
        for _ in range(self.burn_factor * n):
            i = int(rng.integers(n))
            s_new = float(np.interp(rng.random(), cdf, nodes))
            th_new = float(np.pi - rng.random() * 2.0 * np.pi)
            r_new = float(np.interp(s_new, branch.s, branch.radius))
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            d_old = surface_distances(s[i:i + 1], th[i:i + 1], r[i:i + 1], s[mask], th[mask], r[mask])[0]
            d_new = surface_distances(np.array([s_new]), np.array([th_new]), np.array([r_new]), s[mask], th[mask], r[mask])[0]
            delta = int((d_new < R).sum()) - int((d_old < R).sum())
            if delta <= 0 or math.log(max(rng.random(), 1e-300)) < lg * delta:
                s[i], th[i], r[i] = s_new, th_new, r_new
        order = np.argsort(s, kind="stable")
        return s[order], th[order]

    def sample_positions(self, branch: PositionBranch, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sample_configuration(branch, n, rng)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "radius_nm": self.radius_nm, "log_gamma": self.log_gamma, "alpha": self.alpha, "burn_factor": self.burn_factor,
                "fit_info": self.fit_info}


# ---------------------------------------------------------------------------------------------------------------------
# P1b2: Strauss process with a piecewise-constant pair potential (several interaction radii)
# ---------------------------------------------------------------------------------------------------------------------


class MultiStraussModel:
    """Multi-Strauss process (piecewise-constant pair potential; Penttinen 1984, Baddeley & Turner 2000) on the trunk surface.

    Shells k = 1..K: R_{k-1} <= d < R_k (R_0 = 0); f(P) ~ prod lambda(p_i) prod_k gamma_k^{n_k(P)}, n_k the number of pairs in
    shell k, 0 <= gamma_k <= 1 (log gamma_k in [LOG_GAMMA_MIN, 0]). K = 1 is ``StraussModel``; gamma_k = 1 switches a shell
    off, gamma_1 = gamma_2 gives a single Strauss with radius R_2. Why: with one radius P1b still places nearest
    neighbours 65 nm closer than real ones (2026-11-09): a strong short-range core and a weaker longer-range repulsion
    cannot both be described by one (R, gamma).

    Fit: Besag's pseudo-likelihood as for P1b. For fixed radii, alpha is profiled out in closed form and the remaining
    function of the log gammas, n log sum_g w_g exp(T_g . lg) - lg . t_data, is convex (log-sum-exp minus linear) and is
    minimised by L-BFGS-B in the box [LOG_GAMMA_MIN, 0]^K; the radii by the profile over all increasing K-subsets of a grid.
    The grid integral is accumulated as weights over the distinct shell-count vectors T_g, so the profile costs one pass of
    distances per branch. Simulation: the same Metropolis-Hastings as P1b, acceptance min(1, exp(sum_k lg_k delta_k)).
    """

    kind = "multi_strauss"

    def __init__(self, base: PositionModel, radii_nm: Sequence[float] = (), log_gammas: Sequence[float] = (), alpha: float = 0.0,
                 burn_factor: int = 50) -> None:
        self.base, self.alpha, self.burn_factor = base, float(alpha), int(burn_factor)
        self.radii_nm = [float(r) for r in radii_nm]
        self.log_gammas = [float(g) for g in log_gammas]
        if len(self.radii_nm) != len(self.log_gammas):
            raise ValueError("radii_nm and log_gammas must have the same length")
        if any(b <= a for a, b in zip(self.radii_nm, self.radii_nm[1:])):
            raise ValueError("radii_nm must be strictly increasing")
        self.fit_info: Dict[str, Any] = {}

    @property
    def gammas(self) -> list:
        return [math.exp(g) for g in self.log_gammas]

    # ---- shell counts --------------------------------------------------------------------------------------------------
    @staticmethod
    def _cumulative_counts(d: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """[rows, len(radii)]: number of points closer than each radius (d of one row against all points)."""
        d_sorted = np.sort(d, axis=1)
        return np.stack([(d_sorted < R).sum(axis=1) for R in radii], axis=1) if d.shape[1] else np.zeros((d.shape[0], len(radii)), int)

    @staticmethod
    def _shells(cum: np.ndarray, idx: Tuple[int, ...]) -> np.ndarray:
        """Shell counts T [rows, K] from cumulative counts for the radii with grid indices ``idx`` (increasing)."""
        c = cum[:, list(idx)]
        return np.diff(np.c_[np.zeros(len(c), dtype=c.dtype), c], axis=1)

    def _branch_terms(self, branch: PositionBranch, radii: np.ndarray, n_angles: int):
        s, th = np.asarray(branch.positions, float), np.asarray(branch.angles, float)
        n = len(s)
        r_pts = np.interp(s, branch.s, branch.radius)
        f_nodes = _density_on_nodes(self.base, branch)
        offset = float(np.log(n * np.maximum(np.interp(s, branch.s, f_nodes), 1e-300) / (2.0 * np.pi)).sum())
        d_pp = surface_distances(s, th, r_pts, s, th, r_pts)
        np.fill_diagonal(d_pp, np.inf)
        ds = np.diff(branch.s)
        w_s = np.zeros(len(branch.s))
        w_s[:-1] += ds / 2.0
        w_s[1:] += ds / 2.0
        phi = -np.pi + (np.arange(n_angles) + 0.5) * (2.0 * np.pi / n_angles)
        gs, gth = np.repeat(branch.s, n_angles), np.tile(phi, len(branch.s))
        gr = np.repeat(branch.radius, n_angles)
        gw = np.repeat(w_s * n * f_nodes / (2.0 * np.pi), n_angles) * (2.0 * np.pi / n_angles)
        d_gp = surface_distances(gs, gth, gr, s, th, r_pts)
        return n, offset, self._cumulative_counts(d_pp, radii).sum(axis=0), self._cumulative_counts(d_gp, radii), gw

    @staticmethod
    def _optimise(n_total: int, offset_total: float, t_data: np.ndarray, T: np.ndarray, w: np.ndarray):
        """max log PL over lg in [LOG_GAMMA_MIN, 0]^K, alpha profiled out. Returns (log PL, lg, alpha, log PL at lg = 0)."""
        from scipy.optimize import minimize

        T = T.astype(float)
        const = n_total * math.log(n_total) + offset_total - n_total

        def neg(lg):
            e = w * np.exp(T @ lg)
            integral = float(e.sum())
            val = n_total * math.log(integral) - float(lg @ t_data) - const
            grad = n_total * (T.T @ e) / integral - t_data
            return val, grad

        K = T.shape[1]
        x0 = np.where(t_data > 0, -0.5, LOG_GAMMA_MIN / 2.0)
        res = minimize(neg, x0, jac=True, method="L-BFGS-B", bounds=[(LOG_GAMMA_MIN, 0.0)] * K, options={"ftol": 1e-12, "gtol": 1e-8})
        lg = np.asarray(res.x, dtype=float)
        alpha = math.log(n_total / float((w * np.exp(T @ lg)).sum()))
        return -float(neg(lg)[0]), lg, alpha, -float(neg(np.zeros(K))[0])

    def fit(self, branches: Sequence[PositionBranch], *, radii: Optional[Sequence[float]] = None, n_shells: int = 2, d_min_nm: float = 50.0,
            n_radii: int = 12, n_angles: int = 32, max_branches: int = 3000, seed: int = 0) -> "MultiStraussModel":
        from itertools import combinations

        use = [b for b in branches if b.positions is not None and b.angles is not None and len(b.positions) >= 1]
        if not use:
            raise ValueError("P1b2 needs branches with observed positions AND angles")
        rng = np.random.default_rng(seed)
        if len(use) > max_branches:
            use = [use[i] for i in rng.choice(len(use), max_branches, replace=False)]
        if radii is None:
            nn = []
            for b in use:
                if len(b.positions) < 2:
                    continue
                r = np.interp(b.positions, b.s, b.radius)
                d = surface_distances(b.positions, b.angles, r, b.positions, b.angles, r)
                np.fill_diagonal(d, np.inf)
                nn.append(d.min(axis=1))
            # longer range than P1b: the outer shell may extend past the typical nearest neighbour
            r_max = float(np.percentile(np.concatenate(nn), 97)) if nn else 3000.0
            radii = np.linspace(max(d_min_nm, 1.0), max(r_max, d_min_nm + 1.0), n_radii)
        radii = np.asarray(sorted(float(r) for r in radii), dtype=float)
        if len(radii) < n_shells:
            raise ValueError(f"need at least {n_shells} radii in the grid")
        combos = [c for k in range(1, n_shells + 1) for c in combinations(range(len(radii)), k)]
        n_total, off_total = 0, 0.0
        t_cum = np.zeros(len(radii))
        acc: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {c: {} for c in combos}
        for b in use:
            n, off, t_c, g_cum, gw = self._branch_terms(b, radii, n_angles)
            n_total += n
            off_total += off
            t_cum += t_c
            for c in combos:
                T = self._shells(g_cum, c)
                code = T @ ((n + 1) ** np.arange(len(c)))   # counts <= n: a unique integer per count vector (1-D unique is fast)
                uniq, first, inv = np.unique(code, return_index=True, return_inverse=True)
                wsum = np.bincount(inv.ravel(), weights=gw, minlength=len(uniq))
                table = acc[c]
                for row, wv in zip(map(tuple, T[first].tolist()), wsum):
                    table[row] = table.get(row, 0.0) + float(wv)
        profile = []
        for c in combos:
            t_data = np.diff(np.r_[0.0, t_cum[list(c)]])
            rows = np.array(list(acc[c].keys()), dtype=float).reshape(-1, len(c))
            w = np.array(list(acc[c].values()), dtype=float)
            pl, lg, alpha, pl0 = self._optimise(n_total, off_total, t_data, rows, w)
            profile.append({"radii_nm": [float(radii[i]) for i in c], "log_pl": pl, "log_gammas": lg.tolist(),
                            "gammas": np.exp(lg).tolist(), "alpha": alpha, "log_pl_gain_over_poisson": pl - pl0,
                            "mean_pairs_in_shell": (t_data / n_total).tolist()})
        best_k = {k: max((p for p in profile if len(p["radii_nm"]) == k), key=lambda p: p["log_pl"]) for k in range(1, n_shells + 1)}
        best = best_k[n_shells]
        self.radii_nm, self.log_gammas, self.alpha = list(best["radii_nm"]), list(best["log_gammas"]), best["alpha"]
        edge = {float(radii[0]), float(radii[-1])}
        self.fit_info = {"radii_nm": self.radii_nm, "gammas": self.gammas, "log_gammas": self.log_gammas, "alpha": self.alpha,
                         "n_shells": n_shells, "n_branches": len(use), "n_points": int(n_total), "n_angles": n_angles,
                         "radius_grid_nm": radii.tolist(),
                         "best_by_n_shells": {str(k): v for k, v in best_k.items()},
                         # nested models: the gain is >= 0 by construction; a pseudo-likelihood is not a likelihood, so no
                         # chi-square test - the choice is made by the Monte Carlo check (fit_p1.py)
                         "log_pl_gain_over_one_radius": best["log_pl"] - best_k[1]["log_pl"],
                         "radius_at_grid_edge": bool(any(r in edge for r in self.radii_nm[-1:])),
                         "profile": sorted(profile, key=lambda p: -p["log_pl"])[:50]}
        return self

    # ---- simulation ------------------------------------------------------------------------------------------------------
    def _shell_counts_1(self, d: np.ndarray) -> np.ndarray:
        cum = np.array([(d < R).sum() for R in self.radii_nm])
        return np.diff(np.r_[0, cum])

    def sample_configuration(self, branch: PositionBranch, n: int, rng: np.random.Generator):
        s, th = _base_configuration(self.base, branch, n, rng)
        lg = np.asarray(self.log_gammas, dtype=float)
        if n < 2 or not self.radii_nm or np.all(lg >= 0.0):
            return s, th
        cdf, nodes = _cdf(self.base, branch)
        r = np.interp(s, branch.s, branch.radius)
        for _ in range(self.burn_factor * n):
            i = int(rng.integers(n))
            s_new = float(np.interp(rng.random(), cdf, nodes))
            th_new = float(np.pi - rng.random() * 2.0 * np.pi)
            r_new = float(np.interp(s_new, branch.s, branch.radius))
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            d_old = surface_distances(s[i:i + 1], th[i:i + 1], r[i:i + 1], s[mask], th[mask], r[mask])[0]
            d_new = surface_distances(np.array([s_new]), np.array([th_new]), np.array([r_new]), s[mask], th[mask], r[mask])[0]
            log_ratio = float(lg @ (self._shell_counts_1(d_new) - self._shell_counts_1(d_old)))
            if log_ratio >= 0.0 or math.log(max(rng.random(), 1e-300)) < log_ratio:
                s[i], th[i], r[i] = s_new, th_new, r_new
        order = np.argsort(s, kind="stable")
        return s[order], th[order]

    def sample_positions(self, branch: PositionBranch, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sample_configuration(branch, n, rng)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "radii_nm": self.radii_nm, "log_gammas": self.log_gammas, "alpha": self.alpha,
                "burn_factor": self.burn_factor, "fit_info": self.fit_info}


def interaction_from_dict(d: Dict[str, Any], base: PositionModel):
    if d["kind"] == "renewal":
        m = RenewalModel(base, d["k"])
    elif d["kind"] == "strauss":
        m = StraussModel(base, d["radius_nm"], d["log_gamma"], d.get("alpha", 0.0), d.get("burn_factor", 50))
    elif d["kind"] == "multi_strauss":
        m = MultiStraussModel(base, d["radii_nm"], d["log_gammas"], d.get("alpha", 0.0), d.get("burn_factor", 50))
    else:
        raise ValueError(f"unknown interaction model {d['kind']!r}")
    m.fit_info = d.get("fit_info", {})
    return m
