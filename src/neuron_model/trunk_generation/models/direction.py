"""Direction model of T0: the trunk heading relaxes to the main axis of the branch (bounded tortuosity).

Why: an Ornstein-Uhlenbeck process on the bending (kappa1, kappa2) integrates to a FREE random walk of the heading,
so the generated tortuosity grows with the length (10 Minnie neurons, dendrites: real median 1.02 / 1.04 / 1.09 for
< 20 / 20-50 / >= 50 um, the kappa-OU gave 1.21 / 2.13 / 2.71). Real dendrites keep an overall direction.

Model (per transverse component, isotropic, nodes ``step`` apart). ``tilt`` = tan of the angle between the node
tangent and the axis, as a 2-vector in the plane normal to the axis (axis = chord direction of the branch)::

    tilt_{j+1} = exp(-step / L_d) tilt_j + step * g_j          (restoring to the axis over the length L_d)
    g_{j+1}    = exp(-step / l_g) g_j + N(0, s_b^2 (1 - exp(-2 step / l_g)))   (smooth turning rate, OU)

``s_b`` is the turning amplitude OF THE BRANCH (random effect): log s_b ~ N(alpha + alpha_group + beta log(r_b / 300 nm),
tau^2), r_b = median radius. A single pooled amplitude is set by the most bent branches (it is the mean square over
all nodes) and makes the typical branch too bent and the turning memory too short (first real fit: kappa_rms x2.7,
kappa autocorrelation at 1 um 0.17 vs 0.51). L_d and l_g are shared; the per-branch variance is profiled out of the
exact AR(1) likelihood of g (closed form), L_d over a grid, l_g by a 1-D search. The transform tilt -> g has a
constant Jacobian, so the likelihoods of different L_d are comparable. With L_d = inf it is the old model.
Generation: s_b drawn for the branch, stationary start (burn-in), tangents t = normalize(tilt, 1) in an axis frame,
then the bending of the minimal (twist-free) rotations between consecutive tangents expressed in the transported frame
- the same (kappa1, kappa2) convention as ``branch_geometry.integrate_bending``, so the rest of T0 is unchanged. Units: nm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ... import branch_geometry as bg

RESTORE_GRID_NM = (1000.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0, 20000.0, 35000.0, 60000.0, 100000.0, math.inf)
MIN_COS = 0.2  # tangents turned more than ~78 deg from the axis are clipped (tilt <= 4.9)


def node_tangents(kappa1: np.ndarray, kappa2: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """(gamma [M,3], tangents [M,3]) rebuilt from the bending (M-1 steps)."""
    gamma, frames = bg.integrate_bending(kappa1, kappa2, step)
    return gamma, frames[:, :, 0]


def axis_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u1, u2 = bg.section_basis(axis)
    return u1, u2


def tilt_series(gamma: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """[M,2] tilt of every node tangent relative to the chord direction of the branch."""
    chord = gamma[-1] - gamma[0]
    norm = float(np.linalg.norm(chord))
    axis = chord / norm if norm > 0 else tangents.mean(0) / max(np.linalg.norm(tangents.mean(0)), 1e-12)
    u1, u2 = axis_basis(axis)
    c = np.clip(tangents @ axis, MIN_COS, None)
    return np.column_stack([tangents @ u1, tangents @ u2]) / c[:, None]


def drive_series(tilt: np.ndarray, step: float, restore_nm: float) -> np.ndarray:
    """[M-1,2] g_j = (tilt_{j+1} - a tilt_j) / step, a = exp(-step / L_d)."""
    a = 0.0 if restore_nm <= 0 else math.exp(-step / restore_nm)
    return (tilt[1:] - a * tilt[:-1]) / step


MIN_DRIVES = 8  # branches with fewer turning-rate values do not inform the per-branch amplitude
R_REF_NM = 300.0


def _branch_sums(g: np.ndarray, mask: np.ndarray, step: np.ndarray, length_nm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per branch: S_b (scaled sum of squares), n_b (number of g values), q_b = 1 - phi^2."""
    phi = np.exp(-step / length_nm)
    q = np.maximum(1.0 - phi**2, 1e-12)
    first = (g[:, 0] ** 2).sum(-1) * mask[:, 0]
    resid = g[:, 1:] - phi[:, None, None] * g[:, :-1]
    m = mask[:, 1:] * mask[:, :-1]
    s = first + ((resid**2).sum(-1) * m).sum(-1) / q
    return s, mask.sum(-1), q


def _profile_nll(length_nm: float, g: np.ndarray, mask: np.ndarray, step: np.ndarray) -> float:
    """Exact AR(1) NLL of isotropic 2-D chains with the per-branch variance maximised out."""
    s, n, q = _branch_sums(g, mask, step, length_nm)
    s = np.maximum(s, 1e-300)
    return 0.5 * float((2.0 * n + 2.0 * n * np.log(s / (2.0 * n)) + 2.0 * (n - 1.0) * np.log(q)).sum())


def _fit_length(g: np.ndarray, mask: np.ndarray, step: np.ndarray) -> Tuple[float, float]:
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda x: _profile_nll(math.exp(x), g, mask, step), bounds=(math.log(50.0), math.log(1e6)), method="bounded")
    return math.exp(float(res.x)), float(res.fun)


@dataclass
class DirectionParams:
    restore_nm: float       # L_d (inf = no restoring)
    drive_std: float        # typical s_b (alpha, branch of radius R_REF_NM), 1/nm
    drive_length_nm: float  # l_g

    def to_dict(self) -> Dict[str, float]:
        return {"restore_nm": None if math.isinf(self.restore_nm) else self.restore_nm, "drive_std": self.drive_std, "drive_length_nm": self.drive_length_nm}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectionParams":
        return cls(math.inf if d["restore_nm"] is None else float(d["restore_nm"]), float(d["drive_std"]), float(d["drive_length_nm"]))


@dataclass
class Amplitude:
    """log s_b = alpha + groups[g] + beta * log(r_b / R_REF_NM) + tau * N(0, 1)."""

    alpha: float
    beta: float = 0.0
    tau: float = 0.0
    groups: Dict[str, float] = field(default_factory=dict)

    def mean_log(self, group: str, radius_nm: Optional[float]) -> float:
        r = R_REF_NM if radius_nm is None or not np.isfinite(radius_nm) or radius_nm <= 0 else float(radius_nm)
        return self.alpha + self.groups.get(group, 0.0) + self.beta * math.log(r / R_REF_NM)

    def to_dict(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta, "tau": self.tau, "groups": dict(self.groups)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Amplitude":
        return cls(float(d["alpha"]), float(d.get("beta", 0.0)), float(d.get("tau", 0.0)), {k: float(v) for k, v in d.get("groups", {}).items()})


@dataclass
class DirectionModel:
    pooled: Optional[DirectionParams] = None
    amplitude: Optional[Amplitude] = None
    profile: Dict[str, float] = field(default_factory=dict)  # restoring length -> nll per value (pooled)

    @property
    def groups(self) -> Dict[str, float]:
        return self.amplitude.groups if self.amplitude is not None else {}

    @staticmethod
    def _arrays(tilts: Sequence[np.ndarray], steps: Sequence[float], restore_nm: float):
        drives = [drive_series(t, h, restore_nm) for t, h in zip(tilts, steps)]
        n = max(len(d) for d in drives)
        g = np.zeros((len(drives), n, 2))
        mask = np.zeros((len(drives), n))
        for i, d in enumerate(drives):
            g[i, : len(d)] = d
            mask[i, : len(d)] = 1.0
        return g, mask, np.asarray(steps, dtype=float)

    def fit(self, branches: Sequence[Any], *, min_group: int = 50, restore_grid: Sequence[float] = RESTORE_GRID_NM) -> "DirectionModel":
        """``branches``: T0Branch-like (kappa1, kappa2, log_radius, step, group)."""
        tilts, steps, groups, radii = [], [], [], []
        for b in branches:
            if b.n_nodes < MIN_DRIVES + 1:
                continue
            gamma, t = node_tangents(b.kappa1, b.kappa2, b.step)
            tilts.append(tilt_series(gamma, t))
            steps.append(float(b.step))
            groups.append(b.group)
            radii.append(float(np.median(np.exp(b.log_radius))))
        if not tilts:
            raise ValueError(f"no branch with at least {MIN_DRIVES + 1} nodes")
        n_values = sum(len(t) - 1 for t in tilts)
        best = None
        for restore in restore_grid:
            g, mask, step = self._arrays(tilts, steps, restore)
            length, nll = _fit_length(g, mask, step)
            self.profile[str(restore)] = nll / n_values
            if best is None or nll < best[0]:
                best = (nll, restore, length)
        _, restore, length = best
        g, mask, step = self._arrays(tilts, steps, restore)
        s, n, _ = _branch_sums(g, mask, step, length)
        y = 0.5 * np.log(np.maximum(s, 1e-300) / (2.0 * n))  # log of the per-branch amplitude
        x = np.log(np.asarray(radii) / R_REF_NM)
        w = n / n.sum()
        design = np.column_stack([np.ones_like(x), x])
        coef = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        resid = y - design @ coef
        groups_arr = np.asarray(groups)
        offsets = {}
        for name in np.unique(groups_arr):
            m = groups_arr == name
            if m.sum() >= min_group:
                offsets[str(name)] = float(np.average(resid[m], weights=n[m]))
                resid[m] -= offsets[str(name)]
        # spread of the true amplitudes = spread of the estimates minus their sampling noise (var log s_hat ~ 1 / (4 n))
        tau2 = float(np.average(resid**2, weights=n)) - float(np.average(1.0 / (4.0 * n), weights=n))
        self.amplitude = Amplitude(float(coef[0]), float(coef[1]), math.sqrt(max(tau2, 0.0025)), offsets)
        self.pooled = DirectionParams(float(restore), math.exp(float(coef[0])), float(length))
        return self

    def sample_tilt(self, n_nodes: int, step: float, group: str, rng: np.random.Generator, radius_nm: Optional[float] = None) -> np.ndarray:
        """[n_nodes,2] tilt from the stationary regime (burn-in), with an amplitude drawn for this branch."""
        p = self.pooled
        amp = self.amplitude or Amplitude(math.log(p.drive_std))
        s_b = math.exp(amp.mean_log(group, radius_nm) + amp.tau * rng.standard_normal())
        a = 1.0 if math.isinf(p.restore_nm) else math.exp(-step / p.restore_nm)
        phi = math.exp(-step / p.drive_length_nm)
        sd = s_b * math.sqrt(max(1.0 - phi**2, 0.0))
        burn = 0 if math.isinf(p.restore_nm) else min(int(math.ceil(5.0 * max(p.restore_nm, p.drive_length_nm) / step)), 5000)
        total = burn + n_nodes
        tilt = np.zeros((total, 2))
        g = rng.normal(0.0, s_b, 2)
        eps = rng.normal(0.0, sd, (total, 2))
        for j in range(total - 1):
            tilt[j + 1] = a * tilt[j] + step * g
            g = phi * g + eps[j]
        return tilt[burn:]

    def sample_bending(self, n_nodes: int, step: float, group: str, rng: np.random.Generator, radius_nm: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(kappa1, kappa2) [n_nodes-1] of a trunk with the sampled tilt (twist-free transport of the frame)."""
        tilt = self.sample_tilt(n_nodes, step, group, rng, radius_nm)
        t = np.column_stack([tilt, np.ones(len(tilt))])
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        return bending_from_tangents(t, step)

    def to_dict(self) -> Dict[str, Any]:
        return {"pooled": self.pooled.to_dict(), "amplitude": self.amplitude.to_dict() if self.amplitude else None,
                "groups": dict(self.groups), "profile": self.profile}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectionModel":
        amp = d.get("amplitude")
        return cls(DirectionParams.from_dict(d["pooled"]), Amplitude.from_dict(amp) if amp else None, dict(d.get("profile", {})))


def bending_from_tangents(t: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """(kappa1, kappa2) [M-1] of unit node tangents [M,3]: rotation vector w of the minimal rotation t_j -> t_{j+1}
    in the frame (t_j, e1_j, e2_j) transported by the same rotations: kappa1 = w.e2 / step, kappa2 = -w.e1 / step
    (``branch_geometry.integrate_bending`` inverts it)."""
    from scipy.spatial.transform import Rotation

    m = len(t)
    e1 = np.cross(t[0], [1.0, 0.0, 0.0] if abs(t[0][0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    k1, k2 = np.zeros(m - 1), np.zeros(m - 1)
    for j in range(m - 1):
        axis = np.cross(t[j], t[j + 1])
        sin = float(np.linalg.norm(axis))
        angle = math.atan2(sin, float(t[j] @ t[j + 1]))
        w = axis / sin * angle if sin > 1e-15 else np.zeros(3)
        e2 = np.cross(t[j], e1)
        k1[j], k2[j] = float(w @ e2) / step, -float(w @ e1) / step
        e1 = Rotation.from_rotvec(w).apply(e1)
        e1 -= (e1 @ t[j + 1]) * t[j + 1]
        e1 /= np.linalg.norm(e1)
    return k1, k2
