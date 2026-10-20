"""Direction model of T0: the trunk heading relaxes to the main axis of the branch (bounded tortuosity).

Why: an Ornstein-Uhlenbeck process on the bending (kappa1, kappa2) integrates to a FREE random walk of the heading,
so the generated tortuosity grows with the length (10 Minnie neurons, dendrites: real median 1.02 / 1.04 / 1.09 for
< 20 / 20-50 / >= 50 um, the kappa-OU gave 1.21 / 2.13 / 2.71). Real dendrites keep an overall direction.

Model (per transverse component, isotropic, nodes ``step`` apart). ``tilt`` = tan of the angle between the node
tangent and the axis, as a 2-vector in the plane normal to the axis (axis = chord direction of the branch)::

    tilt_{j+1} = exp(-step / L_d) tilt_j + step * g_j          (restoring to the axis over the length L_d)
    g_{j+1}    = exp(-step / l_g) g_j + N(0, s_g^2 (1 - exp(-2 step / l_g)))   (smooth turning rate, OU)

With L_d = inf it is the old model (g ~ the bending). The transform tilt -> g has a constant Jacobian, so the exact
AR(1) likelihood of g ranks the restoring lengths L_d (profile over a grid); s_g, l_g by maximum likelihood.
Generation: stationary start (burn-in), tangents t = normalize(tilt, 1) in an axis frame, then the bending of the
minimal (twist-free) rotations between consecutive tangents expressed in the transported frame - the same
(kappa1, kappa2) convention as ``branch_geometry.integrate_bending``, so the rest of T0 is unchanged. Units: nm.
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


def _ar1_nll(params: np.ndarray, g: np.ndarray, mask: np.ndarray, step: np.ndarray) -> float:
    """Negative exact log-likelihood of isotropic 2-D AR(1) chains (stationary start). g [B,n,2], mask [B,n]."""
    log_s, log_l = params
    s2 = math.exp(2.0 * log_s)
    phi = np.exp(-step / math.exp(log_l))[:, None]  # [B,1]
    q = np.maximum(s2 * (1.0 - phi**2), 1e-300)
    first = (g[:, 0] ** 2).sum(-1) / s2 + 2.0 * math.log(s2)
    resid = g[:, 1:] - phi[:, :, None] * g[:, :-1]
    trans = (resid**2).sum(-1) / q + 2.0 * np.log(q)
    m = mask[:, 1:] * mask[:, :-1]
    return 0.5 * float((first * mask[:, 0]).sum() + (trans * m).sum())


def _fit_ar1(g: np.ndarray, mask: np.ndarray, step: np.ndarray) -> Tuple[float, float, float]:
    """(s_g, l_g, nll) by L-BFGS on (log s, log l)."""
    from scipy.optimize import minimize

    v = g[mask.astype(bool)]
    s0 = max(float(np.sqrt(np.mean(v**2))), 1e-12)
    best = None
    for l0 in (500.0, 2000.0, 8000.0):
        res = minimize(_ar1_nll, np.array([math.log(s0), math.log(l0)]), args=(g, mask, step), method="L-BFGS-B",
                       bounds=[(math.log(s0) - 5, math.log(s0) + 5), (math.log(50.0), math.log(1e6))])
        if best is None or res.fun < best.fun:
            best = res
    return math.exp(best.x[0]), math.exp(best.x[1]), float(best.fun)


@dataclass
class DirectionParams:
    restore_nm: float  # L_d (inf = no restoring)
    drive_std: float   # s_g, 1/nm
    drive_length_nm: float  # l_g

    def to_dict(self) -> Dict[str, float]:
        return {"restore_nm": None if math.isinf(self.restore_nm) else self.restore_nm, "drive_std": self.drive_std, "drive_length_nm": self.drive_length_nm}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectionParams":
        return cls(math.inf if d["restore_nm"] is None else float(d["restore_nm"]), float(d["drive_std"]), float(d["drive_length_nm"]))


@dataclass
class DirectionModel:
    pooled: Optional[DirectionParams] = None
    groups: Dict[str, DirectionParams] = field(default_factory=dict)
    profile: Dict[str, float] = field(default_factory=dict)  # restoring length -> nll per node (pooled)

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
        """``branches``: T0Branch-like (kappa1, kappa2, step, group)."""
        tilts, steps, groups = [], [], []
        for b in branches:
            if b.n_nodes < 3:
                continue
            gamma, t = node_tangents(b.kappa1, b.kappa2, b.step)
            tilts.append(tilt_series(gamma, t))
            steps.append(float(b.step))
            groups.append(b.group)
        n_nodes = sum(len(t) - 1 for t in tilts)
        best = None
        for restore in restore_grid:
            g, mask, step = self._arrays(tilts, steps, restore)
            s, l, nll = _fit_ar1(g, mask, step)
            self.profile[str(restore)] = nll / max(n_nodes, 1)
            if best is None or nll < best[0]:
                best = (nll, DirectionParams(float(restore), s, l))
        self.pooled = best[1]
        groups_arr = np.asarray(groups)
        for name in np.unique(groups_arr):
            idx = np.flatnonzero(groups_arr == name)
            if len(idx) < min_group:
                continue
            g, mask, step = self._arrays([tilts[i] for i in idx], [steps[i] for i in idx], self.pooled.restore_nm)
            s, l, _ = _fit_ar1(g, mask, step)
            self.groups[str(name)] = DirectionParams(self.pooled.restore_nm, s, l)
        return self

    def params_for(self, group: str) -> DirectionParams:
        return self.groups.get(group, self.pooled)

    def sample_tilt(self, n_nodes: int, step: float, group: str, rng: np.random.Generator) -> np.ndarray:
        """[n_nodes,2] tilt from the stationary regime (burn-in)."""
        p = self.params_for(group)
        a = 1.0 if math.isinf(p.restore_nm) else math.exp(-step / p.restore_nm)
        phi = math.exp(-step / p.drive_length_nm)
        sd = p.drive_std * math.sqrt(max(1.0 - phi**2, 0.0))
        burn = 0 if math.isinf(p.restore_nm) else min(int(math.ceil(5.0 * max(p.restore_nm, p.drive_length_nm) / step)), 5000)
        total = burn + n_nodes
        tilt = np.zeros((total, 2))
        g = rng.normal(0.0, p.drive_std, 2)
        eps = rng.normal(0.0, sd, (total, 2))
        for j in range(total - 1):
            tilt[j + 1] = a * tilt[j] + step * g
            g = phi * g + eps[j]
        return tilt[burn:]

    def sample_bending(self, n_nodes: int, step: float, group: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """(kappa1, kappa2) [n_nodes-1] of a trunk with the sampled tilt (twist-free transport of the frame)."""
        tilt = self.sample_tilt(n_nodes, step, group, rng)
        t = np.column_stack([tilt, np.ones(len(tilt))])
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        return bending_from_tangents(t, step)

    def to_dict(self) -> Dict[str, Any]:
        return {"pooled": self.pooled.to_dict(), "groups": {g: p.to_dict() for g, p in self.groups.items()}, "profile": self.profile}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DirectionModel":
        return cls(DirectionParams.from_dict(d["pooled"]), {g: DirectionParams.from_dict(p) for g, p in d["groups"].items()}, dict(d.get("profile", {})))


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
