"""Trunk surface for the assembly: a trunk record of the T interface -> dense centreline, closed tube mesh, tube SDF.

Trunk record (module-interfaces-spec.md, section 4.2; ``generate_tp.py`` / ``branch_data.trunk_record``): node arrays
``s`` [M], ``gamma`` [M,3], ``radius`` [M], ``tangent`` / ``normal`` (= e1) / ``binormal`` (= e2) [M,3], nm. The nodes
are 250-500 nm apart; the tube is built on a denser grid (cubic centreline, linear radius, frames re-orthogonalised to
the dense tangent) so the surface does not show the node polygon. Units: nm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from .. import branch_geometry as bg


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


@dataclass
class DenseTrunk:
    s: np.ndarray
    gamma: np.ndarray
    radius: np.ndarray
    tangent: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    _tree: Any = None
    _probe_s: Any = None

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def frame_at(self, s: np.ndarray) -> Dict[str, np.ndarray]:
        """Point, radius and orthonormal frame (t, e1, e2) at arclengths ``s`` (linear between dense samples)."""
        s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, self.length)
        interp = lambda a: np.stack([np.interp(s, self.s, a[:, k]) for k in range(3)], axis=1)
        t = _unit(interp(self.tangent))
        e1 = interp(self.e1)
        e1 = _unit(e1 - (e1 * t).sum(1, keepdims=True) * t)
        e2 = interp(self.e2)
        e2 = e2 - (e2 * t).sum(1, keepdims=True) * t - (e2 * e1).sum(1, keepdims=True) * e1
        return {"point": interp(self.gamma), "radius": np.interp(s, self.s, self.radius), "tangent": t, "e1": e1, "e2": _unit(e2)}

    def surface_frame(self, s: np.ndarray, theta: np.ndarray) -> Dict[str, np.ndarray]:
        """Anchor a = gamma + r n, outward normal n = cos(theta) e1 + sin(theta) e2 (the P convention), tangent."""
        f = self.frame_at(s)
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        n = np.cos(theta)[:, None] * f["e1"] + np.sin(theta)[:, None] * f["e2"]
        return {"anchor": f["point"] + f["radius"][:, None] * n, "normal": n, "tangent": f["tangent"], "radius": f["radius"]}

    def signed_distance(self, points: np.ndarray, probe_step_nm: float = 10.0) -> np.ndarray:
        """Approximate SDF of the round tube: |p - gamma(s*)| - r(s*), s* = nearest centreline sample (negative inside).
        Exact for points whose nearest centre point is unique (|SDF| < 1 / kappa), which is the case near the surface."""
        if self._tree is None:
            n = max(2, int(np.ceil(self.length / probe_step_nm)) + 1)
            self._probe_s = np.linspace(0.0, self.length, n)
            probe = np.stack([np.interp(self._probe_s, self.s, self.gamma[:, k]) for k in range(3)], axis=1)
            self._tree = cKDTree(probe)
        d, idx = self._tree.query(np.asarray(points, dtype=float))
        return d - np.interp(self._probe_s[idx], self.s, self.radius)


def densify(trunk: Dict[str, Any], step_nm: float = 100.0) -> DenseTrunk:
    s = np.asarray(trunk["s"], dtype=float)
    gamma = np.asarray(trunk["gamma"], dtype=float)
    length = float(s[-1])
    n = max(2, int(np.ceil(length / step_nm)) + 1)
    sd = np.linspace(0.0, length, n)
    if len(s) >= 4:
        spline = CubicSpline(s, gamma, axis=0)
        g, t = spline(sd), _unit(spline(sd, 1))
    else:
        g = np.stack([np.interp(sd, s, gamma[:, k]) for k in range(3)], axis=1)
        t = _unit(np.stack([np.interp(sd, s, np.asarray(trunk["tangent"])[:, k]) for k in range(3)], axis=1))
    interp = lambda a: np.stack([np.interp(sd, s, np.asarray(a, dtype=float)[:, k]) for k in range(3)], axis=1)
    e1 = interp(trunk["normal"])
    e1 = _unit(e1 - (e1 * t).sum(1, keepdims=True) * t)
    e2 = interp(trunk["binormal"])
    e2 = _unit(e2 - (e2 * t).sum(1, keepdims=True) * t - (e2 * e1).sum(1, keepdims=True) * e1)
    return DenseTrunk(s=sd, gamma=g, radius=np.interp(sd, s, np.asarray(trunk["radius"], dtype=float)), tangent=t, e1=e1, e2=e2)


def trunk_mesh(trunk: DenseTrunk, n_theta: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Closed round tube (flat fan caps at both ends), outward faces."""
    return bg.tube_mesh(trunk.gamma, trunk.radius, trunk.e1, trunk.e2, n_theta=n_theta, caps=True)
