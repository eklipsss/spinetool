"""Placement generation (p-module-technical-spec.md, sections 5, 6, 9).

For every trunk record: count N (P0-count, a neuron effect shared by the branches of one neuron), positions by the inverse
CDF, uniform angles, rejection of configurations with two bases closer than ``d_min`` (up to ``max_attempts`` redraws),
anchor points ``a_i = gamma(s_i) + r(s_i) (cos(theta) e1 + sin(theta) e2)`` and outward normals. Units: nm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .models.position import PositionBranch, sample_angles


def surface_min_distance(s: np.ndarray, theta: np.ndarray, radius_at: np.ndarray) -> float:
    """Smallest pairwise surface distance sqrt(ds^2 + (r_mean dtheta)^2) (section 6); inf for fewer than two points."""
    n = len(s)
    if n < 2:
        return float("inf")
    ds = s[:, None] - s[None, :]
    dth = np.abs(theta[:, None] - theta[None, :])
    dth = np.minimum(dth, 2 * np.pi - dth)
    rbar = 0.5 * (radius_at[:, None] + radius_at[None, :])
    d = np.sqrt(ds ** 2 + (rbar * dth) ** 2)
    d[np.arange(n), np.arange(n)] = np.inf
    return float(d.min())


def anchors(trunk: Dict[str, Any], s: np.ndarray, theta: np.ndarray) -> Dict[str, np.ndarray]:
    """Anchor points and outward normals on the tube (frames e1 = ``normal``, e2 = ``binormal`` of the trunk record)."""
    grid = trunk["s"]
    interp = lambda a: np.stack([np.interp(s, grid, a[:, k]) for k in range(3)], axis=1)
    e1, e2 = interp(trunk["normal"]), interp(trunk["binormal"])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = e2 - (e2 * e1).sum(1, keepdims=True) * e1
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    n_hat = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    r = np.interp(s, grid, trunk["radius"])
    return {"anchor": interp(trunk["gamma"]) + r[:, None] * n_hat, "normal": n_hat, "radius": r}


def trunk_to_position_branch(trunk: Dict[str, Any], soma_start_nm: Optional[float] = None) -> PositionBranch:
    kappa = np.hypot(trunk["kappa1"], trunk["kappa2"])
    return PositionBranch(np.asarray(trunk["s"], dtype=float), np.asarray(trunk["radius"], dtype=float), kappa, soma_start_nm)


def generate_placements(
    trunks: Sequence[Dict[str, Any]],
    counts: Sequence[int],
    position_model: Any,
    *,
    d_min_nm: float = 0.0,
    soma_start_nm: Optional[Sequence[Optional[float]]] = None,
    seed: int = 0,
    max_attempts: int = 20,
) -> List[Dict[str, Any]]:
    """One placement record per trunk; ``counts`` come from ``CountModel.sample``."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (trunk, n) in enumerate(zip(trunks, counts)):
        branch = trunk_to_position_branch(trunk, None if soma_start_nm is None else soma_start_nm[i])
        n = int(n)
        attempts, ok = 0, n < 2 or d_min_nm <= 0.0
        s = np.zeros(0)
        theta = np.zeros(0)
        min_dist = float("inf")
        while attempts < max_attempts:
            attempts += 1
            s = position_model.sample_positions(branch, n, rng)
            theta = sample_angles(n, rng)
            min_dist = surface_min_distance(s, theta, np.interp(s, branch.s, branch.radius))
            if d_min_nm <= 0.0 or min_dist >= d_min_nm:
                ok = True
                break
        geom = anchors(trunk, s, theta) if n else {"anchor": np.zeros((0, 3)), "normal": np.zeros((0, 3)), "radius": np.zeros(0)}
        out.append({
            "n_spines": n, "s": s, "u": s / trunk["length"], "theta": theta,
            "anchor": geom["anchor"], "normal": geom["normal"],
            "attempts": attempts, "d_min_nm": float(d_min_nm), "min_distance_nm": min_dist, "d_min_satisfied": bool(ok),
            "seed": seed,
        })
    return out


def covariates_from_trunk(trunk: Dict[str, Any], request: Dict[str, Any], neuron: str):
    """Count covariates computed on a (generated or real) trunk record; ``request`` carries the conditions,
    ``is_terminal`` and (optionally) ``soma_distance_start_nm`` - quantities T does not generate."""
    from .covariates import BranchCovariates

    s, r, gamma = np.asarray(trunk["s"], float), np.asarray(trunk["radius"], float), np.asarray(trunk["gamma"], float)
    length = float(s[-1])
    d0 = request.get("soma_distance_start_nm")
    chord = float(np.linalg.norm(gamma[-1] - gamma[0]))
    return BranchCovariates(neuron, -1, length, float(np.trapezoid(2 * np.pi * r, s)), float(np.trapezoid(r, s) / max(length, 1e-9)),
                            None if d0 is None else float(d0) + 0.5 * length, length / max(chord, 1e-9), bool(request.get("is_terminal", False)),
                            request.get("species"), request.get("cell_type_binary"), request.get("compartment"))


def pack_records(trunks: Sequence[Dict[str, Any]], fields: Sequence[str] = ("s", "gamma", "radius", "normal", "binormal", "tangent")) -> Dict[str, np.ndarray]:
    """Concatenate trunk records into flat arrays + offsets (one file for a whole generated set, not one per branch)."""
    n = np.array([len(t["s"]) for t in trunks], dtype=np.int64)
    out = {"offsets": np.r_[0, np.cumsum(n)].astype(np.int64)}
    for f in fields:
        if all(f in t for t in trunks):
            out[f] = np.concatenate([np.asarray(t[f]) for t in trunks]) if trunks else np.zeros(0)
    return out
