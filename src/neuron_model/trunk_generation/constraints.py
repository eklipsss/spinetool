"""Validity checks of a generated branch (t-module-technical-spec.md, section 7).

Every sample, whatever the model, passes the same checks; the thresholds are calibrated on the training branches
(real branches must pass with <= 1 % rejections, otherwise the thresholds are wrong, not the branches)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .. import branch_geometry as bg


@dataclass(frozen=True)
class TrunkLimits:
    radius_min_nm: float = 80.0
    radius_max_nm: float = 1500.0
    c_r_kappa: float = 0.5  # max r * kappa (necessary for a fold-free tube; 0.5 leaves a margin for the discretisation)
    kappa_max: Optional[float] = None  # nm^-1; None = not checked
    tortuosity_max: Optional[float] = None
    check_contact: bool = True

    @classmethod
    def from_data(cls, kappa: Sequence[np.ndarray], tortuosity: Sequence[float], *, quantile: float = 99.9, safety: float = 1.2, **kwargs: Any) -> "TrunkLimits":
        """kappa_max / tortuosity_max from the training branches (99.9 % quantile x ``safety`` / quantile)."""
        flat = np.concatenate([np.asarray(k, dtype=float)[np.isfinite(k)] for k in kappa])
        return cls(kappa_max=float(np.percentile(flat, quantile) * safety), tortuosity_max=float(np.percentile(np.asarray(tortuosity, dtype=float), quantile)), **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_trunk(gamma: np.ndarray, radius: np.ndarray, kappa1: np.ndarray, kappa2: np.ndarray, step: float, limits: TrunkLimits) -> Dict[str, Any]:
    """``{"passed": bool, "reasons": [...], metrics...}``; kappa arrays have M-1 steps (a trailing NaN is ignored)."""
    m = len(radius)
    k = np.hypot(np.asarray(kappa1, dtype=float)[: m - 1], np.asarray(kappa2, dtype=float)[: m - 1])
    nodes = np.arange(m) * step
    reasons = []
    if not (np.isfinite(radius).all() and np.isfinite(gamma).all() and np.isfinite(k).all()):
        return {"passed": False, "reasons": ["non_finite"]}
    if radius.min() < limits.radius_min_nm or radius.max() > limits.radius_max_nm:
        reasons.append("radius_range")
    r_mid = 0.5 * (radius[:-1] + radius[1:])
    max_r_kappa = float((r_mid * k).max()) if len(k) else 0.0
    if max_r_kappa >= limits.c_r_kappa:
        reasons.append("r_kappa")
    if limits.kappa_max is not None and len(k) and k.max() > limits.kappa_max:
        reasons.append("kappa_max")
    chord = float(np.linalg.norm(gamma[-1] - gamma[0]))
    tortuosity = float(nodes[-1] / max(chord, 1e-9))
    if limits.tortuosity_max is not None and tortuosity > limits.tortuosity_max:
        reasons.append("tortuosity")
    contact, margin = (False, float("inf"))
    if limits.check_contact and not reasons:
        contact, margin = bg.tube_self_contact(gamma, radius, nodes, step)
        if contact:
            reasons.append("self_contact")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "max_r_kappa": max_r_kappa,
        "kappa_max_observed": float(k.max()) if len(k) else 0.0,
        "tortuosity": tortuosity,
        "contact_margin_nm": float(margin),
    }
