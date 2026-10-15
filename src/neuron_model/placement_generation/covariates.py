"""Branch-level covariates of the count model (p-module-technical-spec.md, section 3).

``BranchCovariates`` holds what is needed per branch; :class:`CountDesign` turns a list of them into the
design matrix (B-splines of the mean radius and of the distance to the soma, log tortuosity, terminal flag,
log-length effect and the condition dummies), standardising every continuous covariate with the TRAINING data.
A missing distance to the soma is never replaced by a Euclidean distance: it gets its own indicator column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..trunk_generation.conditions import DesignEncoder


@dataclass
class BranchCovariates:
    neuron: str
    n_spines: int  # N_b (spines on the centreline span, false spines excluded); -1 when unknown (generation)
    length_nm: float
    area_nm2: float
    mean_radius_nm: float
    soma_distance_nm: Optional[float]  # branch midpoint, along the dendritic path
    tortuosity: float
    is_terminal: bool
    species: Optional[str]
    cell_type_binary: Optional[str]
    compartment: Optional[str]


def _bspline(x: np.ndarray, n_basis: int, lo: float, hi: float, degree: int = 3) -> np.ndarray:
    from scipy.interpolate import BSpline

    u = np.clip((np.asarray(x, dtype=float) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    interior = np.linspace(0.0, 1.0, n_basis - degree + 1)[1:-1]
    knots = np.r_[[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    return BSpline.design_matrix(u, knots, degree).toarray()


class CountDesign:
    """Fitted on the training branches, applied to any branch (also at generation)."""

    def __init__(self, n_basis: int = 5) -> None:
        self.n_basis = n_basis
        self.encoder: Optional[DesignEncoder] = None
        self.radius_range = (0.0, 1.0)
        self.soma_range = (0.0, 1.0)

    def fit(self, branches: Sequence[BranchCovariates]) -> "CountDesign":
        self.encoder = DesignEncoder.fit([b.length_nm for b in branches])
        radius = np.array([b.mean_radius_nm for b in branches])
        self.radius_range = (float(np.percentile(radius, 1)), float(np.percentile(radius, 99)))
        soma = np.array([b.soma_distance_nm for b in branches if b.soma_distance_nm is not None])
        self.soma_range = (float(np.percentile(soma, 1)), float(np.percentile(soma, 99))) if len(soma) else (0.0, 1.0)
        return self

    @property
    def names(self) -> List[str]:
        enc = self.encoder.names[1:]  # drop the intercept (the splines sum to one and carry it)
        return (
            [f"radius_spline{i}" for i in range(self.n_basis)]
            + [f"soma_spline{i}" for i in range(self.n_basis)]
            + ["soma_missing", "log_tortuosity", "is_terminal"]
            + list(enc)
        )

    def matrix(self, branches: Sequence[BranchCovariates]) -> np.ndarray:
        radius = _bspline([b.mean_radius_nm for b in branches], self.n_basis, *self.radius_range)
        has = np.array([b.soma_distance_nm is not None for b in branches])
        soma_values = np.array([b.soma_distance_nm if b.soma_distance_nm is not None else self.soma_range[0] for b in branches], dtype=float)
        soma = _bspline(soma_values, self.n_basis, *self.soma_range) * has[:, None]
        # a missing distance: the soma spline block is zero and the intercept-like role is taken by the radius block
        misc = np.column_stack([~has, np.log(np.maximum([b.tortuosity for b in branches], 1.0)), [float(b.is_terminal) for b in branches]]).astype(float)
        cond = np.vstack([self.encoder.row(b.species, b.cell_type_binary, b.compartment, b.length_nm)[1:] for b in branches])
        return np.hstack([radius, soma, misc, cond])

    def inflation_matrix(self, branches: Sequence[BranchCovariates]) -> np.ndarray:
        """w_b = (1, z(log L), is_terminal, e(C)) of the zero-inflation / hurdle part."""
        rows = [np.r_[self.encoder.row(b.species, b.cell_type_binary, b.compartment, b.length_nm), float(b.is_terminal)] for b in branches]
        return np.vstack(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_basis": self.n_basis, "encoder": self.encoder.to_dict(), "radius_range": list(self.radius_range), "soma_range": list(self.soma_range)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CountDesign":
        obj = cls(d["n_basis"])
        obj.encoder = DesignEncoder.from_dict(d["encoder"])
        obj.radius_range, obj.soma_range = tuple(d["radius_range"]), tuple(d["soma_range"])
        return obj
