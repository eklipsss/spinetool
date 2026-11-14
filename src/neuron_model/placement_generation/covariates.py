"""Branch-level covariates of the count model (p-module-technical-spec.md, section 3).

``BranchCovariates`` holds what is needed per branch; :class:`CountDesign` turns a list of them into the
design matrix (B-splines of the mean radius and of the distance to the soma, log tortuosity, terminal flag,
log-length effect and the condition dummies), standardising every continuous covariate with the TRAINING data.
A missing distance to the soma is never replaced by a Euclidean distance: it gets its own indicator column.

Detailed cell type (2026-11-15, ``subtypes=True``): one indicator column per detailed type with at least
``subtype_min_neurons`` training neurons (``Unsure*`` and unknown types get none), LAST in the mean design. The binary type
column stays, so a subtype coefficient is the deviation of the subtype from the mean of its binary type; ``CountModel``
puts a Gaussian prior N(0, tau^2) on these coefficients (``subtype_prior_sd``), which pulls a subtype with few neurons
towards its binary type. Reason: neuron cross-validation (chain X) found obs / model 0.32 for 5P-NP, 1.10 for 23P,
0.55 for BPC, 1.2-1.4 for MC under the binary-type model.
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
    cell_type: Optional[str] = None  # detailed type (23P, 5P-NP, BPC, ...); used only by a design with subtypes


def _bspline(x: np.ndarray, n_basis: int, lo: float, hi: float, degree: int = 3) -> np.ndarray:
    from scipy.interpolate import BSpline

    u = np.clip((np.asarray(x, dtype=float) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    interior = np.linspace(0.0, 1.0, n_basis - degree + 1)[1:-1]
    knots = np.r_[[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    return BSpline.design_matrix(u, knots, degree).toarray()


class CountDesign:
    """Fitted on the training branches, applied to any branch (also at generation)."""

    def __init__(self, n_basis: int = 5, subtypes: bool = False, subtype_min_neurons: int = 3) -> None:
        self.n_basis = n_basis
        self.encoder: Optional[DesignEncoder] = None
        self.radius_range = (0.0, 1.0)
        self.soma_range = (0.0, 1.0)
        self.use_subtypes, self.subtype_min_neurons = bool(subtypes), int(subtype_min_neurons)
        self.subtypes: List[str] = []  # detailed types with a column (fitted)

    @staticmethod
    def _subtype(b: BranchCovariates) -> Optional[str]:
        t = getattr(b, "cell_type", None)
        if t is None or (isinstance(t, float) and np.isnan(t)):
            return None
        t = str(t)
        return None if not t or t.lower().startswith("unsure") or t.lower() in ("none", "nan") else t

    def fit(self, branches: Sequence[BranchCovariates]) -> "CountDesign":
        self.encoder = DesignEncoder.fit([b.length_nm for b in branches])
        if self.use_subtypes:
            neurons: Dict[str, set] = {}
            for b in branches:
                t = self._subtype(b)
                if t is not None:
                    neurons.setdefault(t, set()).add(b.neuron)
            self.subtypes = sorted(t for t, n in neurons.items() if len(n) >= self.subtype_min_neurons)
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
            + [f"subtype_{t}" for t in self.subtypes]
        )

    @property
    def n_subtype(self) -> int:
        """Number of detailed-type columns (the LAST columns of ``matrix``)."""
        return len(self.subtypes)

    def subtype_matrix(self, branches: Sequence[BranchCovariates]) -> np.ndarray:
        col = {t: i for i, t in enumerate(self.subtypes)}
        out = np.zeros((len(branches), len(self.subtypes)))
        for k, b in enumerate(branches):
            j = col.get(self._subtype(b))
            if j is not None:
                out[k, j] = 1.0
        return out

    def matrix(self, branches: Sequence[BranchCovariates]) -> np.ndarray:
        radius = _bspline([b.mean_radius_nm for b in branches], self.n_basis, *self.radius_range)
        has = np.array([b.soma_distance_nm is not None for b in branches])
        soma_values = np.array([b.soma_distance_nm if b.soma_distance_nm is not None else self.soma_range[0] for b in branches], dtype=float)
        soma = _bspline(soma_values, self.n_basis, *self.soma_range) * has[:, None]
        # a missing distance: the soma spline block is zero and the intercept-like role is taken by the radius block
        misc = np.column_stack([~has, np.log(np.maximum([b.tortuosity for b in branches], 1.0)), [float(b.is_terminal) for b in branches]]).astype(float)
        cond = np.vstack([self.encoder.row(b.species, b.cell_type_binary, b.compartment, b.length_nm)[1:] for b in branches])
        return np.hstack([radius, soma, misc, cond, self.subtype_matrix(branches)])

    def inflation_matrix(self, branches: Sequence[BranchCovariates]) -> np.ndarray:
        """w_b = (1, z(log L), is_terminal, e(C)) of the zero-inflation / hurdle part."""
        rows = [np.r_[self.encoder.row(b.species, b.cell_type_binary, b.compartment, b.length_nm), float(b.is_terminal)] for b in branches]
        return np.vstack(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_basis": self.n_basis, "encoder": self.encoder.to_dict(), "radius_range": list(self.radius_range), "soma_range": list(self.soma_range),
                "subtypes": list(self.subtypes), "subtype_min_neurons": self.subtype_min_neurons}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CountDesign":
        obj = cls(d["n_basis"])
        obj.encoder = DesignEncoder.from_dict(d["encoder"])
        obj.radius_range, obj.soma_range = tuple(d["radius_range"]), tuple(d["soma_range"])
        obj.subtypes = list(d.get("subtypes", []))  # old designs: none
        obj.use_subtypes, obj.subtype_min_neurons = bool(obj.subtypes), int(d.get("subtype_min_neurons", 3))
        return obj
