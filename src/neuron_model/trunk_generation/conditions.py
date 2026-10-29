"""Condition vector e(C) of modules T and P (docs/neuron-model/module-interfaces-spec.md, section 7).

Categorical features are one-hot with an explicit "unknown" category (a feature missing in a dataset is
never mistaken for a real value) and a known-mask; the branch length L enters as log L. Two views:

- :func:`encode_conditions`  fixed-size vector for neural nets (T1, T2, P3);
- :func:`design_row`         regression design (dummies against a reference category + standardised log L)
                             for the statistical models (T0, P0).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

SPECIES = ("mouse", "human")
CELL_TYPE_BINARY = ("excitatory", "inhibitory")
COMPARTMENT = ("apical", "basal", "uncertain")
CATEGORIES = {"species": SPECIES, "cell_type_binary": CELL_TYPE_BINARY, "compartment": COMPARTMENT}
CONDITION_DIM = sum(len(c) + 1 for c in CATEGORIES.values()) + len(CATEGORIES)  # one-hot (+unknown) and known masks


def _known(value: Any, categories: Sequence[str]) -> bool:
    return isinstance(value, str) and value in categories


def encode_conditions(species: Optional[str], cell_type_binary: Optional[str], compartment: Optional[str]) -> np.ndarray:
    """[CONDITION_DIM] float32: one-hot with unknown slot for each feature, then three known flags."""
    values = {"species": species, "cell_type_binary": cell_type_binary, "compartment": compartment}
    parts, flags = [], []
    for name, categories in CATEGORIES.items():
        one_hot = np.zeros(len(categories) + 1, dtype=np.float32)
        if _known(values[name], categories):
            one_hot[categories.index(values[name])] = 1.0
        else:
            one_hot[-1] = 1.0
        parts.append(one_hot)
        flags.append(1.0 if _known(values[name], categories) else 0.0)
    return np.concatenate(parts + [np.asarray(flags, dtype=np.float32)])


class DesignEncoder:
    """Regression design: ``[1, z(log L), dummies...]`` (reference: mouse / excitatory / apical; unknown values
    get their own dummy). ``z`` standardises ``log L`` with the statistics of the training data.

    With ``soma`` = (mean, std) of ``log(1 + d / 1 um)`` (``d`` = distance of the branch midpoint to the soma along the dendrite)
    the design gets three more columns: ``z_log_soma`` (0 when unknown), ``soma=unknown`` and ``is_terminal``. Used by T0
    (distal branches are thinner and nearly spine-free: a trunk generated without it has a typical radius whatever its
    distance to the soma, 2026-10-29); P0 keeps the plain design (it has the soma distance in its own covariates)."""

    def __init__(self, log_length_mean: float = 0.0, log_length_std: float = 1.0, soma: Optional[Tuple[float, float]] = None) -> None:
        self.log_length_mean = float(log_length_mean)
        self.log_length_std = float(log_length_std) if log_length_std > 0 else 1.0
        self.soma = None if soma is None else (float(soma[0]), float(soma[1]) if soma[1] > 0 else 1.0)

    @classmethod
    def fit(cls, lengths_nm: Sequence[float], soma_distances_nm: Optional[Sequence[float]] = None) -> "DesignEncoder":
        logs = np.log(np.asarray(lengths_nm, dtype=float))
        soma = None
        if soma_distances_nm is not None:
            d = np.asarray(soma_distances_nm, dtype=float)
            d = np.log1p(d[np.isfinite(d)] / 1000.0)
            if len(d) >= 2:
                soma = (float(d.mean()), float(d.std()))
        return cls(float(logs.mean()), float(logs.std()), soma)

    @property
    def names(self) -> Tuple[str, ...]:
        names = ["intercept", "z_log_length"]
        for feature, categories in CATEGORIES.items():
            names += [f"{feature}={c}" for c in categories[1:]] + [f"{feature}=unknown"]
        if self.soma is not None:
            names += ["z_log_soma", "soma=unknown", "is_terminal"]
        return tuple(names)

    def z_length(self, length_nm: float) -> float:
        return (math.log(length_nm) - self.log_length_mean) / self.log_length_std

    def row(self, species: Optional[str], cell_type_binary: Optional[str], compartment: Optional[str], length_nm: float,
            soma_distance_nm: Optional[float] = None, is_terminal: Optional[bool] = None) -> np.ndarray:
        values = {"species": species, "cell_type_binary": cell_type_binary, "compartment": compartment}
        out = [1.0, self.z_length(length_nm)]
        for feature, categories in CATEGORIES.items():
            value = values[feature]
            known = _known(value, categories)
            out += [1.0 if known and value == c else 0.0 for c in categories[1:]]
            out.append(0.0 if known else 1.0)
        if self.soma is not None:
            known_soma = soma_distance_nm is not None and np.isfinite(soma_distance_nm)
            z = (math.log1p(float(soma_distance_nm) / 1000.0) - self.soma[0]) / self.soma[1] if known_soma else 0.0
            out += [z, 0.0 if known_soma else 1.0, 1.0 if is_terminal else 0.0]
        return np.asarray(out, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"log_length_mean": self.log_length_mean, "log_length_std": self.log_length_std}
        if self.soma is not None:
            out["soma_log_mean"], out["soma_log_std"] = self.soma
        return out

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DesignEncoder":
        soma = (values["soma_log_mean"], values["soma_log_std"]) if "soma_log_mean" in values else None
        return cls(values["log_length_mean"], values["log_length_std"], soma)


def group_key(species: Optional[str], cell_type_binary: Optional[str], compartment: Optional[str]) -> str:
    """Group of the hierarchical shrinkage: species | cell type | compartment (unknown values -> "unknown")."""
    def part(value, categories):
        return value if _known(value, categories) else "unknown"

    return "|".join((part(species, SPECIES), part(cell_type_binary, CELL_TYPE_BINARY), part(compartment, COMPARTMENT)))
