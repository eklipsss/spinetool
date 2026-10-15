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
    get their own dummy). ``z`` standardises ``log L`` with the statistics of the training data."""

    def __init__(self, log_length_mean: float = 0.0, log_length_std: float = 1.0) -> None:
        self.log_length_mean = float(log_length_mean)
        self.log_length_std = float(log_length_std) if log_length_std > 0 else 1.0

    @classmethod
    def fit(cls, lengths_nm: Sequence[float]) -> "DesignEncoder":
        logs = np.log(np.asarray(lengths_nm, dtype=float))
        return cls(float(logs.mean()), float(logs.std()))

    @property
    def names(self) -> Tuple[str, ...]:
        names = ["intercept", "z_log_length"]
        for feature, categories in CATEGORIES.items():
            names += [f"{feature}={c}" for c in categories[1:]] + [f"{feature}=unknown"]
        return tuple(names)

    def z_length(self, length_nm: float) -> float:
        return (math.log(length_nm) - self.log_length_mean) / self.log_length_std

    def row(self, species: Optional[str], cell_type_binary: Optional[str], compartment: Optional[str], length_nm: float) -> np.ndarray:
        values = {"species": species, "cell_type_binary": cell_type_binary, "compartment": compartment}
        out = [1.0, self.z_length(length_nm)]
        for feature, categories in CATEGORIES.items():
            value = values[feature]
            known = _known(value, categories)
            out += [1.0 if known and value == c else 0.0 for c in categories[1:]]
            out.append(0.0 if known else 1.0)
        return np.asarray(out, dtype=float)

    def to_dict(self) -> Dict[str, float]:
        return {"log_length_mean": self.log_length_mean, "log_length_std": self.log_length_std}

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "DesignEncoder":
        return cls(values["log_length_mean"], values["log_length_std"])


def group_key(species: Optional[str], cell_type_binary: Optional[str], compartment: Optional[str]) -> str:
    """Group of the hierarchical shrinkage: species | cell type | compartment (unknown values -> "unknown")."""
    def part(value, categories):
        return value if _known(value, categories) else "unknown"

    return "|".join((part(species, SPECIES), part(cell_type_binary, CELL_TYPE_BINARY), part(compartment, COMPARTMENT)))
