"""Branch <-> feature sequence (t-module-technical-spec.md, section 3).

A branch is a sequence of nodes every ~500 nm carrying three numbers (kappa1, kappa2, log r): bending of the
rotation-minimising frame (nm^-1) and log radius. Global coordinates are NOT part of the representation: they
contain the arbitrary position / orientation of the branch. The geometry is rebuilt by integrating the frame
(``branch_geometry.integrate_bending``) from the canonical start (gamma_0 = 0, R_0 = I), and the frame is then
re-aligned so that e1(0) is the bending direction (the convention the preprocessing uses for theta).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .. import branch_geometry as bg

CHANNELS = ("kappa1", "kappa2", "log_radius")
M_MAX = 512


@dataclass
class BranchGeometry:
    """Preprocessed geometry of one merged branch (``branch_geometry.npz``)."""

    s: np.ndarray  # [M] nm
    gamma: np.ndarray  # [M,3]
    radius: np.ndarray  # [M]
    tangent: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    kappa1: np.ndarray  # [M], last node NaN
    kappa2: np.ndarray

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def step(self) -> float:
        return float(self.s[1] - self.s[0])

    @property
    def n_nodes(self) -> int:
        return len(self.s)


def load_geometry(path: Path) -> BranchGeometry:
    d = np.load(path)
    return BranchGeometry(
        s=d["s"], gamma=d["gamma"], radius=d["radius"], tangent=d["frame_tangent"], e1=d["frame_e1"], e2=d["frame_e2"],
        kappa1=d["kappa1"], kappa2=d["kappa2"],
    )


def sequence_features(geom: BranchGeometry) -> np.ndarray:
    """[M,3] (kappa1, kappa2, log r); the bending of the last node is undefined (NaN)."""
    return np.column_stack([geom.kappa1, geom.kappa2, np.log(geom.radius)])


def valid_step_mask(n_nodes: int) -> np.ndarray:
    """Nodes that carry a bending value (all but the last)."""
    mask = np.ones(n_nodes, dtype=bool)
    mask[-1] = False
    return mask


def pad_sequence(features: np.ndarray, m_max: int = M_MAX) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X [m_max,3] zero padded with NaN -> 0, node mask [m_max], step mask [m_max])."""
    m = len(features)
    if m > m_max:
        raise ValueError(f"branch has {m} nodes > M_MAX={m_max}")
    x = np.zeros((m_max, features.shape[1]), dtype=np.float32)
    x[:m] = np.nan_to_num(features, nan=0.0)
    node = np.zeros(m_max, dtype=bool)
    node[:m] = True
    step = np.zeros(m_max, dtype=bool)
    step[: m - 1] = True
    return x, node, step


class FeatureStats:
    """Per-channel mean / std over the valid entries of the training sequences (NOT per object)."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    @classmethod
    def fit(cls, sequences: Sequence[np.ndarray]) -> "FeatureStats":
        stacked = np.vstack(sequences)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        return cls(mean, np.where(std > 0, std, 1.0))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, values: Dict[str, list]) -> "FeatureStats":
        return cls(values["mean"], values["std"])


def canonicalize_frames(gamma: np.ndarray, frames: np.ndarray, straight_fraction: float = 0.02) -> Tuple[np.ndarray, str]:
    """Rotate (e1, e2) about t by one angle so that e1(0) is the bending direction (RMF has zero twist, so a
    rigid rotation of the whole frame field keeps it an RMF). Returns (frames [m,3,3], reference method)."""
    t0 = frames[0][:, 0]
    target, method = bg.initial_normal(gamma, t0, straight_fraction)
    e1_0, e2_0 = frames[0][:, 1], frames[0][:, 2]
    phi = math.atan2(float(e2_0 @ target), float(e1_0 @ target))
    c, s = math.cos(phi), math.sin(phi)
    out = frames.copy()
    out[:, :, 1] = c * frames[:, :, 1] + s * frames[:, :, 2]
    out[:, :, 2] = -s * frames[:, :, 1] + c * frames[:, :, 2]
    return out, method


def geometry_from_features(kappa1: np.ndarray, kappa2: np.ndarray, log_radius: np.ndarray, step: float, *, straight_fraction: float = 0.02) -> Dict[str, np.ndarray]:
    """Rebuild a branch from features. ``kappa*`` carry M-1 steps (extra entries are ignored), ``log_radius`` M nodes."""
    m = len(log_radius)
    k1, k2 = np.asarray(kappa1, dtype=float)[: m - 1], np.asarray(kappa2, dtype=float)[: m - 1]
    gamma, frames = bg.integrate_bending(k1, k2, step)
    frames, reference = canonicalize_frames(gamma, frames, straight_fraction)
    return {
        "s": np.arange(m) * step,
        "gamma": gamma,
        "radius": np.exp(np.asarray(log_radius, dtype=float)),
        "tangent": frames[:, :, 0],
        "e1": frames[:, :, 1],
        "e2": frames[:, :, 2],
        "kappa1": np.r_[k1, np.nan],
        "kappa2": np.r_[k2, np.nan],
        "length": float((m - 1) * step),
        "frame_reference": reference,
    }
