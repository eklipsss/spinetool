"""VAE losses (spec, "Функция потерь VAE").

Baseline: ``L = L_sdf + beta * L_KL``. After a stable baseline the geometric
regularisers are switched on ONE AT A TIME (weights 0 = off) to measure each
one's contribution::

    L = l_sdf L_sdf + l_surface L_surface + l_eik L_eikonal + l_normal L_normal + beta L_KL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn import functional as F

from ...data.datasets import SAMPLE_SURFACE


@dataclass(frozen=True)
class LossWeights:
    sdf: float = 1.0
    surface: float = 0.0
    eikonal: float = 0.0
    normal: float = 0.0
    beta: float = 1e-3
    kl_warmup_steps: int = 0
    free_bits: Optional[float] = None
    sdf_loss: str = "l1"          # "l1" or "smooth_l1"
    smooth_l1_beta: float = 0.01  # model units
    sdf_clamp: Optional[float] = None  # DeepSDF-style truncation of |d| (model units); None = off

    @property
    def needs_gradient(self) -> bool:
        return self.eikonal > 0 or self.normal > 0


def sdf_loss(pred: torch.Tensor, target: torch.Tensor, weights: LossWeights) -> torch.Tensor:
    if weights.sdf_clamp is not None:
        pred = pred.clamp(-weights.sdf_clamp, weights.sdf_clamp)
        target = target.clamp(-weights.sdf_clamp, weights.sdf_clamp)
    if weights.sdf_loss == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=weights.smooth_l1_beta)
    if weights.sdf_loss == "l1":
        return (pred - target).abs().mean()
    raise ValueError(f"Unknown sdf_loss {weights.sdf_loss!r}")


def surface_loss(pred: torch.Tensor, sample_type: torch.Tensor) -> torch.Tensor:
    mask = sample_type == SAMPLE_SURFACE
    return pred[mask].abs().mean() if mask.any() else pred.new_zeros(())


def sdf_gradient(pred: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """``d pred / d query`` with a graph kept for the regularisers' own backward pass."""
    return torch.autograd.grad(pred, query, grad_outputs=torch.ones_like(pred), create_graph=True)[0]


def eikonal_loss(grad: torch.Tensor) -> torch.Tensor:
    return (grad.norm(dim=-1) - 1.0).pow(2).mean()


def normal_loss(grad: torch.Tensor, normals: torch.Tensor, has_normal: torch.Tensor, sample_type: torch.Tensor) -> torch.Tensor:
    """``1 - cos(grad f, n)`` on surface samples that carry a mesh normal."""
    mask = has_normal & (sample_type == SAMPLE_SURFACE)
    if not mask.any():
        return grad.new_zeros(())
    return (1.0 - F.cosine_similarity(grad[mask], normals[mask], dim=-1)).mean()


def combine(terms: Dict[str, torch.Tensor], weights: LossWeights, kl_w: float) -> torch.Tensor:
    total = weights.sdf * terms["sdf_loss"] + kl_w * terms["kl_loss"]
    if weights.surface:
        total = total + weights.surface * terms["surface_loss"]
    if weights.eikonal:
        total = total + weights.eikonal * terms["eikonal_loss"]
    if weights.normal:
        total = total + weights.normal * terms["normal_loss"]
    return total
