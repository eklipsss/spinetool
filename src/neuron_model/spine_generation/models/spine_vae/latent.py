"""Diagonal-Gaussian VAE latent: h -> (mu, log sigma^2) -> z, KL, free bits, latent statistics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

LOGVAR_RANGE = (-20.0, 10.0)


class GaussianLatent(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)
        nn.init.zeros_(self.logvar.weight)  # start near the prior's unit variance
        nn.init.zeros_(self.logvar.bias)

    def forward(self, h: torch.Tensor, *, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(z, mu, logvar)``; ``sample=False`` gives the deterministic ``z = mu`` (reconstruction)."""
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(*LOGVAR_RANGE)
        if not sample:
            return mu, mu, logvar
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std), mu, logvar


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``[B, D]`` KL(q(z|x) || N(0, I)) per latent dimension."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: Optional[float] = None) -> torch.Tensor:
    """Batch mean of the KL summed over dimensions.

    ``free_bits`` (nats per dimension): dimensions whose batch-mean KL is below
    the threshold are not penalised (Kingma et al. 2016) - a remedy for
    posterior collapse, together with KL warm-up (``kl_weight``).
    """
    per_dim = kl_per_dim(mu, logvar).mean(dim=0)
    if free_bits:
        per_dim = torch.clamp(per_dim, min=float(free_bits))
    return per_dim.sum()


def kl_weight(step: int, beta: float, warmup_steps: int = 0) -> float:
    """Linear KL warm-up from 0 to ``beta`` over ``warmup_steps``."""
    if warmup_steps <= 0:
        return float(beta)
    return float(beta) * min(1.0, step / float(warmup_steps))


@torch.no_grad()
def latent_statistics(mu: torch.Tensor, logvar: torch.Tensor, active_threshold: float = 0.01) -> Dict[str, float]:
    """``mean_abs_mu``, ``mean_std_z``, ``active_latent_dimensions`` (Burda et al.: Var_x[mu_d] > threshold), raw KL."""
    std = torch.exp(0.5 * logvar)
    active = (mu.var(dim=0, unbiased=False) > active_threshold).sum() if mu.shape[0] > 1 else torch.tensor(float("nan"))
    return {
        "mean_abs_mu": float(mu.abs().mean()),
        "mean_std_z": float(std.mean()),
        "active_latent_dimensions": float(active),
        "kl_raw": float(kl_per_dim(mu, logvar).sum(dim=1).mean()),
    }
