"""SIREN SDF decoder ``f_theta(q, z[, e(c)]) -> d`` (Sitzmann et al. 2020).

Hidden layers ``h_{l+1} = sin(omega * (W h_l + b))``, linear output. The
special initialisation is mandatory: first layer ``U(-1/in, 1/in)``, later
layers ``U(-sqrt(6/in)/omega, sqrt(6/in)/omega)`` - it keeps activations and
gradients at a stable scale; without it the network either over-smooths the
surface or becomes numerically unstable (spec).

``cond_dim > 0`` reserves the conditional input ``e(c)`` (zeros in the first,
unconditional version; spec: interfaces must allow adding it later).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class SirenConfig:
    hidden_dim: int = 256
    hidden_layers: int = 5
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0
    output_dim: int = 1
    cond_dim: int = 0


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega_0: float, is_first: bool) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            bound = 1.0 / in_dim if is_first else math.sqrt(6.0 / in_dim) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenSDFDecoder(nn.Module):
    def __init__(self, latent_dim: int, cfg: SirenConfig = SirenConfig(), coord_dim: int = 3) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = coord_dim + latent_dim + cfg.cond_dim
        layers = [SineLayer(in_dim, cfg.hidden_dim, cfg.first_omega_0, is_first=True)]
        layers += [SineLayer(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_omega_0, is_first=False) for _ in range(cfg.hidden_layers - 1)]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / cfg.hidden_dim) / cfg.hidden_omega_0
            self.out.weight.uniform_(-bound, bound)

    def forward(self, query: torch.Tensor, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``query [B, M, 3]``, ``z [B, D]`` (``cond [B, C]``) -> ``sdf [B, M]``."""
        b, m, _ = query.shape
        parts = [query, z[:, None, :].expand(b, m, z.shape[-1])]
        if self.cfg.cond_dim:
            cond = query.new_zeros(b, self.cfg.cond_dim) if cond is None else cond
            parts.append(cond[:, None, :].expand(b, m, self.cfg.cond_dim))
        elif cond is not None:
            raise ValueError("decoder built without a conditional input (cond_dim=0)")
        return self.out(self.net(torch.cat(parts, dim=-1))).squeeze(-1)
