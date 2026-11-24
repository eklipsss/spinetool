"""VN + FoldingNet autoencoder of spine point clouds (the architecture used in the lab for spine embeddings, 2026-11-24).

Encoder: Vector Neurons DGCNN (``vn.VNDGCNNEncoder``) -> rotation-INVARIANT code z (projections of the global VN
features onto the equivariant frame R, then an MLP). Decoder: FoldingNet (Yang et al., CVPR 2018): a fixed 2-D grid
(``grid: plane``, as in the paper) or points on a sphere (``grid: sphere``, closed surface without a seam) is folded twice
by MLPs conditioned on z into a 3-D cloud; any number of grid points can be decoded, so the model trains on 2048 points
and generates 8192 (the input of our Poisson chain d8_pw8).

Orientation (``mode``): our spines live in a frame tied to the dendrite (origin = attachment centre, +y away from the
trunk), so orientation is part of the shape (docs/neuron-model/24.11-bridge-matching.md).
  * ``invariant``  - the decoder output IS the cloud: the code cannot tell a spine from its rotated copy, so the decoder
                     has to guess the orientation from the invariant shape (the plain VN embedding; tests the claim that
                     an invariant encoder loses the tilt of the spine);
  * ``frame``      - the decoder outputs the cloud in the canonical frame, rotated back by R: p = p_can R. The
                     autoencoder is then fully rotation-equivariant and loses nothing; for generation the latent is
                     (z, first two rows of R) and the flow models both (``latent_vector`` / ``split_latent``).
Loss: symmetric Chamfer distance (squared, mean over both directions), in model units (coord_scale: nm -> um).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from .vn import VNDGCNNEncoder, rotation_from_6d


@dataclasses.dataclass
class VNFoldConfig:
    mode: str = "frame"                    # frame | invariant
    latent_dim: int = 512
    encoder_channels: Tuple[int, ...] = (21, 42, 85, 170)
    k_nn: int = 16
    grid: str = "plane"                    # plane | sphere
    grid_points: int = 2025                # training decode size (45 x 45)
    grid_extent: float = 0.3               # plane grid in [-extent, extent]^2 (FoldingNet)
    hidden_dim: int = 512

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["encoder_channels"] = list(self.encoder_channels)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VNFoldConfig":
        d = dict(d)
        if "encoder_channels" in d:
            d["encoder_channels"] = tuple(int(c) for c in d["encoder_channels"])
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})


def make_grid(n: int, kind: str, extent: float = 0.3, seed: int = 0) -> torch.Tensor:
    """[n, 2] plane grid (m x m with m = ceil(sqrt n), a seeded subset of n points if m^2 > n) or [n, 3] Fibonacci sphere."""
    if kind == "sphere":
        i = torch.arange(n, dtype=torch.float32) + 0.5
        phi = torch.acos(1 - 2 * i / n)
        theta = math.pi * (1 + 5 ** 0.5) * i
        return torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=1)
    if kind != "plane":
        raise ValueError(f"grid must be plane or sphere, got {kind!r}")
    m = int(math.ceil(math.sqrt(n)))
    lin = torch.linspace(-extent, extent, m)
    grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(-1, 2)
    if len(grid) > n:
        grid = grid[torch.randperm(len(grid), generator=torch.Generator().manual_seed(seed))[:n]]
    return grid


def _mlp(c_in: int, hidden: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, c_out, 1))


class FoldingDecoder(nn.Module):
    def __init__(self, latent_dim: int, grid_dim: int, hidden: int) -> None:
        super().__init__()
        self.fold1 = _mlp(latent_dim + grid_dim, hidden, 3)
        self.fold2 = _mlp(latent_dim + 3, hidden, 3)

    def forward(self, z: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """z [B, D], grid [n, g] -> points [B, n, 3]."""
        b, n = z.shape[0], grid.shape[0]
        zz = z.unsqueeze(-1).expand(b, z.shape[1], n)
        g = grid.T.unsqueeze(0).expand(b, grid.shape[1], n).to(z)
        x = self.fold1(torch.cat([zz, g], dim=1))
        x = self.fold2(torch.cat([zz, x], dim=1))
        return x.transpose(1, 2)


class VNFoldingAE(nn.Module):
    def __init__(self, cfg: VNFoldConfig) -> None:
        super().__init__()
        if cfg.mode not in ("frame", "invariant"):
            raise ValueError(f"mode must be frame or invariant, got {cfg.mode!r}")
        self.cfg = cfg
        self.encoder = VNDGCNNEncoder(cfg.encoder_channels, k=cfg.k_nn)
        c = self.encoder.out_channels
        self.to_code = nn.Sequential(nn.Linear(3 * c, 2 * cfg.latent_dim), nn.ReLU(), nn.Linear(2 * cfg.latent_dim, cfg.latent_dim))
        self.decoder = FoldingDecoder(cfg.latent_dim, 3 if cfg.grid == "sphere" else 2, cfg.hidden_dim)
        self.register_buffer("train_grid", make_grid(cfg.grid_points, cfg.grid, cfg.grid_extent), persistent=False)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """points [B, N, 3] -> (invariant code z [B, D], frame R [B, 3, 3])."""
        g, frame = self.encoder(points)
        invariant = torch.einsum("bci,bji->bcj", g, frame)               # projections onto e1, e2, e3: [B, C, 3]
        return self.to_code(invariant.reshape(invariant.shape[0], -1)), frame

    def decode(self, z: torch.Tensor, frame: Optional[torch.Tensor] = None, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        grid = self.train_grid if grid is None else grid.to(z.device)
        canonical = self.decoder(z, grid)
        if self.cfg.mode == "frame":
            if frame is None:
                raise ValueError("mode 'frame' needs the frame to decode")
            return canonical @ frame                                      # p = p_can R
        return canonical

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        z, frame = self.encode(points)
        return self.decode(z, frame)

    # ------------------------------------------------------------------ latent vector for the flow
    def latent_vector(self, z: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """The vector the generative prior models: z, plus the first two rows of R in mode 'frame' (6-D rotation)."""
        return torch.cat([z, frame[:, :2].reshape(-1, 6)], dim=1) if self.cfg.mode == "frame" else z

    def split_latent(self, v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        d = self.cfg.latent_dim
        if self.cfg.mode == "frame":
            return v[:, :d], rotation_from_6d(v[:, d:d + 6])
        return v[:, :d], None


def chamfer(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric squared Chamfer per cloud [B]: mean_x min_y |x - y|^2 + mean_y min_x |x - y|^2."""
    d = torch.cdist(pred, target) ** 2
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)
