"""PointNeXt encoder (Qian et al. 2022): hierarchical local aggregation -> global shape vector ``h``.

Local groups are built **during the forward pass** (spec requirement: radii /
neighbour counts / depth changeable by config, nothing precomputed):

    stem MLP on [xyz, normals?]
    for each stage: SetAbstraction (subsample by `stride` with FPS, group kNN,
                    features ++ relative offsets -> MLP -> max over neighbours)
                    + `blocks` InvResMLP (local aggregation + inverted-bottleneck
                    pointwise MLP, residual)
    permutation-invariant pooling (max, or max ++ avg) -> Linear -> h

Normalisation is LayerNorm over channels by default (stable with the small
per-GPU batches of 8192-point clouds); ``norm: batch`` switches to BatchNorm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PointNeXtConfig:
    input_points: int = 8192
    use_normals: bool = True
    base_width: int = 64
    stage_widths: Sequence[int] = (64, 128, 256, 512)
    strides: Sequence[int] = (1, 4, 4, 4)       # stage 0 keeps all points (stem stage)
    blocks: Sequence[int] = (0, 1, 1, 1)        # InvResMLP blocks per stage
    k_neighbors: int = 32
    expansion: int = 4
    pooling: str = "max"                        # "max" or "max_avg"
    global_feature_dim: int = 512
    sampler: str = "fps"                        # "fps" or "random" (faster, for ablation)
    norm: str = "layer"                         # "layer" or "batch"

    def __post_init__(self) -> None:
        n = len(self.stage_widths)
        if not (len(self.strides) == len(self.blocks) == n):
            raise ValueError("stage_widths, strides and blocks must have the same length")
        if self.pooling not in ("max", "max_avg") or self.sampler not in ("fps", "random") or self.norm not in ("layer", "batch"):
            raise ValueError(f"invalid pooling/sampler/norm: {self.pooling}/{self.sampler}/{self.norm}")


def farthest_point_sample(xyz: torch.Tensor, m: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """``[B, m]`` indices by iterative farthest point sampling (first point random)."""
    b, n, _ = xyz.shape
    idx = torch.empty(b, m, dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), float("inf"), device=xyz.device)
    farthest = torch.randint(n, (b,), device="cpu", generator=generator).to(xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(m):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return idx


def knn(query: torch.Tensor, points: torch.Tensor, k: int) -> torch.Tensor:
    """``[B, M, k]`` indices of the k nearest ``points`` for every ``query`` point."""
    return torch.cdist(query, points).topk(min(k, points.shape[1]), dim=-1, largest=False).indices


def gather(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``values [B, N, C]``, ``idx [B, ...]`` -> ``[B, ..., C]``."""
    b = values.shape[0]
    batch = torch.arange(b, device=values.device).view(b, *([1] * (idx.dim() - 1)))
    return values[batch, idx]


class ChannelNorm(nn.Module):
    """Norm over the last (channel) dim for tensors ``[..., C]``."""

    def __init__(self, channels: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(channels) if kind == "layer" else nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "layer":
            return self.norm(x)
        shape = x.shape
        return self.norm(x.reshape(-1, shape[-1])).reshape(shape)


def mlp(in_ch: int, out_ch: int, norm: str, act: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(in_ch, out_ch, bias=False), ChannelNorm(out_ch, norm)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    """Subsample -> group kNN -> [f_j, p_j - p_i] -> MLP -> max over the group."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, k: int, sampler: str, norm: str) -> None:
        super().__init__()
        self.stride, self.k, self.sampler = stride, k, sampler
        self.mlp = nn.Sequential(mlp(in_ch + 3, out_ch, norm), mlp(out_ch, out_ch, norm))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        b, n, _ = xyz.shape
        m = max(1, n // self.stride)
        if self.stride == 1:
            idx = torch.arange(n, device=xyz.device).expand(b, n)
        elif self.sampler == "fps":
            idx = farthest_point_sample(xyz, m)
        else:
            idx = torch.stack([torch.randperm(n, device=xyz.device)[:m] for _ in range(b)])
        new_xyz = gather(xyz, idx)
        group = knn(new_xyz, xyz, self.k)
        offsets = gather(xyz, group) - new_xyz.unsqueeze(2)
        grouped = torch.cat([gather(feat, group), offsets], dim=-1)
        return new_xyz, self.mlp(grouped).max(dim=2).values


class InvResMLP(nn.Module):
    """Local aggregation (kNN, max) + inverted-bottleneck pointwise MLP, with a residual connection."""

    def __init__(self, ch: int, k: int, expansion: int, norm: str) -> None:
        super().__init__()
        self.k = k
        self.local = mlp(ch + 3, ch, norm)
        self.pointwise = nn.Sequential(mlp(ch, ch * expansion, norm), mlp(ch * expansion, ch, norm, act=False))
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        group = knn(xyz, xyz, self.k)
        offsets = gather(xyz, group) - xyz.unsqueeze(2)
        local = self.local(torch.cat([gather(feat, group), offsets], dim=-1)).max(dim=2).values
        return self.act(feat + self.pointwise(local))


class PointNeXtEncoder(nn.Module):
    """``(xyz [B,N,3], normals [B,N,3] | None) -> h [B, global_feature_dim]``."""

    def __init__(self, cfg: PointNeXtConfig = PointNeXtConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        in_ch = 3 + (3 if cfg.use_normals else 0)
        self.stem = mlp(in_ch, cfg.base_width, cfg.norm)
        stages = []
        blocks = []
        ch = cfg.base_width
        for width, stride, n_blocks in zip(cfg.stage_widths, cfg.strides, cfg.blocks):
            stages.append(SetAbstraction(ch, width, stride, cfg.k_neighbors, cfg.sampler, cfg.norm))
            blocks.append(nn.ModuleList(InvResMLP(width, cfg.k_neighbors, cfg.expansion, cfg.norm) for _ in range(n_blocks)))
            ch = width
        self.stages = nn.ModuleList(stages)
        self.blocks = nn.ModuleList(blocks)
        pooled = ch * (2 if cfg.pooling == "max_avg" else 1)
        self.head = nn.Sequential(nn.Linear(pooled, cfg.global_feature_dim), nn.ReLU(inplace=True), nn.Linear(cfg.global_feature_dim, cfg.global_feature_dim))

    def forward(self, xyz: torch.Tensor, normals: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.use_normals:
            if normals is None:
                raise ValueError("encoder built with use_normals=True but no normals given")
            feat = torch.cat([xyz, normals], dim=-1)
        else:
            feat = xyz
        feat = self.stem(feat)
        for stage, blocks in zip(self.stages, self.blocks):
            xyz, feat = stage(xyz, feat)
            for block in blocks:
                feat = block(xyz, feat)
        pooled = feat.max(dim=1).values
        if self.cfg.pooling == "max_avg":
            pooled = torch.cat([pooled, feat.mean(dim=1)], dim=-1)
        return self.head(pooled)
