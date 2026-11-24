"""Vector Neurons layers (Deng et al., ICCV 2021, arXiv 2104.12229) and a VN-DGCNN point-cloud encoder.

A VN feature is a list of 3-D vectors: tensor ``[B, C, 3, ...]`` (C channels, the 3 = space). Every layer commutes with
a rotation Q applied to the input points (v -> Q v for every vector): linear maps mix channels only, the nonlinearity
and the normalisation act on vector lengths and on projections onto learned directions. Rotations are about the ORIGIN:
the clouds are not centred, the origin of the spine frame (the attachment centre) is meaningful.

Invariant features (VN paper, section 3.4): an equivariant frame R = [e1; e2; e3] (rows, orthonormalised from three
learned vector channels) gives ``V R^T`` (projections of every vector onto the frame), unchanged by rotations of the
input; R itself rotates with the input (R(Qx) = R(x) Q^T).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

EPS = 1e-6


class VNLinear(nn.Module):
    """[B, C_in, 3, ...] -> [B, C_out, 3, ...]: a channel-mixing matrix (no bias: a bias vector would break equivariance)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c_out, c_in) / c_in ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oc,bc...->bo...", self.weight, x)


class VNBatchNorm(nn.Module):
    """Batch normalisation of the vector LENGTHS per channel, directions kept."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=2) + EPS                       # [B, C, ...]
        shape = norm.shape
        normed = self.bn(norm.reshape(shape[0], shape[1], -1)).reshape(shape)
        return x / norm.unsqueeze(2) * normed.unsqueeze(2)


class VNLeakyReLU(nn.Module):
    """For every channel a learned direction k = W x; the component of x against k is removed where <x, k> < 0
    (leaky with slope ``negative_slope``)."""

    def __init__(self, channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.direction = VNLinear(channels, channels)
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.direction(x)
        dot = (x * d).sum(dim=2, keepdim=True)
        d_norm2 = (d * d).sum(dim=2, keepdim=True) + EPS
        projected = x - (dot / d_norm2) * d
        out = torch.where(dot >= 0, x, projected)
        return self.negative_slope * x + (1.0 - self.negative_slope) * out


class VNBlock(nn.Module):
    """VNLinear -> VNBatchNorm -> VNLeakyReLU."""

    def __init__(self, c_in: int, c_out: int, negative_slope: float = 0.2, batch_norm: bool = True) -> None:
        super().__init__()
        self.linear = VNLinear(c_in, c_out)
        self.bn = VNBatchNorm(c_out) if batch_norm else nn.Identity()
        self.act = VNLeakyReLU(c_out, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.linear(x)))


def knn_edge_features(points: torch.Tensor, k: int) -> torch.Tensor:
    """points [B, N, 3] -> VN edge features [B, 2, 3, N, k]: (x_j - x_i, x_i) for the k nearest neighbours j of i
    (the point itself excluded). Distances directly as |a - b| (a^2 + b^2 - 2ab loses precision on close points)."""
    b, n, _ = points.shape
    k = min(int(k), n - 1)
    dist = torch.cdist(points, points)                                   # [B, N, N]
    idx = dist.topk(k + 1, dim=-1, largest=False).indices[..., 1:]       # [B, N, k]
    nbr = torch.gather(points.unsqueeze(1).expand(b, n, n, 3), 2, idx.unsqueeze(-1).expand(b, n, k, 3))  # [B, N, k, 3]
    center = points.unsqueeze(2).expand(b, n, k, 3)
    edge = torch.stack([nbr - center, center], dim=1)                    # [B, 2, N, k, 3]
    return edge.permute(0, 1, 4, 2, 3).contiguous()                      # [B, 2, 3, N, k]


def gram_schmidt(v: torch.Tensor) -> torch.Tensor:
    """v [B, 3, 3] (three vectors as rows) -> rotation R [B, 3, 3] with rows e1, e2, e3 = e1 x e2 (det +1)."""
    e1 = v[:, 0] / (v[:, 0].norm(dim=-1, keepdim=True) + EPS)
    u2 = v[:, 1] - (v[:, 1] * e1).sum(-1, keepdim=True) * e1
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + EPS)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=1)


def rotation_from_6d(x: torch.Tensor) -> torch.Tensor:
    """[B, 6] (first two rows of a rotation, possibly perturbed) -> rotation [B, 3, 3] (Zhou et al. 2019 continuity)."""
    return gram_schmidt(torch.cat([x.reshape(-1, 2, 3), torch.zeros(x.shape[0], 1, 3, dtype=x.dtype, device=x.device)], dim=1))


class VNDGCNNEncoder(nn.Module):
    """VN-DGCNN-lite: one VN edge convolution on the k-NN graph (mean over neighbours), point-wise VN blocks with skip
    concatenation, mean pooling over points -> global equivariant features [B, C, 3]; a VN layer gives three vectors
    for the frame R. Returns (global features [B, C, 3], frame R [B, 3, 3])."""

    def __init__(self, channels: Sequence[int] = (21, 42, 85, 170), k: int = 16, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.k = int(k)
        self.edge = VNBlock(2, channels[0], negative_slope)
        self.blocks = nn.ModuleList(VNBlock(channels[i], channels[i + 1], negative_slope) for i in range(len(channels) - 1))
        self.out_channels = int(sum(channels))
        self.frame = nn.Sequential(VNBlock(self.out_channels, self.out_channels // 2, negative_slope, batch_norm=False),
                                   VNLinear(self.out_channels // 2, 3))

    def forward(self, points: torch.Tensor):
        x = self.edge(knn_edge_features(points, self.k)).mean(dim=-1)    # [B, C0, 3, N]
        feats = [x]
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        g = torch.cat(feats, dim=1).mean(dim=-1)                          # [B, C, 3]
        frame = gram_schmidt(self.frame(g))                               # [B, 3, 3], rows rotate with the input
        return g, frame
