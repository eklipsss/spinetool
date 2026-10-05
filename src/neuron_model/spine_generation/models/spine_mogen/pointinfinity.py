"""PyTorch port of MoGen's PointInfinity + kNN velocity network.

Reference (JAX/Flax, Apache-2.0): google-research/connectomics,
``connectomics/mogen/models/pointinfinity.py`` + ``minimal_vit.py`` +
``minimal_network_utils.py``. Ported so the pretrained ``mouse_mixed``
checkpoint can be fine-tuned in the same PyTorch stack as the other models
(JAX-GPU is not supported on native Windows). Numerical equivalence with the
JAX implementation is checked by ``tests`` + the parity script (plan stage 4.3).

Details kept exactly as in the reference:

- point tokens: ``[p_i, p_j1 - p_i, ..., p_jk - p_i]`` with the kNN of the
  *current* noisy cloud; for ``N <= 8192`` the neighbours include the point
  itself (``spatial.knn``), for ``N > 8192`` they exclude it (``kdnn`` + drop first);
- time ``t`` -> sinusoidal embedding (``t * 1000``, max period 10000) -> MLP ->
  one extra latent token; condition ``cond`` -> Dense -> one more latent token
  (unconditional generation feeds zeros - as MoGen's own inference does);
- block = read (latents attend to points) -> MLP -> ``n_subblocks`` x
  (latent self-attention -> MLP) -> write (points attend to latents) -> MLP,
  pre-norm RMSNorm (eps 1e-6) everywhere, residual connections;
- attention with q/k RMSNorm + learned bias per head (``normalize_qk=True``),
  zero-initialised output projection and zero-initialised last MLP layer;
- GELU is the tanh approximation (Flax ``nn.gelu`` default).

Module names mirror the Flax parameter tree (``PointInfinityBlock_i``,
``MultiHeadDotProductAttention_j``, ...) so weight conversion is a direct
name mapping (``weights.py``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

RMS_EPS = 1e-6
KNN_EXCLUDE_SELF_ABOVE = 8192


@dataclass(frozen=True)
class PointInfinityConfig:
    """Defaults = ``mouse_mixed`` (``pfty_*`` in its config.json) = the spec's parameters."""

    point_dim: int = 128
    latent_dim: int = 256
    n_latents: int = 256
    n_blocks: int = 4
    n_subblocks: int = 2
    n_heads: int = 8
    k_nn: int = 16
    coord_dim: int = 3
    cond_dim: int = 0  # input width of the cond Dense (0 = no cond token); read from the checkpoint

    def to_dict(self):
        return asdict(self)

    @property
    def token_in_dim(self) -> int:
        return self.coord_dim * (1 + self.k_nn)


class RMSNorm(nn.Module):
    """Flax ``nn.RMSNorm``: ``x * rsqrt(mean(x^2) + eps) * scale`` (computed in float32)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)
        return (normed * self.scale.float()).to(x.dtype)


class RMSNormWithBias(nn.Module):
    """``minimal_vit.RMSNormWithBias`` (used as q/k norm, per head over head_dim)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.RMSNorm_0 = RMSNorm(dim)
        self.param = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.RMSNorm_0(x) + self.param


class MultiHeadDotProductAttention(nn.Module):
    """Flax MHDPA with ``normalize_qk=True``; q from ``inputs_q``, k/v from ``inputs_kv``.

    Projections map each input from its own width to ``num_heads x head_dim``
    where ``qkv_features = width(inputs_q)``; the output projection returns to
    ``width(inputs_q)``.
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int) -> None:
        super().__init__()
        if q_dim % num_heads:
            raise ValueError(f"q_dim {q_dim} not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.query = nn.Linear(q_dim, q_dim)
        self.key = nn.Linear(kv_dim, q_dim)
        self.value = nn.Linear(kv_dim, q_dim)
        self.query_ln = RMSNormWithBias(self.head_dim)
        self.key_ln = RMSNormWithBias(self.head_dim)
        self.out = nn.Linear(q_dim, q_dim)
        for layer in (self.query, self.key, self.value):
            nn.init.xavier_uniform_(layer.weight)  # Flax default lecun_normal ~ similar scale; only matters for scratch
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.out.weight)  # out_init = zeros_init()
        nn.init.zeros_(self.out.bias)

    def forward(self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor) -> torch.Tensor:
        b, lq, _ = inputs_q.shape
        lk = inputs_kv.shape[1]
        q = self.query_ln(self.query(inputs_q).view(b, lq, self.num_heads, self.head_dim))
        k = self.key_ln(self.key(inputs_kv).view(b, lk, self.num_heads, self.head_dim))
        v = self.value(inputs_kv).view(b, lk, self.num_heads, self.head_dim)
        # scaled_dot_product_attention expects [B, H, L, D]; its default scale is 1/sqrt(D) as in Flax
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(x.transpose(1, 2).reshape(b, lq, self.num_heads * self.head_dim))


class MlpBlock(nn.Module):
    """Dense(4d) -> GELU(tanh) -> Dense(d, zero-init)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.Dense_0 = nn.Linear(dim, 4 * dim)
        self.Dense_1 = nn.Linear(4 * dim, dim)
        nn.init.xavier_uniform_(self.Dense_0.weight)
        nn.init.zeros_(self.Dense_0.bias)
        nn.init.zeros_(self.Dense_1.weight)  # kernel_init = zeros
        nn.init.zeros_(self.Dense_1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Dense_1(F.gelu(self.Dense_0(x), approximate="tanh"))


class PointInfinityBlock(nn.Module):
    """read -> MLP -> n_subblocks x (self-attn -> MLP) -> write -> MLP (names follow Flax creation order)."""

    def __init__(self, cfg: PointInfinityConfig) -> None:
        super().__init__()
        p, l, h, s = cfg.point_dim, cfg.latent_dim, cfg.n_heads, cfg.n_subblocks
        self.n_subblocks = s
        # read
        self.MultiHeadDotProductAttention_0 = MultiHeadDotProductAttention(l, p, h)
        self.RMSNorm_0 = RMSNorm(l)  # z (queries)
        self.RMSNorm_1 = RMSNorm(p)  # x (keys/values)
        self.MlpBlock_0 = MlpBlock(l)
        self.RMSNorm_2 = RMSNorm(l)
        # latent self-attention subblocks: RMSNorm_{3+2i} (z_normed), MHDPA_{1+i}, MlpBlock_{1+i}, RMSNorm_{4+2i}
        for i in range(s):
            setattr(self, f"RMSNorm_{3 + 2 * i}", RMSNorm(l))
            setattr(self, f"MultiHeadDotProductAttention_{1 + i}", MultiHeadDotProductAttention(l, l, h))
            setattr(self, f"MlpBlock_{1 + i}", MlpBlock(l))
            setattr(self, f"RMSNorm_{4 + 2 * i}", RMSNorm(l))
        # write
        w = 3 + 2 * s
        self.write_attn_name = f"MultiHeadDotProductAttention_{1 + s}"
        setattr(self, self.write_attn_name, MultiHeadDotProductAttention(p, l, h))
        setattr(self, f"RMSNorm_{w}", RMSNorm(p))      # x (queries)
        setattr(self, f"RMSNorm_{w + 1}", RMSNorm(l))  # z (keys/values)
        setattr(self, f"MlpBlock_{1 + s}", MlpBlock(p))
        setattr(self, f"RMSNorm_{w + 2}", RMSNorm(p))
        self._w = w

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        s, w = self.n_subblocks, self._w
        z = z + self.MultiHeadDotProductAttention_0(self.RMSNorm_0(z), self.RMSNorm_1(x))
        z = z + self.MlpBlock_0(self.RMSNorm_2(z))
        for i in range(s):
            z_normed = getattr(self, f"RMSNorm_{3 + 2 * i}")(z)
            z = z + getattr(self, f"MultiHeadDotProductAttention_{1 + i}")(z_normed, z_normed)
            z = z + getattr(self, f"MlpBlock_{1 + i}")(getattr(self, f"RMSNorm_{4 + 2 * i}")(z))
        x = x + getattr(self, self.write_attn_name)(getattr(self, f"RMSNorm_{w}")(x), getattr(self, f"RMSNorm_{w + 1}")(z))
        x = x + getattr(self, f"MlpBlock_{1 + s}")(getattr(self, f"RMSNorm_{w + 2}")(x))
        return x, z


def timestep_embedding(t: torch.Tensor, dim: int, max_time: float = 1.0) -> torch.Tensor:
    """``minimal_network_utils.get_timestep_embedding`` (``[sin, cos]``, t scaled by 1000/max_time)."""
    t = t.float() * (1000.0 / max_time)
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -(math.log(10000.0) / (half - 1)))
    emb = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def knn_indices(coord: torch.Tensor, k: int) -> torch.Tensor:
    """``[B, N, k]`` nearest-neighbour indices; self included for N <= 8192, excluded above (as MoGen).

    Dense ``[B, N, N]`` distances like the reference ``spatial.knn``: at N=8192
    that is 256 MB per sample in fp32 - keep the per-step batch modest (or use
    gradient accumulation) on the 24 GB card.
    """
    n = coord.shape[1]
    exclude_self = n > KNN_EXCLUDE_SELF_ABOVE
    # Direct difference (matches connectomics.jax.spatial.squared_distances), not the
    # expand-of-squares identity (sq_a + sq_b - 2ab): the latter cancels two large,
    # nearly-equal terms down to the tiny true distance, which is numerically unstable
    # in fp32 and was observed to occasionally pick a different nearest neighbour than
    # the reference JAX implementation (check_mogen_parity.py) - same asymptotic cost
    # (still an [B, N, N] matrix), just computed in a stable order.
    dist = ((coord[:, :, None, :] - coord[:, None, :, :]) ** 2).sum(-1)
    idx = torch.topk(-dist, k + int(exclude_self), dim=-1).indices
    return idx[..., 1:] if exclude_self else idx


def knn_features(coord: torch.Tensor, k: int) -> torch.Tensor:
    """``[B, N, 3k]`` relative neighbour offsets ``p_j - p_i`` (neighbour order = nearest first)."""
    b, n, d = coord.shape
    idx = knn_indices(coord, k)
    neighbours = coord[torch.arange(b, device=coord.device)[:, None, None], idx]  # [B, N, k, 3]
    return (neighbours - coord[:, :, None, :]).reshape(b, n, k * d)


class PointInfinity(nn.Module):
    """Velocity network ``v_theta(x_t, t[, cond])`` -> ``[B, N, 3]``."""

    def __init__(self, cfg: PointInfinityConfig = PointInfinityConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        l = cfg.latent_dim
        self.MlpBlock_0 = MlpBlock(l)  # time embedding MLP
        self.Dense_0 = nn.Linear(cfg.cond_dim, l) if cfg.cond_dim > 0 else None  # cond -> latent token
        self.Dense_1 = nn.Linear(cfg.token_in_dim, cfg.point_dim)  # point tokenizer
        self.Embed_0 = nn.Embedding(cfg.n_latents, l)
        self.blocks = nn.ModuleList(PointInfinityBlock(cfg) for _ in range(cfg.n_blocks))
        self.Dense_2 = nn.Linear(cfg.point_dim, cfg.coord_dim)  # velocity head
        nn.init.normal_(self.Embed_0.weight, std=l ** -0.5)  # Flax Embed default: variance scaling 1/fan_in

    def forward(self, coord: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = coord.shape[0]
        t_emb = self.MlpBlock_0(timestep_embedding(t.reshape(b), self.cfg.latent_dim)[:, None, :])[:, 0, :]
        x = coord
        if self.cfg.k_nn > 0:
            x = torch.cat([x, knn_features(coord, self.cfg.k_nn)], dim=-1)
        x = self.Dense_1(x)
        z = self.Embed_0.weight[None].expand(b, -1, -1)
        tokens = [z, t_emb[:, None, :]]
        if self.Dense_0 is not None:
            if cond is None:
                cond = coord.new_zeros(b, self.cfg.cond_dim)  # unconditional: all-zero cond (= fully dropped)
            tokens.append(self.Dense_0(cond)[:, None, :])
        elif cond is not None:
            raise ValueError("model was built without a cond input (cond_dim=0)")
        z = torch.cat(tokens, dim=1)
        for block in self.blocks:
            x, z = block(x, z)
        return self.Dense_2(x)
