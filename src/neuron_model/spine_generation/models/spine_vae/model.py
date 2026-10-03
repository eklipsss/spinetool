"""SpineVAE = PointNeXt encoder -> Gaussian latent -> SIREN SDF decoder.

- training: point cloud (+ normals) -> mu, log sigma^2 -> z (reparameterised)
  -> SDF at a minibatch of query points (the full 3D grid is never decoded in training);
- reconstruction: ``z = mu``; prior generation: ``z ~ N(0, I)`` (used for the
  comparison with Flow Matching).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from .latent import GaussianLatent, kl_loss, kl_weight, latent_statistics
from .losses import LossWeights, combine, eikonal_loss, normal_loss, sdf_gradient, sdf_loss, surface_loss
from .pointnext_encoder import PointNeXtConfig, PointNeXtEncoder
from .siren_sdf_decoder import SirenConfig, SirenSDFDecoder


@dataclass(frozen=True)
class SpineVAEConfig:
    latent_dim: int = 128
    encoder: PointNeXtConfig = field(default_factory=PointNeXtConfig)
    decoder: SirenConfig = field(default_factory=SirenConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SpineVAEConfig":
        encoder = dict(values.get("encoder", {}))
        encoder.pop("type", None)
        decoder = dict(values.get("decoder", {}))
        decoder.pop("type", None)
        for key in ("stage_widths", "strides", "blocks"):
            if key in encoder:
                encoder[key] = tuple(encoder[key])
        return cls(latent_dim=int(values.get("latent_dim", 128)), encoder=PointNeXtConfig(**encoder), decoder=SirenConfig(**decoder))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpineVAE(nn.Module):
    def __init__(self, cfg: SpineVAEConfig = SpineVAEConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = PointNeXtEncoder(cfg.encoder)
        self.latent = GaussianLatent(cfg.encoder.global_feature_dim, cfg.latent_dim)
        self.decoder = SirenSDFDecoder(cfg.latent_dim, cfg.decoder)

    def encode(self, points: torch.Tensor, normals: Optional[torch.Tensor] = None, *, sample: bool = True):
        return self.latent(self.encoder(points, normals if self.cfg.encoder.use_normals else None), sample=sample)

    def decode(self, query: torch.Tensor, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(query, z, cond)

    def sample_prior(self, n: int, *, generator: Optional[torch.Generator] = None, device=None) -> torch.Tensor:
        return torch.randn(n, self.cfg.latent_dim, generator=generator).to(device or next(self.parameters()).device)

    def compute_losses(self, batch: Mapping[str, torch.Tensor], weights: LossWeights, step: int, *, sample: bool = True) -> Dict[str, torch.Tensor]:
        """All loss terms + the weighted total (``total_loss``) for one batch of ``SDFDataset`` items."""
        z, mu, logvar = self.encode(batch["points"], batch.get("normals"), sample=sample)
        query = batch["query_points"]
        if weights.needs_gradient:
            query = query.detach().requires_grad_(True)
        pred = self.decode(query, z)
        terms: Dict[str, torch.Tensor] = {
            "sdf_loss": sdf_loss(pred, batch["sdf"], weights),
            "surface_loss": surface_loss(pred, batch["sample_type"]),
            "kl_loss": kl_loss(mu, logvar, weights.free_bits),
        }
        if weights.needs_gradient:
            grad = sdf_gradient(pred, query)
            terms["eikonal_loss"] = eikonal_loss(grad)
            terms["normal_loss"] = normal_loss(grad, batch["query_normals"], batch["query_has_normal"], batch["sample_type"])
        kl_w = kl_weight(step, weights.beta, weights.kl_warmup_steps)
        terms["total_loss"] = combine(terms, weights, kl_w)
        terms["kl_weight"] = torch.tensor(kl_w)
        terms["_mu"], terms["_logvar"] = mu.detach(), logvar.detach()
        return terms


def loss_logs(terms: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    logs = {k: float(v.detach()) for k, v in terms.items() if not k.startswith("_")}
    logs.update(latent_statistics(terms["_mu"], terms["_logvar"]))
    return logs
