"""Joint fine-tuning of the VAE decoder and per-spine codes (auto-decoder, DeepSDF style).

Why (2026-11-08, chains K and L, dev-val evaluation 2026-11-08_180137_generator-evaluation-anchored-val):
- with the encoder's codes (z = mu) the decoder fills the necks (ConvexHullRatio -0.6 SD), with the code of each spine fitted
  directly (decoder frozen) the necks are there (bias 0.07-0.09) -> the encoder is the bottleneck;
- but a flow prior on fitted codes, unpenalised (chain K) or anchored to the posterior (w = 1 nm, chain L), does not give
  necks in GENERATION: the single-component meshes are as convex as with encoder codes (ConvexHullRatio -0.69...-0.86 vs
  -0.69, the real_mc control -0.29) and 27-42 % of the meshes break at the neck (head detached). A fitted code is good for
  ITS spine; the frozen decoder's field around it either fills or breaks the neck.
So the decoder itself must learn necks on the codes the prior is fitted to: decoder and codes are trained together on the
training spines (the encoder is not used), starting from the base VAE decoder and the anchored codes.

Per batch of training spines i (their SDF samples, as the VAE training):

    z_i  = c_i + code_noise * sigma_i * eps                      (eps ~ N(0, I); 0 = plain DeepSDF)
    L    = SDF(decode(q, z_i)) + far + eikonal + normal          (weights of the base run's loss config)
         + anchor_weight_nm * mean_d(((c_i - mu_i) / sigma_i)^2)  (MAP penalty of the code fit, chain L)

mu_i / sigma_i are the encoder's posterior of spine i. ``code_noise = 1`` trains the decoder on z ~ N(c_i, sigma_i^2), the
same neighbourhood the VAE decoder was trained on, but around a free code instead of the encoder's mean (a semi-amortised
VAE): it asks the decoder to keep the neck in a neighbourhood of the code, which is what prior samples need.
The codes are an ``nn.Embedding(sparse=True)`` with ``SparseAdam``: a code is updated only when its spine is in the batch
(dense Adam would keep moving every code by momentum). Units: the loss lives in coord-scaled units as in training; the
anchor weight is in nm of SDF loss (multiplied by ``coord_scale``), as ``decoder_study.optimize_latents``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np


class _KeysOnly:
    """Stand-in for the point-cloud part of ``SDFDataset``: the auto-decoder needs only the spine key, not 8192 points."""

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = list(keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {"spine_key": self.keys[i]}


def query_dataset(index, data_cfg: Mapping[str, Any], far_bbox=None):
    """``SDFDataset`` of the spines of ``index`` that reads only the SDF samples (random queries per item, as training)."""
    from ...data.datasets import SDFDataset

    ds = SDFDataset(index, int(data_cfg["n_points"]), dict(data_cfg["query_counts"]), random_variant=True,
                    coord_scale=float(data_cfg["coord_scale"]), use_normals=bool(data_cfg.get("use_normals", True)),
                    variants=tuple(int(v) for v in data_cfg.get("variants", (1, 2, 3, 4))),
                    far_queries=int(data_cfg.get("far_queries", 0)), far_bbox=far_bbox,
                    far_clamp_nm=float(data_cfg.get("far_clamp_nm", 300.0)))
    ds.pointclouds = _KeysOnly(index.keys)
    return ds


def _cosine(warmup: int, total: int, floor: float) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

    return factor


def finetune_autodecoder(model, codes: np.ndarray, mu: np.ndarray, sigma: np.ndarray, keys: Sequence[str], dataset, *, weights,
                         coord_scale: float, steps: int, batch_size: int = 16, lr: float = 3e-5, lr_codes: float = 5e-3,
                         min_lr_ratio: float = 0.1, warmup_steps: int = 500, anchor_weight_nm: float = 1.0, code_noise: float = 0.0,
                         clip_norm: Optional[float] = 1.0, seed: int = 0, device: Any = "cpu", precision: Optional[str] = None,
                         num_workers: int = 0, log_every: int = 100, log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
                         checkpoint_every: Optional[int] = None, checkpoint_fn: Optional[Callable[[int, np.ndarray], None]] = None):
    """Train ``model.decoder`` and the codes (``codes [n, D]``, row i = ``keys[i]``) on ``dataset`` (``query_dataset``).

    The encoder and the latent head stay frozen (unused). Returns ``(codes [n, D] numpy, history)``; ``history`` holds a row
    every ``log_every`` steps: the training terms, and on the same batch WITHOUT gradient the SDF loss at the clean code
    and at a code perturbed by its posterior sigma (nm) - how fast the field degrades around the code - plus the median
    ((c - mu) / sigma)^2 of the batch and the mean code norm.
    """
    import torch

    from ...data.datasets import make_loader
    from ...experiment.training import autocast, infinite
    from .losses import eikonal_loss, far_field_loss, normal_loss, sdf_gradient, sdf_loss, surface_loss

    row_of = {k: i for i, k in enumerate(keys)}
    missing = [k for k in dataset.pointclouds.keys if k not in row_of]
    if missing:
        raise ValueError(f"{len(missing)} spines of the dataset have no code (e.g. {missing[0]})")
    device = torch.device(device)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.decoder.train()

    table = torch.nn.Embedding(len(keys), codes.shape[1], sparse=True).to(device)
    with torch.no_grad():
        table.weight.copy_(torch.as_tensor(codes, dtype=torch.float32))
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
    sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    opt_dec = torch.optim.Adam([p for p in model.decoder.parameters()], lr=lr)
    opt_code = torch.optim.SparseAdam(list(table.parameters()), lr=lr_codes)
    factor = _cosine(int(warmup_steps), int(steps), float(min_lr_ratio))
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, factor) for o in (opt_dec, opt_code)]
    lam = float(anchor_weight_nm) * coord_scale
    loader = make_loader(dataset, batch_size=batch_size, shuffle=True, seed=seed, num_workers=num_workers, drop_last=True)
    batches = infinite(loader)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    history: List[Dict[str, float]] = []
    started = time.time()

    def field_loss(q, target, z):
        pred = model.decode(q, z)
        return sdf_loss(pred, target, weights), pred

    for step in range(1, int(steps) + 1):
        batch = next(batches)
        idx = torch.as_tensor([row_of[k] for k in batch["spine_key"]], device=device)
        q = batch["query_points"].to(device)
        target = batch["sdf"].to(device)
        if weights.needs_gradient:
            q = q.detach().requires_grad_(True)
        c = table(idx)
        s = sig_t[idx]
        z = c if code_noise <= 0 else c + float(code_noise) * s * torch.randn(c.shape, generator=gen).to(device)
        with autocast(device, precision):
            loss_sdf, pred = field_loss(q, target, z)
            terms = {"sdf_loss": loss_sdf}
            total = weights.sdf * loss_sdf
            if weights.surface:
                terms["surface_loss"] = surface_loss(pred, batch["sample_type"].to(device))
                total = total + weights.surface * terms["surface_loss"]
            if weights.far and "far_points" in batch:
                terms["far_loss"] = far_field_loss(model.decode(batch["far_points"].to(device), z), batch["far_lower_bound"].to(device))
                total = total + weights.far * terms["far_loss"]
        if weights.needs_gradient:
            grad = sdf_gradient(pred.float(), q)
            terms["eikonal_loss"] = eikonal_loss(grad)
            terms["normal_loss"] = normal_loss(grad, batch["query_normals"].to(device), batch["query_has_normal"].to(device),
                                               batch["sample_type"].to(device))
            total = total + weights.eikonal * terms["eikonal_loss"] + weights.normal * terms["normal_loss"]
        dist2 = (((c - mu_t[idx]) / s) ** 2).mean(dim=1)
        terms["anchor_dist2"] = dist2.mean()
        if lam > 0:
            total = total + lam * terms["anchor_dist2"]
        opt_dec.zero_grad(set_to_none=True)
        opt_code.zero_grad(set_to_none=True)
        total.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), float(clip_norm))
        opt_dec.step()
        opt_code.step()
        for sch in scheds:
            sch.step()

        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                qd = q.detach()
                clean, _ = field_loss(qd, target, c.detach())
                eps = torch.randn(c.shape, generator=gen).to(device)
                noisy, _ = field_loss(qd, target, c.detach() + s * eps)
            row = {"step": step, "total_loss": float(total.detach()), **{k: float(v.detach()) for k, v in terms.items()},
                   "sdf_clean_nm": float(clean) / coord_scale, "sdf_sigma_perturbed_nm": float(noisy) / coord_scale,
                   "anchor_dist2_median": float(dist2.detach().median()), "code_norm_mean": float(c.detach().norm(dim=1).mean()),
                   "lr": float(opt_dec.param_groups[0]["lr"]), "elapsed_s": time.time() - started}
            history.append(row)
            if log_fn is not None:
                log_fn(row)
        if checkpoint_fn is not None and ((checkpoint_every and step % checkpoint_every == 0) or step == steps):
            checkpoint_fn(step, table.weight.detach().cpu().numpy())
    model.eval()
    for p in model.decoder.parameters():
        p.requires_grad_(False)
    return table.weight.detach().cpu().numpy().astype(np.float64), history
