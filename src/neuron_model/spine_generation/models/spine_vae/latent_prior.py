"""Latent space of a trained SpineVAE: encoding real spines, diagnostics of the latent, learned priors.

Why: generating with ``z ~ N(0, I)`` gave poor diversity and coverage, because with the tiny KL weight (beta 1e-4) the
encoder codes of real spines (the aggregate posterior q(z) = E_x q(z|x)) do not fill N(0, I) and prior samples fall into
"holes". A full-covariance Gaussian fitted to the codes recovered about half of the gap (2026-10-22). This module
provides:

- ``load_vae_run`` / ``encode_spines``: codes mu and log sigma^2 of real spines, for several point-cloud resamplings;
- ``latent_report``: does the code identify the spine (resampling noise and posterior width against the spread between
  spines, retrieval accuracy), how the aggregate posterior differs from N(0, I) (spectrum, effective rank), and how far
  samples of a prior land from the real codes (``hole_ratio``);
- priors with one interface (``log_prob``, ``sample``, ``state_dict``-like save/load): ``StandardNormalPrior``,
  ``GaussianPrior`` (full covariance) and ``FlowPrior`` - a RealNVP normalizing flow (affine coupling layers) that maps
  the codes to N(0, I) bijectively; sampling is ``eps ~ N(0, T^2 I) -> z = f^{-1}(eps)``.

All in model units of the latent (no physical units involved).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# loading and encoding
# ---------------------------------------------------------------------------


def load_vae_run(run_dir: Path, checkpoint: str = "latest", device: Any = None):
    """``(model, run, cfg, checkpoint_id, device)``; EMA weights when the checkpoint has them (as ``generate_vae.py``)."""
    from ...experiment.checkpoint import CheckpointManager
    from ...experiment.run import open_run
    from ...experiment.training import get_device
    from .model import SpineVAE, SpineVAEConfig

    run = open_run(Path(run_dir))
    cfg = run.load_config()
    model = SpineVAE(SpineVAEConfig.from_mapping(json.loads((run.root / "model_config.json").read_text(encoding="utf-8"))))
    payload = CheckpointManager(run.checkpoints).load(checkpoint, restore_rng=False)
    state = payload["state"]
    model.load_state_dict(state["ema"]["model"] if "ema" in state else state["model"])
    device = device if device is not None else get_device(cfg.get("device", "auto"))
    model.to(device).eval()
    return model, run, cfg, f"{run.experiment_id}:{checkpoint}:step{payload['step']}", device


def load_split_index(cfg: Dict[str, Any], split: str, n: int, seed: int, *, manifests: Optional[Sequence[Path]] = None,
                     splits: Optional[Path] = None, path_mode: Optional[str] = None):
    """``n`` random spines of ``split`` (the run's manifest / split unless overridden); spines outside the split dropped."""
    from ...data.index import load_spine_index
    from ...data.splits import load_splits
    from ...experiment.config import resolve_path

    data = cfg["data"]
    manifests = manifests or [resolve_path(p) for p in data["manifests"]]
    index = load_spine_index(manifests, path_mode=path_mode or data.get("path_mode", "relative"), check_files=())
    index = index.with_split(load_splits(splits or resolve_path(data["splits"])), split, drop_unassigned=True)
    if len(index) == 0:
        raise SystemExit(f"no spines in split {split!r}")
    order = np.random.default_rng(seed).permutation(len(index))[:n]
    return index.subset(index.frame["spine_key"].iloc[order])


@torch.no_grad()
def encode_spines(model, index, data_cfg: Dict[str, Any], *, variants: Sequence[int] = (1,), device: Any = "cpu",
                  batch_size: int = 16, progress: bool = False) -> Dict[str, Any]:
    """Encoder output for every spine of ``index`` and every point-cloud resampling variant.

    Returns ``mu [n, V, D]``, ``logvar [n, V, D]``, ``keys`` (n spine keys), ``variants``. Batched: with
    ``latent_mu_norm: batch`` the eval mode uses running statistics, so the batch size does not change the codes."""
    from ...data.datasets import PointCloudDataset

    mus, logvars = [], []
    for v in variants:
        ds = PointCloudDataset(index, int(data_cfg["n_points"]), variants=(int(v),), random_variant=False,
                               coord_scale=float(data_cfg["coord_scale"]), use_normals=bool(data_cfg.get("use_normals", True)))
        mu_v, lv_v = [], []
        starts = range(0, len(ds), batch_size)
        if progress:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc=f"encode variant {v}")
        for b in starts:
            items = [ds[i] for i in range(b, min(b + batch_size, len(ds)))]
            pts = torch.stack([it["points"] for it in items]).to(device)
            nrm = torch.stack([it["normals"] for it in items]).to(device) if "normals" in items[0] else None
            _, mu, logvar = model.encode(pts, nrm, sample=False)
            mu_v.append(mu.cpu().numpy())
            lv_v.append(logvar.cpu().numpy())
        mus.append(np.concatenate(mu_v))
        logvars.append(np.concatenate(lv_v))
    return {"mu": np.stack(mus, axis=1).astype(np.float64), "logvar": np.stack(logvars, axis=1).astype(np.float64),
            "keys": list(index.keys), "variants": [int(v) for v in variants]}


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _nn_distances(a: np.ndarray, b: np.ndarray, *, exclude_self: bool = False, chunk: int = 2048) -> np.ndarray:
    """Distance from every row of ``a`` to its nearest row of ``b`` (Euclidean)."""
    out = np.empty(len(a))
    b_sq = (b ** 2).sum(1)
    for s in range(0, len(a), chunk):
        block = a[s: s + chunk]
        d2 = (block ** 2).sum(1)[:, None] + b_sq[None, :] - 2.0 * block @ b.T
        if exclude_self:
            idx = np.arange(s, s + len(block))
            d2[np.arange(len(block)), idx] = np.inf
        out[s: s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def hole_ratio(samples: np.ndarray, codes: np.ndarray) -> Dict[str, float]:
    """How far prior samples land from the real codes, relative to how close real codes are to each other.

    ``median_ratio`` = median NN distance (sample -> code) / median NN distance (code -> other code). ~1: samples fall
    where codes are (the prior covers the data); >> 1: samples fall into empty regions ("holes"), the decoder is asked
    for codes it never saw. A prior that is too NARROW (all samples at the centre) also has a small ``median_ratio``,
    so ``coverage_ratio`` = median NN distance (code -> sample) / the same base is the other half: large when real codes
    have no sample near them (missing diversity). Both depend on the number of samples: compare priors with the same
    number of samples and with real codes of the same number as the floor. Use codes NOT used to fit the prior
    (validation) for an honest number."""
    s2c = _nn_distances(samples, codes)
    c2c = _nn_distances(codes, codes, exclude_self=True)
    c2s = _nn_distances(codes, samples)
    base = float(np.median(c2c))
    return {"median_ratio": float(np.median(s2c) / base), "p90_ratio": float(np.quantile(s2c, 0.9) / base),
            "coverage_ratio": float(np.median(c2s) / base), "median_code_nn": base}


def _spectrum(x: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(np.cov(x, rowvar=False)))[::-1].clip(min=0.0)


def latent_report(mu: np.ndarray, logvar: np.ndarray, *, seed: int = 0, n_prior_samples: int = 4000) -> Dict[str, Any]:
    """Diagnostics of the codes ``mu [n, V, D]`` / ``logvar [n, V, D]`` (V >= 1 point-cloud resamplings per spine).

    - ``between_std``: per-dimension spread of the spine codes (mean over variants) - the signal;
    - ``resampling_to_between``: spread of mu of the SAME spine over resamplings / between-spine spread (median over
      dimensions). Small = the encoder sees the shape, not the sampling of points;
    - ``posterior_to_between``: posterior sigma / between-spine spread. Small = a sample z ~ q(z|x) stays at its own
      spine (the VAE is nearly an autoencoder); ~1 = the posterior covers other spines too;
    - ``retrieval_resampling`` / ``retrieval_posterior``: fraction of spines whose code from another resampling / whose
      posterior sample is nearest to their OWN mean code among all spines (1 = the code identifies the spine);
    - aggregate posterior vs N(0, I): mean norm, spectrum, effective rank (participation ratio), share of the first 1/5
      principal components, KL(fitted Gaussian || N(0, I));
    - ``hole_ratio`` of N(0, I) samples and of samples from a Gaussian fitted to half of the spines (scored against the
      other half)."""
    rng = np.random.default_rng(seed)
    n, n_var, dim = mu.shape
    centre = mu.mean(axis=1)                       # [n, D] code of a spine
    between = centre.std(axis=0)                    # [D]
    sigma = np.exp(0.5 * logvar).mean(axis=(0, 1))  # [D]
    report: Dict[str, Any] = {"n_spines": int(n), "n_variants": int(n_var), "latent_dim": int(dim)}
    safe = np.maximum(between, 1e-12)
    report["between_std_median"] = float(np.median(between))
    report["posterior_sigma_median"] = float(np.median(sigma))
    report["posterior_to_between_median"] = float(np.median(sigma / safe))
    report["active_dimensions"] = int((centre.var(axis=0) > 0.01 * centre.var(axis=0).max()).sum())
    if n_var > 1:
        within = mu.std(axis=1).mean(axis=0)
        report["resampling_to_between_median"] = float(np.median(within / safe))
        # retrieval: code of variant k against the mean code of the other variants
        hits = []
        for k in range(n_var):
            others = np.delete(mu, k, axis=1).mean(axis=1)
            q = mu[:, k]
            d2 = (q ** 2).sum(1)[:, None] + (others ** 2).sum(1)[None, :] - 2.0 * q @ others.T
            hits.append(np.mean(d2.argmin(1) == np.arange(n)))
        report["retrieval_resampling"] = float(np.mean(hits))
    z = centre + np.exp(0.5 * logvar).mean(axis=1) * rng.standard_normal(centre.shape)
    d2 = (z ** 2).sum(1)[:, None] + (centre ** 2).sum(1)[None, :] - 2.0 * z @ centre.T
    report["retrieval_posterior"] = float(np.mean(d2.argmin(1) == np.arange(n)))

    spec = _spectrum(centre)
    report["aggregate_mean_norm"] = float(np.linalg.norm(centre.mean(axis=0)))
    report["aggregate_total_variance"] = float(spec.sum())
    report["aggregate_effective_rank"] = float(spec.sum() ** 2 / max((spec ** 2).sum(), 1e-300))
    report["aggregate_top_eigenvalues"] = [float(v) for v in spec[:8]]
    report["aggregate_first_pc_share"] = float(spec[0] / max(spec.sum(), 1e-300))
    k5 = max(1, dim // 5)
    report["aggregate_top_fifth_share"] = float(spec[:k5].sum() / max(spec.sum(), 1e-300))
    gauss = GaussianPrior.fit(centre)
    report["kl_fitted_gaussian_to_standard_normal"] = float(gauss.kl_to_standard_normal())

    order = rng.permutation(n)
    fit_half, test_half = centre[order[: n // 2]], centre[order[n // 2:]]
    if len(test_half) >= 2 and len(fit_half) > dim:
        m = min(n_prior_samples, 4 * len(test_half))
        report["hole_ratio_standard_normal"] = hole_ratio(rng.standard_normal((m, dim)), test_half)
        report["hole_ratio_fitted_gaussian"] = hole_ratio(GaussianPrior.fit(fit_half).sample_np(m, rng=rng), test_half)
        report["hole_ratio_real_codes"] = hole_ratio(fit_half[: m], test_half)  # the floor: other real spines
    return report


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------


class StandardNormalPrior:
    kind = "standard_normal"

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        return -0.5 * ((z ** 2).sum(1) + self.dim * LOG_2PI)

    def sample(self, n: int, *, temperature: float = 1.0, seed: int = 0) -> np.ndarray:
        return temperature * np.random.default_rng(seed).standard_normal((n, self.dim))


class GaussianPrior:
    """Full-covariance Gaussian fitted to codes (``reg`` is added to the diagonal)."""

    kind = "gaussian"

    def __init__(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.chol = np.linalg.cholesky(self.cov)

    @classmethod
    def fit(cls, codes: np.ndarray, reg: float = 1e-6) -> "GaussianPrior":
        cov = np.cov(codes, rowvar=False) + reg * np.eye(codes.shape[1]) * max(np.trace(np.cov(codes, rowvar=False)) / codes.shape[1], 1e-12)
        return cls(codes.mean(axis=0), cov)

    @property
    def dim(self) -> int:
        return len(self.mean)

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        diff = np.linalg.solve(self.chol, (z - self.mean).T)
        logdet = 2.0 * np.log(np.diag(self.chol)).sum()
        return -0.5 * ((diff ** 2).sum(0) + logdet + self.dim * LOG_2PI)

    def sample_np(self, n: int, *, temperature: float = 1.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        return self.mean + temperature * rng.standard_normal((n, self.dim)) @ self.chol.T

    def sample(self, n: int, *, temperature: float = 1.0, seed: int = 0) -> np.ndarray:
        return self.sample_np(n, temperature=temperature, rng=np.random.default_rng(seed))

    def kl_to_standard_normal(self) -> float:
        d = self.dim
        logdet = 2.0 * np.log(np.diag(self.chol)).sum()
        return 0.5 * (np.trace(self.cov) + self.mean @ self.mean - d - logdet)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "mean": self.mean.tolist(), "cov": self.cov.tolist()}


class _Coupling(nn.Module):
    """Affine coupling: the dimensions with mask 1 pass unchanged and set scale/shift of the others."""

    def __init__(self, mask: torch.Tensor, hidden: int, depth: int) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        dim = mask.numel()
        layers: List[nn.Module] = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 2 * dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)  # identity at start
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x_masked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self.net(x_masked).chunk(2, dim=-1)
        s = torch.tanh(s) * (1.0 - self.mask)  # bounded log-scale: stable training
        return s, t * (1.0 - self.mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self._params(x * self.mask)
        return x * self.mask + (1.0 - self.mask) * (x * torch.exp(s) + t), s.sum(-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        s, t = self._params(y * self.mask)
        return y * self.mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))


class RealNVP(nn.Module):
    """z -> eps: fixed whitening (mean, Cholesky of the code covariance) followed by ``n_layers`` affine couplings with
    random half masks. Whitening first means the flow starts AS the fitted Gaussian and only learns the non-Gaussian
    remainder (it can never be worse than the Gaussian on the training codes)."""

    def __init__(self, dim: int, *, n_layers: int = 8, hidden: int = 256, depth: int = 2, seed: int = 0,
                 mean: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.dim, self.n_layers, self.hidden, self.depth, self.seed = int(dim), int(n_layers), int(hidden), int(depth), int(seed)
        gen = torch.Generator().manual_seed(seed)
        masks = []
        for _ in range(n_layers):
            m = torch.zeros(dim)
            m[torch.randperm(dim, generator=gen)[: dim // 2]] = 1.0
            masks.append(m)
        self.layers = nn.ModuleList(_Coupling(m, hidden, depth) for m in masks)
        mean = np.zeros(dim) if mean is None else mean
        cov = np.eye(dim) if cov is None else cov
        chol = np.linalg.cholesky(cov)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("chol", torch.as_tensor(chol, dtype=torch.float32))
        self.register_buffer("chol_inv", torch.as_tensor(np.linalg.inv(chol), dtype=torch.float32))
        self.register_buffer("log_det_whiten", torch.tensor(-float(np.log(np.diag(chol)).sum())))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(eps, log|det d eps / d z|)``."""
        x = (z - self.mean) @ self.chol_inv.T
        log_det = self.log_det_whiten.expand(z.shape[0]).clone()
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, eps: torch.Tensor) -> torch.Tensor:
        x = eps
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x @ self.chol.T + self.mean

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        eps, log_det = self(z)
        return -0.5 * ((eps ** 2).sum(-1) + self.dim * LOG_2PI) + log_det


class FlowPrior:
    kind = "flow"

    def __init__(self, flow: RealNVP) -> None:
        self.flow = flow.eval()

    @property
    def dim(self) -> int:
        return self.flow.dim

    @torch.no_grad()
    def log_prob(self, z: np.ndarray) -> np.ndarray:
        return self.flow.log_prob(torch.as_tensor(z, dtype=torch.float32)).double().numpy()

    @torch.no_grad()
    def sample(self, n: int, *, temperature: float = 1.0, seed: int = 0) -> np.ndarray:
        eps = temperature * torch.randn(n, self.dim, generator=torch.Generator().manual_seed(seed))
        return self.flow.inverse(eps).double().numpy()

    def save(self, path: Path) -> None:
        f = self.flow
        torch.save({"kind": self.kind, "dim": f.dim, "n_layers": f.n_layers, "hidden": f.hidden, "depth": f.depth, "seed": f.seed,
                    "state": f.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "FlowPrior":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        flow = RealNVP(payload["dim"], n_layers=payload["n_layers"], hidden=payload["hidden"], depth=payload["depth"], seed=payload["seed"])
        flow.load_state_dict(payload["state"])
        return cls(flow)


def fit_flow(train: np.ndarray, val: Optional[np.ndarray] = None, *, n_layers: int = 8, hidden: int = 256, depth: int = 2,
             epochs: int = 300, batch_size: int = 512, lr: float = 1e-3, weight_decay: float = 1e-5, patience: int = 30,
             noise_std: float = 0.0, seed: int = 0, log=print) -> Tuple[FlowPrior, Dict[str, Any]]:
    """Maximum likelihood fit of a RealNVP to codes ``train [n, D]``; early stopping on the validation NLL (best state kept).

    ``noise_std``: Gaussian noise added to every training batch in WHITENED units (dequantisation-like regularisation,
    keeps the flow from collapsing onto the finite set of training codes; 0.05-0.1 when the codes are few)."""
    torch.manual_seed(seed)
    gauss = GaussianPrior.fit(train)
    flow = RealNVP(train.shape[1], n_layers=n_layers, hidden=hidden, depth=depth, seed=seed, mean=gauss.mean, cov=gauss.cov)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    x_train = torch.as_tensor(train, dtype=torch.float32)
    x_val = None if val is None else torch.as_tensor(val, dtype=torch.float32)
    chol = flow.chol
    dim = train.shape[1]
    history: List[Dict[str, float]] = []
    best = (math.inf, None, -1)
    gen = torch.Generator().manual_seed(seed)

    # Curves with ONE estimator: eval mode, no noise, a fixed subset of the training codes of the size of the validation set,
    # the mean NLL per dimension and its standard error over codes. ``train_nll_per_dim`` (epoch mean in training mode with
    # noise and changing weights) is kept in the log only.
    n_eval = len(x_val) if x_val is not None else min(len(x_train), 1500)
    x_train_eval = x_train[torch.as_tensor(np.sort(np.random.default_rng(seed + 7).permutation(len(x_train))[:n_eval]))]

    def _nll_stats(x: torch.Tensor) -> Tuple[float, float]:
        with torch.no_grad():
            per_code = -flow.log_prob(x).double() / dim
        n = len(per_code)
        return float(per_code.mean()), float(per_code.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")

    flow.eval()
    init = {"train_eval_nll_per_dim": _nll_stats(x_train_eval)[0]}
    if x_val is not None:
        init["val_nll_per_dim"] = _nll_stats(x_val)[0]  # epoch "0": the flow is the fitted Gaussian
    for epoch in range(epochs):
        flow.train()
        perm = torch.randperm(len(x_train), generator=gen)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            batch = x_train[perm[s: s + batch_size]]
            if noise_std > 0:
                batch = batch + noise_std * torch.randn(batch.shape, generator=gen) @ chol.T
            loss = -flow.log_prob(batch).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            total += float(loss.detach()) * len(batch)
        sched.step()
        row = {"epoch": epoch, "train_nll_per_dim": total / len(x_train) / dim}
        flow.eval()
        row["train_eval_nll_per_dim"], row["train_eval_se"] = _nll_stats(x_train_eval)
        row["lr"] = float(opt.param_groups[0]["lr"])
        if x_val is not None:
            row["val_nll_per_dim"], row["val_se"] = _nll_stats(x_val)
            score = row["val_nll_per_dim"]
        else:
            score = row["train_nll_per_dim"]
        history.append(row)
        if score < best[0]:
            best = (score, {k: v.detach().clone() for k, v in flow.state_dict().items()}, epoch)
        if log and (epoch % 25 == 0 or epoch == epochs - 1):
            log(f"[flow] epoch {epoch}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"))
        if epoch - best[2] >= patience:
            if log:
                log(f"[flow] early stop at epoch {epoch} (best {best[2]})")
            break
    flow.load_state_dict(best[1])
    return FlowPrior(flow), {"best_epoch": best[2], "best_score_per_dim": best[0], "history": history, "init": init,
                             "stopped_epoch": len(history) - 1, "patience": patience, "n_eval_codes": int(n_eval)}


def nll_per_dim(prior, codes: np.ndarray) -> float:
    """Mean negative log-likelihood of ``codes`` under ``prior``, nats per latent dimension."""
    return float(-np.mean(prior.log_prob(codes)) / codes.shape[1])


def save_prior(prior, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(prior, FlowPrior):
        path = directory / "flow_prior.pt"
        prior.save(path)
    elif isinstance(prior, GaussianPrior):
        path = directory / "gaussian_prior.npz"
        np.savez(path, mean=prior.mean, cov=prior.cov)
    else:
        raise TypeError(type(prior))
    return path


def load_prior(path: Path):
    path = Path(path)
    if path.is_dir():
        path = path / "flow_prior.pt" if (path / "flow_prior.pt").exists() else path / "gaussian_prior.npz"
    if path.suffix == ".pt":
        return FlowPrior.load(path)
    data = np.load(path)
    return GaussianPrior(data["mean"], data["cov"])


def repeat_statistics(table, metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per metric, from a table with columns ``spine``, ``kind`` (real / mu / posterior / resampling) and the metrics:

    - ``between_sd``: sd over spines of the metric of the mesh decoded from z = mu;
    - ``within_<kind>_sd``: pooled sd over the repeats of one spine (mu + the ``kind`` repeats), RMS over spines;
    - ``within_to_between_<kind>`` and ``icc_<kind>`` = between^2 / (between^2 + within^2);
    - ``recon_bias_sd``: mean(mu - real) / sd(real); ``recon_median_rel_err``: median |mu - real| / |real|."""
    import pandas as pd

    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        real = table[table["kind"] == "real"].set_index("spine")[m].astype(float)
        mu_vals = table[table["kind"] == "mu"].set_index("spine")[m].astype(float)
        between = float(mu_vals.std())
        entry: Dict[str, float] = {"between_sd": between}
        for kind in ("posterior", "resampling"):
            sub = table[table["kind"].isin([kind, "mu"])]
            if not (sub["kind"] == kind).any():
                continue
            sds = sub.groupby("spine")[m].agg(lambda x: float(np.nanstd(x.astype(float), ddof=1)) if x.notna().sum() > 1 else np.nan)
            within = float(np.sqrt(np.nanmean(sds.to_numpy() ** 2))) if sds.notna().any() else float("nan")
            entry[f"within_{kind}_sd"] = within
            entry[f"within_to_between_{kind}"] = within / between if between > 0 else float("nan")
            entry[f"icc_{kind}"] = between ** 2 / (between ** 2 + within ** 2) if np.isfinite(within) and between > 0 else float("nan")
        pair = pd.concat([mu_vals.rename("mu"), real.rename("real")], axis=1).dropna()
        sd_real = float(real.std())
        entry["recon_bias_sd"] = float((pair["mu"] - pair["real"]).mean() / sd_real) if sd_real > 0 and len(pair) else float("nan")
        entry["recon_median_rel_err"] = float(((pair["mu"] - pair["real"]).abs() / pair["real"].abs()).median()) if len(pair) else float("nan")
        entry["n_pairs"] = int(len(pair))
        out[m] = entry
    return out
