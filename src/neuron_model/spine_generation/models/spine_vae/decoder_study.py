"""Which VAE decoder reconstructs real spines without a systematic shape bias?  (statistics of compare_vae_decoders.py)

The untouched-test evaluation (2026-11-06) showed that the gap between the VAE and real spines is the decoder, not the
Marching-Cubes grid nor the prior: against real spines through the SAME chain (exact SDF -> MC on the frozen bbox, the
``real_mc`` control) the classifier still reaches AUC 0.95 (0.98 at 256^3), and the reconstructions of real codes are
already biased (ConvexHullRatio -0.97 SD, i.e. too convex: necks filled; base area +0.88 SD). A prior cannot remove a bias
the decoder has on real codes, so decoder variants are compared on RECONSTRUCTIONS (z = mu) of the same validation spines.

The design is PAIRED: every spine has its reference (the real mesh through exact SDF + MC at the generation resolution) and
one reconstruction per decoder. Per metric m and decoder k:

    bias_k(m) = mean_i (x_ik - r_i) / sd(r)          (standardised paired bias; 0 = no systematic error)

with a cluster bootstrap over NEURONS (spines of one neuron are not independent). The summary score of a decoder is the
mean |bias| over the metrics. A variant is compared with the baseline by the paired difference of the scores within the
same bootstrap replicate (same resampled neurons for both), so the interval reflects only the difference.

Two independent trainings of the SAME recipe give the floor of training noise: a variant that differs from the baseline by
less than the two baselines differ from each other is not an improvement.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def surface_distances(a, b, *, n_points: int = 20_000, seed: int = 0) -> Dict[str, float]:
    """Symmetric surface error between two meshes (same units as the meshes): Chamfer-L1 (mean of the two mean distances)
    and the 95th percentile of the symmetric distance (HD95). ``nan`` when one mesh is missing."""
    from scipy.spatial import cKDTree

    if a is None or b is None or len(getattr(a, "faces", [])) == 0 or len(getattr(b, "faces", [])) == 0:
        return {"chamfer_l1": float("nan"), "hd95": float("nan")}
    import trimesh

    pa, _ = trimesh.sample.sample_surface(a, n_points, seed=seed)
    pb, _ = trimesh.sample.sample_surface(b, n_points, seed=seed + 1)
    dab, _ = cKDTree(pb).query(pa)
    dba, _ = cKDTree(pa).query(pb)
    return {"chamfer_l1": float(0.5 * (dab.mean() + dba.mean())), "hd95": float(np.quantile(np.r_[dab, dba], 0.95))}


def _paired(table: pd.DataFrame, kind: str, reference: str, metrics: Sequence[str]):
    ref = table[table["kind"] == reference].set_index("spine_key")
    rec = table[table["kind"] == kind].set_index("spine_key")
    keys = ref.index.intersection(rec.index)
    groups = ref.loc[keys, "neuron_id"].to_numpy()
    d = rec.loc[keys, list(metrics)].to_numpy(float) - ref.loc[keys, list(metrics)].to_numpy(float)
    sd = ref[list(metrics)].astype(float).std().to_numpy()
    return keys, groups, d, sd


def _cluster_means(d: np.ndarray, groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap of the column means of ``d`` [n, M] resampling whole groups; NaN-aware. Returns [n_boot, M]."""
    uniq = np.unique(groups)
    rows = [np.flatnonzero(groups == g) for g in uniq]
    out = np.empty((n_boot, d.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        with np.errstate(invalid="ignore"):
            out[b] = np.nanmean(d[idx], axis=0)
    return out


def paired_bias(table: pd.DataFrame, kinds: Sequence[str], *, reference: str, metrics: Sequence[str], baseline: Optional[str] = None,
                n_boot: int = 2000, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """``table``: one row per (spine_key, kind) with ``neuron_id`` and the metrics; ``reference`` is the kind every other
    kind is paired with. Returns ``per_metric`` (kind, metric, bias_sd + 95 % CI, median relative error, n) and
    ``summary`` (kind, mean |bias| over metrics + CI, and with ``baseline`` the paired difference of that score)."""
    metrics = [m for m in metrics if m in table.columns]
    rng = np.random.default_rng(seed)
    ref_keys = table.loc[table["kind"] == reference, ["spine_key", "neuron_id"]].drop_duplicates("spine_key")
    common = set(ref_keys["spine_key"])
    for k in kinds:
        common &= set(table.loc[table["kind"] == k, "spine_key"])
    sub = table[table["spine_key"].isin(common)]
    # one shared resampling of neurons for every kind -> differences between kinds are paired within a replicate
    groups = ref_keys.set_index("spine_key").loc[sorted(common), "neuron_id"].to_numpy()
    uniq = np.unique(groups)
    rows_of = [np.flatnonzero(groups == g) for g in uniq]
    draws = [np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]
    per_metric, summary, boot_score = [], [], {}
    for k in kinds:
        keys, _, d, sd = _paired(sub, k, reference, metrics)
        order = np.argsort(np.asarray(keys))  # align with sorted(common)
        z = d[order] / np.where(sd > 0, sd, np.nan)
        ref_vals = sub[sub["kind"] == reference].set_index("spine_key").loc[np.asarray(keys)[order], metrics].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bias = np.nanmean(z, axis=0)
            rel = np.nanmedian(np.abs(d[order]) / np.abs(ref_vals), axis=0)
            boot = np.stack([np.nanmean(z[idx], axis=0) for idx in draws])
        lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
        for j, m in enumerate(metrics):
            per_metric.append({"kind": k, "metric": m, "bias_sd": bias[j], "ci_low": lo[j], "ci_high": hi[j],
                               "significant": bool(lo[j] > 0 or hi[j] < 0), "median_rel_error": rel[j], "n": int(np.isfinite(z[:, j]).sum())})
        score_b = np.nanmean(np.abs(boot), axis=1)
        boot_score[k] = score_b
        summary.append({"kind": k, "mean_abs_bias_sd": float(np.nanmean(np.abs(bias))),
                        "ci_low": float(np.nanquantile(score_b, 0.025)), "ci_high": float(np.nanquantile(score_b, 0.975)),
                        "n_significant": int(sum(r["significant"] for r in per_metric if r["kind"] == k)), "n_metrics": len(metrics),
                        "n_spines": int(len(common)), "n_neurons": int(len(uniq))})
    summary = pd.DataFrame(summary)
    if baseline is not None and baseline in boot_score:
        for col in ("diff_vs_baseline", "diff_ci_low", "diff_ci_high"):
            summary[col] = np.nan
        for i, k in enumerate(summary["kind"]):
            delta = boot_score[k] - boot_score[baseline]
            summary.loc[i, "diff_vs_baseline"] = summary.loc[i, "mean_abs_bias_sd"] - float(summary.loc[summary["kind"] == baseline, "mean_abs_bias_sd"].iloc[0])
            summary.loc[i, ["diff_ci_low", "diff_ci_high"]] = np.nanquantile(delta, [0.025, 0.975])
    return {"per_metric": pd.DataFrame(per_metric), "summary": summary}


def cluster_mean_ci(values: pd.Series, groups: pd.Series, *, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean of a per-spine quantity (e.g. Chamfer) with a cluster-bootstrap 95 % interval over neurons."""
    ok = values.notna().to_numpy()
    v, g = values.to_numpy(float)[ok], groups.to_numpy()[ok]
    if not len(v):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    boot = _cluster_means(v[:, None], g, n_boot, np.random.default_rng(seed))[:, 0]
    return {"mean": float(v.mean()), "ci_low": float(np.nanquantile(boot, 0.025)), "ci_high": float(np.nanquantile(boot, 0.975)), "n": int(len(v))}


def optimize_latent(model, z0, sdf_path, *, coord_scale: float, sdf_clamp: Optional[float], far_bbox=None, steps: int = 300,
                    lr: float = 1e-2, n_far: int = 2048, far_clamp_nm: float = 300.0, far_slack_nm: float = 20.0, seed: int = 0, device="cpu"):
    """Auto-decoder fit of the code of ONE spine with the decoder FROZEN (diagnostic of the amortisation gap).

    Starts at the encoder's ``z0`` (= mu) and minimises the training SDF loss (clamped L1 on ALL samples of the spine's
    ``sdf_samples.npz``) plus the far-field hinge outside the spine's box (as in training, otherwise a free code may create
    surface where the decoder was never supervised). If the reconstructions of the optimised codes lose the bias, the
    decoder CAN represent the shape and the encoder (the amortised inference) is the bottleneck; if the bias remains, the
    decoder's capacity / representation is. Returns ``(z [D] numpy, {"loss_start", "loss_end"})``.
    """
    import torch

    from ...data.datasets import SAMPLE_SURFACE

    data = np.load(sdf_path)
    q_np = np.asarray(data["query_points"], dtype=np.float64)
    sample_type = np.asarray(data["sample_type"])
    q = torch.as_tensor(q_np * coord_scale, dtype=torch.float32, device=device).unsqueeze(0)
    target = torch.as_tensor(np.asarray(data["sdf"], dtype=np.float64) * coord_scale, dtype=torch.float32, device=device).unsqueeze(0)
    far_p = far_b = None
    if far_bbox is not None and n_far > 0:
        rng = np.random.default_rng(seed)
        surface = q_np[sample_type == SAMPLE_SURFACE]
        low, high = surface.min(axis=0) - far_slack_nm, surface.max(axis=0) + far_slack_nm
        g_low, g_high = np.asarray(far_bbox[0], dtype=np.float64), np.asarray(far_bbox[1], dtype=np.float64)
        p = g_low + rng.random((n_far, 3)) * (g_high - g_low)
        bound = np.minimum(np.maximum(np.maximum(low - p, p - high), 0.0).max(axis=1), far_clamp_nm)
        far_p = torch.as_tensor(p * coord_scale, dtype=torch.float32, device=device).unsqueeze(0)
        far_b = torch.as_tensor(bound * coord_scale, dtype=torch.float32, device=device).unsqueeze(0)

    for prm in model.parameters():
        prm.requires_grad_(False)
    z = torch.as_tensor(np.asarray(z0), dtype=torch.float32, device=device).reshape(1, -1).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    def loss_fn():
        pred, tgt = model.decode(q, z), target
        if sdf_clamp is not None:
            pred, tgt = pred.clamp(-sdf_clamp, sdf_clamp), tgt.clamp(-sdf_clamp, sdf_clamp)
        loss = (pred - tgt).abs().mean()
        if far_p is not None:
            active = far_b > 0
            if bool(active.any()):
                loss = loss + torch.relu(far_b - model.decode(far_p, z))[active].mean()
        return loss

    with torch.no_grad():
        start = float(loss_fn())
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
    with torch.no_grad():
        end = float(loss_fn())
    return z.detach()[0].cpu().numpy(), {"loss_start": start / coord_scale, "loss_end": end / coord_scale}
