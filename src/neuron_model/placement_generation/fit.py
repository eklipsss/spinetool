"""Fitting and validation of P0 (p-module-technical-spec.md, sections 4-6, 8).

``fit_p0`` fits every count variant of the grid (family x exposure x neuron effect) on the train branches, scores
them by the held-out negative log-likelihood and calibration on the validation branches, fits the positional
density (full covariate set + one ablation per covariate) and estimates d_min from the observed spines.
Returns a JSON-serialisable report and the chosen models. Units: nm.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .covariates import BranchCovariates, CountDesign
from .generate import surface_min_distance
from .models.count import CountModel, build_count_data
from .models.position import PositionBranch, PositionModel


def zero_fraction_by_length(model: CountModel, data, lengths: np.ndarray, rng: np.random.Generator, n_rep: int = 50, n_bins: int = 4) -> List[Dict[str, float]]:
    """Empirical vs model P(N = 0) and mean N in length quartiles (the model one by simulation, neuron effect drawn)."""
    sims = np.stack([model.sample(data, rng) for _ in range(n_rep)])
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        m = (lengths >= edges[i]) & (lengths <= edges[i + 1] if i == n_bins - 1 else lengths < edges[i + 1])
        if not m.any():
            continue
        out.append({"length_from_nm": float(edges[i]), "length_to_nm": float(edges[i + 1]), "n": int(m.sum()),
                    "zero_fraction_data": float((data.y[m] == 0).mean()), "zero_fraction_model": float((sims[:, m] == 0).mean()),
                    "mean_data": float(data.y[m].mean()), "mean_model": float(sims[:, m].mean())})
    return out


def dispersion_index(y: np.ndarray, lengths: np.ndarray, n_bins: int = 5) -> List[Dict[str, float]]:
    """Var/mean of the counts in length quantile bins (diagnostic only: > 1 overdispersion, < 1 underdispersion)."""
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        m = (lengths >= edges[i]) & (lengths <= edges[i + 1])
        if m.sum() < 5 or y[m].mean() <= 0:
            continue
        out.append({"length_from_nm": float(edges[i]), "length_to_nm": float(edges[i + 1]), "n": int(m.sum()), "D": float(y[m].var(ddof=1) / y[m].mean())})
    return out


def estimate_d_min(spines: pd.DataFrame, quantile: float = 1.0) -> Dict[str, Any]:
    """Small quantile of the nearest-neighbour surface distances sqrt(ds^2 + (r dtheta)^2) between attachment points of
    one branch (section 6), and the attachment radius sqrt(A_cap / pi) for comparison."""
    from ..branch_data import key_series

    sp = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].fillna(False).astype(bool)]
    nn = []
    for _, g in sp.groupby(key_series(sp)):
        if len(g) < 2:
            continue
        s, th, r = (g[c].to_numpy(dtype=float) for c in ("s", "theta", "radius_at_s"))
        ds = s[:, None] - s[None, :]
        dth = np.abs(th[:, None] - th[None, :])
        dth = np.minimum(dth, 2 * np.pi - dth)
        d = np.sqrt(ds ** 2 + (0.5 * (r[:, None] + r[None, :]) * dth) ** 2)
        np.fill_diagonal(d, np.inf)
        nn.append(d.min(axis=1))
    nn_all = np.concatenate(nn) if nn else np.zeros(0)
    cap = sp["attachment_cap_area"].to_numpy(dtype=float) if "attachment_cap_area" in sp.columns else np.zeros(0)
    cap = cap[np.isfinite(cap)]
    return {
        "d_min_nm": float(np.percentile(nn_all, quantile)) if len(nn_all) else 0.0,
        "quantile_percent": quantile,
        "nn_distance_median_nm": float(np.median(nn_all)) if len(nn_all) else None,
        "n_pairs": int(len(nn_all)),
        "attachment_radius_median_nm": float(np.median(np.sqrt(cap / np.pi))) if len(cap) else None,
    }


def fit_p0(
    train_cov: Sequence[BranchCovariates],
    val_cov: Sequence[BranchCovariates],
    train_pos: Sequence[PositionBranch],
    val_pos: Sequence[PositionBranch],
    *,
    families: Sequence[str] = ("poisson", "nb", "zinb", "hurdle"),
    exposures: Sequence[str] = ("length", "area"),
    random_effects: Sequence[bool] = (False, True),
    position_use: Sequence[str] = ("radius", "kappa", "soma", "end"),
    n_basis: int = 5,
    smooth_penalty: float = 1.0,
    max_iter: int = 300,
    seed: int = 0,
    log=print,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    design = CountDesign(n_basis).fit(train_cov)
    variants = []
    best = None
    lengths_val = np.array([b.length_nm for b in val_cov])
    for family, exposure, re in itertools.product(families, exposures, random_effects):
        dtr, dva = build_count_data(design, train_cov, exposure), build_count_data(design, val_cov, exposure)
        model = CountModel(family, exposure, re, n_basis=n_basis, smooth_penalty=smooth_penalty).fit(dtr, max_iter=max_iter)
        nll = model.neg_loglik(dva) / max(len(val_cov), 1)
        resid = model.randomized_quantile_residuals(dva, rng)
        row = {"family": family, "exposure": exposure, "random_effect": re, "val_nll_per_branch": float(nll),
               "train_nll_per_branch": float(model.fit_info["nll"] / max(len(train_cov), 1)),
               "dispersion_k": None if family == "poisson" else float(model.dispersion), "sigma_b": float(model.sigma_b),
               "residual_mean": float(resid.mean()), "residual_std": float(resid.std())}
        variants.append(row)
        log(f"[p0-count] {family:8s} {exposure:6s} re={str(re):5s}  val NLL/branch {nll:.4f}")
        if best is None or nll < best[0]:
            best = (nll, model, row)
    _, count_model, chosen = best
    dva = build_count_data(design, val_cov, count_model.exposure)
    calibration = zero_fraction_by_length(count_model, dva, lengths_val, rng)

    # positional density: the full set and one ablation per covariate
    pos_rows = []
    n_val_pos = sum(len(b.positions) for b in val_pos if b.positions is not None)
    full = PositionModel(use=position_use).fit(train_pos)
    full_ll = full.loglik(val_pos) / max(n_val_pos, 1)
    pos_rows.append({"use": list(position_use), "val_loglik_per_spine": float(full_ll)})
    flat = PositionModel(use=()).fit(train_pos)
    pos_rows.append({"use": [], "val_loglik_per_spine": float(flat.loglik(val_pos) / max(n_val_pos, 1))})
    for drop in position_use:
        use = [u for u in position_use if u != drop]
        m = PositionModel(use=use).fit(train_pos)
        pos_rows.append({"use": use, "dropped": drop, "val_loglik_per_spine": float(m.loglik(val_pos) / max(n_val_pos, 1)),
                         "gain_of_dropped": float(full_ll - m.loglik(val_pos) / max(n_val_pos, 1))})
    log(f"[p0-position] val loglik/spine: full {full_ll:.4f}, uniform-in-u {pos_rows[1]['val_loglik_per_spine']:.4f}")
    report = {
        "n_train_branches": len(train_cov), "n_val_branches": len(val_cov),
        "n_train_positions": int(sum(len(b.positions) for b in train_pos if b.positions is not None)), "n_val_positions": int(n_val_pos),
        "count_variants": variants, "count_chosen": chosen, "count_calibration_val": calibration,
        "dispersion_index_train": dispersion_index(np.array([b.n_spines for b in train_cov], dtype=float), np.array([b.length_nm for b in train_cov])),
        "position_variants": pos_rows, "position_coefficients": dict(zip(full.names, map(float, full.beta))),
    }
    return {"design": design, "count_model": count_model, "position_model": full, "report": report}


def load_p0_bundle(path) -> Dict[str, Any]:
    """``p0_bundle.json`` of fit_p0.py -> {design, count_model, position_model, d_min_nm}."""
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"design": CountDesign.from_dict(d["design"]), "count_model": CountModel.from_dict(d["count_model"]),
            "position_model": PositionModel.from_dict(d["position_model"]), "d_min_nm": float(d["d_min_nm"])}
