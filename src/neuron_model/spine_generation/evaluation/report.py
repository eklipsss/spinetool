"""Evaluation of one generator against real spines + the cross-model report (plan stage 7).

``evaluate_generated_set`` scores ONE generated set (one generation seed) against the real
TEST split, with standardisation statistics and the MMD bandwidth taken from the real
TRAIN split (so every model is measured on one fixed scale). ``evaluate_model`` runs it
for every seed and aggregates (spec protocol: 1000 samples x 5 seeds). ``write_report``
stores JSON + CSV tables + a markdown summary of the spec's selection criteria:

    valid_mesh_rate (up), MMD_morphometrics (down), Wasserstein on key morphometrics (down),
    Coverage-CD (up), 1-NNA-CD (-> 0.5), diversity similarity to real (-> 1),
    memorization rate (down)

Surface metrics (Chamfer-based) are quadratic in the set size and run on a fixed-size
random subset of the real test split (``surface_reference_size``, equal to the number of
generated clouds used, because 1-NNA needs equal sets); morphometric metrics use all rows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ...spine_morphometrics import HISTOGRAM_METRICS
from ..reconstruction.mesh_validation import aggregate_validity
from .bootstrap import aggregate_over_seeds
from .classifier import real_vs_generated_classifier
from .distribution_metrics import (
    JOINT_FEATURES,
    UNIVARIATE_FEATURES,
    Standardizer,
    correlation_difference,
    histogram_distances,
    median_heuristic_sigma,
    rbf_mmd2,
    univariate_table,
)
from .diversity import diversity_report
from .memorization import chamfer_dcr, memorization_report
from .point_cloud_metrics import chamfer_matrix, surface_metrics


@dataclass(frozen=True)
class EvaluationConfig:
    surface_reference_size: int = 1000   # clouds per set for Chamfer-based metrics
    chamfer_scale: float = 1e-3          # nm -> um, readability only (one global factor)
    device: str = "cpu"
    pair_chunk: int = 16
    memorization_repeats: int = 20
    memorization_k_candidates: int = 50  # Chamfer-DCR: train candidates re-ranked per query
    classifier_model: str = "logistic"
    seed: int = 0
    key_morphometrics: Sequence[str] = field(default_factory=lambda: ("Length", "Volume", "Area", "OpenAngle", "CVD"))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvaluationConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: (tuple(v) if k == "key_morphometrics" else v) for k, v in values.items() if k in known})


@dataclass(frozen=True)
class RealReference:
    """Everything about the real data that is shared by all models (fit once, reuse)."""

    train_morph: pd.DataFrame
    test_morph: pd.DataFrame
    standardizer: Standardizer
    mmd_sigma: float
    test_clouds: Optional[np.ndarray] = None    # [n, P, 3], physical units
    # FULL train split for Chamfer-space memorization (may be a memmap; only candidate rows
    # are read) + its precomputed cloud_descriptors (otherwise recomputed per model)
    train_clouds: Optional[np.ndarray] = None
    train_descriptors: Optional[np.ndarray] = None


def build_real_reference(
    train_morph: pd.DataFrame,
    test_morph: pd.DataFrame,
    *,
    test_clouds: Optional[np.ndarray] = None,
    train_clouds: Optional[np.ndarray] = None,
    features: Sequence[str] = JOINT_FEATURES,
    seed: int = 0,
) -> RealReference:
    """Fit the train standardiser, fix the MMD bandwidth (median heuristic on real train data)
    and precompute train cloud descriptors - all shared by every evaluated model."""
    from .point_cloud_metrics import cloud_descriptors

    standardizer = Standardizer.fit(train_morph, features)
    train_x = standardizer.transform(train_morph)
    sigma = median_heuristic_sigma(train_x, train_x, seed=seed)
    descriptors = cloud_descriptors(train_clouds) if train_clouds is not None else None
    return RealReference(train_morph, test_morph, standardizer, sigma, test_clouds, train_clouds, descriptors)


def _subsample(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


def evaluate_generated_set(
    real: RealReference,
    gen_morph: pd.DataFrame,
    *,
    gen_clouds: Optional[np.ndarray] = None,
    validity_raw: Optional[Sequence[Mapping[str, Any]]] = None,
    validity_postprocessed: Optional[Sequence[Mapping[str, Any]]] = None,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    result: Dict[str, Any] = {"n_generated": int(len(gen_morph))}

    if validity_raw is not None:
        result["validity_raw"] = aggregate_validity(validity_raw)
    if validity_postprocessed is not None:
        result["validity_postprocessed"] = aggregate_validity(validity_postprocessed)
    if "attachment_found" in gen_morph.columns:
        result["attachment_found_rate"] = float(gen_morph["attachment_found"].mean())
    if "attachment_method" in gen_morph.columns:
        result["attachment_methods"] = {str(k): int(v) for k, v in gen_morph["attachment_method"].value_counts(dropna=False).items()}

    # --- per-feature distributions (W1 normalised by the TRAIN std: same scale for all models)
    scales = dict(zip(real.standardizer.features, real.standardizer.std))
    train_std = real.train_morph[[f for f in UNIVARIATE_FEATURES if f in real.train_morph.columns]].astype(float).std()
    scales.update({k: float(v) for k, v in train_std.items() if k not in scales})
    result["univariate"] = univariate_table(real.test_morph, gen_morph, UNIVARIATE_FEATURES, scales=scales)
    for metric in HISTOGRAM_METRICS:
        if metric in gen_morph.columns and metric in real.test_morph.columns:
            result[f"histogram_{metric}"] = histogram_distances(real.test_morph[metric], gen_morph[metric])

    # --- joint morphometric distribution (spec's 10-feature vector, train-standardised)
    real_x = real.standardizer.transform(real.test_morph)
    gen_x = real.standardizer.transform(gen_morph)
    result["n_generated_with_all_joint_features"] = int(len(gen_x))
    if len(gen_x) >= 2 and len(real_x) >= 2:
        mmd2, sigma = rbf_mmd2(real_x, gen_x, sigma=real.mmd_sigma)
        corr = correlation_difference(real_x, gen_x)
        result["joint"] = {
            "mmd2_morphometrics": mmd2,
            "mmd_sigma": sigma,
            "correlation_frobenius": corr["frobenius"],
            "correlation_max_abs": corr["max_abs"],
        }
        if min(len(real_x), len(gen_x)) >= 5:
            result["classifier"] = real_vs_generated_classifier(real_x, gen_x, model=cfg.classifier_model, seed=cfg.seed)
        train_x = real.standardizer.transform(real.train_morph)
        result["memorization"] = memorization_report(gen_x=gen_x, train_x=train_x, test_x=real_x, n_repeats=cfg.memorization_repeats, seed=cfg.seed)
    else:
        result["joint"] = {"skipped": "fewer than 2 complete feature vectors (attachment finder missing?)"}

    # --- surface level (Chamfer on sampled clouds) + Chamfer-space diversity / memorization
    if gen_clouds is not None and real.test_clouds is not None and len(gen_clouds) >= 2:
        k = min(cfg.surface_reference_size, len(gen_clouds), len(real.test_clouds))
        g = np.asarray(gen_clouds)[_subsample(len(gen_clouds), k, rng)]
        r = np.asarray(real.test_clouds)[_subsample(len(real.test_clouds), k, rng)]
        chamfer = dict(device=cfg.device, pair_chunk=cfg.pair_chunk, scale=cfg.chamfer_scale)
        d_rr, d_gg, d_rg = chamfer_matrix(r, **chamfer), chamfer_matrix(g, **chamfer), chamfer_matrix(r, g, **chamfer)
        result["surface"] = {**surface_metrics(d_rr, d_gg, d_rg), "n_per_set": int(k), "chamfer_units": "um^2" if cfg.chamfer_scale == 1e-3 else f"(scale {cfg.chamfer_scale})^2"}
        if len(real_x) >= 2 and len(gen_x) >= 2:
            result["diversity"] = diversity_report(real_real_chamfer=d_rr, gen_gen_chamfer=d_gg, real_x=real_x, gen_x=gen_x, features=real.standardizer.features)
        if real.train_clouds is not None and len(real.train_clouds) >= 2:
            # DCR against the FULL train split: generated vs held-out real test clouds
            mem = chamfer_dcr(
                g, r, real.train_clouds,
                k_candidates=cfg.memorization_k_candidates, device=cfg.device, scale=cfg.chamfer_scale,
                train_descriptors=real.train_descriptors,
            )
            result.setdefault("memorization", {}).update({f"chamfer_{key}": v for key, v in mem.items()})
    return result


def headline(result: Mapping[str, Any], key_morphometrics: Sequence[str]) -> Dict[str, float]:
    """The spec's selection criteria as flat numbers (for seed aggregation and the summary table)."""
    flat: Dict[str, float] = {}
    for section in ("validity_raw", "validity_postprocessed"):
        if section in result:
            flat[f"{section}.valid_mesh_rate"] = result[section].get("valid_mesh_rate", np.nan)
    if "joint" in result and "mmd2_morphometrics" in result["joint"]:
        flat["mmd2_morphometrics"] = result["joint"]["mmd2_morphometrics"]
        flat["correlation_frobenius"] = result["joint"]["correlation_frobenius"]
    uni = result.get("univariate")
    if isinstance(uni, pd.DataFrame) and len(uni):
        for _, row in uni[uni["feature"].isin(key_morphometrics)].iterrows():
            flat[f"w1_normalized.{row['feature']}"] = row["w1_normalized"]
    if "surface" in result:
        flat["coverage_cd"] = result["surface"]["coverage_cd"]
        flat["one_nna_cd"] = result["surface"]["one_nna_cd"]
        flat["mmd_cd"] = result["surface"]["mmd_cd"]
    if "diversity" in result:
        flat["diversity.chamfer_mean_ratio"] = result["diversity"]["chamfer_mean_ratio"]
        flat["diversity.morphometric_mean_ratio"] = result["diversity"]["morphometric_mean_ratio"]
    mem = result.get("memorization", {})
    for key in ("morphometric_dcr_median_ratio", "morphometric_near_copy_rate", "chamfer_dcr_median_ratio", "chamfer_near_copy_rate"):
        if key in mem:
            flat[f"memorization.{key}"] = mem[key]
    if "classifier" in result:
        flat["classifier_accuracy"] = result["classifier"]["accuracy"]
    if "attachment_found_rate" in result:
        flat["attachment_found_rate"] = result["attachment_found_rate"]
    return flat


def evaluate_model(
    real: RealReference,
    per_seed: Mapping[int, Mapping[str, Any]],
    cfg: EvaluationConfig = EvaluationConfig(),
) -> Dict[str, Any]:
    """``per_seed[seed] = {"gen_morph": DataFrame, "gen_clouds": array?, "validity_raw": [...]?, ...}``."""
    results = {seed: evaluate_generated_set(real, cfg=cfg, **inputs) for seed, inputs in per_seed.items()}
    headlines = {seed: headline(res, cfg.key_morphometrics) for seed, res in results.items()}
    return {"per_seed": results, "headline_per_seed": headlines, "headline": aggregate_over_seeds(headlines), "config": asdict(cfg)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


CRITERIA = (
    ("validity_raw.valid_mesh_rate", "↑"),
    ("validity_postprocessed.valid_mesh_rate", "↑"),
    ("mmd2_morphometrics", "↓"),
    ("correlation_frobenius", "↓"),
    ("coverage_cd", "↑"),
    ("one_nna_cd", "→ 0.5"),
    ("mmd_cd", "↓"),
    ("diversity.chamfer_mean_ratio", "→ 1"),
    ("diversity.morphometric_mean_ratio", "→ 1"),
    ("memorization.morphometric_dcr_median_ratio", "→ 1 (≪1 = copies)"),
    ("memorization.chamfer_dcr_median_ratio", "→ 1 (≪1 = copies)"),
    ("memorization.morphometric_near_copy_rate", "→ 0.05"),
    ("memorization.chamfer_near_copy_rate", "→ 0.05"),
    ("classifier_accuracy", "→ 0.5"),
)


def summary_table(models: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Rows = criteria (incl. w1_normalized.* found), columns = models, cells = ``mean ± std`` over seeds."""
    keys = [k for k, _ in CRITERIA]
    extra = sorted({k for m in models.values() for k in m["headline"] if k.startswith("w1_normalized.")})
    rows = []
    for key, direction in list(CRITERIA) + [(k, "↓") for k in extra]:
        row = {"criterion": key, "target": direction}
        present = False
        for name, model in models.items():
            stats = model["headline"].get(key)
            if stats:
                present = True
                row[name] = f"{stats['mean']:.4g} ± {stats['std']:.2g}" if stats["n_seeds"] > 1 else f"{stats['mean']:.4g}"
            else:
                row[name] = "—"
        if present:
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(models: Mapping[str, Mapping[str, Any]], out_dir: Path) -> Path:
    """``models[name] = evaluate_model(...)`` -> ``out_dir/{report.json, summary.csv, summary.md, univariate_<model>_seed<s>.csv}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(_jsonable(models), indent=2, ensure_ascii=False), encoding="utf-8")
    for name, model in models.items():
        for seed, res in model["per_seed"].items():
            if isinstance(res.get("univariate"), pd.DataFrame):
                res["univariate"].to_csv(out_dir / f"univariate_{name}_seed{seed}.csv", index=False)
    table = summary_table(models)
    table.to_csv(out_dir / "summary.csv", index=False)
    lines = [
        "# S-module generator comparison",
        "",
        "Cells: mean ± std over generation seeds. Targets follow the spec's selection criteria;",
        "the choice must not rest on a single metric.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
