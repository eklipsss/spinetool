"""Surface-to-surface geometry metrics (reconstruction calibration, later generation eval).

All distances are in the units of the inputs (physical units for calibration).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree


def sample_surface(mesh: Any, n_points: int, seed: int = 0) -> np.ndarray:
    """Area-weighted surface sample (same scheme as the preprocessing point clouds)."""
    from ...spine_sampling import sample_surface_area_weighted

    points, _, _ = sample_surface_area_weighted(mesh, n_points, np.random.default_rng(seed))
    return np.asarray(points, dtype=np.float64)


def nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every point of ``a``: distance to the nearest point of ``b``."""
    return cKDTree(np.asarray(b, dtype=np.float64)).query(np.asarray(a, dtype=np.float64), k=1)[0]


def chamfer_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    f_score_thresholds: Sequence[float] = (),
) -> Dict[str, float]:
    """Symmetric Chamfer (L1 = mean distance, L2 = mean squared), Hausdorff (max, 95th pct), F-score@tau."""
    d_pred_ref = nearest_distances(pred, ref)  # accuracy
    d_ref_pred = nearest_distances(ref, pred)  # completeness
    result = {
        "chamfer_l1": float(0.5 * (d_pred_ref.mean() + d_ref_pred.mean())),
        "chamfer_l2": float(0.5 * ((d_pred_ref**2).mean() + (d_ref_pred**2).mean())),
        "accuracy_mean": float(d_pred_ref.mean()),
        "completeness_mean": float(d_ref_pred.mean()),
        "hausdorff": float(max(d_pred_ref.max(), d_ref_pred.max())),
        "hausdorff_95": float(max(np.percentile(d_pred_ref, 95), np.percentile(d_ref_pred, 95))),
    }
    for tau in f_score_thresholds:
        precision = float((d_pred_ref <= tau).mean())
        recall = float((d_ref_pred <= tau).mean())
        f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        result[f"f_score@{tau:g}"] = f
    return result


def neck_region_chamfer(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    quantile: float = 0.3,
) -> Dict[str, float]:
    """Chamfer restricted to the base/neck: points closer to ``origin`` (the attachment
    centre in local coordinates) than the ``quantile`` of the reference's distances.

    Axis-agnostic on purpose; the thin neck is where Poisson/Marching Cubes are
    most likely to over-smooth (spec risk), and a whole-surface Chamfer would
    hide it behind the much larger head.
    """
    origin = np.asarray(origin, dtype=np.float64)
    radius = float(np.quantile(np.linalg.norm(ref - origin, axis=1), quantile))
    ref_neck = ref[np.linalg.norm(ref - origin, axis=1) <= radius]
    pred_neck = pred[np.linalg.norm(pred - origin, axis=1) <= radius]
    if len(ref_neck) == 0 or len(pred_neck) == 0:
        return {"neck_radius": radius, "neck_chamfer_l1": float("nan"), "neck_n_pred": int(len(pred_neck))}
    d1 = nearest_distances(pred_neck, ref)
    d2 = nearest_distances(ref_neck, pred)
    return {
        "neck_radius": radius,
        "neck_chamfer_l1": float(0.5 * (d1.mean() + d2.mean())),
        "neck_completeness_mean": float(d2.mean()),
        "neck_n_pred": int(len(pred_neck)),
    }


def relative_difference(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None or reference == 0:
        return None
    return float((value - reference) / abs(reference))


def compare_meshes(
    pred_mesh: Any,
    ref_mesh: Any,
    *,
    n_points: int = 8192,
    f_score_thresholds: Sequence[float] = (),
    neck_quantile: float = 0.3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Full reconstruction-vs-reference comparison used by the calibration."""
    pred = sample_surface(pred_mesh, n_points, seed)
    ref = sample_surface(ref_mesh, n_points, seed + 1)
    result: Dict[str, Any] = chamfer_metrics(pred, ref, f_score_thresholds=f_score_thresholds)
    result.update(neck_region_chamfer(pred, ref, quantile=neck_quantile))
    pred_volume = float(pred_mesh.volume) if pred_mesh.is_watertight else None
    ref_volume = float(ref_mesh.volume) if ref_mesh.is_watertight else None
    result["volume_rel_diff"] = relative_difference(pred_volume, ref_volume)
    result["area_rel_diff"] = relative_difference(float(pred_mesh.area), float(ref_mesh.area))
    return result
