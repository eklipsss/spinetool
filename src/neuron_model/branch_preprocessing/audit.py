"""Reverse-reconstruction audit (spec section 9): is a centreline + one radius enough for module T?

For sampled real branches the round tube X(s, theta) = gamma + r (cos theta e1 + sin theta e2) built from
the preprocessed geometry is compared with the shaft-only surface of the same branch (the mesh without
spine vertices, false spines kept). Both are restricted to the interior of the centreline span (the trimmed
ends and their branching regions are not represented by the tube) and sampled uniformly by area.

Metrics per branch (nm): chamfer_sq_nm2 = mean d^2 (shaft->tube) + mean d^2 (tube->shaft) (spec definition),
chamfer_rms_nm = sqrt(chamfer_sq / 2), chamfer_l1_nm, hd95_nm, area_rel_error (tube vs shaft-only area; the shaft
has holes where spines were removed, so it is biased towards the tube being larger). The shaft surface is open
(holes, cut ends), so the volume error of the specification is not computed.
Acceptance (spec): median chamfer_rms <= 0.25 r_med.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .. import branch_geometry as bg
from .config import BranchPreprocessingConfig
from .inputs import load_members, load_spine_table
from .process import branch_segments, load_branch_surface


def _sample_triangles(vertices: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return np.zeros((0, 3))
    pick = rng.choice(len(faces), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _area(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def audit_branch(cfg: BranchPreprocessingConfig, row: pd.Series, spine_info: Dict, rng: np.random.Generator, n_points: int = 20000) -> Dict[str, Any]:
    from scipy.spatial import cKDTree

    geom = np.load(row["geometry_path"])
    s, gamma, radius = geom["s"], geom["gamma"], geom["radius"]
    e1, e2 = geom["frame_e1"], geom["frame_e2"]
    r_med = float(np.median(radius))
    length = float(s[-1])
    margin = cfg.end_distance_factor * r_med
    limb = int(row["limb_id"])
    members = [int(b) for b in str(row["member_branch_ids"]).split("|")]
    neuron_dir = cfg.dataset_root / str(row["neuron_id"])
    member_dirs = {b: neuron_dir / f"limb_{limb:03d}" / f"branch_{b:03d}" for b in members}
    vertices, faces, spine_mask = load_branch_surface(cfg, member_dirs, branch_segments(member_dirs, spine_info, limb))
    shaft_faces = faces[~spine_mask[faces].any(axis=1)]
    # restrict the shaft to the interior of the centreline span and to the vicinity of the tube
    dense_s = np.linspace(0.0, length, max(int(length / 25.0), 10))
    dense = np.column_stack([np.interp(dense_s, s, gamma[:, k]) for k in range(3)])
    centroids = vertices[shaft_faces].mean(axis=1)
    dist, idx = cKDTree(dense).query(centroids)
    s_near = dense_s[idx]
    keep = (s_near > margin) & (s_near < length - margin) & (dist < cfg.max_section_radius_nm)
    shaft_faces = shaft_faces[keep]
    inner = (s > margin) & (s < length - margin)
    if shaft_faces.size == 0 or inner.sum() < 3:
        return {"status": "too_short_for_audit"}
    tv, tf = bg.tube_mesh(gamma[inner], radius[inner], e1[inner], e2[inner], n_theta=32, caps=False)
    shaft_pts = _sample_triangles(vertices, shaft_faces, n_points, rng)
    tube_pts = _sample_triangles(tv, tf, n_points, rng)
    d1, _ = cKDTree(tube_pts).query(shaft_pts)
    d2, _ = cKDTree(shaft_pts).query(tube_pts)
    # core of the trunk: shaft points within 1.75 r_med of the centreline. Farther points are protrusions the tube cannot
    # represent by design (unsegmented spines / stubs, false spines kept in the shaft, bits of the neighbouring branch)
    d_line, _ = cKDTree(dense).query(shaft_pts)
    core = d_line < 1.75 * r_med
    chamfer_sq = float(np.mean(d1**2) + np.mean(d2**2))
    a_shaft, a_tube = _area(vertices, shaft_faces), _area(tv, tf)
    ellipse = {}
    if "ellipse_a" in geom.files:
        ev, ef = bg.ellipse_tube_mesh(gamma[inner], geom["ellipse_a"][inner], geom["ellipse_b"][inner], geom["ellipse_angle"][inner], e1[inner], e2[inner], n_theta=32)
        ell_pts = _sample_triangles(ev, ef, n_points, rng)
        e1d, _ = cKDTree(ell_pts).query(shaft_pts)
        e2d, _ = cKDTree(shaft_pts).query(ell_pts)
        ecd = float(np.mean(e1d**2) + np.mean(e2d**2))
        a_ell = _area(ev, ef)
        ellipse = {
            "ellipse_chamfer_rms_nm": float(np.sqrt(ecd / 2.0)),
            "ellipse_chamfer_rms_over_r": float(np.sqrt(ecd / 2.0) / r_med),
            "ellipse_chamfer_l1_nm": float(0.5 * (e1d.mean() + e2d.mean())),
            "ellipse_hd95_nm": float(max(np.percentile(e1d, 95), np.percentile(e2d, 95))),
            "ellipse_area_rel_error": float(abs(a_ell - a_shaft) / a_shaft) if a_shaft > 0 else float("nan"),
            "ellipse_ratio_median": float(np.median(geom["ellipse_b"] / geom["ellipse_a"])),
            # the median distance from the shaft to each tube, split: d1 = shaft -> tube (the tube misses the surface)
            "circle_shaft_to_tube_median_nm": float(np.median(d1)),
            "ellipse_shaft_to_tube_median_nm": float(np.median(e1d)),
            "ellipse_shaft_to_tube_p90_over_r": float(np.percentile(e1d, 90) / r_med),
            "ellipse_tube_to_shaft_p90_over_r": float(np.percentile(e2d, 90) / r_med),
            "ellipse_shaft_to_tube_rms_trim95_over_r": float(np.sqrt(np.mean(np.sort(e1d**2)[: int(0.95 * len(e1d))])) / r_med),
            "ellipse_core_shaft_to_tube_median_over_r": float(np.median(e1d[core]) / r_med) if core.any() else float("nan"),
            "ellipse_core_shaft_to_tube_rms_over_r": float(np.sqrt(np.mean(e1d[core] ** 2)) / r_med) if core.any() else float("nan"),
        }
    return {
        **ellipse,
        "status": "ok",
        "r_med_nm": r_med,
        "length_nm": length,
        "chamfer_sq_nm2": chamfer_sq,
        "chamfer_rms_nm": float(np.sqrt(chamfer_sq / 2.0)),
        "chamfer_l1_nm": float(0.5 * (d1.mean() + d2.mean())),
        "hd95_nm": float(max(np.percentile(d1, 95), np.percentile(d2, 95))),
        "area_shaft_nm2": a_shaft,
        "area_tube_nm2": a_tube,
        "area_rel_error": float(abs(a_tube - a_shaft) / a_shaft) if a_shaft > 0 else float("nan"),
        "chamfer_rms_over_r": float(np.sqrt(chamfer_sq / 2.0) / r_med),
        # the two directions separately: shaft -> tube (does the tube reach every part of the surface? protrusions, false
        # spines kept in the shaft, flattened sections) and tube -> shaft (does the tube cover places without surface? the
        # holes left by the removed spines, cut ends)
        "shaft_to_tube_median_over_r": float(np.median(d1) / r_med),
        "shaft_to_tube_p90_over_r": float(np.percentile(d1, 90) / r_med),
        "tube_to_shaft_median_over_r": float(np.median(d2) / r_med),
        "tube_to_shaft_p90_over_r": float(np.percentile(d2, 90) / r_med),
        "shaft_to_tube_rms_trim95_over_r": float(np.sqrt(np.mean(np.sort(d1**2)[: int(0.95 * len(d1))])) / r_med),
        "protrusion_area_fraction": float(1.0 - core.mean()),
        "core_shaft_to_tube_median_over_r": float(np.median(d1[core]) / r_med) if core.any() else float("nan"),
        "core_shaft_to_tube_rms_over_r": float(np.sqrt(np.mean(d1[core] ** 2)) / r_med) if core.any() else float("nan"),
    }


def run_audit(cfg: BranchPreprocessingConfig, *, per_stratum: int = 25, seed: int = 0, exclude_test: bool = True) -> Dict[str, Any]:
    """Stratified sample (length quartiles) of processed branches -> per-branch metrics + summary."""
    branches = pd.read_parquet(cfg.out_root / "branches.parquet")
    pool = branches[(branches["status"] == "ok") & branches["geometry_path"].notna()]
    if exclude_test and "split" in pool.columns:
        pool = pool[pool["split"].astype(str) != "test"]  # the test part is never used for choices
    if pool.empty:
        raise SystemExit("no processed branches with geometry to audit")
    rng = np.random.default_rng(seed)
    quartile = pd.qcut(pool["length_nm"], q=min(4, len(pool)), labels=False, duplicates="drop")
    sample = pd.concat([g.sample(min(per_stratum, len(g)), random_state=seed) for _, g in pool.groupby(quartile)])
    spines = load_spine_table(cfg, sample["neuron_id"].astype(str).unique())
    spine_info: Dict[str, Dict] = {}
    for neuron_id, g in spines.groupby("neuron_id"):
        spine_info[str(neuron_id)] = {(r["limb_id"], r["branch_id"], r["spine_id"]): r for r in g.to_dict("records")}
    rows = []
    from tqdm.auto import tqdm

    for _, row in tqdm(list(sample.iterrows()), desc="audit"):
        key = {"neuron_id": row["neuron_id"], "limb_id": int(row["limb_id"]), "merged_branch_id": int(row["merged_branch_id"]), "train_eligible_T": bool(row["train_eligible_T"])}
        try:
            rows.append({**key, **audit_branch(cfg, row, spine_info.get(str(row["neuron_id"]), {}), rng)})
        except Exception as exc:
            rows.append({**key, "status": "failed", "error": repr(exc)})
    table = pd.DataFrame(rows)
    out_dir = cfg.out_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "reconstruction_circle.parquet", index=False)
    ok = table[table["status"] == "ok"]
    summary: Dict[str, Any] = {"n_audited": int(len(table)), "n_ok": int(len(ok))}
    if len(ok):
        summary.update(
            median_chamfer_rms_nm=float(ok["chamfer_rms_nm"].median()),
            median_chamfer_rms_over_r=float(ok["chamfer_rms_over_r"].median()),
            fraction_chamfer_rms_le_quarter_r=float((ok["chamfer_rms_over_r"] <= 0.25).mean()),
            median_hd95_nm=float(ok["hd95_nm"].median()),
            median_area_rel_error=float(ok["area_rel_error"].median()),
            acceptance_median_le_quarter_r=bool(ok["chamfer_rms_over_r"].median() <= 0.25),
        )
        for col in ("shaft_to_tube_median_over_r", "shaft_to_tube_p90_over_r", "tube_to_shaft_median_over_r", "tube_to_shaft_p90_over_r", "shaft_to_tube_rms_trim95_over_r",
                    "protrusion_area_fraction", "core_shaft_to_tube_median_over_r", "core_shaft_to_tube_rms_over_r",
                    "ellipse_core_shaft_to_tube_median_over_r", "ellipse_core_shaft_to_tube_rms_over_r"):
            if col in ok.columns:
                summary[f"median_{col}"] = float(ok[col].median())
        if "ellipse_chamfer_rms_over_r" in ok.columns:
            e = ok[ok["ellipse_chamfer_rms_over_r"].notna()]
            summary.update(
                n_with_ellipse=int(len(e)),
                ellipse_median_chamfer_rms_nm=float(e["ellipse_chamfer_rms_nm"].median()),
                ellipse_median_chamfer_rms_over_r=float(e["ellipse_chamfer_rms_over_r"].median()),
                ellipse_fraction_chamfer_rms_le_quarter_r=float((e["ellipse_chamfer_rms_over_r"] <= 0.25).mean()),
                ellipse_median_hd95_nm=float(e["ellipse_hd95_nm"].median()),
                ellipse_median_area_rel_error=float(e["ellipse_area_rel_error"].median()),
                ellipse_ratio_b_over_a_median=float(e["ellipse_ratio_median"].median()),
                circle_median_chamfer_rms_over_r_same_branches=float(e["chamfer_rms_over_r"].median()),
                median_shaft_to_tube_nm_circle=float(e["circle_shaft_to_tube_median_nm"].median()),
                median_shaft_to_tube_nm_ellipse=float(e["ellipse_shaft_to_tube_median_nm"].median()),
                ellipse_median_shaft_to_tube_p90_over_r=float(e["ellipse_shaft_to_tube_p90_over_r"].median()),
                ellipse_median_tube_to_shaft_p90_over_r=float(e["ellipse_tube_to_shaft_p90_over_r"].median()),
                ellipse_median_shaft_to_tube_rms_trim95_over_r=float(e["ellipse_shaft_to_tube_rms_trim95_over_r"].median()),
                ellipse_acceptance_median_le_quarter_r=bool(e["ellipse_chamfer_rms_over_r"].median() <= 0.25),
            )
    (out_dir / "reconstruction_circle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
