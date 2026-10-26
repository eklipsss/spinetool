"""Visual check of the LabID attachment regions (notebooks/neuron-model/labid-attachment-visual-check.ipynb).

For a few chosen spines (valid and needs_review): convert only their branches (``labid_layout.convert``), rerun the S
stages 1-3 + QC on just those spines (mac scale, <= 10 spines), and draw two panels per spine:

1. the dendrite (``branch_mesh.off`` = surface mesh with spines, grey), the spine (green), its attachment cap (pink);
2. the sealed spine mesh alone with the cap (pink).

The cap = ``attachment_cap_face_indices`` of stage 1 (faces of the sealed mesh that seal the attachment loop), exactly
what the training data use. Spines are picked from the QC table of the full run (copied from Windows), or, without it,
from a local run on the first branches. Colours: html-materials-style-guide.md (pink #d6418f, green #3f8a2a). Units: nm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PINK, GREEN, GREY = "#d6418f", "#3f8a2a", "#b9bcc1"
MAX_SPINES = 10  # mac: tiny runs only (CLAUDE.md)


def pick_spines(quality: pd.DataFrame, n_valid: int = 3, n_review: int = 3, seed: int = 0, max_branches: int = 4) -> pd.DataFrame:
    """Spines to show from a ``mesh_quality.parquet`` of the full run: ``n_valid`` valid + ``n_review`` needs_review, drawn
    from at most ``max_branches`` branches (each shown branch is converted and preprocessed locally)."""
    q = quality.copy()
    q["neuron_id"] = q["neuron_id"].astype(str)
    q["spine_id"] = q["spine_id"].astype(str)
    rng = np.random.default_rng(seed)
    review_branches = q.loc[q["status"] == "needs_review", "neuron_id"].unique()
    valid_branches = q.loc[q["status"] == "valid", "neuron_id"].unique()
    both = np.intersect1d(review_branches, valid_branches)
    pool = both if len(both) else np.union1d(review_branches, valid_branches)
    branches = rng.permutation(pool)[:max_branches]
    sub = q[q["neuron_id"].isin(branches)]
    out = []
    for status, n in (("valid", n_valid), ("needs_review", n_review)):
        cand = sub[sub["status"] == status]
        out.append(cand.iloc[rng.permutation(len(cand))[:n]])
    picked = pd.concat(out, ignore_index=True)
    if len(picked) > MAX_SPINES:
        raise ValueError(f"{len(picked)} spines > {MAX_SPINES}: mac runs stay tiny")
    return picked


def prepare(labid_root: Path, work_root: Path, keys: Sequence[tuple], config_path: Path, metadata_root: Path) -> Dict[str, Any]:
    """Convert the branches of ``keys`` ((neuron_id, spine_id) pairs), run seal / false-spine / orient / QC on those
    spines only. Returns {"cfg", "records": {(neuron, spine): SpineRecord}, "conversion": DataFrame}."""
    from .labid_layout import convert
    from .spine_preprocessing import (
        SpinePreprocessingConfig,
        discover_spines,
        run_false_spine_detection_stage,
        run_mesh_qc_stage,
        run_orientation_stage,
        run_sealing_stage,
    )

    keys = [(str(n), str(s)) for n, s in keys]
    if len(keys) > MAX_SPINES:
        raise ValueError(f"{len(keys)} spines > {MAX_SPINES}: mac runs stay tiny")
    units = sorted({n for n, _ in keys})
    conversion = convert(Path(labid_root), Path(work_root), units=units, log=None)
    raw = Path(work_root)
    cfg = SpinePreprocessingConfig.from_yaml(Path(config_path), raw_data_root=raw, metadata_root=Path(metadata_root),
                                             output_root=raw / "labid" / "preprocessed", dataset="labid", workers=1, force=True)
    records = {}
    for unit in units:
        for r in discover_spines(SpinePreprocessingConfig.from_yaml(Path(config_path), raw_data_root=raw, metadata_root=Path(metadata_root),
                                                                   dataset="labid", neuron_id=unit, workers=1)):
            if (r.neuron_id, r.spine_id) in keys:
                records[(r.neuron_id, r.spine_id)] = r
    chosen = [records[k] for k in keys if k in records]
    for stage in (run_sealing_stage, run_false_spine_detection_stage, run_orientation_stage):
        stage(cfg, chosen)
    qc = run_mesh_qc_stage(cfg, chosen)
    return {"cfg": cfg, "records": {k: records[k] for k in keys if k in records}, "conversion": conversion, "qc": qc}


def _mesh3d(vertices: np.ndarray, faces: np.ndarray, color: str, name: str, opacity: float = 1.0, scene: str = "scene"):
    import plotly.graph_objects as go

    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color,
                     opacity=opacity, name=name, showlegend=True, flatshading=True, scene=scene,
                     lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.1}, hoverinfo="name")


def spine_figure(record, cfg, *, local_qc: Optional[Dict[str, Any]] = None, windows_row: Optional[Dict[str, Any]] = None,
                 conversion_row: Optional[Dict[str, Any]] = None, context_factor: float = 2.0, context_min_nm: float = 2500.0):
    """Two panels: dendrite with the spine (green) and its cap (pink); the sealed spine with the cap (pink). The dendrite is
    cut to the faces within max(context_factor x spine size, context_min_nm) of the spine centre (the whole branch makes
    the spine a few pixels)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .spine_geometry import load_trimesh
    from .spine_preprocessing import attachment_json_path, quality_json_path, sealed_mesh_path

    branch = load_trimesh(record.branch_path / "branch_mesh.off", process=False)
    opened = load_trimesh(record.source_path, process=False)  # the spine as it was on the dendrite (seal removed)
    sealed_path = sealed_mesh_path(record, cfg)
    if not sealed_path.exists():
        raise FileNotFoundError(f"stage 1 did not write {sealed_path}")
    sealed = load_trimesh(sealed_path, process=False)
    attachment = json.loads(attachment_json_path(record, cfg).read_text(encoding="utf-8"))
    cap = np.asarray(attachment.get("attachment_cap_face_indices") or [], dtype=int)
    qpath = quality_json_path(record, cfg)
    quality = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else {}

    bv, bf = np.asarray(branch.vertices), np.asarray(branch.faces)
    sv, sf = np.asarray(sealed.vertices), np.asarray(sealed.faces)
    half = 0.5 * max(context_factor * float(np.ptp(sv, axis=0).max()), context_min_nm)
    centre = 0.5 * (sv.min(0) + sv.max(0))
    keep = (np.abs(bv[bf].mean(1) - centre) < half).all(axis=1)  # a cube centred on the spine
    bf = bf[keep]
    cap_mask = np.zeros(len(sf), dtype=bool)
    cap_mask[cap[cap < len(sf)]] = True

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.02,
                        subplot_titles=("дендрит: шипик зелёный, область крепления розовая", "запаянный шипик: область крепления розовая"))
    fig.add_trace(_mesh3d(bv, bf, GREY, "дендрит", opacity=0.5), row=1, col=1)
    fig.add_trace(_mesh3d(np.asarray(opened.vertices), np.asarray(opened.faces), GREEN, "шипик"), row=1, col=1)
    if cap_mask.any():
        fig.add_trace(_mesh3d(sv, sf[cap_mask], PINK, "область крепления"), row=1, col=1)
    corners = np.array([centre - half, centre + half])  # invisible: fixes the scene box, the spine stays in the middle
    fig.add_trace(go.Scatter3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], mode="markers", marker={"size": 0.1, "opacity": 0},
                               showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(_mesh3d(sv, sf[~cap_mask], GREEN, "шипик (запаянный)", opacity=0.55), row=1, col=2)
    if cap_mask.any():
        fig.add_trace(_mesh3d(sv, sf[cap_mask], PINK, "область крепления "), row=1, col=2)

    # cameras: the cap normal n (outward from the spine = towards the shaft) and the spine axis; left panel looks at the
    # spine from the side (profile on the dendrite), right panel obliquely from below, so the cap is visible
    tri = sv[sf]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 x area
    n = normals[cap_mask].sum(0) if cap_mask.any() else (sv.mean(0) - sv[sf[cap_mask]].reshape(-1, 3).mean(0) if cap_mask.any() else np.array([0, 0, -1.0]))
    n = n / max(np.linalg.norm(n), 1e-12)
    near = bv[np.linalg.norm(bv - sv.mean(0), axis=1) < 3.0 * max(float(np.ptp(sv, axis=0).max()), 1.0)]
    trunk_axis = np.linalg.svd(near - near.mean(0), full_matrices=False)[2][0] if len(near) > 3 else np.array([1.0, 0, 0])
    side = np.cross(n, trunk_axis)
    side = side / max(np.linalg.norm(side), 1e-12)
    hidden = {"visible": False}
    eye = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    scene_left = {f"{ax}axis": hidden for ax in "xyz"}
    scene_left.update(aspectmode="cube", camera={"eye": eye(0.95 * side)})
    oblique = n + 0.8 * side
    fig.update_layout(scene=scene_left, scene2={"aspectmode": "data", "xaxis": hidden, "yaxis": hidden, "zaxis": hidden,
                                                "camera": {"eye": eye(1.8 * oblique / np.linalg.norm(oblique))}})

    local_qc = local_qc or {}
    status, reason = local_qc.get("status"), local_qc.get("reason")
    bits = [f"<b>{record.neuron_id} / spine {record.spine_id}</b>",
            f"QC здесь (mac): {status}" + (f" ({reason})" if reason else ""),
            f"заплатка лаборатории: {conversion_row.get('unseal_status')}" if conversion_row else "",
            f"крепление этапа 1: {attachment.get('attachment_method')}, граней {len(cap)}, площадь {float(attachment.get('attachment_cap_area') or 0) / 1e6:.3f} мкм²"]
    if windows_row:
        bits.append(f"QC на Windows: {windows_row.get('status')}" + (f" ({windows_row.get('reason')})" if windows_row.get("reason") else ""))
    if quality.get("original_has_self_intersections") is not None:
        bits.append(f"самопересечения в исходной сетке: {quality.get('original_has_self_intersections')}")
    fig.update_layout(title={"text": "<br>".join(b for b in bits if b), "font": {"size": 12}, "y": 0.98, "yanchor": "top"}, height=640,
                      margin={"l": 0, "r": 0, "t": 170, "b": 0},
                      paper_bgcolor="white", legend={"orientation": "h", "y": -0.02})
    return fig


def local_candidates(labid_root: Path, max_spines: int = MAX_SPINES, start: int = 0) -> List[tuple]:
    """Without the Windows QC table: (branch id, spine id) of the first LabID branches (from ``start``), at most
    ``max_spines`` spines; the statuses then come from the local QC run."""
    from .labid_layout import discover_raw

    keys: List[tuple] = []
    for item in discover_raw(Path(labid_root)).iloc[start:].to_dict("records"):
        spines = sorted(Path(item["path"]).glob("spine_*.off"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
        ids = ["".join(ch for ch in p.stem if ch.isdigit()) for p in spines]
        keys.extend((item["neuron_id"], sid) for sid in ids[: max_spines - len(keys)])
        if len(keys) >= max_spines:
            break
    return keys
