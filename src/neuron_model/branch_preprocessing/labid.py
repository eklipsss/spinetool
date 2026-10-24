"""LabID inputs of the branch preprocessing (the same B2-B4 geometry as Minnie65, ``process.branch_geometry``).

Layout: ``<raw>/labid/<group>/<dendrite>/`` with ``pure_surface_mesh.off`` (the shaft WITHOUT spines, closed),
``surface_mesh.off`` (shaft + spines) and ``spine_<n>.off`` (closed spine meshes, cut from ``surface_mesh``).
Groups: ``Wt``, ``Wt_old`` (healthy, two imaging series), ``Ab`` (disease), ``Ab+9009`` / ``Wt+9009`` (treated).
Light microscopy (confocal), coordinates in MICROMETRES (converted to nm on reading, ``UNIT_NM``; the voxel size is not
stored in the tif files - 22 nm in xy and ~97 nm in z were inferred from the mesh extents, to be confirmed by the lab).

Differences from Minnie65 (no skeleton, limbs, soma or S preprocessing):
- raw polyline = longest path of the CGAL skeleton of the closed shaft (``surface_mesh_skeletonization`` in a
  subprocess); it only orders the sections, the centreline comes from the sections as for Minnie;
- every spine file counts as a real spine (no false-spine verdicts); its attachment centre is the area-weighted centroid
  of its seal cap = the faces of the spine mesh that are not faces of ``surface_mesh`` (one patch at the shaft, its
  centre lies within tens of nm of the shaft surface); fallback: the spine vertex nearest to the shaft;
- the branch ends are the image borders, not branch points: no soma distance, no proximal end (the skeleton order is
  kept, ``proximal_end_method = unknown``), ``is_terminal`` unknown; the length is a property of the field of view;
- cell = ``neuron_id``: ``<group>_<number before the dash>`` for names like ``ab23-1_processed`` / ``24-2`` / ``1003-22``
  (ASSUMPTION: the first number is the cell/animal, the second the dendrite), the sanitised full name otherwise
  (``3_full_res (10-2)``); dendrites of one cell share ``limb_id`` 0 and get ``merged_branch_id`` 0, 1, ...
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BranchPreprocessingConfig
from .process import STATUS_FAILED, STATUS_OK, STATUS_TOO_SHORT, _atomic_parquet, _finalize_eligibility, branch_geometry

UNIT_NM = 1000.0  # LabID meshes are in micrometres
GROUPS = {
    # group folder -> (health, treatment, imaging series)
    "wt": ("healthy", None, "wt"),
    "wt_old": ("healthy", None, "wt_old"),
    "wt+9009": ("healthy", "9009", "wt+9009"),
    "ab": ("disease", None, "ab"),
    "ab+9009": ("disease", "9009", "ab+9009"),
}
_NAME = re.compile(r"^(?:ab)?(\d+)(?:-(\d+))?$", re.IGNORECASE)


class _NoSoma:
    def distance(self, limb: int, point: np.ndarray) -> float:
        return float("nan")


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9+]+", "_", text).strip("_")


def cell_of(group: str, dendrite: str) -> str:
    name = re.sub(r"(_processed)+(-\d+)?$", "", dendrite)
    m = _NAME.match(name)
    return _sanitize(f"{group}_{m.group(1)}" if m else f"{group}_{name}")


def discover(cfg: BranchPreprocessingConfig) -> pd.DataFrame:
    """One row per dendrite folder that has a shaft mesh: group, dendrite, path, neuron_id (cell), merged_branch_id."""
    rows = []
    root = cfg.dataset_root
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for d in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if (d / "pure_surface_mesh.off").exists():
                rows.append({"group": group_dir.name, "dendrite": d.name, "path": str(d), "neuron_id": cell_of(group_dir.name, d.name)})
    frame = pd.DataFrame(rows, columns=["group", "dendrite", "path", "neuron_id"])
    frame["merged_branch_id"] = frame.groupby("neuron_id").cumcount() if len(frame) else []
    if cfg.neuron_ids:
        frame = frame[frame["neuron_id"].isin({str(n) for n in cfg.neuron_ids})]
    return frame.reset_index(drop=True)


def _load(path: Path):
    from ..spine_geometry import load_trimesh

    return load_trimesh(path, process=False)


def attachment_of_spine(spine, full_vertices_tree, full_faces: set, shaft) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    """(centre nm, cap area nm^2, method) of a closed LabID spine mesh (already in nm)."""
    _, idx = full_vertices_tree.query(spine.vertices)
    mapped = np.sort(idx[spine.faces], axis=1)
    new = np.fromiter((tuple(r) not in full_faces for r in mapped), dtype=bool, count=len(mapped))
    if new.any() and new.sum() < len(new):
        area = spine.area_faces[new]
        if area.sum() > 0:
            centre = (spine.triangles_center[new] * area[:, None]).sum(0) / area.sum()
            return centre, float(area.sum()), "seal_cap"
    _, dist, _ = shaft.nearest.on_surface(spine.vertices)
    return np.asarray(spine.vertices[int(np.argmin(dist))], dtype=float), None, "nearest_vertex"


def skeleton_polyline(shaft_nm, timeout: float = 300.0) -> np.ndarray:
    from ..spine_geometry import cgal_skeleton_segments_from_trimesh_isolated, longest_path_polyline

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shaft.off"
        shaft_nm.export(path)
        segments = cgal_skeleton_segments_from_trimesh_isolated(path, timeout=timeout)
    return longest_path_polyline(segments)


def process_dendrite(cfg: BranchPreprocessingConfig, ctx: Dict[str, Any], item: Dict[str, Any], out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import trimesh
    from scipy.spatial import cKDTree

    d = Path(item["path"])
    neuron_id, merged_id, limb = ctx["neuron_id"], int(item["merged_branch_id"]), 0
    health, treatment, series = GROUPS.get(item["group"].lower(), (None, None, item["group"]))
    pure = _load(d / "pure_surface_mesh.off")
    shaft = trimesh.Trimesh(np.asarray(pure.vertices, dtype=float) * UNIT_NM, np.asarray(pure.faces), process=False)
    full = _load(d / "surface_mesh.off")
    full_v = np.asarray(full.vertices, dtype=float) * UNIT_NM
    full_faces = set(map(tuple, np.sort(np.asarray(full.faces), axis=1)))
    tree = cKDTree(full_v)
    spine_paths = sorted(d.glob("spine_*.off"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))

    row: Dict[str, Any] = {
        "dataset": cfg.dataset, "neuron_id": neuron_id, "limb_id": limb, "merged_branch_id": merged_id,
        "member_branch_ids": str(merged_id), "member_count": 1, "merged_parent_branch_id": -1, "merged_branch_depth": None,
        "is_terminal": None, "source_dir": f"{item['group']}/{item['dendrite']}", "labid_group": item["group"],
        "species": "mouse", "health": health, "treatment": treatment, "imaging_series": series,
        "cell_type": None, "cell_type_binary": None, "compartment": None,
        "split": ctx["split"], "status": STATUS_OK, "error": None,
        "preprocessing_version": cfg.algorithm_version, "config_hash": ctx["config_hash"], "geometry_path": None,
        "merge_max_gap_nm": 0.0, "skeleton_topology_ok": True, "proximal_end_method": "unknown",
    }
    segments, spine_rows = [], []
    for path in spine_paths:
        spine_id = re.sub(r"\D", "", path.stem)
        mesh = _load(path)
        spine = trimesh.Trimesh(np.asarray(mesh.vertices, dtype=float) * UNIT_NM, np.asarray(mesh.faces), process=False)
        centre, cap_area, method = attachment_of_spine(spine, tree, full_faces, shaft)
        info = {"origin_global": centre.tolist() if centre is not None else None, "attachment_cap_area": cap_area, "attachment_loop_area": None,
                "attachment_ambiguous": method != "seal_cap"}
        segments.append({"branch": merged_id, "spine_id": spine_id, "path": path, "status": "valid", "info": info, "method": method})
        spine_rows.append({"dataset": cfg.dataset, "neuron_id": neuron_id, "limb_id": limb, "merged_branch_id": merged_id, "original_branch_id": merged_id,
                           "spine_id": spine_id, "status": "valid", "has_position": False, "split": ctx["split"], "attachment_method": method})
    row.update(n_segments_total=len(segments), n_false=0, n_unknown=0, n_spines=len(segments), unknown_fraction=0.0)

    polyline = skeleton_polyline(shaft)
    row["table_length_nm"] = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())  # skeleton length (no table for LabID)
    if row["table_length_nm"] < cfg.length_min_p_nm:
        row["status"] = STATUS_TOO_SHORT
        _finalize_eligibility(cfg, row)
        return row, spine_rows
    vertices, faces = np.asarray(shaft.vertices), np.asarray(shaft.faces, dtype=np.int64)
    return branch_geometry(cfg, ctx, row, segments, spine_rows, polyline, vertices, faces, np.zeros(len(vertices), dtype=bool), out_dir, limb, merged_id)


def process_cell(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Runner unit for LabID: every dendrite of one cell; writes the same per-neuron tables as ``process.process_neuron``."""
    started = time.perf_counter()
    cfg = BranchPreprocessingConfig.from_mapping(payload["config"])
    neuron_id = payload["neuron_id"]
    out_dir = cfg.out_root / neuron_id
    ctx = {"neuron_id": neuron_id, "split": payload.get("split"), "soma": _NoSoma(), "soma_midpoints": {}, "config_hash": payload["config_hash"]}
    items = payload["dendrites"]
    partial = bool(cfg.limit_branches or cfg.merged_branch_ids)
    if cfg.merged_branch_ids:
        items = [i for i in items if int(i["merged_branch_id"]) in {int(b) for b in cfg.merged_branch_ids}]
    if cfg.limit_branches:
        items = items[: int(cfg.limit_branches)]
    rows, spine_rows = [], []
    for item in items:
        try:
            row, srows = process_dendrite(cfg, ctx, item, out_dir)
        except Exception as exc:  # one bad dendrite must not stop the cell
            row = {"dataset": cfg.dataset, "neuron_id": neuron_id, "limb_id": 0, "merged_branch_id": int(item["merged_branch_id"]),
                   "source_dir": f"{item['group']}/{item['dendrite']}", "status": STATUS_FAILED,
                   "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                   "split": ctx["split"], "config_hash": ctx["config_hash"], "preprocessing_version": cfg.algorithm_version}
            _finalize_eligibility(cfg, row)
            srows = []
        rows.append(row)
        spine_rows.extend(srows)
    branches = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(branches, out_dir / "branches.parquet")
    _atomic_parquet(pd.DataFrame(spine_rows), out_dir / "branch_spines.parquet")
    summary = {
        "neuron_id": neuron_id, "config_hash": payload["config_hash"], "n_branches": int(len(branches)),
        "n_failed": int((branches["status"] == STATUS_FAILED).sum()), "n_ok": int((branches["status"] == STATUS_OK).sum()),
        "n_train_eligible_T": int(branches.get("train_eligible_T", pd.Series(dtype=bool)).sum()),
        "n_train_eligible_P": int(branches.get("train_eligible_P", pd.Series(dtype=bool)).sum()),
        "seconds": round(time.perf_counter() - started, 2), "partial": partial, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = out_dir / "_done.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "_done.json")
    return summary


def build_payloads(cfg: BranchPreprocessingConfig, cells: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    from .inputs import neuron_split_lookup

    frame = discover(cfg)
    split = neuron_split_lookup(cfg.splits)
    if cells is not None:
        frame = frame[frame["neuron_id"].isin(set(cells))]
    if cfg.limit_neurons:
        keep = sorted(frame["neuron_id"].unique())[: int(cfg.limit_neurons)]
        frame = frame[frame["neuron_id"].isin(keep)]
    return [{"neuron_id": n, "config": cfg.to_dict(), "config_hash": cfg.config_hash(), "dendrites": g.to_dict("records"), "split": split.get(n)}
            for n, g in frame.groupby("neuron_id", sort=True)]
