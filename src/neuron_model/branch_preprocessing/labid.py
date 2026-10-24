"""LabID inputs of the branch preprocessing (the same B2-B4 geometry as Minnie65, ``process.branch_geometry``).

Input = the tree written by ``scripts/neuron_model/convert_labid_layout.py`` (``labid_layout.convert``), NOT the raw
LabID folders: ``<raw>/labid/<branch id>/limb_000/branch_000/`` with ``shaft_mesh.off`` (the shaft without spines,
closed), ``branch_skeleton.npy`` (longest path of the CGAL skeleton of the shaft), ``spines/spine_<n>.off`` (OPEN spine
meshes: the lab's seal removed) and ``_converted.json`` (group, source folder, scales). The conversion applies the
units (um -> nm) and the optional z scale, so the S preprocessing and this one see the same geometry; a new scale =
a new converted tree and new outputs of both.

LabID is light microscopy (confocal): the voxel size is not stored in the tif files (22 nm in xy and ~97 nm in z were
inferred from the mesh extents, to be confirmed by the lab).

Differences from Minnie65 (no limbs, soma, merge table):
- every LabID folder is a separate branch and its own grouping unit (``neuron_id`` = branch id, ``merged_branch_id`` 0;
  user decision 2026-10-24: whether two folders come from one dendrite / cell is unknown; the numbers of the folder
  name are kept as ``name_number_1/2`` only);
- with ``--s-preprocessed-root <converted>/labid/preprocessed`` (S preprocessing of the same tree) the attachment
  centre, areas and false-spine verdict come from the S tables, exactly as for Minnie65 (``attachment_method =
  s_preprocessing``); without them (or for a spine missing there) every spine counts as real and its attachment centre
  is the centroid of the boundary loop of the open spine mesh (= where it was cut from the dendrite surface);
- the branch ends are the image borders, not branch points: no soma distance, no proximal end (the skeleton order is
  kept, ``proximal_end_method = unknown``), ``is_terminal`` unknown; the length is a property of the field of view.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BranchPreprocessingConfig
from .process import STATUS_FAILED, STATUS_OK, STATUS_TOO_SHORT, _atomic_parquet, _finalize_eligibility, branch_geometry

from ..metadata import LABID_GROUPS

GROUPS = {k.lower(): v for k, v in LABID_GROUPS.items()}


class _NoSoma:
    def distance(self, limb: int, point: np.ndarray) -> float:
        return float("nan")


def discover(cfg: BranchPreprocessingConfig) -> pd.DataFrame:
    """One row per converted LabID branch: group, dendrite (source folder), path (branch dir), neuron_id, merged_branch_id."""
    rows = []
    root = cfg.dataset_root
    if not root.is_dir():
        return pd.DataFrame(columns=["group", "dendrite", "path", "neuron_id", "merged_branch_id"])
    for marker in sorted(root.glob("*/limb_000/branch_*/_converted.json")):
        meta = json.loads(marker.read_text(encoding="utf-8"))
        branch_dir = marker.parent
        rows.append({"group": meta.get("group"), "dendrite": meta.get("dendrite"), "path": str(branch_dir), "neuron_id": branch_dir.parent.parent.name,
                     "merged_branch_id": int(branch_dir.name.split("_")[-1]), "name_number_1": meta.get("name_number_1"),
                     "name_number_2": meta.get("name_number_2"), "unit_nm": meta.get("unit_nm"), "z_scale": meta.get("z_scale")})
    if not rows and any(root.glob("*/*/pure_surface_mesh.off")):
        raise ValueError(f"{root} is the RAW LabID layout: convert it first (scripts/neuron_model/convert_labid_layout.py) and pass the "
                         f"converted root as --raw-data-root")
    frame = pd.DataFrame(rows)
    if cfg.neuron_ids and len(frame):
        frame = frame[frame["neuron_id"].isin({str(n) for n in cfg.neuron_ids})]
    return frame.reset_index(drop=True)


def _load(path: Path):
    from ..spine_geometry import load_trimesh

    return load_trimesh(path, process=False)


def loop_centre(spine) -> Optional[np.ndarray]:
    """Centroid of the longest boundary loop of an open spine mesh (the cut from the dendrite surface), or None."""
    from ..spine_geometry import boundary_loops

    loops = boundary_loops(spine)
    if not loops:
        return None
    best = max(loops, key=lambda loop: loop.perimeter)
    return np.asarray(best.center, dtype=float)


def process_dendrite(cfg: BranchPreprocessingConfig, ctx: Dict[str, Any], item: Dict[str, Any], out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from ..spine_geometry import load_skeleton_segments

    d = Path(item["path"])
    neuron_id, merged_id, limb = ctx["neuron_id"], int(item["merged_branch_id"]), 0
    health, treatment, series = GROUPS.get(str(item["group"]).lower(), (None, None, item["group"]))
    shaft = _load(d / "shaft_mesh.off")
    spine_paths = sorted((d / "spines").glob("spine_*.off"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    row: Dict[str, Any] = {
        "dataset": cfg.dataset, "neuron_id": neuron_id, "limb_id": limb, "merged_branch_id": merged_id,
        "member_branch_ids": str(merged_id), "member_count": 1, "merged_parent_branch_id": -1, "merged_branch_depth": None,
        "is_terminal": None, "source_dir": f"{item['group']}/{item['dendrite']}", "labid_group": item["group"],
        "name_number_1": item.get("name_number_1"), "name_number_2": item.get("name_number_2"),
        "unit_nm": item.get("unit_nm"), "z_scale": item.get("z_scale"),
        "species": "mouse", "health": health, "treatment": treatment, "imaging_series": series,
        "cell_type": None, "cell_type_binary": None, "compartment": None,
        "split": ctx["split"], "status": STATUS_OK, "error": None,
        "preprocessing_version": cfg.algorithm_version, "config_hash": ctx["config_hash"], "geometry_path": None,
        "merge_max_gap_nm": 0.0, "skeleton_topology_ok": True, "proximal_end_method": "unknown",
    }
    segments, spine_rows = [], []
    for path in spine_paths:
        spine_id = re.sub(r"\D", "", path.stem)
        s_row = ctx.get("s_spines", {}).get((f"{merged_id:03d}", spine_id))
        s_origin = None if s_row is None else s_row.get("origin_global")
        if s_origin is not None and np.size(s_origin) == 3 and np.all(np.isfinite(np.asarray(s_origin, dtype=float))):
            # S preprocessing of the same converted tree: its stage-1 loop centre and areas, its verdict
            status = s_row.get("status") if s_row.get("status") in ("valid", "invalid") else "valid"
            info = {"origin_global": list(np.asarray(s_origin, dtype=float)), "attachment_cap_area": s_row.get("attachment_cap_area"),
                    "attachment_loop_area": s_row.get("attachment_loop_area"), "attachment_ambiguous": s_row.get("attachment_ambiguous")}
            method = "s_preprocessing"
        else:
            centre = loop_centre(_load(path))
            info = {"origin_global": centre.tolist() if centre is not None else None, "attachment_cap_area": None, "attachment_loop_area": None,
                    "attachment_ambiguous": centre is None}
            method, status = ("boundary_loop" if centre is not None else "none"), "valid"
        segments.append({"branch": merged_id, "spine_id": spine_id, "path": path, "status": status, "info": info, "method": method})
        spine_rows.append({"dataset": cfg.dataset, "neuron_id": neuron_id, "limb_id": limb, "merged_branch_id": merged_id, "original_branch_id": merged_id,
                           "spine_id": spine_id, "status": status, "has_position": False, "split": ctx["split"], "attachment_method": method})
    n_false = sum(seg["status"] == "invalid" for seg in segments)
    row.update(n_segments_total=len(segments), n_false=n_false, n_unknown=0, n_spines=len(segments) - n_false, unknown_fraction=0.0)

    segs = load_skeleton_segments(d / "branch_skeleton.npy")
    polyline = np.vstack([segs[0][0]] + [end for _, end in segs])  # the conversion stores one ordered path as segments
    row["table_length_nm"] = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())  # skeleton length (no table for LabID)
    if row["table_length_nm"] < cfg.length_min_p_nm:
        row["status"] = STATUS_TOO_SHORT
        _finalize_eligibility(cfg, row)
        return row, spine_rows
    vertices, faces = np.asarray(shaft.vertices, dtype=float), np.asarray(shaft.faces, dtype=np.int64)
    return branch_geometry(cfg, ctx, row, segments, spine_rows, polyline, vertices, faces, np.zeros(len(vertices), dtype=bool), out_dir, limb, merged_id)


def process_cell(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Runner unit for LabID: every dendrite of one cell; writes the same per-neuron tables as ``process.process_neuron``."""
    started = time.perf_counter()
    cfg = BranchPreprocessingConfig.from_mapping(payload["config"])
    neuron_id = payload["neuron_id"]
    out_dir = cfg.out_root / neuron_id
    ctx = {"neuron_id": neuron_id, "split": payload.get("split"), "soma": _NoSoma(), "soma_midpoints": {}, "config_hash": payload["config_hash"],
           "s_spines": {(str(r["branch_id"]), str(r["spine_id"])): r for r in payload.get("s_spines", [])}}
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
    # S tables of the converted tree (--s-preprocessed-root <converted>/labid/preprocessed), when present
    s_groups: Dict[str, List[Dict[str, Any]]] = {}
    if cfg.s_preprocessed_root and (Path(cfg.s_preprocessed_root) / "manifest.parquet").exists():
        from .inputs import load_spine_table

        table = load_spine_table(cfg, sorted(frame["neuron_id"].unique()))
        keep = ["branch_id", "spine_id", "status", "origin_global", "attachment_loop_area", "attachment_cap_area", "attachment_ambiguous"]
        for n, g in table.groupby("neuron_id"):
            s_groups[str(n)] = g[[c for c in keep if c in g.columns]].to_dict("records")
        print(f"[branches] labid: S tables {cfg.s_preprocessed_root}: {len(table)} spines of {len(s_groups)} cells", flush=True)
    return [{"neuron_id": n, "config": cfg.to_dict(), "config_hash": cfg.config_hash(), "dendrites": g.to_dict("records"), "split": split.get(n),
             "s_spines": s_groups.get(str(n), [])}
            for n, g in frame.groupby("neuron_id", sort=True)]
