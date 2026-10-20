"""Per-neuron / per-merged-branch processing (stages B1-B4, counts, QC).

``process_neuron(payload)`` is the unit of work of the runner: it processes every merged branch of
one neuron and writes, under ``<out_root>/<neuron_id>/``:

- ``limb_<L>/merged_<B>/branch_geometry.npz``  nodes s_j, gamma, radius, RMF, kappa1/kappa2, raw skeleton, offsets;
- ``limb_<L>/merged_<B>/branch_sections.npz``  per-section estimates before smoothing + flags;
- ``branches.parquet`` (one row per merged branch), ``branch_spines.parquet`` (one row per spine segment);
- ``_done.json`` (written last; the runner skips neurons whose marker has the current config hash).

A failing branch becomes a row with ``status="failed"`` and the error text; the neuron goes on.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import branch_geometry as bg
from .config import BranchPreprocessingConfig
from .inputs import as_float_array
from .soma import SomaDistance

STATUS_OK = "ok"
STATUS_TOO_SHORT = "too_short"
STATUS_NO_CENTERLINE = "no_centerline"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _load_vertices_faces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    from ..spine_geometry import load_trimesh

    mesh = load_trimesh(path, process=False)
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def _skeleton_chain(member_segments: List[List[Tuple[np.ndarray, np.ndarray]]], join_distance: float) -> Dict[str, Any]:
    """Ordered polyline of a merged branch from the skeleton segments of its members.

    Connected pieces (members usually share their junction vertex) are path-ordered; separate pieces
    are chained greedily by their nearest ends. Returns the polyline, the largest join gap and
    whether some piece is not a simple path (branching / cycle)."""
    import networkx as nx

    from ..spine_geometry import graph_from_segments

    segments = [seg for member in member_segments for seg in member]
    graph = graph_from_segments(segments)
    pieces = []
    topology_ok = True
    for comp in nx.connected_components(graph):
        sub = graph.subgraph(comp)
        if sub.number_of_nodes() < 2:
            continue
        degrees = dict(sub.degree())
        is_path = sub.number_of_edges() == sub.number_of_nodes() - 1 and max(degrees.values()) <= 2
        topology_ok &= is_path
        ends = [n for n, d in degrees.items() if d == 1]
        if is_path and len(ends) == 2:
            order = nx.shortest_path(sub, ends[0], ends[1])
        else:  # fall back to the longest shortest-path between leaves (or any two nodes)
            candidates = ends if len(ends) >= 2 else list(sub.nodes)
            best, order = -1.0, None
            for i, src in enumerate(candidates):
                lengths, paths = nx.single_source_dijkstra(sub, src, weight="length")
                for dst in candidates[i + 1:]:
                    if lengths.get(dst, -1.0) > best:
                        best, order = lengths[dst], paths[dst]
        pieces.append(np.vstack([sub.nodes[n]["pos"] for n in order]))
    if not pieces:
        raise ValueError("empty skeleton")
    pieces.sort(key=lambda p: -float(bg.arclength(p)[-1]))
    n_pieces = len(pieces)
    chain = pieces.pop(0)
    max_gap = 0.0
    while pieces:
        best = None
        for idx, piece in enumerate(pieces):
            for at_end in (True, False):
                anchor = chain[-1] if at_end else chain[0]
                for reverse in (False, True):
                    candidate = piece[::-1] if reverse else piece
                    gap = float(np.linalg.norm((candidate[0] if at_end else candidate[-1]) - anchor))
                    if best is None or gap < best[0]:
                        best = (gap, idx, at_end, candidate)
        gap, idx, at_end, candidate = best
        chain = np.vstack([chain, candidate]) if at_end else np.vstack([candidate, chain])
        pieces.pop(idx)
        max_gap = max(max_gap, gap)
    return {"polyline": chain, "max_gap": max_gap, "n_pieces": n_pieces, "topology_ok": topology_ok}


def _spine_files(branch_dir: Path) -> Dict[str, Path]:
    spines = branch_dir / "spines"
    if not spines.is_dir():
        return {}
    out = {}
    for path in spines.glob("spine_*.off"):
        out[path.stem.split("_", 1)[1]] = path
    return out


def _json_centroid(path: Any) -> Optional[np.ndarray]:
    if not isinstance(path, str) or not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fd:
            return as_float_array(json.load(fd).get("attachment_loop_centroid_global"))
    except (OSError, ValueError):
        return None


def branch_segments(member_dirs: Dict[int, Path], spine_info: Dict[Tuple[str, str, str], Dict[str, Any]], limb: int) -> List[Dict[str, Any]]:
    """Spine segments of the member branches with their S verdict: valid / invalid (false spine) / unknown."""
    limb_str = f"{limb:03d}"
    segments = []
    for b, d in member_dirs.items():
        for spine_id, path in sorted(_spine_files(d).items()):
            info = spine_info.get((limb_str, f"{b:03d}", spine_id), {})
            status = info.get("status")
            status = status if status in ("valid", "invalid") else "unknown"
            segments.append({"branch": b, "spine_id": spine_id, "path": path, "status": status, "info": info})
    return segments


def load_branch_surface(cfg: BranchPreprocessingConfig, member_dirs: Dict[int, Path], segments: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, faces, spine_mask) of the merged branch mesh. ``spine_mask`` marks the vertices of real
    spines and of segments without a verdict; false spines (``invalid``) are part of the shaft and stay."""
    with ThreadPoolExecutor(max_workers=max(1, int(cfg.io_threads))) as pool:
        mesh_parts = list(pool.map(_load_vertices_faces, [d / "branch_mesh.off" for d in member_dirs.values()]))
        exclude = [s["path"] for s in segments if s["status"] != "invalid"]
        spine_vertices = list(pool.map(lambda p: _load_vertices_faces(p)[0], exclude))
    vertices = np.vstack([v for v, _ in mesh_parts])
    offsets = np.cumsum([0] + [len(v) for v, _ in mesh_parts[:-1]])
    faces = np.vstack([f + o for (_, f), o in zip(mesh_parts, offsets)])
    spine_v = np.vstack(spine_vertices) if spine_vertices else np.zeros((0, 3))
    return vertices, faces, bg.match_vertices(vertices, spine_v, cfg.vertex_decimals)


# ---------------------------------------------------------------------------
# centreline geometry (shared with scripts/neuron_model/study_centerline_smoothing.py)
# ---------------------------------------------------------------------------


def adaptive_smoothing_nm(cfg: BranchPreprocessingConfig, r_med: float) -> float:
    """Width of the centreline smoothing: fixed ``centerline_smoothing_nm``, or with ``centerline_smoothing_factor`` > 0
    clip(factor * r_med, centerline_smoothing_min_nm, centerline_smoothing_nm): thin dendrites wind on a sub-micron scale
    comparable to their radius, a 1 um smoothing takes the line out of the tube."""
    if cfg.centerline_smoothing_factor and np.isfinite(r_med):
        return float(np.clip(cfg.centerline_smoothing_factor * r_med, cfg.centerline_smoothing_min_nm, cfg.centerline_smoothing_nm))
    return float(cfg.centerline_smoothing_nm)


def adaptive_node_step_nm(cfg: BranchPreprocessingConfig, smoothing_nm: float) -> float:
    """Node spacing: fixed ``node_step_nm``, or with ``node_step_per_smoothing`` > 0 clip(f * smoothing, node_step_min_nm,
    node_step_nm) so that the nodes resolve what the smoothed line still contains."""
    if cfg.node_step_per_smoothing:
        return float(np.clip(cfg.node_step_per_smoothing * smoothing_nm, cfg.node_step_min_nm, cfg.node_step_nm))
    return float(cfg.node_step_nm)


def centerline_geometry(centers, radii, *, smoothing_nm: float, radius_smoothing_nm: float, node_step_nm: float, straight_fraction: float, max_nodes: int = 512,
                        ellipse: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
    """Reliable section centres (ordered) -> smoothed line, node grid, radius profile, RMF, bending, round trip.

    ``ellipse`` = (a, b, major-axis unit vectors [n,3]) of the same sections (NaN where no ellipse): adds the node profiles
    ``ellipse_a``, ``ellipse_b`` and ``ellipse_angle`` (angle of the major axis in the RMF, from e1 towards e2, in (-pi/2, pi/2])."""
    from scipy.spatial import cKDTree

    centerline = bg.smooth_centerline(centers, smoothing_nm=smoothing_nm)
    length = centerline.length
    dense_tree = cKDTree(centerline.dense)
    _, idx = dense_tree.query(centers)
    s_rel = centerline.s[idx]
    order = np.argsort(s_rel)
    s_rel, r_rel = s_rel[order], np.asarray(radii)[order]
    keep = np.r_[True, np.diff(s_rel) > 1e-6]
    s_rel, r_rel = s_rel[keep], r_rel[keep]
    gaps = np.diff(np.r_[0.0, s_rel, length])
    step = max(float(node_step_nm), length / max(max_nodes - 1, 1))  # at most max_nodes nodes (the T representation)
    nodes, ds_b = bg.node_grid(length, step)
    if len(s_rel) >= 5:
        radius = bg.smooth_series(s_rel, r_rel, radius_smoothing_nm)(np.clip(nodes, s_rel[0], s_rel[-1]))
    else:
        radius = np.interp(nodes, s_rel, r_rel)
    radius = np.clip(radius, 0.5 * float(r_rel.min()), None)
    gamma = centerline.at(nodes)
    t_nodes = centerline.tangent_at(nodes)
    e1_0, reference = bg.initial_normal(gamma, t_nodes[0], straight_fraction)
    t_nodes, e1, e2 = bg.rotation_minimizing_frames(gamma, t_nodes, e1_0)
    k1, k2, twist = bg.bending_from_frames(t_nodes, e1, e2, ds_b)
    frame0 = np.stack([t_nodes[0], e1[0], e2[0]], axis=1)
    rebuilt, _ = bg.integrate_bending(k1, k2, ds_b, gamma[0], frame0)
    out = {
        "centerline": centerline, "length": length, "longest_gap": float(gaps.max()) if len(gaps) else float("nan"),
        "nodes": nodes, "step": ds_b, "radius": radius, "gamma": gamma, "tangent": t_nodes, "e1": e1, "e2": e2,
        "frame_reference": reference, "kappa1": k1, "kappa2": k2, "twist": twist, "rebuilt": rebuilt,
        "ellipse_a": None, "ellipse_b": None, "ellipse_angle": None, "ellipse_valid_fraction": 0.0,
    }
    if ellipse is not None:
        out.update(_ellipse_profile(centerline.s[idx], ellipse, nodes, e1, e2, radius_smoothing_nm))
    return out


def _ellipse_profile(s_c: np.ndarray, ellipse, nodes: np.ndarray, e1: np.ndarray, e2: np.ndarray, smoothing_nm: float) -> Dict[str, Any]:
    """Section ellipses -> smoothed a(s), b(s) and the major-axis angle in the RMF at the nodes (doubled-angle smoothing: an
    ellipse is symmetric under a rotation by pi). Fewer than 5 valid sections -> no profile (None)."""
    a, b, major = (np.asarray(x, dtype=float) for x in ellipse)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(major).all(axis=1)
    res: Dict[str, Any] = {"ellipse_valid_fraction": float(valid.mean()) if len(a) else 0.0}
    if valid.sum() < 5:
        return res
    s_v, a_v, b_v, m_v = s_c[valid], a[valid], b[valid], major[valid]
    e1_s = np.column_stack([np.interp(s_v, nodes, e1[:, k]) for k in range(3)])
    e2_s = np.column_stack([np.interp(s_v, nodes, e2[:, k]) for k in range(3)])
    phi = np.arctan2((m_v * e2_s).sum(1), (m_v * e1_s).sum(1))
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    order = np.argsort(s_v)
    s_v, a_v, b_v, c2, s2 = s_v[order], a_v[order], b_v[order], c2[order], s2[order]
    keep = np.r_[True, np.diff(s_v) > 1e-6]
    s_v, a_v, b_v, c2, s2 = s_v[keep], a_v[keep], b_v[keep], c2[keep], s2[keep]
    if len(s_v) < 5:
        return res
    at = np.clip(nodes, s_v[0], s_v[-1])
    sm = lambda y: bg.smooth_series(s_v, y, smoothing_nm)(at)
    a_n, b_n = np.clip(sm(a_v), 1.0, None), np.clip(sm(b_v), 1.0, None)
    ang = 0.5 * np.arctan2(sm(s2), sm(c2))
    swap = b_n > a_n  # keep a >= b: swapping the axes turns the major direction by pi/2
    a_n[swap], b_n[swap] = b_n[swap].copy(), a_n[swap].copy()
    ang[swap] = ang[swap] + np.pi / 2
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2
    res.update(ellipse_a=a_n, ellipse_b=b_n, ellipse_angle=ang)
    return res


def line_inside_fraction(gamma: np.ndarray, triangles: np.ndarray, *, margin: float) -> float:
    """Fraction of the node polyline (nodes + midpoints, ends within ``margin`` excluded) inside the branch mesh
    (winding number > 0.5; the mesh is open only at the cut ends, which the margin keeps away from)."""
    from ..spine_sampling import winding_numbers

    pts = np.vstack([gamma, 0.5 * (gamma[1:] + gamma[:-1])])
    s = np.r_[bg.arclength(gamma), 0.5 * (bg.arclength(gamma)[1:] + bg.arclength(gamma)[:-1])]
    inner = (s > margin) & (s < s.max() - margin)
    if inner.sum() < 3:
        inner = np.ones(len(pts), dtype=bool)
    return float((winding_numbers(pts[inner], triangles) > 0.5).mean())


# ---------------------------------------------------------------------------
# one merged branch
# ---------------------------------------------------------------------------


def process_branch(cfg: BranchPreprocessingConfig, ctx: Dict[str, Any], group: pd.DataFrame, out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    neuron_id = ctx["neuron_id"]
    limb = int(group["limb_id"].iloc[0])
    merged_id = int(group["merged_branch_id"].iloc[0])
    members = sorted(int(b) for b in group["original_branch_id"])
    limb_dir = Path(ctx["neuron_dir"]) / f"limb_{limb:03d}"
    member_dirs = {b: limb_dir / f"branch_{b:03d}" for b in members}
    meta = ctx["metadata"]
    row: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "neuron_id": neuron_id,
        "limb_id": limb,
        "merged_branch_id": merged_id,
        "member_branch_ids": "|".join(map(str, members)),
        "member_count": len(members),
        "merged_parent_branch_id": int(group["merged_parent_branch_id"].iloc[0]),
        "merged_branch_depth": int(group["merged_branch_depth"].iloc[0]),
        "is_terminal": bool(group["merged_is_terminal"].iloc[0]),
        "table_length_nm": float(group["merged_length_nm"].iloc[0]),
        "species": meta["species"],
        "health": meta["health"],
        "cell_type": meta["cell_type"],
        "cell_type_binary": meta["cell_type_binary"],
        "compartment": meta["compartment_by_limb"].get(limb),
        "split": ctx["split"],
        "status": STATUS_OK,
        "error": None,
        "preprocessing_version": cfg.algorithm_version,
        "config_hash": ctx["config_hash"],
        "geometry_path": None,
    }

    # ---- spine segments and their S verdicts --------------------------------------------
    segments = branch_segments(member_dirs, ctx["spines"], limb)
    n_total = len(segments)
    n_false = sum(s["status"] == "invalid" for s in segments)
    n_unknown = sum(s["status"] == "unknown" for s in segments)
    row.update(
        n_segments_total=n_total,
        n_false=n_false,
        n_unknown=n_unknown,
        n_spines=n_total - n_false,
        unknown_fraction=(n_unknown / n_total) if n_total else 0.0,
    )
    spine_rows: List[Dict[str, Any]] = [
        {
            "dataset": cfg.dataset,
            "neuron_id": neuron_id,
            "limb_id": limb,
            "merged_branch_id": merged_id,
            "original_branch_id": s["branch"],
            "spine_id": s["spine_id"],
            "status": s["status"],
            "has_position": False,
            "split": ctx["split"],
        }
        for s in segments
    ]

    if row["table_length_nm"] < cfg.length_min_p_nm:
        row["status"] = STATUS_TOO_SHORT
        _finalize_eligibility(cfg, row)
        return row, spine_rows

    # ---- B1: skeleton chain, orientation, mesh, shaft-only vertices ----------------------
    from ..spine_geometry import load_skeleton_segments

    member_segments = [load_skeleton_segments(d / "branch_skeleton.npy") for d in member_dirs.values()]
    chain = _skeleton_chain(member_segments, cfg.join_distance_nm)
    polyline = chain["polyline"]
    row["merge_max_gap_nm"] = chain["max_gap"]
    row["skeleton_topology_ok"] = bool(chain["topology_ok"])
    soma: SomaDistance = ctx["soma"]
    d_first, d_last = soma.distance(limb, polyline[0]), soma.distance(limb, polyline[-1])
    if np.isfinite(d_first) and np.isfinite(d_last):
        if d_last < d_first:
            polyline = polyline[::-1]
        row["proximal_end_method"] = "limb_graph"
    else:
        row["proximal_end_method"] = "unknown"

    vertices, faces, spine_mask = load_branch_surface(cfg, member_dirs, segments)
    shaft = vertices[~spine_mask]
    row["n_mesh_vertices"] = int(len(vertices))
    row["n_spine_vertices_removed"] = int(spine_mask.sum())

    # ---- B2: sections ------------------------------------------------------------------
    g = bg.resample_polyline(polyline, cfg.initial_step_nm)
    k = max(1, int(round(cfg.tangent_half_window_nm / cfg.initial_step_nm)))
    tangents = bg.smoothed_tangents(g, k)
    from scipy.spatial import cKDTree

    tree = cKDTree(shaft) if len(shaft) else None
    kwargs = dict(slab=cfg.slab_half_thickness_nm, cluster_eps=cfg.cluster_eps_nm, robust_k=cfg.robust_k, min_points=cfg.min_section_points, tree=tree, fit_ellipse=cfg.fit_ellipse)
    sections = bg.fit_sections(g, tangents, shaft, max_r=cfg.max_section_radius_nm, **kwargs)
    max_r_used = cfg.max_section_radius_nm
    if cfg.adaptive_max_radius_factor and np.isfinite(sections.radii).any():
        r1 = float(np.nanmedian(sections.radii))
        if cfg.adaptive_max_radius_factor * r1 > cfg.max_section_radius_nm:
            max_r_used = cfg.adaptive_max_radius_factor * r1
            sections = bg.fit_sections(g, tangents, shaft, max_r=max_r_used, **kwargs)
    row["section_max_radius_nm"] = max_r_used
    inside = None
    fitted = np.isfinite(sections.radii)
    if cfg.outside_mesh_check and fitted.any():
        from ..spine_sampling import winding_numbers

        inside = np.zeros(len(g), dtype=bool)
        inside[fitted] = winding_numbers(sections.centers[fitted], vertices[faces]) > 0.5
    flags, r_med = bg.flag_sections(
        sections,
        min_points=cfg.min_section_points,
        large_radius_factor=cfg.large_radius_factor,
        kink_factor=cfg.kink_factor,
        kink_floor_nm=cfg.kink_floor_nm,
        end_distance_factor=cfg.end_distance_factor,
        inside=inside,
    )
    reliable = flags == 0
    row["n_sections"] = int(len(g))
    row["n_reliable_sections"] = int(reliable.sum())
    row["section_invalid_fraction"] = float(1.0 - reliable.mean()) if len(g) else 1.0
    for name, count in bg.flag_reasons(flags).items():
        row[f"sections_{name}"] = count
    row["radius_median_nm"] = r_med
    offset = np.linalg.norm(sections.points - sections.centers, axis=1)
    geometry_dir = out_dir / f"limb_{limb:03d}" / f"merged_{merged_id}"
    _atomic_npz(
        geometry_dir / "branch_sections.npz",
        points=sections.points, tangents=sections.tangents, centers=sections.centers, radii=sections.radii,
        n_points=sections.n_points, flags=flags,
        **({"ellipse_center": sections.ellipse_center, "ellipse_a": sections.ellipse_a, "ellipse_b": sections.ellipse_b, "ellipse_major": sections.ellipse_major}
           if sections.ellipse_a is not None else {}),
    )
    if reliable.sum() < cfg.min_reliable_sections:
        row["status"] = STATUS_NO_CENTERLINE
        _finalize_eligibility(cfg, row)
        return row, spine_rows

    s_init = bg.arclength(g)
    first, last = int(np.flatnonzero(reliable)[0]), int(np.flatnonzero(reliable)[-1])
    row["trim_start_nm"] = float(s_init[first])
    row["trim_end_nm"] = float(s_init[-1] - s_init[last])

    r_med_fit = float(r_med) if r_med is not None and np.isfinite(r_med) else float(np.nanmedian(sections.radii[reliable]))
    smoothing = adaptive_smoothing_nm(cfg, r_med_fit)
    line = centerline_geometry(
        sections.centers[reliable], sections.radii[reliable], smoothing_nm=smoothing, radius_smoothing_nm=max(cfg.radius_smoothing_nm, smoothing),
        node_step_nm=adaptive_node_step_nm(cfg, smoothing), straight_fraction=cfg.straight_fraction, max_nodes=cfg.max_nodes,
        ellipse=(sections.ellipse_a[reliable], sections.ellipse_b[reliable], sections.ellipse_major[reliable]) if sections.ellipse_a is not None else None,
    )
    row["ellipse_valid_fraction"] = line["ellipse_valid_fraction"]
    if line["ellipse_a"] is not None:
        ratio = line["ellipse_b"] / line["ellipse_a"]
        row["ellipse_ratio_median"] = float(np.median(ratio))
        row["ellipse_a_median_nm"] = float(np.median(line["ellipse_a"]))
        row["ellipse_b_median_nm"] = float(np.median(line["ellipse_b"]))
        row["ellipse_req_median_nm"] = float(np.median(np.sqrt(line["ellipse_a"] * line["ellipse_b"])))
    centerline, length = line["centerline"], line["length"]
    row["length_nm"] = length
    row["centerline_rms_to_centers_nm"] = centerline.rms_to_centers
    row["centerline_smoothing_used_nm"] = float(smoothing)
    row["longest_gap_nm"] = line["longest_gap"]
    nodes, ds_b, radius, gamma = line["nodes"], line["step"], line["radius"], line["gamma"]
    t_nodes, e1, e2, reference = line["tangent"], line["e1"], line["e2"], line["frame_reference"]
    k1, k2, twist, rebuilt = line["kappa1"], line["kappa2"], line["twist"], line["rebuilt"]
    row["centerline_inside_fraction"] = line_inside_fraction(gamma, vertices[faces], margin=cfg.end_distance_factor * r_med_fit)
    kappa = np.hypot(k1, k2)
    contact, margin = bg.tube_self_contact(gamma, radius, nodes, ds_b)
    r_mid = 0.5 * (radius[:-1] + radius[1:])
    row.update(
        n_nodes=int(len(nodes)),
        step_nm=float(ds_b),
        frame_reference=reference,
        radius_proximal_nm=float(np.mean(radius[nodes <= 0.1 * length])),
        radius_distal_nm=float(np.mean(radius[nodes >= 0.9 * length])),
        tortuosity=float(length / max(np.linalg.norm(gamma[-1] - gamma[0]), 1e-9)),
        kappa_rms=float(np.sqrt(np.mean(kappa**2))) if len(kappa) else 0.0,
        kappa_max=float(kappa.max()) if len(kappa) else 0.0,
        max_r_kappa=float((r_mid * kappa).max()) if len(kappa) else 0.0,
        twist_max_per_nm=float(twist.max()) if len(twist) else 0.0,
        gamma_roundtrip_error_nm=float(np.max(np.linalg.norm(rebuilt - gamma, axis=1))),
        tube_self_contact=bool(contact),
        tube_contact_margin_nm=float(margin),
        area_shaft_nm2=float(_trapezoid(2.0 * np.pi * radius, nodes)),
    )
    shaft_faces = faces[~spine_mask[faces].any(axis=1)]
    tri = vertices[shaft_faces]
    row["area_shaft_from_mesh_nm2"] = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())

    # soma distance
    d_start = soma.distance(limb, gamma[0])
    row["soma_distance_start_nm"] = d_start if np.isfinite(d_start) else None
    row["soma_distance_source"] = "limb_graph" if np.isfinite(d_start) else None
    table_mid = [ctx["soma_midpoints"].get(p) for p in group["prefix"]]
    table_mid = [v for v in table_mid if v is not None]
    row["soma_midpoint_table_nm"] = float(table_mid[0]) if table_mid else None
    if table_mid and np.isfinite(d_start):
        ours = d_start + 0.5 * length
        row["soma_distance_mismatch"] = bool(abs(ours - table_mid[0]) > cfg.soma_mismatch_fraction * max(abs(table_mid[0]), 1.0))
    else:
        row["soma_distance_mismatch"] = None

    # ---- B4: attachment coordinates --------------------------------------------------------
    centroids, owners = [], []
    for i, s in enumerate(segments):
        if s["status"] != "valid":
            continue
        info = s["info"]
        a = as_float_array(info.get("origin_global"))
        source = "oriented"
        if a is None and info.get("seal_status") == "success":
            a, source = _json_centroid(info.get("attachment_region_path")), "json"
        if a is None:
            continue
        centroids.append(a)
        owners.append((i, source))
    n_in_range = n_with_position = n_end_clamped = 0
    if centroids:
        proj = bg.project_points(np.vstack(centroids), gamma, nodes, t_nodes, e1)
        r_at = np.interp(proj.s, nodes, radius)
        kappa_nodes = np.r_[kappa, kappa[-1:]] if len(kappa) else np.zeros(len(nodes))
        k_at = np.interp(proj.s, nodes, kappa_nodes)
        dist_perp = np.linalg.norm(proj.v_perp, axis=1)
        for n, (i, source) in enumerate(owners):
            clamped = bool(proj.at_end[n] != 0 and proj.beyond[n] > cfg.end_clamp_factor * r_at[n])
            outlier = position_outlier(cfg, dist_perp[n] - r_at[n], r_at[n])
            info = segments[i]["info"]
            spine_rows[i].update(
                has_position=not clamped and not outlier,
                s=float(proj.s[n]),
                u=float(proj.s[n] / length),
                theta=float(proj.theta[n]),
                dist_perp=float(dist_perp[n]),
                dist_perp_minus_radius=float(dist_perp[n] - r_at[n]),
                radius_at_s=float(r_at[n]),
                kappa_at_s=float(k_at[n]),
                soma_distance_nm=float(d_start + proj.s[n]) if np.isfinite(d_start) else None,
                attachment_loop_area=_num(info.get("attachment_loop_area")),
                attachment_cap_area=_num(info.get("attachment_cap_area")),
                attachment_ambiguous=_bool(info.get("attachment_ambiguous")),
                projects_to_end=clamped,
                position_outlier=outlier,
                centroid_source=source,
            )
            n_end_clamped += clamped
            n_with_position += (not clamped and not outlier)
    row["n_end_clamped"] = int(n_end_clamped)
    row["n_spines_in_range"] = int(row["n_spines"] - n_end_clamped)
    row["n_with_position"] = int(n_with_position)

    kappa1_nodes = np.r_[k1, np.nan]
    kappa2_nodes = np.r_[k2, np.nan]
    geometry_path = geometry_dir / "branch_geometry.npz"
    _atomic_npz(
        geometry_path,
        s=nodes, gamma=gamma, radius=radius, frame_tangent=t_nodes, frame_e1=e1, frame_e2=e2,
        kappa1=kappa1_nodes, kappa2=kappa2_nodes, skeleton=polyline, skeleton_offset=offset,
        skeleton_offset_relative=_safe_ratio(offset, sections.radii),
        **({"ellipse_a": line["ellipse_a"], "ellipse_b": line["ellipse_b"], "ellipse_angle": line["ellipse_angle"]} if line["ellipse_a"] is not None else {}),
    )
    row["geometry_path"] = str(geometry_path)
    _finalize_eligibility(cfg, row)
    return row, spine_rows


def position_outlier(cfg: BranchPreprocessingConfig, d: float, r: float) -> bool:
    """Attachment centre too far inside (d < -inside_factor r) or outside (d > max(factor r, floor)) the trunk surface,
    d = |v_perp| - r(s). The centre lies systematically outside (the spine cut crosses the neck above the shaft)."""
    return bool(d < -cfg.position_inside_factor * r or d > max(cfg.position_outlier_factor * r, cfg.position_outlier_floor_nm))


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    return float(fn(y, x))


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def _num(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _bool(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return bool(value)


def _finalize_eligibility(cfg: BranchPreprocessingConfig, row: Dict[str, Any]) -> None:
    """QC flags and the two independent training-eligibility flags (spec, section 10)."""
    ok = row["status"] == STATUS_OK
    reasons: List[str] = []
    if row["status"] != STATUS_OK:
        reasons.append(row["status"])
    length = row.get("length_nm")
    r_med = row.get("radius_median_nm")
    qc_merge = ok and row.get("merge_max_gap_nm", 0.0) <= cfg.join_distance_nm and bool(row.get("skeleton_topology_ok", False))
    qc_cross = ok and row.get("section_invalid_fraction", 1.0) <= cfg.invalid_section_fraction_max
    qc_centerline = ok and row.get("longest_gap_nm", float("inf")) <= cfg.long_gap_nm and (
        row.get("centerline_rms_to_centers_nm", float("inf")) <= max(cfg.spline_rms_factor * (r_med or float("nan")), cfg.spline_rms_floor_nm)
    ) and row.get("centerline_inside_fraction", 1.0) >= cfg.centerline_inside_min
    # the double-reflection RMF leaves a discretisation twist ~0.05 kappa^2 * step (synthetic check): with the adaptive
    # smoothing kappa of thin branches reaches ~2e-3 /nm and the twist ~4e-5 /nm, so the limit is relative to kappa_max
    qc_frame = ok and row.get("twist_max_per_nm", float("inf")) <= max(cfg.twist_tolerance_per_nm, cfg.twist_relative_max * row.get("kappa_max", 0.0))
    qc_tube = ok and not row.get("tube_self_contact", True) and row.get("max_r_kappa", float("inf")) < 1.0
    qc_soma = ok and row.get("soma_distance_start_nm") is not None and not row.get("soma_distance_mismatch")
    in_range = row.get("n_spines_in_range")
    qc_spines = row.get("unknown_fraction", 1.0) <= cfg.unknown_fraction_max and (
        not in_range or row.get("n_with_position", 0) >= cfg.position_fraction_min * in_range
    )
    radius_ok = r_med is not None and np.isfinite(r_med) and cfg.radius_min_nm <= r_med <= cfg.radius_max_nm
    trim_ok = row.get("trim_start_nm", float("inf")) <= cfg.trim_max_nm and row.get("trim_end_nm", float("inf")) <= cfg.trim_max_nm
    for name, value in (("merge", qc_merge), ("cross_section", qc_cross), ("centerline", qc_centerline), ("frame", qc_frame), ("tube", qc_tube), ("spines", qc_spines), ("soma", qc_soma)):
        row[f"qc_{name}"] = bool(value)
        if ok and not value:
            reasons.append(name)
    if ok and not radius_ok:
        reasons.append("radius_range")
    if ok and not trim_ok:
        reasons.append("trim")
    length_ok_t = length is not None and cfg.length_min_t_nm <= length <= cfg.length_max_t_nm
    length_ok_p = length is not None and length >= cfg.length_min_p_nm
    row["train_eligible_T"] = bool(ok and length_ok_t and qc_merge and qc_cross and qc_centerline and qc_frame and qc_tube and radius_ok and trim_ok)
    row["train_eligible_P"] = bool(ok and length_ok_p and qc_merge and qc_centerline and qc_spines and radius_ok)
    row["qc_reason"] = ";".join(reasons) if reasons else None


# ---------------------------------------------------------------------------
# one neuron
# ---------------------------------------------------------------------------


def process_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Runner unit: every merged branch of one neuron. Returns a small summary dict."""
    started = time.perf_counter()
    cfg = BranchPreprocessingConfig.from_mapping(payload["config"])
    neuron_id = payload["neuron_id"]
    out_dir = cfg.out_root / neuron_id
    members = pd.DataFrame(payload["members"])
    ctx = {
        "neuron_id": neuron_id,
        "neuron_dir": cfg.dataset_root / neuron_id,
        "metadata": payload["metadata"],
        "split": payload.get("split"),
        "spines": {(r["limb_id"], r["branch_id"], r["spine_id"]): r for r in payload["spines"]},
        "soma": SomaDistance(cfg.dataset_root / neuron_id),
        "soma_midpoints": payload.get("soma_midpoints", {}),
        "config_hash": payload["config_hash"],
    }
    groups = list(members.groupby(["limb_id", "merged_branch_id"], sort=True))
    partial = False
    if cfg.merged_branch_ids:
        wanted = {int(b) for b in cfg.merged_branch_ids}
        groups = [(k, g) for k, g in groups if int(k[1]) in wanted]
        partial = True
    if cfg.limit_branches:
        groups = groups[: int(cfg.limit_branches)]
        partial = True
    rows, spine_rows = [], []
    for (_, merged_id), group in groups:
        try:
            row, srows = process_branch(cfg, ctx, group, out_dir)
        except Exception as exc:  # one bad branch must not stop the neuron
            row = {
                "dataset": cfg.dataset, "neuron_id": neuron_id, "limb_id": int(group["limb_id"].iloc[0]),
                "merged_branch_id": int(merged_id), "status": STATUS_FAILED,
                "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                "split": ctx["split"], "config_hash": ctx["config_hash"], "preprocessing_version": cfg.algorithm_version,
            }
            _finalize_eligibility(cfg, row)
            srows = []
        rows.append(row)
        spine_rows.extend(srows)
    branches = pd.DataFrame(rows)
    _atomic_parquet(branches, out_dir / "branches.parquet")
    _atomic_parquet(pd.DataFrame(spine_rows), out_dir / "branch_spines.parquet")
    summary = {
        "neuron_id": neuron_id,
        "config_hash": payload["config_hash"],
        "n_branches": int(len(branches)),
        "n_failed": int((branches["status"] == STATUS_FAILED).sum()),
        "n_ok": int((branches["status"] == STATUS_OK).sum()),
        "n_train_eligible_T": int(branches.get("train_eligible_T", pd.Series(dtype=bool)).sum()),
        "n_train_eligible_P": int(branches.get("train_eligible_P", pd.Series(dtype=bool)).sum()),
        "seconds": round(time.perf_counter() - started, 2),
        "partial": partial,  # a branch subset: never counted as a finished neuron
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = out_dir / "_done.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "_done.json")
    return summary
