"""Data of the neuron compartment viewer (scripts/neuron_model/view_neuron_compartments.py): whole neurons with their
limbs labelled (our direction rule, the colleague's length rule), branch radius, synapse points, initial limb directions,
and close-ups of single branches with their segmented spines. Pure functions, no HTML.

Coordinates on the page: um, relative to the soma centre (the data stay in nm until here). Display only: the branch
skeleton of every original branch is ordered into a polyline and resampled every ``step_nm``; meshes are decimated by
``html_kit.pack_mesh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

UM = 1000.0


def branch_polyline(segments: np.ndarray, step_nm: float = 500.0) -> np.ndarray:
    """Ordered points of a branch skeleton (segment soup [S, 2, 3], nm), resampled every ``step_nm``. A branch skeleton
    is an unbranched path; when it is not (or is broken), the longest walk from an end is used and the rest dropped."""
    seg = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if len(seg) == 0:
        return np.zeros((0, 3))
    pts = seg.reshape(-1, 3)
    nodes, inv = np.unique(np.round(pts, 1), axis=0, return_inverse=True)
    inv = inv.reshape(-1, 2)
    inv = inv[inv[:, 0] != inv[:, 1]]
    if len(inv) == 0:
        return nodes[:1]
    adj: Dict[int, List[int]] = {}
    for a, b in inv:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    ends = [n for n, nb in adj.items() if len(nb) == 1]
    start = ends[0] if ends else next(iter(adj))
    order, seen, cur = [start], {start}, start
    while True:
        nxt = [n for n in adj[cur] if n not in seen]
        if not nxt:
            break
        cur = nxt[0]
        seen.add(cur)
        order.append(cur)
    line = nodes[order]
    if len(line) < 2:
        return line
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    if d[-1] <= step_nm:
        return line[[0, -1]]
    s = np.r_[np.arange(0.0, d[-1], step_nm), d[-1]]
    return np.stack([np.interp(s, d, line[:, k]) for k in range(3)], axis=1)


def soma_center(neuron_dir: Path) -> Optional[np.ndarray]:
    from .compartments import soma_center as _sc

    return _sc(neuron_dir)


def neuron_skeleton(neuron_dir: Path, *, step_nm: float = 500.0) -> List[Dict[str, Any]]:
    """[{limb, branch, points [K, 3] nm}] for every ``limb_*/branch_*/branch_skeleton.npy``."""
    out = []
    for f in sorted(Path(neuron_dir).glob("limb_*/branch_*/branch_skeleton.npy")):
        limb = int(f.parent.parent.name.split("_")[-1])
        branch = int(f.parent.name.split("_")[-1])
        pts = branch_polyline(np.load(f), step_nm)
        if len(pts) >= 2:
            out.append({"limb": limb, "branch": branch, "points": pts})
    return out


def branch_radius_map(branches: pd.DataFrame, neuron_id: Any) -> Dict[Tuple[int, int], float]:
    """{(limb, original branch): median radius nm of its merged branch} from ``branches.parquet`` (member_branch_ids)."""
    b = branches[branches["neuron_id"].astype(str) == str(neuron_id)]
    out: Dict[Tuple[int, int], float] = {}
    for limb, members, r in zip(b["limb_id"], b["member_branch_ids"], b["radius_median_nm"]):
        for m in str(members).split("|"):
            if m.strip().lstrip("-").isdigit():
                out[(int(limb), int(m))] = float(r) if pd.notna(r) else float("nan")
    return out


def neuron_payload(neuron_dir: Path, *, neuron_id: str, cell_type: Optional[str], ours: Mapping[int, str], table: Mapping[int, str],
                   limb_info: pd.DataFrame, radius: Mapping[Tuple[int, int], float], points: pd.DataFrame, pia: Sequence[float],
                   features: Optional[pd.DataFrame] = None, step_nm: float = 500.0, soma_faces: int = 3000,
                   why: str = "") -> Dict[str, Any]:
    """Everything the page needs for one neuron (coordinates in um relative to the soma centre)."""
    from .reports.html_kit import clean_floats, pack_array, pack_mesh
    from .spine_geometry import load_trimesh

    neuron_dir = Path(neuron_dir)
    center = soma_center(neuron_dir)
    if center is None:
        raise FileNotFoundError(f"no soma mesh in {neuron_dir}")
    skel = neuron_skeleton(neuron_dir, step_nm=step_nm)
    xyz = np.concatenate([b["points"] for b in skel]) if skel else np.zeros((0, 3))
    offsets = np.cumsum([0] + [len(b["points"]) for b in skel])
    soma = load_trimesh(neuron_dir / "soma" / "soma_mesh.off", process=False)
    sv = (np.asarray(soma.vertices, float) - center) / UM
    pts = points[points["neuron_id"].astype(str) == str(neuron_id)]
    kinds = {"pre": 0, "post_shaft": 1, "post_spine": 2, "post_soma": 3}
    pts = pts[pts["kind"].isin(kinds)]
    limbs = []
    for _, row in limb_info.iterrows():
        limb = int(row["limb_id"])
        p = pts[pts["limb_id"] == limb] if "limb_id" in pts.columns else pts.iloc[:0]
        near = p["dist_nm"] <= 3000 if "dist_nm" in p.columns else np.ones(len(p), bool)
        limbs.append({"limb": limb, "ours": ours.get(limb, "not_applicable"), "table": table.get(limb),
                      "radius_nm": clean_floats([row.get("limb_radius_nm")])[0], "cable_mm": clean_floats([row.get("cable_nm", np.nan) / 1e6], 3)[0],
                      "cos": clean_floats([row.get("cos_pia")], 3)[0], "n_branches": clean_floats([row.get("limb_n_branches")])[0],
                      "pre": int(((p["kind"] == "pre") & near).sum()), "post": int((p["kind"].isin(["post_shaft", "post_spine"]) & near).sum())})
    arrows = []
    if features is not None and len(features):
        f = features[features["neuron_id"].astype(str) == str(neuron_id)]
        for _, row in f.iterrows():
            if row.get("dir_init") is None or row.get("root") is None:
                continue
            root = (np.asarray(row["root"], float) - center) / UM
            d = np.asarray(row["dir_init"], float)
            arrows.append({"limb": int(row["limb_id"]), "from": clean_floats(root, 5), "to": clean_floats(root + 30.0 * d, 5)})
    return {
        "neuron_id": str(neuron_id), "cell_type": cell_type, "why": why,
        "xyz": pack_array((xyz - center) / UM if len(xyz) else xyz, "float32"),
        "offsets": [int(o) for o in offsets],
        "branch_limb": [b["limb"] for b in skel], "branch_id": [b["branch"] for b in skel],
        "branch_radius": clean_floats([radius.get((b["limb"], b["branch"]), np.nan) for b in skel], 4),
        "soma": pack_mesh(sv, np.asarray(soma.faces), max_faces=soma_faces),
        "points": {"xyz": pack_array(((pts[["x", "y", "z"]].to_numpy(float) - center) / UM) if len(pts) else np.zeros((0, 3)), "float32"),
                   "kind": [kinds[k] for k in pts["kind"]], "limb": [int(v) for v in pts["limb_id"]] if "limb_id" in pts.columns else []},
        "limbs": limbs, "arrows": arrows, "pia": clean_floats(np.asarray(pia, float) / np.linalg.norm(pia), 5),
    }


def spine_example_payload(neuron_dir: Path, *, neuron_id: str, limb: int, branch: int, points: pd.DataFrame, title: str,
                          branch_faces: int = 25000, spine_faces: int = 400, max_spines: int = 120,
                          info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One original branch: its mesh, the meshes of its segmented spines (``spines/spine_*.off``) and the synapse
    points assigned to it (um, relative to the branch centroid)."""
    from .reports.html_kit import pack_array, pack_mesh
    from .spine_geometry import load_trimesh

    d = Path(neuron_dir) / f"limb_{limb:03d}" / f"branch_{branch:03d}"
    mesh = load_trimesh(d / "branch_mesh.off", process=False)
    v = np.asarray(mesh.vertices, float)
    c = v.mean(axis=0)
    spines = []
    for f in sorted((d / "spines").glob("spine_*.off"))[:max_spines]:
        try:
            s = load_trimesh(f, process=False)
        except Exception:  # noqa: BLE001 - a broken spine file only drops that spine from the picture
            continue
        spines.append(pack_mesh((np.asarray(s.vertices, float) - c) / UM, np.asarray(s.faces), max_faces=spine_faces))
    p = points[(points["neuron_id"].astype(str) == str(neuron_id)) & (points["limb_id"] == limb) & (points["branch_id"] == branch)]
    kinds = {"pre": 0, "post_shaft": 1, "post_spine": 2, "post_soma": 3}
    p = p[p["kind"].isin(kinds)]
    return {"neuron_id": str(neuron_id), "limb": int(limb), "branch": int(branch), "title": title, "info": info or {},
            "mesh": pack_mesh((v - c) / UM, np.asarray(mesh.faces), max_faces=branch_faces), "spines": spines,
            "n_spine_files": int(len(list((d / "spines").glob("spine_*.off")))) if (d / "spines").exists() else 0,
            "points": {"xyz": pack_array(((p[["x", "y", "z"]].to_numpy(float) - c) / UM) if len(p) else np.zeros((0, 3)), "float32"),
                       "kind": [kinds[k] for k in p["kind"]]}}
