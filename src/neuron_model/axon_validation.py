"""Check axon labels of limbs against synapses (independent ground truth from the Minnie65 annotation tables).

Why: interneurons have almost no spines on their dendrites, so their axon cannot be found by "thin and without spines".
The candidate rule "thin and long" (limb radius < 180 nm and cable > 1.5 mm, ``rule_axon``) was tuned on excitatory
neurons, whose axon labels come from another heuristic table. Synapses give labels that do not depend on any geometry
rule:

* the PRESYNAPTIC point of a synapse made by the neuron (``pre_pt_root_id`` = neuron) lies on its axon (boutons);
* the POSTSYNAPTIC point of a synapse onto the neuron (``post_pt_root_id`` = neuron), tagged ``shaft`` or ``spine``,
  lies on a dendrite (axo-axonic synapses on the axon initial segment exist, chandelier cells, but are rare);
* manual myelination points tagged ``t`` lie on an axon.

Source: ``minnie65_metadata/vortex_compartment_targets.parquet`` (synapses with a compartment tag of the target) and
``vortex_manual_myelination_v0.parquet``. Positions are in voxels of 4 x 4 x 40 nm (``dataset_info.json``,
``viewer_resolution_*``; checked against the soma mesh centroid of a local neuron, 0.3 um).

Each point goes to the nearest limb skeleton node of its neuron (``limb_*/limb_skeleton.npy``, nodes and segment
midpoints) or to the soma (vertices of ``soma/soma_mesh.off``), within ``max_dist_nm``. A limb with enough points is
called "axon by synapses" when the presynaptic / myelin share is high and "dendrite by synapses" when it is low; the
geometric labels are then compared with it per limb and per neuron. Units: nm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

VOXEL_NM = np.array([4.0, 4.0, 40.0])
SOMA_LIMB = -1
AXON_KINDS = ("pre", "myelin")
DENDRITE_KINDS = ("post_shaft", "post_spine")


def _xyz(col: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=float) for v in col]) * VOXEL_NM if len(col) else np.zeros((0, 3))


def load_synapse_points(metadata_dir: Path, neurons: Iterable[int]) -> pd.DataFrame:
    """Labelled points (nm) of the given neurons: columns ``neuron_id`` (int64), ``kind``, ``x``, ``y``, ``z``.

    kind: ``pre`` (presynaptic point of a synapse the neuron makes), ``post_shaft`` / ``post_spine`` / ``post_soma``
    (postsynaptic point of a synapse onto the neuron by its target tag), ``myelin`` (manual myelination tag ``t``).
    """
    ids = {int(n) for n in neurons}
    metadata_dir = Path(metadata_dir)
    parts = []
    syn = pd.read_parquet(metadata_dir / "vortex_compartment_targets.parquet",
                          columns=["tag", "pre_pt_root_id", "post_pt_root_id", "pre_pt_position", "post_pt_position"])
    pre = syn[syn["pre_pt_root_id"].astype("int64").isin(ids)]
    parts.append(pd.DataFrame({"neuron_id": pre["pre_pt_root_id"].astype("int64").to_numpy(), "kind": "pre",
                               **dict(zip("xyz", _xyz(pre["pre_pt_position"]).T))}))
    post = syn[syn["post_pt_root_id"].astype("int64").isin(ids) & syn["tag"].isin(["shaft", "spine", "soma"])]
    parts.append(pd.DataFrame({"neuron_id": post["post_pt_root_id"].astype("int64").to_numpy(), "kind": "post_" + post["tag"].to_numpy().astype(object),
                               **dict(zip("xyz", _xyz(post["post_pt_position"]).T))}))
    path = metadata_dir / "vortex_manual_myelination_v0.parquet"
    if path.exists():
        my = pd.read_parquet(path, columns=["tag", "pt_root_id", "pt_position"])
        my = my[(my["tag"].astype(str) == "t") & my["pt_root_id"].astype("int64").isin(ids)]
        parts.append(pd.DataFrame({"neuron_id": my["pt_root_id"].astype("int64").to_numpy(), "kind": "myelin",
                                   **dict(zip("xyz", _xyz(my["pt_position"]).T))}))
    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else \
        pd.DataFrame(columns=["neuron_id", "kind", "x", "y", "z"])
    return out


def _segment_points(path: Path) -> np.ndarray:
    seg = np.asarray(np.load(path), dtype=float).reshape(-1, 2, 3)
    return np.concatenate([seg[:, 0], seg[:, 1], seg.mean(axis=1)])


def neuron_index(neuron_dir: Path, *, branches: bool = False):
    """(points [M, 3] nm, limb id per point, branch id per point) of one neuron folder; soma mesh vertices get
    SOMA_LIMB and branch -1. ``branches=False``: limb skeletons only, branch id -1 (one file per limb); ``True``: the
    branch skeletons (``limb_*/branch_*/branch_skeleton.npy``, one file per branch) instead."""
    neuron_dir = Path(neuron_dir)
    pts, lab, br = [], [], []
    for d in sorted(neuron_dir.glob("limb_*")):
        limb = int(d.name.split("_")[-1])
        files = sorted(d.glob("branch_*/branch_skeleton.npy")) if branches else []
        if files:
            for f in files:
                p = _segment_points(f)
                pts.append(p)
                lab.append(np.full(len(p), limb))
                br.append(np.full(len(p), int(f.parent.name.split("_")[-1])))
            continue
        f = d / "limb_skeleton.npy"
        if not f.exists():
            continue
        p = _segment_points(f)
        pts.append(p)
        lab.append(np.full(len(p), limb))
        br.append(np.full(len(p), -1))
    soma = neuron_dir / "soma" / "soma_mesh.off"
    if soma.exists():
        from .spine_geometry import load_trimesh

        v = np.asarray(load_trimesh(soma, process=False).vertices, dtype=float)
        pts.append(v)
        lab.append(np.full(len(v), SOMA_LIMB))
        br.append(np.full(len(v), -1))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(pts), np.concatenate(lab), np.concatenate(br)


def branch_spine_counts(neuron_dir: Path) -> dict:
    """{(limb, branch): number of spines} from the neuron's ``summary.json`` (empty when absent)."""
    import json

    path = Path(neuron_dir) / "summary.json"
    if not path.exists():
        return {}
    s = json.loads(path.read_text(encoding="utf-8"))
    return {(li, bi): int(n) for li, limb in enumerate(s.get("limbs", [])) for bi, n in enumerate(limb.get("spines_per_branch", []))}


def assign_points(points: np.ndarray, index_points: np.ndarray, *labels: np.ndarray):
    """(label of the nearest index point for each label array ..., distance nm) for each point; -2 when there is no
    index (no skeleton found)."""
    from scipy.spatial import cKDTree

    if len(points) == 0 or len(index_points) == 0:
        return (*[np.full(len(points), -2) for _ in labels], np.full(len(points), np.inf))
    dist, idx = cKDTree(index_points).query(points)
    return (*[lab[idx] for lab in labels], dist)


def limb_evidence(assigned: pd.DataFrame, *, max_dist_nm: float, min_points: int = 3, axon_share: float = 0.8,
                  dendrite_share: float = 0.2) -> pd.DataFrame:
    """Per (neuron, limb): counts of axon-type (pre, myelin) and dendrite-type (post shaft/spine) points within
    ``max_dist_nm`` and the label ``by_synapses``: axon / dendrite / mixed / too_few."""
    a = assigned[(assigned["dist_nm"] <= max_dist_nm) & (assigned["limb_id"] >= 0)]
    a = a.assign(axon_pt=a["kind"].isin(AXON_KINDS), dend_pt=a["kind"].isin(DENDRITE_KINDS))
    g = a.groupby(["neuron_id", "limb_id"]).agg(n_axon_points=("axon_pt", "sum"), n_dendrite_points=("dend_pt", "sum"),
                                               n_pre=("kind", lambda s: int((s == "pre").sum()))).reset_index()
    n = g["n_axon_points"] + g["n_dendrite_points"]
    share = g["n_axon_points"] / n.where(n > 0)
    g["axon_point_share"] = share
    g["by_synapses"] = np.select([n < min_points, share >= axon_share, share <= dendrite_share], ["too_few", "axon", "dendrite"], "mixed")
    return g


def per_neuron(assigned: pd.DataFrame, limbs: pd.DataFrame, label_cols: Sequence[str], *, max_dist_nm: float) -> pd.DataFrame:
    """Per neuron: number of presynaptic / dendritic points (assigned within ``max_dist_nm``, soma excluded), the share of
    presynaptic points on limbs labelled axon by each geometric label, the share of dendritic points on them (should be
    ~0), the limb with most presynaptic points and whether each label marks it."""
    a = assigned[(assigned["dist_nm"] <= max_dist_nm) & (assigned["limb_id"] >= 0)]
    lab = limbs.set_index(["neuron_id", "limb_id"])[list(label_cols)].fillna(False).astype(bool)
    n_axon_limbs = lab.groupby(level=0).sum()
    rows = []
    for nid, g in a.groupby("neuron_id"):
        keys = list(zip(g["neuron_id"], g["limb_id"]))
        flags = lab.reindex(keys).fillna(False).to_numpy(dtype=bool)
        is_pre = (g["kind"] == "pre").to_numpy()
        is_dend = g["kind"].isin(DENDRITE_KINDS).to_numpy()
        row = {"neuron_id": nid, "n_pre": int(is_pre.sum()), "n_dendrite_points": int(is_dend.sum()),
               "n_limbs_with_pre": int(g.loc[is_pre, "limb_id"].nunique())}
        top = g.loc[is_pre, "limb_id"].value_counts()
        row["top_pre_limb"] = int(top.index[0]) if len(top) else None
        row["top_pre_limb_share"] = float(top.iloc[0] / top.sum()) if len(top) else np.nan
        for j, c in enumerate(label_cols):
            row[f"pre_on_{c}"] = float(flags[is_pre, j].mean()) if is_pre.any() else np.nan
            row[f"dendrite_points_on_{c}"] = float(flags[is_dend, j].mean()) if is_dend.any() else np.nan
            row[f"n_limbs_{c}"] = int(n_axon_limbs.loc[nid, c]) if nid in n_axon_limbs.index else 0
            top_key = (nid, row["top_pre_limb"])
            row[f"top_pre_limb_is_{c}"] = (bool(lab.loc[top_key, c]) if top_key in lab.index else False) if row["top_pre_limb"] is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean and 95 % percentile interval over units (neurons)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
