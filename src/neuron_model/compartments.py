"""Apical / basal limbs by DIRECTION towards the pia (an alternative to the cable-length rule of the Minnie65 table).

The apical dendrite of a pyramidal neuron is the one dendrite that leaves the soma towards the pial surface; the basal
dendrites leave in all directions. The length rule of the existing table (``legacy/apical_basal_spatial_stats.py``:
apical = the longest non-axon limb when it is >= 2x longer than the second) fails where basal dendrites are long or the
apical trunk is cut by the volume boundary (H01, a thin slab). Direction does not depend on the length: a truncated
apical trunk still points to the pia.

- ``limb_features``: per limb of a neuron the root (skeleton node closest to the soma centroid), the cable length, and
  the initial direction of the dendrite = from the root to the mean of the nodes at a path distance of
  ``init_range_nm`` (default 10-30 um) along the limb skeleton; also the direction of the whole limb (centroid).
- ``estimate_pia_axis``: the pia direction from data - the least-squares direction along which the cortical layer of
  the neurons decreases with the soma position (layer from h01_layer.csv, or from the Minnie65 cell type names 23P / 4P
  / 5P-* / 6P-*); returns a unit vector pointing to the pia and R^2 of the fit.
- ``classify_by_direction``: per neuron, among the non-axon limbs the one whose initial direction is closest to the pia
  is apical when cos >= ``min_cos`` and no other limb points there as well (second cos < ``min_cos`` or the winner is
  >= ``length_ratio`` times longer), or when it clearly leads (cos >= ``min_cos_gap`` and >= ``gap`` above the second:
  the pia axis is only approximate, the cortex is tilted / curved in the volume); otherwise every limb of the neuron is ``uncertain``. Neurons of a type without an
  apical dendrite (interneurons) are left ``not_applicable``.

Units: nm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LAYER_OF_MINNIE_TYPE = {"23P": 2.5, "4P": 4.0, "5P-IT": 5.0, "5P-NP": 5.0, "5P-PT": 5.0, "6P-CT": 6.0, "6P-IT": 6.0, "6P-U": 6.0, "WM-P": 7.0}


def layer_value(text: Any) -> Optional[float]:
    """'Layer 2' -> 2, 'White matter' -> 7, Minnie types via LAYER_OF_MINNIE_TYPE; else None."""
    if not isinstance(text, str):
        return None
    if text in LAYER_OF_MINNIE_TYPE:
        return LAYER_OF_MINNIE_TYPE[text]
    t = text.strip().lower()
    if t.startswith("layer"):
        try:
            return float(t.split()[-1])
        except ValueError:
            return None
    if "white" in t:
        return 7.0
    return None


def soma_center(neuron_dir: Path) -> Optional[np.ndarray]:
    from .spine_geometry import load_trimesh

    path = Path(neuron_dir) / "soma" / "soma_mesh.off"
    if not path.exists():
        return None
    return np.asarray(load_trimesh(path, process=False).vertices, dtype=float).mean(axis=0)


def _skeleton_graph(segments: np.ndarray):
    """(node coordinates [M, 3], csr adjacency weighted by the segment length) of a segment soup [S, 2, 3]."""
    from scipy.sparse import coo_matrix

    pts = segments.reshape(-1, 3)
    nodes, inverse = np.unique(np.round(pts, 3), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1, 2)
    w = np.linalg.norm(segments[:, 0] - segments[:, 1], axis=1)
    keep = inverse[:, 0] != inverse[:, 1]
    i, j, w = inverse[keep, 0], inverse[keep, 1], w[keep]
    m = len(nodes)
    graph = coo_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(m, m)).tocsr()
    return nodes, graph


def limb_features(neuron_dir: Path, center: np.ndarray, *, init_range_nm: Tuple[float, float] = (10_000.0, 30_000.0)) -> List[Dict[str, Any]]:
    """One dict per ``limb_*/limb_skeleton.npy`` of the neuron (see module docstring)."""
    from scipy.sparse.csgraph import dijkstra

    rows = []
    for path in sorted(Path(neuron_dir).glob("limb_*/limb_skeleton.npy")):
        limb = int(path.parent.name.split("_")[-1])
        seg = np.asarray(np.load(path), dtype=float).reshape(-1, 2, 3)
        row: Dict[str, Any] = {"limb_id": limb, "n_segments": int(len(seg))}
        if len(seg) == 0:
            rows.append(row)
            continue
        nodes, graph = _skeleton_graph(seg)
        root = int(np.argmin(np.linalg.norm(nodes - center, axis=1)))
        dist = dijkstra(graph, indices=root)
        row["cable_nm"] = float(np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1).sum())
        row["root_to_soma_nm"] = float(np.linalg.norm(nodes[root] - center))
        row["root"] = nodes[root].tolist()
        a, b = init_range_nm
        band = np.isfinite(dist) & (dist >= a) & (dist <= b)
        if not band.any():  # a short limb: its farthest reachable nodes
            reach = np.isfinite(dist)
            band = reach & (dist >= 0.5 * dist[reach].max())
        vec = nodes[band].mean(axis=0) - nodes[root]
        row["dir_init"] = (vec / np.linalg.norm(vec)).tolist() if np.linalg.norm(vec) > 0 else None
        far = nodes[np.isfinite(dist)].mean(axis=0) - center
        row["dir_limb"] = (far / np.linalg.norm(far)).tolist() if np.linalg.norm(far) > 0 else None
        row["max_path_nm"] = float(dist[np.isfinite(dist)].max())
        rows.append(row)
    return rows


def estimate_pia_axis(centers: np.ndarray, layers: np.ndarray, *, min_spread_ratio: float = 0.2) -> Tuple[np.ndarray, float]:
    """Unit vector towards the pia: layer ~ a + g . x by least squares, axis = -g / |g|; R^2 of the fit.
    Coordinates along which the somata spread less than ``min_spread_ratio`` of the widest one (a thin slab: H01 z)
    are left out (g = 0 there): a slope along a thin direction is estimated from noise and tilts the axis."""
    centers, layers = np.asarray(centers, float), np.asarray(layers, float)
    ok = np.isfinite(layers) & np.isfinite(centers).all(axis=1)
    x = centers[ok] - centers[ok].mean(axis=0)
    y = layers[ok] - layers[ok].mean()
    spread = x.std(axis=0)
    use = spread >= min_spread_ratio * spread.max()
    g = np.zeros(x.shape[1])
    g[use], *_ = np.linalg.lstsq(x[:, use], y, rcond=None)
    pred = x @ g
    r2 = 1.0 - float(((y - pred) ** 2).sum() / max((y ** 2).sum(), 1e-12))
    return -g / np.linalg.norm(g), r2


def classify_by_direction(limbs: pd.DataFrame, pia: np.ndarray, *, min_cos: float = 0.6, length_ratio: float = 2.0,
                          min_cos_gap: float = 0.3, gap: float = 0.4,
                          applicable: Optional[Dict[str, bool]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``limbs``: rows with ``neuron_id``, ``limb_id``, ``dir_init``, ``cable_nm``, ``axon_like`` (bool). Returns the limbs
    with ``cos_pia`` and ``compartment_dir`` + one diagnostic row per neuron."""
    pia = np.asarray(pia, float) / np.linalg.norm(pia)
    limbs = limbs.copy()
    limbs["cos_pia"] = [float(np.dot(d, pia)) if d is not None and not (isinstance(d, float) and np.isnan(d)) else np.nan for d in limbs["dir_init"]]
    labels, diag = [], []
    for neuron, g in limbs.groupby("neuron_id", sort=False):
        lab = pd.Series("uncertain", index=g.index)
        lab[g["axon_like"].astype(bool)] = "axon_like"
        d = g[~g["axon_like"].astype(bool) & g["cos_pia"].notna()].sort_values("cos_pia", ascending=False)
        if applicable is not None and not applicable.get(str(neuron), True):
            lab[~g["axon_like"].astype(bool)] = "not_applicable"
            labels.append(lab)
            diag.append({"neuron_id": neuron, "confident": False, "reason": "no apical dendrite for this cell type"})
            continue
        winner = None
        reason = "no dendritic limb"
        if len(d):
            best = d.iloc[0]
            second_cos = float(d.iloc[1]["cos_pia"]) if len(d) > 1 else -1.0
            second_cable = float(d.iloc[1]["cable_nm"]) if len(d) > 1 else 0.0
            b = float(best["cos_pia"])
            clear_lead = b >= min_cos_gap and b - second_cos >= gap  # the pia axis is only approximate (tilted cortex)
            if b < min_cos and not clear_lead:
                reason = f"no limb towards the pia (best cos {b:.2f}, second {second_cos:.2f})"
            elif second_cos >= min_cos and best["cable_nm"] < length_ratio * second_cable:
                reason = f"two limbs towards the pia (cos {b:.2f} / {second_cos:.2f})"
            else:
                winner = best.name
                reason = "ok" if b >= min_cos else "ok (clear lead)"
        if winner is not None:
            lab[~g["axon_like"].astype(bool)] = "basal"
            lab[winner] = "apical"
        labels.append(lab)
        diag.append({"neuron_id": neuron, "confident": winner is not None, "reason": reason,
                     "apical_limb": None if winner is None else int(limbs.loc[winner, "limb_id"]),
                     "best_cos": float(d.iloc[0]["cos_pia"]) if len(d) else np.nan, "second_cos": float(d.iloc[1]["cos_pia"]) if len(d) > 1 else np.nan})
    limbs["compartment_dir"] = pd.concat(labels) if labels else pd.Series(dtype=object)
    return limbs, pd.DataFrame(diag)
