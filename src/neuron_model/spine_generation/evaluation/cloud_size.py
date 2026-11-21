"""Size of point clouds BEFORE surface reconstruction: does the generator inflate the spine, or does Poisson?

Context (chains AF / AI, 2026-11-21): generated MoGen meshes are larger than the real ones (median volume x1.22 for A,
x1.44 for v3, real spines through the same Poisson chain x1.03), while the length is reproduced. Two explanations
predict different things:
  * the generator makes fatter shapes -> the cloud itself is thicker (radial size, hull volume) by the same factor;
  * the cloud is fine but noisier (points scattered off the surface) and Poisson wraps the noise -> cloud size near
    the real one, but larger surface noise and a larger mesh volume / cloud hull volume ratio than for real clouds.

Per cloud (local spine frame, +y = spine axis, nm): length along y, radial rms and head radius around the axis
through the centroid, convex hull volume, nearest-neighbour spacing, local surface noise (rms distance of a point to
the plane of its k neighbours), and - if a reconstructed mesh is next to the cloud - mesh volume and mesh / hull.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

MEASURES = ("length_nm", "extent_x_nm", "extent_z_nm", "radial_rms_nm", "head_radius_nm", "neck_radius_nm",
            "hull_volume_nm3", "nn_spacing_nm", "surface_noise_nm", "mesh_volume_nm3", "mesh_over_hull")


def cloud_measures(points: np.ndarray, *, k: int = 16, n_noise: int = 1024, seed: int = 0) -> Dict[str, float]:
    from scipy.spatial import ConvexHull, cKDTree

    p = np.asarray(points, dtype=np.float64)
    p = p[np.all(np.isfinite(p), axis=1)]
    y = p[:, 1]
    lo, hi = np.percentile(y, [0.5, 99.5])
    cx, cz = p[:, 0].mean(), p[:, 2].mean()
    r = np.hypot(p[:, 0] - cx, p[:, 2] - cz)
    length = hi - lo
    upper = y > lo + 0.5 * length
    band = (y > lo + 0.15 * length) & (y < lo + 0.4 * length)
    out = {
        "n_points": float(len(p)),
        "length_nm": float(length),
        "extent_x_nm": float(np.subtract(*np.percentile(p[:, 0], [99.5, 0.5]))),
        "extent_z_nm": float(np.subtract(*np.percentile(p[:, 2], [99.5, 0.5]))),
        "radial_rms_nm": float(np.sqrt(np.mean(r ** 2))),
        "head_radius_nm": float(np.percentile(r[upper], 90)) if upper.sum() > 10 else float("nan"),
        "neck_radius_nm": float(np.median(r[band])) if band.sum() > 10 else float("nan"),
    }
    try:
        out["hull_volume_nm3"] = float(ConvexHull(p).volume)
    except Exception:  # noqa: BLE001 - degenerate cloud
        out["hull_volume_nm3"] = float("nan")
    tree = cKDTree(p)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p), min(n_noise, len(p)), replace=False)
    dist, nbr = tree.query(p[idx], k=k + 1)
    out["nn_spacing_nm"] = float(np.median(dist[:, 1]))
    q = p[nbr[:, 1:]] - p[nbr[:, 1:]].mean(axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", q, q) / k)
    out["surface_noise_nm"] = float(np.median(np.sqrt(np.maximum(ev[:, 0], 0.0))))
    return out


def mesh_volume(path: Path) -> float:
    import trimesh

    try:
        m = trimesh.load(path, process=False, force="mesh")
        return float(abs(m.volume)) if m.is_watertight else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def measure_generated_dir(root: Path, *, mesh_name: str = "generated_mesh_postprocessed.off", limit: Optional[int] = None) -> pd.DataFrame:
    """``<root>/sample_*/generated_pointcloud.ply`` (+ the mesh if present)."""
    from ..experiment.artifacts import read_pointcloud_ply

    rows = []
    dirs = sorted(d for d in Path(root).glob("sample_*") if (d / "generated_pointcloud.ply").exists())
    for d in dirs[:limit]:
        pts, _ = read_pointcloud_ply(d / "generated_pointcloud.ply")
        row = {"sample": d.name, **cloud_measures(pts)}
        mesh = d / mesh_name
        row["mesh_volume_nm3"] = mesh_volume(mesh) if mesh.exists() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def measure_real_controls(root: Path, cloud_path_of: Mapping[str, str], *, n_points: int = 8192,
                          mesh_name: str = "generated_mesh_postprocessed.off", limit: Optional[int] = None) -> pd.DataFrame:
    """Real-control sets of ``make_real_controls.py`` (``generation_metadata.json`` holds the spine key): the cloud is
    the real preprocessing cloud of that spine (the one Poisson saw), the mesh is its Poisson reconstruction."""
    import json

    from ...pointcloud_io import load_points

    rows = []
    dirs = sorted(d for d in Path(root).glob("sample_*") if (d / "generation_metadata.json").exists())
    for d in dirs[:limit]:
        key = json.loads((d / "generation_metadata.json").read_text(encoding="utf-8"))["spine_key"]
        if key not in cloud_path_of:
            continue
        row = {"sample": d.name, "spine_key": key, **cloud_measures(load_points(cloud_path_of[key], n_points, 1))}
        mesh = d / mesh_name
        row["mesh_volume_nm3"] = mesh_volume(mesh) if mesh.exists() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(tables: Mapping[str, pd.DataFrame], reference: str, *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per set and measure: median, p90, p99 and median / reference median with a 95 % bootstrap interval
    (both sides resampled; the reference by neuron when it has ``spine_key``)."""
    rng = np.random.default_rng(seed)
    ref = tables[reference]
    groups = None
    if "spine_key" in ref.columns:
        neuron = ref["spine_key"].astype(str).str.split("/").str[:2].str.join("/")
        groups = [np.flatnonzero(neuron.to_numpy() == n) for n in neuron.unique()]
    rows = []
    for label, t in tables.items():
        for m in MEASURES:
            if m == "mesh_over_hull":
                if not {"mesh_volume_nm3", "hull_volume_nm3"} <= set(t.columns) & set(ref.columns):
                    continue
                v =(t["mesh_volume_nm3"] / t["hull_volume_nm3"]).to_numpy(float)
                rv = (ref["mesh_volume_nm3"] / ref["hull_volume_nm3"]).to_numpy(float)
            elif m in t.columns and m in ref.columns:
                v, rv = t[m].to_numpy(float), ref[m].to_numpy(float)
            else:
                continue
            ok, rok = np.isfinite(v), np.isfinite(rv)
            if ok.sum() < 5 or rok.sum() < 5:
                continue
            ratios = []
            for _ in range(n_boot):
                gv = v[ok][rng.integers(0, ok.sum(), ok.sum())]
                if groups is not None:
                    pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                    rr = rv[pick]
                    rr = rr[np.isfinite(rr)]
                else:
                    rr = rv[rok][rng.integers(0, rok.sum(), rok.sum())]
                ratios.append(np.median(gv) / np.median(rr))
            rows.append({"set": label, "measure": m, "n": int(ok.sum()), "median": float(np.median(v[ok])),
                         "p90": float(np.percentile(v[ok], 90)), "p99": float(np.percentile(v[ok], 99)),
                         "ratio_to_ref": float(np.median(v[ok]) / np.median(rv[rok])),
                         "ratio_lo": float(np.percentile(ratios, 2.5)), "ratio_hi": float(np.percentile(ratios, 97.5)),
                         "p99_ratio_to_ref": float(np.percentile(v[ok], 99) / np.percentile(rv[rok], 99))})
    return pd.DataFrame(rows)
