"""Load evaluation inputs: generated sample sets (artifact layout) and real reference data.

Generated layout (written by ``scripts/neuron_model/generate_mogen.py`` /
``experiment.artifacts``)::

    <model_root>/seed_<s>/sample_<i>/generated_mesh_raw.off
                                     generated_mesh_postprocessed.off
                                     mesh_validation_raw.json
                                     mesh_validation_postprocessed.json

``mesh_kind`` = ``raw`` or ``postprocessed`` - the spec evaluates both separately (a model
that only looks good after heavy fixing must not win).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.index import SpineIndex, pointcloud_variant_path

MESH_KINDS = ("raw", "postprocessed")


def discover_seed_dirs(model_root: Path) -> Dict[int, Path]:
    """``{seed: <model_root>/seed_<seed>}`` for every seed directory present."""
    found = {}
    for path in sorted(Path(model_root).glob("seed_*")):
        match = re.fullmatch(r"seed_(\d+)", path.name)
        if match and path.is_dir():
            found[int(match.group(1))] = path
    return found


def load_generated_set(seed_dir: Path, mesh_kind: str = "raw") -> Tuple[List[Optional[Any]], List[Dict[str, Any]], List[str]]:
    """``(meshes, validity_reports, sample_ids)``; a sample whose mesh file is missing (no
    surface extracted) is kept as ``None`` with a ``mesh_present=False`` report, so failures
    count against the model instead of silently disappearing."""
    from ...spine_geometry import load_trimesh

    if mesh_kind not in MESH_KINDS:
        raise ValueError(f"mesh_kind must be one of {MESH_KINDS}")
    meshes, reports, ids = [], [], []
    for sample_dir in sorted(Path(seed_dir).glob("sample_*")):
        mesh_path = sample_dir / f"generated_mesh_{mesh_kind}.off"
        report_path = sample_dir / f"mesh_validation_{mesh_kind}.json"
        mesh = load_trimesh(mesh_path, process=False) if mesh_path.exists() else None
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"mesh_present": mesh is not None, "is_valid": False}
        if mesh is None:
            report = {**report, "mesh_present": False, "is_valid": False}
        meshes.append(mesh)
        reports.append(report)
        ids.append(sample_dir.name)
    return meshes, reports, ids


def load_real_clouds(index: SpineIndex, n_points: int, *, variant: int = 1) -> np.ndarray:
    """``[n, n_points, 3]`` precomputed surface samples (``pointcloud_<n>_<variant>.npz``) of the indexed spines."""
    clouds = []
    for path in index.frame[f"pointcloud_{n_points}_path"]:
        with np.load(pointcloud_variant_path(path, variant)) as data:
            clouds.append(np.asarray(data["points"], dtype=np.float32))
    return np.stack(clouds) if clouds else np.empty((0, n_points, 3), dtype=np.float32)


def build_cloud_cache(index: SpineIndex, n_points: int, out_path: Path, *, variant: int = 1) -> Path:
    """Stream all indexed clouds into one ``.npy`` (open later with ``np.load(..., mmap_mode="r")``).

    One pass over many small npz files (slow from the NAS - do it once, on a local disk);
    afterwards Chamfer-DCR reads only the candidate rows it needs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.npy")
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(index), n_points, 3))
    for i, path in enumerate(index.frame[f"pointcloud_{n_points}_path"]):
        with np.load(pointcloud_variant_path(path, variant)) as data:
            cache[i] = data["points"]
    cache.flush()
    del cache
    tmp.replace(out_path)
    out_path.with_suffix(".keys.json").write_text(json.dumps(index.keys), encoding="utf-8")
    return out_path


def open_cloud_cache(path: Path, index: SpineIndex) -> np.ndarray:
    """Memmap of a cache built by :func:`build_cloud_cache`; refuses a cache of other spines/order."""
    path = Path(path)
    keys = json.loads(path.with_suffix(".keys.json").read_text(encoding="utf-8"))
    if keys != index.keys:
        raise ValueError(f"cloud cache {path} was built for a different spine list - rebuild it")
    return np.load(path, mmap_mode="r")
