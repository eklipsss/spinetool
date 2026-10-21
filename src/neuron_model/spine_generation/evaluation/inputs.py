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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ...pointcloud_io import load_points
from ..data.index import SpineIndex

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


def threaded_map(fn: Callable[[Any], Any], items: Sequence[Any], *, threads: int = 32, label: str = "") -> Iterator[Any]:
    """``fn`` over ``items`` in threads, results in input order; prints progress every ~10 %.

    The reads are NAS round trips (latency, not bandwidth), so threads give a near-linear speed-up."""
    n = len(items)
    step = max(1, n // 10)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for i, result in enumerate(pool.map(fn, items), start=1):
            if label and (i % step == 0 or i == n):
                print(f"  {label}: {i}/{n} ({time.perf_counter() - started:.0f} s)", flush=True)
            yield result


def load_real_clouds(index: SpineIndex, n_points: int, *, variant: int = 1, threads: int = 32) -> np.ndarray:
    """``[n, n_points, 3]`` precomputed surface samples (size ``n_points``, resampling ``variant``) of the indexed spines."""
    paths = list(index.frame[f"pointcloud_{n_points}_path"])
    clouds = list(threaded_map(lambda path: np.asarray(load_points(path, n_points, variant), dtype=np.float32), paths,
                               threads=threads, label="test clouds"))
    return np.stack(clouds) if clouds else np.empty((0, n_points, 3), dtype=np.float32)


def build_cloud_cache(index: SpineIndex, n_points: int, out_path: Path, *, variant: int = 1, threads: int = 32) -> Path:
    """Stream all indexed clouds into one ``.npy`` (open later with ``np.load(..., mmap_mode="r")``).

    One pass over many small npz files (slow from the NAS - do it once, on a local disk);
    afterwards Chamfer-DCR reads only the candidate rows it needs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.npy")
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(index), n_points, 3))
    paths = list(index.frame[f"pointcloud_{n_points}_path"])
    for i, points in enumerate(threaded_map(lambda path: load_points(path, n_points, variant), paths, threads=threads, label="train cloud cache")):
        cache[i] = points
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
