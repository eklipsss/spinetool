"""Generated-sample artifacts, stored separately at every stage (spec requirement).

Each sample gets its own directory ``<root>/<tag>/sample_<index>/`` holding
(file names from s-module-technical-spec.md)::

    generation_metadata.json          # seed, checkpoint id, model/generation config, timings
    latent.npy                        # VAE: latent vector z
    sdf_evaluation_config.json        # VAE: bbox, grid resolution, chunking, iso level
    generated_pointcloud.ply          # MoGen: raw generator output (physical units)
    flow_inference_config.json        # MoGen: solver, steps, schedule, coordinate scale
    estimated_normals.ply             # MoGen: points + estimated normals fed to Poisson
    poisson_config.json               # MoGen: Screened Poisson parameters
    generated_mesh_raw.off            # extracted surface before any postprocessing
    generated_mesh_postprocessed.off  # after the allowed light postprocessing only
    mesh_validation_raw.json / mesh_validation_postprocessed.json

so a generator error can be told apart from a surface-extraction or
postprocessing error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_mesh_off(path: Path, mesh: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = mesh.export(file_type="off")
    tmp.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    tmp.replace(path)


def write_pointcloud_ply(path: Path, points: np.ndarray, normals: Optional[np.ndarray] = None) -> None:
    """Binary little-endian PLY with float32 x,y,z (+ nx,ny,nz)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    columns = [points]
    props = ["x", "y", "z"]
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        if len(normals) != len(points):
            raise ValueError("points and normals must have the same length")
        columns.append(normals)
        props += ["nx", "ny", "nz"]
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {len(points)}\n"
    header += "".join(f"property float {name}\n" for name in props)
    header += "end_header\n"
    body = np.ascontiguousarray(np.hstack(columns).astype("<f4")).tobytes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(header.encode("ascii") + body)
    tmp.replace(path)


def read_pointcloud_ply(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reader for the files written by :func:`write_pointcloud_ply`."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"Unsupported PLY format in {path}")
    n = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    props = [line.split()[2] for line in header if line.startswith("property float")]
    table = np.frombuffer(data[end:], dtype="<f4", count=n * len(props)).reshape(n, len(props))
    points = table[:, :3].astype(np.float32)
    normals = table[:, 3:6].astype(np.float32) if len(props) >= 6 else None
    return points, normals


class SampleWriter:
    """Writes per-sample artifact directories under ``root`` (a run's ``samples/`` or ``meshes/``)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def sample_dir(self, tag: str, index: int) -> Path:
        path = self.root / tag / f"sample_{index:05d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_metadata(
        self,
        tag: str,
        index: int,
        *,
        seed: int,
        checkpoint_id: Optional[str],
        generation_config: Mapping[str, Any],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = self.sample_dir(tag, index) / "generation_metadata.json"
        write_json(
            path,
            {
                "sample_index": index,
                "generation_seed": int(seed),
                "checkpoint_id": checkpoint_id,
                "generation_config": dict(generation_config),
                **dict(extra or {}),
            },
        )
        return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
