"""Point-cloud storage of the preprocessed spines (stage 6).

Format ``packed_v1`` (current): every size x resampling variant of a spine in ONE file
``pointclouds.npz`` with keys ``<field>_<n_points>_<variant>`` (``points_8192_2``,
``normals_8192_2``, ...). One file instead of ``len(sizes) * len(seeds)`` (12) per spine:
on the NAS the cost is dominated by the number of files created, not by bytes.

``.npz`` is a zip of ``.npy`` members and ``np.load`` is lazy, so :func:`load_pointcloud`
reads only the members of the requested variant. A packed file can be split back with
:func:`unpack_pointclouds` (or opened with any zip tool).

Legacy format (preprocessing before ``packed_v1``): one ``pointcloud_<n>_<variant>.npz``
per size/variant with keys ``points, normals, ...``; manifests point at the variant-1
file. :func:`load_pointcloud` reads both, so older preprocessed data keeps working.

Always go through :func:`load_pointcloud` (it closes the file - an open ``np.load``
handle blocks renaming/deleting the file on Windows - and avoids reading all 12 variants
by accident, e.g. ``dict(np.load(path))``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union

import numpy as np

PACKED_NAME = "pointclouds.npz"
FIELDS = ("points", "normals", "face_indices", "is_attachment_cap", "seed", "sampling_seed")
_LEGACY_RE = re.compile(r"^pointcloud_(\d+)_(\d+)$")
_PACKED_KEY_RE = re.compile(r"^points_(\d+)_(\d+)$")

PathLike = Union[str, Path]


def packed_key(field: str, n_points: int, variant: int) -> str:
    return f"{field}_{int(n_points)}_{int(variant)}"


def is_packed(path: PathLike) -> bool:
    return Path(path).name == PACKED_NAME


def legacy_variant_path(path: PathLike, variant: int) -> Path:
    """``.../pointcloud_8192_1.npz`` -> ``.../pointcloud_8192_<variant>.npz`` (legacy format)."""
    path = Path(path)
    prefix, _, _ = path.stem.rpartition("_")
    return path.with_name(f"{prefix}_{int(variant)}{path.suffix}")


def write_pointclouds(path: PathLike, clouds: Mapping[Tuple[int, int], Mapping[str, np.ndarray]]) -> None:
    """Write ``{(n_points, variant): {field: array}}`` as one packed npz, atomically
    (tmp file + rename, like every preprocessing output). The parent directory must exist."""
    path = Path(path)
    arrays = {
        packed_key(field, n_points, variant): value
        for (n_points, variant), fields in clouds.items()
        for field, value in fields.items()
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as fd:
        np.savez_compressed(fd, **arrays)
    tmp_path.replace(path)


def load_pointcloud(path: PathLike, n_points: int, variant: int = 1) -> Dict[str, np.ndarray]:
    """``{"points", "normals", "face_indices", "is_attachment_cap", "seed", "sampling_seed"}``
    of one size/variant - the same keys as a legacy per-variant file.

    ``path``: a packed ``pointclouds.npz`` or (legacy) any ``pointcloud_<n>_<v>.npz`` of
    the spine (the manifest stores the variant-1 one).
    """
    path = Path(path)
    if is_packed(path):
        with np.load(path) as data:
            key = packed_key("points", n_points, variant)
            if key not in data.files:
                raise KeyError(f"{path} has no point cloud of {n_points} points, variant {variant}")
            return {
                field: np.asarray(data[packed_key(field, n_points, variant)])
                for field in FIELDS
                if packed_key(field, n_points, variant) in data.files
            }
    match = _LEGACY_RE.match(path.stem)
    if match is not None and int(match.group(1)) != int(n_points):
        path = path.with_name(f"pointcloud_{int(n_points)}_1{path.suffix}")
    with np.load(legacy_variant_path(path, variant)) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_points(path: PathLike, n_points: int, variant: int = 1) -> np.ndarray:
    """Only the ``points`` array (``float32 [n_points, 3]``)."""
    path = Path(path)
    if is_packed(path):
        with np.load(path) as data:
            return np.asarray(data[packed_key("points", n_points, variant)])
    return load_pointcloud(path, n_points, variant)["points"]


def available_pointclouds(path: PathLike) -> List[Tuple[int, int]]:
    """``[(n_points, variant), ...]`` stored in a packed file."""
    with np.load(Path(path)) as data:
        found = (_PACKED_KEY_RE.match(name) for name in data.files)
        return sorted((int(m.group(1)), int(m.group(2))) for m in found if m)


def unpack_pointclouds(path: PathLike, out_dir: PathLike) -> List[Path]:
    """Split a packed file back into legacy ``pointcloud_<n>_<v>.npz`` files in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for n_points, variant in available_pointclouds(path):
        target = out_dir / f"pointcloud_{n_points}_{variant}.npz"
        np.savez_compressed(target, **load_pointcloud(path, n_points, variant))
        written.append(target)
    return written
