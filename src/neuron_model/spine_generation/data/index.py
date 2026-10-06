"""Index of preprocessed spines, built from one or more ``manifest.parquet`` files.

Models never walk the file system: every spine and all its derived files come
from the manifest written by the preprocessing pipeline
(``s-module-preprocessing.md`` §4.2). One row per spine, keyed by
``spine_key = "<dataset>/<neuron_id>/<limb_id>/<branch_id>/<spine_id>"`` (the
preprocessing ``SpineRecord.object_id``).

Paths: the manifest stores absolute paths from the machine that ran the
preprocessing (e.g. ``O:\\Datasets\\...`` on the Windows workstation). With
``path_mode="relative"`` (default) each file is re-rooted next to the manifest
using the fixed layout ``<manifest_dir>/<neuron>/limb_<L>/branch_<B>/spines/spine_<S>/<file name>``,
so a preprocessed folder copied to another machine works unchanged.
``path_mode="absolute"`` uses the stored paths as they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from ...pointcloud_io import is_packed, legacy_variant_path

KEY_COLUMNS = ("dataset", "neuron_id", "limb_id", "branch_id", "spine_id")
PATH_COLUMNS = (
    "local_mesh_path",
    "sealed_mesh_path",
    "local_sealed_mesh_path",
    "attachment_region_path",
    "transform_path",
    "sdf_samples_path",
    "metadata_path",
    "morphometrics_path",
    "quality_path",
    "pointclouds_path",  # packed point clouds (pointcloud_io.py); absent in older manifests
)
POINTCLOUD_SIZES = (2048, 4096, 8192)


def spine_key(dataset: str, neuron_id: str, limb_id: str, branch_id: str, spine_id: str) -> str:
    return "/".join(str(part) for part in (dataset, neuron_id, limb_id, branch_id, spine_id))


def spine_relative_dir(neuron_id: str, limb_id: str, branch_id: str, spine_id: str) -> Path:
    return Path(str(neuron_id)) / f"limb_{limb_id}" / f"branch_{branch_id}" / "spines" / f"spine_{spine_id}"


def pointcloud_variant_path(variant_1_path: Path, variant: int) -> Path:
    """Legacy (per-variant files) only: ``.../pointcloud_8192_1.npz`` -> ``..._<variant>.npz``.

    A packed ``pointclouds.npz`` holds every variant, so it is returned unchanged. Read
    point clouds with ``src.neuron_model.pointcloud_io.load_pointcloud`` (handles both).
    """
    path = Path(variant_1_path)
    if is_packed(path):
        return path
    return legacy_variant_path(path, variant)


def _file_name(stored: object) -> Optional[str]:
    if stored is None or (isinstance(stored, float) and pd.isna(stored)) or str(stored) == "":
        return None
    text = str(stored)
    # Works for both "O:\\a\\b.npz" and "/a/b.npz" regardless of the current OS.
    return PureWindowsPath(text).name if "\\" in text else Path(text).name


@dataclass(frozen=True)
class SpineIndex:
    """A filtered, path-resolved view of the manifest(s)."""

    frame: pd.DataFrame
    dataset_versions: tuple  # distinct (preprocessing_version, config_hash) pairs present

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def keys(self) -> List[str]:
        return self.frame["spine_key"].tolist()

    @property
    def dataset_version(self) -> str:
        """Single string recorded in run_info.json (``<preprocessing_version>:<config_hash>``, '+'-joined)."""
        return "+".join(f"{v}:{h}" for v, h in self.dataset_versions)

    def subset(self, keys: Iterable[str]) -> "SpineIndex":
        wanted = set(keys)
        return SpineIndex(self.frame[self.frame["spine_key"].isin(wanted)].reset_index(drop=True), self.dataset_versions)

    def with_split(self, splits: pd.DataFrame, split: Optional[str] = None) -> "SpineIndex":
        """Attach the ``split`` column (from ``splits_v*.parquet``); optionally keep one split only."""
        frame = self.frame.drop(columns=["split", "group_id"], errors="ignore").merge(
            splits[["spine_key", "group_id", "split"]], on="spine_key", how="left"
        )
        missing = frame["split"].isna().sum()
        if missing:
            raise ValueError(f"{missing} spines of the index have no split assignment - rebuild or version the split")
        if split is not None:
            frame = frame[frame["split"] == split]
        return SpineIndex(frame.reset_index(drop=True), self.dataset_versions)

    def pointcloud_path(self, row: pd.Series, n_points: int, variant: int = 1) -> Path:
        """File holding that size/variant (the packed file, or the legacy per-variant one)."""
        return pointcloud_variant_path(Path(row[f"pointcloud_{n_points}_path"]), variant)


def load_spine_index(
    manifest_paths: Sequence[Path],
    *,
    require_train_eligible: bool = True,
    path_mode: str = "relative",
    check_files: Sequence[str] = ("pointcloud_8192_path", "sdf_samples_path"),
) -> SpineIndex:
    """Read and concatenate manifests, keep training-eligible spines, resolve paths.

    ``check_files`` lists path columns that must exist for a spine to be kept;
    spines with missing files are dropped (and counted in ``frame.attrs``).
    """
    if path_mode not in {"relative", "absolute"}:
        raise ValueError("path_mode must be 'relative' or 'absolute'")
    frames = []
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        frame = pd.read_parquet(manifest_path)
        for column in KEY_COLUMNS:
            frame[column] = frame[column].astype(str)
        frame["manifest_path"] = str(manifest_path)
        if path_mode == "relative":
            frame = _reroot(frame, manifest_path.parent)
        frames.append(frame)
    if not frames:
        raise ValueError("No manifest paths given")
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "spine_key", [spine_key(*row) for row in frame[list(KEY_COLUMNS)].itertuples(index=False)])
    if frame["spine_key"].duplicated().any():
        dup = frame.loc[frame["spine_key"].duplicated(), "spine_key"].head(5).tolist()
        raise ValueError(f"Duplicate spine keys across manifests, e.g. {dup}")

    n_total = len(frame)
    if require_train_eligible:
        frame = frame[frame["train_eligible"].fillna(False).astype(bool)]
    n_eligible = len(frame)
    for column in check_files:
        exists = frame[column].map(lambda p: p is not None and Path(p).exists())
        frame = frame[exists]
    frame = frame.sort_values("spine_key").reset_index(drop=True)
    frame.attrs.update({"n_manifest": n_total, "n_train_eligible": n_eligible, "n_with_files": len(frame)})

    versions = tuple(
        sorted({(str(v), str(h)) for v, h in frame[["preprocessing_version", "config_hash"]].itertuples(index=False)})
    )
    return SpineIndex(frame, versions)


def _reroot(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    frame = frame.copy()
    rel_dirs = [
        root / spine_relative_dir(n, l, b, s)
        for n, l, b, s in frame[["neuron_id", "limb_id", "branch_id", "spine_id"]].itertuples(index=False)
    ]
    columns = [c for c in (*PATH_COLUMNS, *(f"pointcloud_{n}_path" for n in POINTCLOUD_SIZES)) if c in frame.columns]
    for column in columns:
        frame[column] = [
            None if (name := _file_name(stored)) is None else str(spine_dir / name)
            for stored, spine_dir in zip(frame[column], rel_dirs)
        ]
    return frame
