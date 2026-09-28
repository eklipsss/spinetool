from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .manifest import write_table_manifest
from .metadata import (
    dataset_metadata,
    id_to_int,
    identity_columns,
    load_metadata_tables,
    merge_metadata_status,
)
from .schemas import SPINES_INFO_COLUMNS
from .spine_geometry import load_trimesh
from .spine_preprocessing import (
    ALGORITHM_VERSION,
    SpinePreprocessingConfig,
    SpineRecord,
    discover_spines,
    load_stage_table,
    local_mesh_path,
    local_sealed_mesh_path,
    sealed_mesh_path,
)

INFO_VERSION = "spines-info-v1"


@dataclass(frozen=True)
class SpinesInfoConfig:
    raw_data_root: Path
    # Output location of the aggregated spines_info / spine_morphology tables.
    processed_root: Path = Path("data/processed/neuron-model/spines")
    # Where the per-spine preprocessing outputs live (None -> <dataset_root>/preprocessed).
    preprocessing_output_root: Optional[Path] = None
    metadata_root: Path = Path("datasets")
    output_path: Optional[Path] = None
    dataset: Optional[str] = None
    neuron_id: Optional[str] = None
    limb_id: Optional[str] = None
    branch_id: Optional[str] = None
    spine_id: Optional[str] = None
    limit: Optional[int] = None
    morphology_output_path: Optional[Path] = None
    batch_size: int = 64
    force: bool = False
    resume: bool = True

    def normalized(self) -> "SpinesInfoConfig":
        output_path = self.output_path
        if output_path is None:
            output_path = Path(self.processed_root) / "spines_info.parquet"
        morphology_output_path = self.morphology_output_path
        if morphology_output_path is None:
            morphology_output_path = Path(self.processed_root) / "spine_morphology.parquet"
        return SpinesInfoConfig(
            raw_data_root=Path(self.raw_data_root).expanduser().resolve(),
            processed_root=Path(self.processed_root).expanduser().resolve(),
            preprocessing_output_root=None
            if self.preprocessing_output_root is None
            else Path(self.preprocessing_output_root).expanduser().resolve(),
            metadata_root=Path(self.metadata_root).expanduser().resolve(),
            output_path=Path(output_path).expanduser().resolve(),
            dataset=self.dataset,
            neuron_id=self.neuron_id,
            limb_id=self.limb_id,
            branch_id=self.branch_id,
            spine_id=self.spine_id,
            limit=None if self.limit is None else int(self.limit),
            morphology_output_path=Path(morphology_output_path).expanduser().resolve(),
            batch_size=int(self.batch_size),
            force=bool(self.force),
            resume=bool(self.resume),
        )

    def config_hash(self) -> str:
        payload = asdict(self.normalized())
        for key in ("raw_data_root", "processed_root", "preprocessing_output_root", "metadata_root", "output_path"):
            payload[key] = None if payload[key] is None else str(payload[key])
        for key in ("morphology_output_path", "batch_size", "force", "resume"):
            payload.pop(key, None)
        payload["info_version"] = INFO_VERSION
        payload["preprocessing_version"] = ALGORITHM_VERSION
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def preprocessing_config(self) -> SpinePreprocessingConfig:
        cfg = self.normalized()
        return SpinePreprocessingConfig(
            raw_data_root=cfg.raw_data_root,
            output_root=cfg.preprocessing_output_root,
            metadata_root=cfg.metadata_root,
            dataset=cfg.dataset,
            neuron_id=cfg.neuron_id,
            limb_id=cfg.limb_id,
            branch_id=cfg.branch_id,
            spine_id=cfg.spine_id,
            batch_size=cfg.batch_size,
            limit=cfg.limit,
            force=cfg.force,
            resume=cfg.resume,
        )


def build_spines_info(config: SpinesInfoConfig) -> pd.DataFrame:
    cfg = config.normalized()
    preprocessing_cfg = cfg.preprocessing_config()
    records = discover_spines(preprocessing_cfg)
    stage_tables = _load_stage_tables(preprocessing_cfg, records)
    metadata = load_metadata_tables(cfg.metadata_root)

    rows = []
    config_hash = cfg.config_hash()
    for record in records:
        key = _record_key(record)
        rows.append(
            {
                **identity_columns(record, metadata),
                **_path_columns(record, preprocessing_cfg),
                **dataset_metadata(record, metadata),
                **_attachment_orientation_columns(key, stage_tables),
                **_mesh_qc_columns(record, stage_tables),
                **_status_columns(key, stage_tables),
                "merge_metadata_status": merge_metadata_status(record, metadata),
                "processing_version": INFO_VERSION,
                "config_hash": config_hash,
            }
        )

    frame = _ensure_columns(pd.DataFrame(rows))
    _write_parquet_atomic(frame, cfg.output_path)
    write_table_manifest(
        cfg.output_path,
        frame,
        artifact_name="spines_info",
        processing_version=INFO_VERSION,
        config_hash=config_hash,
        input_paths={
            "raw_data_root": cfg.raw_data_root,
            "processed_root": cfg.processed_root,
            "metadata_root": cfg.metadata_root,
            "preprocessed_root": cfg.preprocessing_output_root or "<dataset_root>/preprocessed",
        },
        status_column="merge_metadata_status",
    )
    return frame


def _load_stage_tables(preprocessing_cfg: SpinePreprocessingConfig, records: Any) -> Dict[str, pd.DataFrame]:
    tables = {
        "sealed": "sealed_spines.parquet",
        "false_spines": "false_spines.parquet",
        "oriented": "oriented_spines.parquet",
    }
    return {name: load_stage_table(preprocessing_cfg, records, table) for name, table in tables.items()}


def _path_columns(record: SpineRecord, preprocessing_cfg: SpinePreprocessingConfig) -> Dict[str, Any]:
    return {
        "source_mesh_path": str(record.source_path),
        "sealed_mesh_path": str(sealed_mesh_path(record, preprocessing_cfg)),
        "local_mesh_path": str(local_mesh_path(record, preprocessing_cfg)),
        "local_sealed_mesh_path": str(local_sealed_mesh_path(record, preprocessing_cfg)),
    }


def _attachment_orientation_columns(key: tuple, stage_tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    orientation = _stage_row(stage_tables.get("oriented"), key)
    origin = _vector_from_row(orientation, "origin_global")
    tangent = _vector_from_row(orientation, "tangent")
    radial = _vector_from_row(orientation, "radial")
    binormal = _vector_from_row(orientation, "binormal")
    projection = _vector_from_row(orientation, "branch_projection")
    return {
        **_xyz("attachment_center_global", origin),
        **_xyz("attachment_projection", projection),
        "attachment_boundary_perimeter": _row_value(orientation, "attachment_loop_perimeter"),
        "attachment_boundary_area": _row_value(orientation, "attachment_loop_area"),
        **_xyz("tangent", tangent),
        **_xyz("radial", radial),
        **_xyz("binormal", binormal),
    }


def _mesh_qc_columns(record: SpineRecord, stage_tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    sealed = _stage_row(stage_tables.get("sealed"), _record_key(record))
    source_vertices = None
    source_faces = None
    try:
        source_mesh = load_trimesh(record.source_path, process=False)
        source_vertices = int(len(source_mesh.vertices))
        source_faces = int(len(source_mesh.faces))
    except Exception:
        pass
    return {
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "n_boundary_loops_before": _row_value(sealed, "n_holes_before"),
        "is_watertight_after": _row_value(sealed, "is_watertight"),
        "is_manifold_after": _row_value(sealed, "is_manifold"),
        "n_connected_components": _row_value(sealed, "n_connected_components"),
    }


def _status_columns(key: tuple, stage_tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    detected = _stage_row(stage_tables.get("false_spines"), key)
    sealed = _stage_row(stage_tables.get("sealed"), key)
    oriented = _stage_row(stage_tables.get("oriented"), key)
    return {
        "detect_status": _row_value(detected, "status"),
        "seal_status": _row_value(sealed, "status"),
        "canonicalize_status": _row_value(oriented, "status"),
    }


def _stage_row(frame: Optional[pd.DataFrame], key: tuple) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    branch_column = "branch_id" if "branch_id" in frame.columns else "original_branch_id"
    mask = (
        (frame["dataset"].astype(str) == str(key[0]))
        & (frame["neuron_id"].astype(str) == str(key[1]))
        & (frame["limb_id"].map(id_to_int) == int(key[2]))
        & (frame[branch_column].map(id_to_int) == int(key[3]))
        & (frame["spine_id"].map(id_to_int) == int(key[4]))
    )
    local = frame.loc[mask]
    if len(local) == 0:
        return None
    return local.iloc[-1]


def _record_key(record: SpineRecord) -> tuple:
    return (
        record.dataset,
        record.neuron_id,
        id_to_int(record.limb_id),
        id_to_int(record.branch_id),
        id_to_int(record.spine_id),
    )


def _vector_from_row(row: Optional[pd.Series], column: str) -> Optional[np.ndarray]:
    if row is None or column not in row.index:
        return None
    value = row.get(column)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def _xyz(prefix: str, value: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if value is None:
        return {f"{prefix}_{axis}": None for axis in ("x", "y", "z")}
    return {
        f"{prefix}_x": float(value[0]),
        f"{prefix}_y": float(value[1]),
        f"{prefix}_z": float(value[2]),
    }


def _row_value(row: Optional[pd.Series], column: str) -> Any:
    if row is None or column not in row.index:
        return None
    value = row.get(column)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in SPINES_INFO_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[SPINES_INFO_COLUMNS]


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)

