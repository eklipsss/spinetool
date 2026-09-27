from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .manifest import write_table_manifest
from .metadata import id_to_int
from .schemas import MORPHOLOGY_COLUMNS, SPINE_MORPHOLOGY_COLUMNS
from .spine_geometry import load_trimesh
from .spine_preprocessing import (
    ALGORITHM_VERSION,
    SpineRecord,
    discover_spines,
    load_stage_table,
    morphometrics_json_path,
    sealed_mesh_path,
)
from .statuses import STATUS_FAILED, STATUS_PENDING, STATUS_PROCESSING, STATUS_SUCCESS

MORPHOLOGY_VERSION = "spine-morphology-v1"
STAGE_MORPHOLOGY = "morphology"


class MorphologyStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS morphology_state (
                dataset TEXT,
                neuron_id TEXT,
                limb_id TEXT,
                original_branch_id TEXT,
                spine_id TEXT,
                stage TEXT,
                status TEXT,
                started_at REAL,
                ended_at REAL,
                duration_s REAL,
                error TEXT,
                input_path TEXT,
                output_path TEXT,
                processing_version TEXT,
                config_hash TEXT,
                PRIMARY KEY (dataset, neuron_id, limb_id, original_branch_id, spine_id, stage)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_status(self, record: SpineRecord, config_hash: str) -> Optional[str]:
        cursor = self.conn.execute(
            """
            SELECT status FROM morphology_state
            WHERE dataset=? AND neuron_id=? AND limb_id=? AND original_branch_id=? AND spine_id=?
              AND stage=? AND config_hash=? AND processing_version=?
            """,
            (
                record.dataset,
                record.neuron_id,
                record.limb_id,
                record.branch_id,
                record.spine_id,
                STAGE_MORPHOLOGY,
                config_hash,
                MORPHOLOGY_VERSION,
            ),
        )
        row = cursor.fetchone()
        return None if row is None else str(row[0])

    def mark(
        self,
        record: SpineRecord,
        status: str,
        *,
        started_at: Optional[float] = None,
        ended_at: Optional[float] = None,
        error: Optional[str] = None,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config_hash: str,
    ) -> None:
        duration = None
        if started_at is not None and ended_at is not None:
            duration = float(ended_at - started_at)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO morphology_state (
                dataset, neuron_id, limb_id, original_branch_id, spine_id, stage,
                status, started_at, ended_at, duration_s, error, input_path,
                output_path, processing_version, config_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.dataset,
                record.neuron_id,
                record.limb_id,
                record.branch_id,
                record.spine_id,
                STAGE_MORPHOLOGY,
                status,
                started_at,
                ended_at,
                duration,
                error,
                None if input_path is None else str(input_path),
                None if output_path is None else str(output_path),
                MORPHOLOGY_VERSION,
                config_hash,
            ),
        )
        self.conn.commit()


def write_spine_morphology_table(config: Any) -> pd.DataFrame:
    cfg = config.normalized()
    records = discover_spines(cfg.preprocessing_config())
    output_path = Path(cfg.morphology_output_path)
    frame = build_spine_morphology_table(cfg, records=records, output_path=output_path)
    _write_parquet_atomic(frame, output_path)
    write_table_manifest(
        output_path,
        frame,
        artifact_name="spine_morphology",
        processing_version=MORPHOLOGY_VERSION,
        config_hash=morphology_config_hash(cfg),
        input_paths={
            "raw_data_root": cfg.raw_data_root,
            "processed_root": cfg.processed_root,
            "preprocessed_root": cfg.preprocessing_output_root or "<dataset_root>/preprocessed",
        },
        status_column="morphology_status",
    )
    return frame


def build_spine_morphology_table(
    config: Any,
    records: Optional[Sequence[SpineRecord]] = None,
    stage_tables: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    cfg = config.normalized()
    records = list(discover_spines(cfg.preprocessing_config()) if records is None else records)
    preprocessing_cfg = cfg.preprocessing_config()
    stage_tables = _load_stage_tables(preprocessing_cfg, records) if stage_tables is None else stage_tables
    output_path = Path(cfg.morphology_output_path if output_path is None else output_path)
    config_hash = morphology_config_hash(cfg)
    store = MorphologyStateStore(Path(cfg.processed_root) / "morphology_state.sqlite")
    existing_rows = _existing_rows_by_key(output_path) if cfg.resume and not cfg.force else {}

    rows = []
    try:
        for batch in _batch_iter(records, getattr(cfg, "batch_size", 64)):
            for record in batch:
                previous = existing_rows.get(_record_key(record))
                if (
                    cfg.resume
                    and not cfg.force
                    and previous is not None
                    and store.get_status(record, config_hash) == STATUS_SUCCESS
                ):
                    rows.append(previous)
                    continue

                row = _base_morphology_row(record)
                row["config_hash"] = config_hash
                input_path = sealed_mesh_path(record, preprocessing_cfg)
                started = time.time()
                store.mark(
                    record,
                    STATUS_PROCESSING,
                    started_at=started,
                    input_path=input_path,
                    output_path=output_path,
                    config_hash=config_hash,
                )
                try:
                    orientation = _stage_row(stage_tables.get("oriented"), _record_key(record))
                    attachment_center = _vector_from_row(orientation, "origin_global")
                    if attachment_center is None:
                        raise ValueError("Missing attachment center from orientation stage.")
                    metrics = calculate_spine_morphology(record, preprocessing_cfg, attachment_center, orientation)
                    row.update(metrics)
                    _write_morphometrics_json(morphometrics_json_path(record, preprocessing_cfg), row)
                    row["morphology_status"] = STATUS_SUCCESS
                    ended = time.time()
                    store.mark(
                        record,
                        STATUS_SUCCESS,
                        started_at=started,
                        ended_at=ended,
                        input_path=input_path,
                        output_path=output_path,
                        config_hash=config_hash,
                    )
                except Exception as exc:
                    ended = time.time()
                    error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    row["morphology_status"] = STATUS_FAILED
                    row["morphology_error"] = error
                    store.mark(
                        record,
                        STATUS_FAILED,
                        started_at=started,
                        ended_at=ended,
                        error=error,
                        input_path=input_path,
                        output_path=output_path,
                        config_hash=config_hash,
                    )
                rows.append(row)
            _write_parquet_atomic(_ensure_morphology_columns(pd.DataFrame(rows)), output_path)
    finally:
        store.close()

    return _ensure_morphology_columns(pd.DataFrame(rows))


def calculate_spine_morphology(
    record: SpineRecord,
    preprocessing_cfg: Any,
    attachment_center: np.ndarray,
    orientation_row: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    from src.spine_analysis.mesh.utils import v_f_to_mesh_isolated
    from src.spine_analysis.shape_metric.float_metric import (
        ConvexHullRatioSpineMetric,
        ConvexHullVolumeSpineMetric,
        VolumeSpineMetric,
    )
    from src.spine_analysis.shape_metric.histogram_metric import OldChordDistributionSpineMetric
    from src.spine_analysis.shape_metric.junction_metric import (
        AreaSpineMetric,
        AverageDistanceSpineMetric,
        CVDSpineMetric,
        LengthAreaRatioSpineMetric,
        LengthSpineMetric,
        LengthVolumeRatioSpineMetric,
        OpenAngleSpineMetric,
    )
    from src.spine_analysis.shape_metric.utils import register_attachment_center

    sealed = load_trimesh(sealed_mesh_path(record, preprocessing_cfg), process=False)
    poly = v_f_to_mesh_isolated(
        np.asarray(sealed.vertices, dtype=float),
        np.asarray(sealed.faces, dtype=int),
    )
    register_attachment_center(poly, np.asarray(attachment_center, dtype=float))

    values: Dict[str, Any] = {
        "OldChordDistribution": _metric_value(OldChordDistributionSpineMetric(poly)),
        "OpenAngle": _metric_value(OpenAngleSpineMetric(poly)),
        "CVD": _metric_value(CVDSpineMetric(poly)),
        "AverageDistance": _metric_value(AverageDistanceSpineMetric(poly)),
        "LengthVolumeRatio": _metric_value(LengthVolumeRatioSpineMetric(poly)),
        "LengthAreaRatio": _metric_value(LengthAreaRatioSpineMetric(poly)),
        "Length": _metric_value(LengthSpineMetric(poly)),
        "Area": _metric_value(AreaSpineMetric(poly)),
        "Volume": _metric_value(VolumeSpineMetric(poly)),
        "ConvexHullVolume": _metric_value(ConvexHullVolumeSpineMetric(poly)),
        "ConvexHullRatio": _metric_value(ConvexHullRatioSpineMetric(poly)),
    }
    junction_area = None
    if orientation_row is not None and "attachment_loop_area" in orientation_row.index:
        junction_area = orientation_row.get("attachment_loop_area")
    values["JunctionArea"] = _safe_float(junction_area)
    return values


def morphology_config_hash(config: Any) -> str:
    payload = {
        "raw_data_root": str(config.raw_data_root),
        "processed_root": str(config.processed_root),
        "dataset": config.dataset,
        "neuron_id": config.neuron_id,
        "limb_id": config.limb_id,
        "branch_id": config.branch_id,
        "spine_id": config.spine_id,
        "limit": config.limit,
        "morphology_version": MORPHOLOGY_VERSION,
        "preprocessing_version": ALGORITHM_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _base_morphology_row(record: SpineRecord) -> Dict[str, Any]:
    return {
        "dataset": record.dataset,
        "neuron_id": record.neuron_id,
        "limb_id": id_to_int(record.limb_id),
        "original_branch_id": id_to_int(record.branch_id),
        "spine_id": id_to_int(record.spine_id),
        "morphology_status": STATUS_PENDING,
        "morphology_error": None,
        **_empty_morphology(),
        "processing_version": MORPHOLOGY_VERSION,
        "config_hash": None,
    }


def _load_stage_tables(preprocessing_cfg: Any, records: Sequence[SpineRecord]) -> Dict[str, pd.DataFrame]:
    return {"oriented": load_stage_table(preprocessing_cfg, records, "oriented_spines.parquet")}


def _write_morphometrics_json(path: Path, row: Dict[str, Any]) -> None:
    if not path.parent.exists():
        return
    payload = {key: row.get(key) for key in SPINE_MORPHOLOGY_COLUMNS}
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fd:
        json.dump(payload, fd, ensure_ascii=False, indent=2, default=_json_default)
    tmp_path.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


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


def _existing_rows_by_key(path: Path) -> Dict[tuple, Dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    result = {}
    for row in frame.to_dict(orient="records"):
        key = (
            str(row["dataset"]),
            str(row["neuron_id"]),
            id_to_int(row["limb_id"]),
            id_to_int(row["original_branch_id"]),
            id_to_int(row["spine_id"]),
        )
        result[key] = row
    return result


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


def _metric_value(metric: Any) -> Any:
    value = metric.value
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _empty_morphology() -> Dict[str, Any]:
    return {column: None for column in MORPHOLOGY_COLUMNS}


def _ensure_morphology_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in SPINE_MORPHOLOGY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[SPINE_MORPHOLOGY_COLUMNS]


def _batch_iter(items: Sequence[SpineRecord], batch_size: int) -> Iterable[Sequence[SpineRecord]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
