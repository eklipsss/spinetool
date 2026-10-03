"""Spine mesh preprocessing for the neuron-model ``S`` module.

Implements ``docs/neuron-model/s-module-preprocessing.md``:

1. ``seal``         sealing + attachment loop / attachment cap;
2. ``false-spine``  false spine detection (skeleton distance profile);
3. ``orient``       local coordinate frame (``orient_spines_w_holes``);
4. ``manifest``     logical branch merge (merged_branch_id), metadata.json, manifest.parquet;
5. ``qc``           geometric QC of ``local_sealed`` meshes;
6. ``pointcloud``   area-weighted point clouds;
7. ``sdf``          SDF sample pools.

Stage 4 does not touch geometry, so it runs last and collects the results of all
other stages. Every spine keeps all its artifacts in
``<dataset>/preprocessed/<neuron>/limb_*/branch_*/spines/spine_<id>/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .manifest import write_table_manifest
from .spine_geometry import (
    apply_homogeneous,
    atomic_export_mesh,
    boundary_edge_count,
    boundary_loops,
    cgal_skeleton_segments_from_trimesh,
    detect_attachment_loop,
    distance_to_segments,
    find_attachment_cap_faces,
    load_skeleton_segments_cached,
    load_trimesh,
    longest_path_polyline,
    mesh_geometry_report,
    orient_spine_path_from_dendrite,
    orient_spines_w_holes,
    require_trimesh,
    sample_polyline_by_arclength,
    select_attachment_patch,
    self_intersection_check,
)
from .spine_sampling import (
    SAMPLE_TYPE_CODES,
    build_sdf_samples,
    sample_surface_area_weighted,
    sdf_quality,
    split_pool,
    stable_seed,
)
from .statuses import (
    DETECT_INVALID,
    DETECT_VALID,
    STATUS_FAILED,
    STATUS_NEEDS_REVIEW,
    STATUS_PROCESSING,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
)

ALGORITHM_VERSION = "spine-preprocessing-v2"

STAGE_SEAL = "seal"
STAGE_FALSE_SPINE = "false_spine_detection"
STAGE_ORIENT = "orient"
STAGE_QC = "mesh_qc"
STAGE_POINTCLOUD = "pointcloud"
STAGE_SDF = "sdf"
STAGE_MORPHOMETRICS = "morphometrics"
STAGE_METADATA = "metadata"
STAGE_MANIFEST = "manifest"

QUALITY_VALID = "valid"
QUALITY_INVALID = "invalid"

DEFAULT_CONFIG_PATH = Path("data/processed/config/spine_preprocessing.yaml")


@functools.lru_cache(maxsize=1)
def _mesh_repair_module():
    """Load mesh_repair.py without importing dendrite_analysis.__init__."""
    module_name = "_neuron_model_dendrite_mesh_repair"
    module_path = Path(__file__).resolve().parents[1] / "dendrite_analysis" / "mesh_repair.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mesh repair module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules[cls.__module__].
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _repair_mesh(mesh: Any, **kwargs: Any):
    return _mesh_repair_module().repair_mesh(mesh, **kwargs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpinePreprocessingConfig:
    raw_data_root: Path
    # None -> <dataset_root>/preprocessed; otherwise <output_root>/<dataset>.
    output_root: Optional[Path] = None
    metadata_root: Path = Path("datasets")

    # selection
    dataset: Optional[str] = None
    neuron_id: Optional[str] = None
    limb_id: Optional[str] = None
    branch_id: Optional[str] = None
    spine_id: Optional[str] = None
    limit: Optional[int] = None

    # execution
    batch_size: int = 64
    workers: int = 1
    force: bool = False
    resume: bool = True
    retry_failed_only: bool = False
    verbose: bool = False
    preprocessing_version: str = "v1"
    physical_unit: str = "nm"

    # stage 1: sealing / attachment loop
    attachment_weight_perimeter: float = 0.4
    attachment_weight_area: float = 0.35
    attachment_weight_distance: float = 0.25
    attachment_ambiguity_ratio: float = 0.85

    # stage 2: false spines
    false_spine_threshold_nm: float = 300.0
    false_spine_threshold_nm_by_dataset: Tuple[Tuple[str, float], ...] = ()
    skeleton_sample_points: int = 10

    # stage 3: orientation
    only_valid_for_orientation: bool = True
    min_tangent_component: float = 0.1

    # stage 5: mesh QC
    transform_tolerance: float = 1e-6
    invariance_rtol: float = 1e-6
    degenerate_area_tol: float = 1e-12
    max_degenerate_faces: int = 0
    check_self_intersections: bool = True
    # True by default: a needs_review mesh (e.g. self_intersections - a
    # small, localized geometric defect, often already present in the raw
    # segmented mesh - see docs/neuron-model/s-module-preprocessing.md 3.8.4)
    # is not excluded from pointcloud/SDF/morphometrics/train_eligible by
    # default; it is still flagged for manual review. Set False for a
    # stricter run that keeps only geometrically pristine (`valid`) meshes.
    allow_needs_review: bool = True

    # stage 6: point clouds
    pointcloud_sizes: Tuple[int, ...] = (2048, 4096, 8192)
    pointcloud_seeds: Tuple[int, ...] = (42, 43, 44, 45)

    # stage 7: SDF
    sdf_pool_size: int = 16384
    sdf_surface_fraction: float = 0.20
    sdf_near_fraction: float = 0.60
    sdf_uniform_fraction: float = 0.20
    sdf_near_sigmas: Tuple[float, ...] = (5.0, 20.0, 60.0)
    sdf_bbox_margin: float = 100.0
    sdf_surface_tolerance: float = 1e-3
    sdf_interior_probe: float = 2.0

    @classmethod
    def from_yaml(cls, path: Path, **overrides: Any) -> "SpinePreprocessingConfig":
        """Build a config from a sectioned YAML file; ``overrides`` win over the file."""
        import yaml

        with Path(path).open("r", encoding="utf-8") as fd:
            data = yaml.safe_load(fd) or {}
        field_names = {field.name for field in dataclasses.fields(cls)}
        values: Dict[str, Any] = {}
        for section, content in data.items():
            items = content.items() if isinstance(content, Mapping) else [(section, content)]
            for key, value in items:
                if key not in field_names:
                    raise KeyError(f"Unknown spine preprocessing config key: {section}.{key}")
                values[key] = value
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values).normalized()

    def normalized(self) -> "SpinePreprocessingConfig":
        by_dataset = self.false_spine_threshold_nm_by_dataset
        if isinstance(by_dataset, Mapping):
            by_dataset = tuple(sorted((str(k), float(v)) for k, v in by_dataset.items()))
        return dataclasses.replace(
            self,
            raw_data_root=Path(self.raw_data_root).expanduser().resolve(),
            output_root=None if self.output_root is None else Path(self.output_root).expanduser().resolve(),
            metadata_root=Path(self.metadata_root).expanduser().resolve(),
            batch_size=max(1, int(self.batch_size)),
            workers=max(1, int(self.workers)),
            limit=None if self.limit is None else int(self.limit),
            false_spine_threshold_nm_by_dataset=tuple((str(k), float(v)) for k, v in by_dataset),
            pointcloud_sizes=tuple(int(v) for v in self.pointcloud_sizes),
            pointcloud_seeds=tuple(int(v) for v in self.pointcloud_seeds),
            sdf_near_sigmas=tuple(float(v) for v in self.sdf_near_sigmas),
        )

    def false_spine_threshold(self, dataset: str) -> float:
        return float(dict(self.false_spine_threshold_nm_by_dataset).get(dataset, self.false_spine_threshold_nm))

    def stage_hash(self, stage: str) -> str:
        """Hash of the parameters that affect the results of ``stage`` (not selection/execution)."""
        payload = {name: _jsonable(getattr(self, name)) for name in STAGE_PARAMS.get(stage, ())}
        payload.update(
            {
                "stage": stage,
                "algorithm_version": ALGORITHM_VERSION,
                "preprocessing_version": self.preprocessing_version,
            }
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


STAGE_PARAMS: Dict[str, Tuple[str, ...]] = {
    STAGE_SEAL: (
        "attachment_weight_perimeter",
        "attachment_weight_area",
        "attachment_weight_distance",
        "attachment_ambiguity_ratio",
        "degenerate_area_tol",
        "check_self_intersections",
    ),
    STAGE_FALSE_SPINE: ("false_spine_threshold_nm", "false_spine_threshold_nm_by_dataset", "skeleton_sample_points"),
    STAGE_ORIENT: ("only_valid_for_orientation", "min_tangent_component", "physical_unit"),
    STAGE_QC: (
        "transform_tolerance",
        "invariance_rtol",
        "degenerate_area_tol",
        "max_degenerate_faces",
        "check_self_intersections",
    ),
    STAGE_POINTCLOUD: ("pointcloud_sizes", "pointcloud_seeds", "allow_needs_review"),
    STAGE_SDF: (
        "sdf_pool_size",
        "sdf_surface_fraction",
        "sdf_near_fraction",
        "sdf_uniform_fraction",
        "sdf_near_sigmas",
        "sdf_bbox_margin",
        "sdf_surface_tolerance",
        "sdf_interior_probe",
        "allow_needs_review",
    ),
    STAGE_MORPHOMETRICS: ("allow_needs_review",),
    STAGE_METADATA: ("metadata_root",),
    STAGE_MANIFEST: ("allow_needs_review", "metadata_root"),
}


def config_hash(cfg: SpinePreprocessingConfig) -> str:
    payload = {name: _jsonable(getattr(cfg, name)) for names in STAGE_PARAMS.values() for name in names}
    payload["algorithm_version"] = ALGORITHM_VERSION
    payload["preprocessing_version"] = cfg.preprocessing_version
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Records and paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpineRecord:
    dataset: str
    neuron_id: str
    limb_id: str
    branch_id: str
    spine_id: str
    source_path: Path
    branch_path: Path
    branch_skeleton_path: Path
    dataset_root: Path

    @property
    def identity(self) -> Dict[str, str]:
        return {
            "dataset": self.dataset,
            "neuron_id": self.neuron_id,
            "limb_id": self.limb_id,
            "branch_id": self.branch_id,
            "spine_id": self.spine_id,
        }

    @property
    def key(self) -> Tuple[str, str, str, str, str]:
        return (self.dataset, self.neuron_id, self.limb_id, self.branch_id, self.spine_id)

    @property
    def object_id(self) -> str:
        return "/".join(self.key)


def discover_spines(config: SpinePreprocessingConfig) -> List[SpineRecord]:
    cfg = config.normalized()
    if cfg.dataset:
        dataset_roots = [(cfg.dataset, cfg.raw_data_root / cfg.dataset)]
        if not dataset_roots[0][1].exists() and cfg.raw_data_root.name == cfg.dataset:
            dataset_roots = [(cfg.dataset, cfg.raw_data_root)]
    else:
        dataset_roots = [
            (path.name, path)
            for path in sorted(cfg.raw_data_root.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]

    records: List[SpineRecord] = []
    for dataset_name, dataset_root in dataset_roots:
        if not dataset_root.exists():
            continue
        for spine_path in sorted(dataset_root.glob("*/limb_*/branch_*/spines/spine_*.off")):
            branch_path = spine_path.parent.parent
            limb_path = branch_path.parent
            neuron_path = limb_path.parent
            record = SpineRecord(
                dataset=dataset_name,
                neuron_id=neuron_path.name,
                limb_id=limb_path.name.replace("limb_", ""),
                branch_id=branch_path.name.replace("branch_", ""),
                spine_id=spine_path.stem.replace("spine_", ""),
                source_path=spine_path,
                branch_path=branch_path,
                branch_skeleton_path=branch_path / "branch_skeleton.npy",
                dataset_root=dataset_root,
            )
            if _record_matches(record, cfg):
                records.append(record)
                if cfg.limit is not None and len(records) >= cfg.limit:
                    return records
    return records


def _record_matches(record: SpineRecord, cfg: SpinePreprocessingConfig) -> bool:
    return all(
        [
            cfg.neuron_id is None or record.neuron_id == cfg.neuron_id,
            cfg.limb_id is None or record.limb_id == _strip_prefix(cfg.limb_id, "limb_"),
            cfg.branch_id is None or record.branch_id == _strip_prefix(cfg.branch_id, "branch_"),
            cfg.spine_id is None or record.spine_id == _strip_prefix(cfg.spine_id, "spine_"),
        ]
    )


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix):] if value.startswith(prefix) else value


def preprocessed_root(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    if cfg.output_root is not None:
        return Path(cfg.output_root) / record.dataset
    return record.dataset_root / "preprocessed"


def spine_output_dir(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    """Mirror of the raw hierarchy: <root>/<neuron>/limb_*/branch_*/spines/spine_<id>/."""
    return (
        preprocessed_root(record, cfg)
        / record.neuron_id
        / record.branch_path.parent.name
        / record.branch_path.name
        / "spines"
        / record.source_path.stem
    )


def sealed_mesh_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / f"sealed_{record.source_path.name}"


def local_mesh_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / f"local_{record.source_path.name}"


def local_sealed_mesh_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / f"local_sealed_{record.source_path.name}"


def attachment_json_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / f"attachment_region_{record.source_path.stem}.json"


def transform_json_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / f"transform_{record.source_path.stem}.json"


def quality_json_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / f"quality_{record.source_path.stem}.json"


def metadata_json_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / "metadata.json"


def morphometrics_json_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / "morphometrics.json"


def pointcloud_path(record: SpineRecord, cfg: SpinePreprocessingConfig, n_points: int, variant: int) -> Path:
    return spine_output_dir(record, cfg) / f"pointcloud_{int(n_points)}_{int(variant)}.npz"


def sdf_samples_path(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return spine_output_dir(record, cfg) / "sdf_samples.npz"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fd:
        json.dump(_jsonable(payload), fd, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fd:
        return json.load(fd)


def _update_json_atomic(path: Path, updates: Mapping[str, Any]) -> None:
    payload = _read_json(path) if path.exists() else {}
    payload.update(_jsonable(updates))
    _write_json_atomic(path, payload)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as fd:
        np.savez_compressed(fd, **arrays)
    tmp_path.replace(path)


def _write_table(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    *,
    artifact_name: Optional[str] = None,
    config_hash: Optional[str] = None,
    input_paths: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_jsonable(row) for row in rows])
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    if artifact_name is not None and config_hash is not None:
        write_table_manifest(
            path,
            frame,
            artifact_name=artifact_name,
            processing_version=ALGORITHM_VERSION,
            config_hash=config_hash,
            input_paths=input_paths,
            status_column="status",
        )
    return frame


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    return tuple(str(row[name]) for name in ("dataset", "neuron_id", "limb_id", "branch_id", "spine_id"))  # type: ignore[return-value]


def _read_table_rows(path: Path) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    if not path.exists():
        return {}
    return {_row_key(row): row for row in pd.read_parquet(path).to_dict(orient="records")}


def load_stage_table(cfg: SpinePreprocessingConfig, records: Sequence[SpineRecord], table_name: str) -> pd.DataFrame:
    """Concatenate ``table_name`` over all preprocessed roots touched by ``records``."""
    frames = []
    for root in sorted({preprocessed_root(record, cfg) for record in records}):
        path = root / table_name
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _base_row(record: SpineRecord) -> Dict[str, Any]:
    return {**record.identity, "source_path": str(record.source_path)}


# ---------------------------------------------------------------------------
# Processing state
# ---------------------------------------------------------------------------


class ProcessingStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_state (
                dataset TEXT,
                neuron_id TEXT,
                limb_id TEXT,
                branch_id TEXT,
                spine_id TEXT,
                stage TEXT,
                status TEXT,
                started_at REAL,
                ended_at REAL,
                duration_s REAL,
                error TEXT,
                input_path TEXT,
                output_path TEXT,
                algorithm_version TEXT,
                config_hash TEXT,
                PRIMARY KEY (dataset, neuron_id, limb_id, branch_id, spine_id, stage)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_state(self, record: SpineRecord, stage: str) -> Optional[Tuple[str, str, str]]:
        """Return ``(status, algorithm_version, config_hash)`` of the last run of ``stage``."""
        cursor = self.conn.execute(
            """
            SELECT status, algorithm_version, config_hash FROM processing_state
            WHERE dataset=? AND neuron_id=? AND limb_id=? AND branch_id=? AND spine_id=? AND stage=?
            """,
            (*record.key, stage),
        )
        row = cursor.fetchone()
        return None if row is None else (str(row[0]), str(row[1]), str(row[2]))

    def get_status(self, record: SpineRecord, stage: str, config_hash: str) -> Optional[str]:
        state = self.get_state(record, stage)
        return _status_from_state(state, config_hash)

    def get_all_states(self, stage: str) -> Dict[Tuple[str, str, str, str, str], Tuple[str, str, str]]:
        """Bulk-fetch ``(status, algorithm_version, config_hash)`` for every record at ``stage``.

        One query instead of one per record: at full Minnie/H01 scale (tens
        of thousands of spines) a per-record ``get_state`` call in the resume
        scan turns into tens of thousands of sqlite round trips; this
        replaces that with a single query plus dict lookups.
        """
        cursor = self.conn.execute(
            """
            SELECT dataset, neuron_id, limb_id, branch_id, spine_id, status, algorithm_version, config_hash
            FROM processing_state WHERE stage=?
            """,
            (stage,),
        )
        return {
            (str(d), str(n), str(l), str(b), str(s)): (str(status), str(av), str(ch))
            for d, n, l, b, s, status, av, ch in cursor.fetchall()
        }

    def mark(
        self,
        record: SpineRecord,
        stage: str,
        status: str,
        *,
        started_at: Optional[float] = None,
        ended_at: Optional[float] = None,
        error: Optional[str] = None,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config_hash: str,
        commit: bool = True,
    ) -> None:
        duration = None
        if started_at is not None and ended_at is not None:
            duration = float(ended_at - started_at)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO processing_state (
                dataset, neuron_id, limb_id, branch_id, spine_id, stage, status,
                started_at, ended_at, duration_s, error, input_path, output_path,
                algorithm_version, config_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *record.key,
                stage,
                status,
                started_at,
                ended_at,
                duration,
                error,
                None if input_path is None else str(input_path),
                None if output_path is None else str(output_path),
                ALGORITHM_VERSION,
                config_hash,
            ),
        )
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()


def _status_from_state(state: Optional[Tuple[str, str, str]], config_hash: str) -> Optional[str]:
    if state is None or state[1] != ALGORITHM_VERSION or state[2] != config_hash:
        return None
    return state[0]


# ---------------------------------------------------------------------------
# Generic stage runner
# ---------------------------------------------------------------------------


class StepTimer:
    """Per-object step timings; printed as ``[i/n] step - t s`` when verbose."""

    def __init__(self, label: str, total: int, verbose: bool) -> None:
        self.label = label
        self.total = total
        self.verbose = verbose
        self.timings: Dict[str, float] = {}
        if verbose:
            print(label, flush=True)

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - started
        self.timings[name] = round(elapsed, 4)
        if self.verbose:
            print(f"[{len(self.timings)}/{self.total}] {name} - {elapsed:.2f} s", flush=True)
        return result


@dataclass(frozen=True)
class StageSpec:
    name: str
    cli_name: str
    table: str
    worker: Callable[[SpineRecord, SpinePreprocessingConfig, StepTimer], Dict[str, Any]]
    n_steps: int
    input_path: Callable[[SpineRecord, SpinePreprocessingConfig], Path]
    output_path: Callable[[SpineRecord, SpinePreprocessingConfig], Path]
    # (record, cfg, context) -> reason to skip, or None
    eligibility: Optional[Callable[[SpineRecord, SpinePreprocessingConfig, Dict[str, Any]], Optional[str]]] = None


def _execute_stage_worker(stage_name: str, record: SpineRecord, cfg: SpinePreprocessingConfig) -> Dict[str, Any]:
    """Top-level (picklable) wrapper that never raises."""
    spec = STAGES[stage_name]
    timer = StepTimer(record.object_id, spec.n_steps, cfg.verbose)
    started = time.time()
    try:
        row = spec.worker(record, cfg, timer)
        return {"row": row, "error": None, "started": started, "ended": time.time(), "timings": timer.timings}
    except Exception as exc:
        return {
            "row": None,
            "error": {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "started": started,
            "ended": time.time(),
            "timings": timer.timings,
        }


def _batch_iter(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _map_chunksize(records: Sequence[SpineRecord], workers: int) -> int:
    """``executor.map`` chunk size: keep same-branch records together.

    ``discover_spines`` yields records sorted by branch, so a contiguous
    chunk of a batch typically covers one or a few whole branches. Sending
    such a chunk to a single worker (instead of the default chunksize=1,
    which round-robins individual records across all workers) lets
    ``load_skeleton_segments_cached`` actually hit its cache instead of
    re-reading ``branch_skeleton.npy`` once per spine - worthwhile when the
    raw dataset sits on a slow network share (see
    docs/neuron-model/windows-workstation-specs.md).

    Sized as ``min(average branch size in this batch, items // workers)``:
    the first term is how much locality is actually available; the second
    keeps at least one chunk per worker so a small or single-branch batch
    doesn't leave most workers idle. Capped at 16.
    """
    n = len(records)
    if workers <= 1 or n <= 1:
        return 1
    n_branches = len({(r.dataset, r.neuron_id, r.limb_id, r.branch_id) for r in records})
    avg_branch_size = max(1, n // max(1, n_branches))
    per_worker = max(1, n // workers)
    return max(1, min(16, avg_branch_size, per_worker))


def _progress(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc)


def _append_errors(root: Path, errors: Sequence[Dict[str, Any]]) -> None:
    if not errors:
        return
    path = root / "errors.parquet"
    frame = pd.DataFrame(list(errors))
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _state_from_row(row: Mapping[str, Any]) -> str:
    status = row.get("_state") or row.get("status")
    return status if status in {STATUS_SUCCESS, STATUS_NEEDS_REVIEW, STATUS_SKIPPED, STATUS_FAILED} else STATUS_SUCCESS


def _print_summary(stage: str, dataset_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    statuses = pd.Series([row.get("status") for row in rows], dtype=object).value_counts(dropna=False)
    print(f"\n[{stage}] {dataset_root}")
    print(f"  Processed:           {len(rows)}")
    for status, count in statuses.items():
        print(f"  {str(status):<20} {int(count)}")


def run_stage(
    stage_name: str,
    config: SpinePreprocessingConfig,
    records: Optional[Sequence[SpineRecord]] = None,
) -> pd.DataFrame:
    """Run one stage over ``records`` with batching, resume, state journal and errors log."""
    spec = STAGES[stage_name]
    cfg = config.normalized()
    records = list(discover_spines(cfg) if records is None else records)
    stage_hash = cfg.stage_hash(spec.name)

    groups: Dict[Path, List[SpineRecord]] = {}
    for record in records:
        groups.setdefault(preprocessed_root(record, cfg), []).append(record)

    executor = ProcessPoolExecutor(max_workers=cfg.workers) if cfg.workers > 1 else None
    result_rows: List[Dict[str, Any]] = []
    try:
        for root, group in groups.items():
            root.mkdir(parents=True, exist_ok=True)
            table_path = root / spec.table
            rows_by_key = _read_table_rows(table_path)
            store = ProcessingStateStore(root / "processing_state.sqlite")
            context: Dict[str, Any] = {"root": root}
            touched: List[Tuple[str, ...]] = []
            todo: List[SpineRecord] = []
            # One query for the whole root instead of one per record - matters
            # once a dataset has tens of thousands of spines (full Minnie/H01
            # on the Windows workstation; see docs/neuron-model/
            # windows-workstation-specs.md for the target hardware profile).
            all_states = store.get_all_states(spec.name)
            try:
                for record in group:
                    touched.append(record.key)
                    if spec.eligibility is not None:
                        reason = spec.eligibility(record, cfg, context)
                        if reason is not None:
                            rows_by_key[record.key] = {**_base_row(record), "status": STATUS_SKIPPED, "reason": reason}
                            store.mark(record, spec.name, STATUS_SKIPPED, error=reason, config_hash=stage_hash, commit=False)
                            continue
                    state = all_states.get(record.key)
                    if cfg.retry_failed_only:
                        if state is not None and state[0] == STATUS_FAILED:
                            todo.append(record)
                        continue
                    done = _status_from_state(state, stage_hash) in {STATUS_SUCCESS, STATUS_NEEDS_REVIEW}
                    if (
                        cfg.resume
                        and not cfg.force
                        and done
                        and record.key in rows_by_key
                        and spec.output_path(record, cfg).exists()
                    ):
                        continue
                    todo.append(record)
                store.commit()

                progress = _progress(len(todo), f"{spec.cli_name} [{root.parent.name}]")
                counters = {STATUS_SUCCESS: 0, STATUS_FAILED: 0, STATUS_NEEDS_REVIEW: 0}
                for batch in _batch_iter(todo, cfg.batch_size):
                    for record in batch:
                        store.mark(
                            record,
                            spec.name,
                            STATUS_PROCESSING,
                            started_at=time.time(),
                            input_path=spec.input_path(record, cfg),
                            output_path=spec.output_path(record, cfg),
                            config_hash=stage_hash,
                            commit=False,
                        )
                    store.commit()

                    if executor is None:
                        outcomes = [_execute_stage_worker(spec.name, record, cfg) for record in batch]
                    else:
                        chunksize = _map_chunksize(batch, cfg.workers)
                        outcomes = list(
                            executor.map(
                                _execute_stage_worker,
                                [spec.name] * len(batch),
                                batch,
                                [cfg] * len(batch),
                                chunksize=chunksize,
                            )
                        )

                    errors = []
                    for record, outcome in zip(batch, outcomes):
                        error = outcome["error"]
                        if error is None:
                            row = {**_base_row(record), **outcome["row"]}
                            state = _state_from_row(row)
                            row.pop("_state", None)
                            error_text = row.get("reason") if state == STATUS_NEEDS_REVIEW else None
                        else:
                            state = STATUS_FAILED
                            error_text = f"{error['exception_type']}: {error['message']}"
                            row = {**_base_row(record), "status": STATUS_FAILED, "error": error_text}
                            errors.append(
                                {
                                    "stage": spec.name,
                                    "object_id": record.object_id,
                                    **error,
                                    "input_path": str(spec.input_path(record, cfg)),
                                    "created_at": outcome["ended"],
                                }
                            )
                        row["step_timings"] = json.dumps(outcome["timings"])
                        row["preprocessing_version"] = cfg.preprocessing_version
                        row["config_hash"] = stage_hash
                        rows_by_key[record.key] = row
                        counters[state] = counters.get(state, 0) + 1
                        store.mark(
                            record,
                            spec.name,
                            state,
                            started_at=outcome["started"],
                            ended_at=outcome["ended"],
                            error=error_text,
                            input_path=spec.input_path(record, cfg),
                            output_path=spec.output_path(record, cfg),
                            config_hash=stage_hash,
                            commit=False,
                        )
                    store.commit()
                    _append_errors(root, errors)
                    _write_table(list(rows_by_key.values()), table_path)
                    if progress is not None:
                        progress.update(len(batch))
                        progress.set_postfix(
                            success=counters[STATUS_SUCCESS],
                            failed=counters[STATUS_FAILED],
                            review=counters[STATUS_NEEDS_REVIEW],
                        )
                if progress is not None:
                    progress.close()
            finally:
                store.close()

            _write_table(
                list(rows_by_key.values()),
                table_path,
                artifact_name=Path(spec.table).stem,
                config_hash=stage_hash,
                input_paths={"dataset_root": group[0].dataset_root, "preprocessed_root": root},
            )
            group_rows = [rows_by_key[key] for key in touched if key in rows_by_key]
            _print_summary(spec.cli_name, group[0].dataset_root, group_rows)
            result_rows.extend(group_rows)
    finally:
        if executor is not None:
            executor.shutdown()

    return pd.DataFrame([_jsonable(row) for row in result_rows])


# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------


def _status_lookup(context: Dict[str, Any], table: str, column: str = "status") -> Dict[Tuple[str, ...], Any]:
    cache_key = f"lookup:{table}:{column}"
    if cache_key not in context:
        rows = _read_table_rows(Path(context["root"]) / table)
        context[cache_key] = {key: row.get(column) for key, row in rows.items()}
    return context[cache_key]


def _training_eligibility(record: SpineRecord, cfg: SpinePreprocessingConfig, context: Dict[str, Any]) -> Optional[str]:
    """Stages 6-7 run only for valid spines with an allowed mesh-quality status."""
    if _status_lookup(context, "false_spines.parquet").get(record.key) != DETECT_VALID:
        return "not_valid_after_false_spine_detection"
    quality = _status_lookup(context, "mesh_quality.parquet").get(record.key)
    allowed = {QUALITY_VALID, STATUS_NEEDS_REVIEW} if cfg.allow_needs_review else {QUALITY_VALID}
    if quality not in allowed:
        return f"mesh_quality_{quality or 'missing'}"
    if not local_sealed_mesh_path(record, cfg).exists():
        return "missing_local_sealed_mesh"
    return None


# ---------------------------------------------------------------------------
# Stage 1: sealing + attachment region
# ---------------------------------------------------------------------------


class SealingQCError(RuntimeError):
    pass


def _seal_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    if not record.branch_skeleton_path.exists():
        raise FileNotFoundError(f"Missing branch skeleton: {record.branch_skeleton_path}")

    mesh = timer.run("load mesh", lambda: load_trimesh(record.source_path, process=False))
    segments = load_skeleton_segments_cached(record.branch_skeleton_path)
    n_boundary_before = boundary_edge_count(mesh)

    detection = timer.run(
        "detect attachment loop",
        lambda: detect_attachment_loop(
            mesh,
            segments,
            weights=(cfg.attachment_weight_perimeter, cfg.attachment_weight_area, cfg.attachment_weight_distance),
            ambiguity_ratio=cfg.attachment_ambiguity_ratio,
        ),
    )
    repaired, _, _ = timer.run(
        "seal holes",
        lambda: _repair_mesh(
            mesh,
            fix_normals=True,
            fill_holes=True,
            remove_degenerate=True,
            stitch_borders=True,
            verbose=False,
        ),
    )

    cap_faces = np.empty(0, dtype=int)
    loop_sealed_indices = np.empty(0, dtype=int)
    if detection.loop is not None:
        cap_faces, loop_sealed_indices = timer.run(
            "identify attachment cap", lambda: find_attachment_cap_faces(mesh, repaired, detection.loop)
        )

    # Fallback for an ambiguous pre-seal detection (s-module-preprocessing.md
    # 3.5.4): detect_attachment_loop's perimeter/area/is_closed come from
    # ordering the *original* mesh's boundary edges into a cycle, which is
    # only well-defined when every boundary vertex has exactly 2 boundary
    # edges - a vertex with more (two holes touching at one point) makes that
    # ordering ambiguous even though sealing itself filled the hole just
    # fine. select_attachment_patch reruns the same scoring on the sealed
    # mesh's own face patches instead (real triangulated area, rim length
    # that doesn't need a cycle order) - well-defined regardless, so it can
    # resolve cases the loop walk can't, and may also break a close_candidates
    # tie differently (real patch area vs. a flat-polygon estimate).
    patch_winner: Optional[Dict[str, Any]] = None
    patch_candidates: List[Dict[str, Any]] = []
    if detection.ambiguous:
        patch_winner, patch_candidates, _patch_reason = timer.run(
            "attachment patch fallback",
            lambda: select_attachment_patch(
                mesh,
                repaired,
                segments,
                weights=(cfg.attachment_weight_perimeter, cfg.attachment_weight_area, cfg.attachment_weight_distance),
                ambiguity_ratio=cfg.attachment_ambiguity_ratio,
            ),
        )
        if patch_winner is not None:
            cap_faces = patch_winner["face_indices"]
            loop_sealed_indices = patch_winner["sealed_vertex_indices"]

    cap_area = float(np.asarray(repaired.area_faces)[cap_faces].sum()) if len(cap_faces) else None

    report = timer.run(
        "QC",
        lambda: mesh_geometry_report(
            repaired,
            degenerate_area_tol=cfg.degenerate_area_tol,
            check_self_intersections=cfg.check_self_intersections,
        ),
    )
    if not (report["is_watertight"] and report["has_positive_finite_volume"] and report["has_finite_coordinates"]):
        raise SealingQCError(
            f"Sealed mesh failed QC: watertight={report['is_watertight']}, "
            f"volume={report['volume']}, boundary_edges={report['n_boundary_edges']}"
        )

    review_reason = None
    if detection.ambiguous and patch_winner is None:
        review_reason = f"attachment_ambiguous:{detection.reason}"
    elif len(cap_faces) == 0:
        review_reason = "attachment_cap_not_found"

    loop = detection.loop
    if patch_winner is not None:
        loop_vertex_indices: Any = patch_winner["vertex_indices"]
        loop_edges: Any = patch_winner["edges"]
        loop_centroid: Any = patch_winner["center"]
        loop_perimeter: Any = patch_winner["perimeter"]
        loop_area: Any = patch_winner["area"]
        loop_distance: Any = patch_winner["distance_to_skeleton"]
        attachment_method = "patch_fallback"
    else:
        loop_vertex_indices = None if loop is None else loop.ordered_vertex_indices
        loop_edges = None if loop is None else loop.edges
        loop_centroid = None if loop is None else loop.center
        loop_perimeter = None if loop is None else loop.perimeter
        loop_area = None if loop is None else loop.area
        loop_distance = None if loop is None else loop.distance_to_skeleton
        attachment_method = "loop"

    attachment = {
        **record.identity,
        "source_mesh_path": str(record.source_path),
        "sealed_mesh_path": str(sealed_mesh_path(record, cfg)),
        "attachment_ambiguous": bool(review_reason is not None),
        "attachment_ambiguity_reason": review_reason,
        "attachment_method": attachment_method,
        "attachment_loop_vertex_indices": loop_vertex_indices,
        "attachment_loop_vertex_indices_sealed": loop_sealed_indices,
        "attachment_loop_edges": loop_edges,
        "attachment_loop_centroid_global": loop_centroid,
        "attachment_loop_perimeter": loop_perimeter,
        "attachment_loop_area": loop_area,
        "attachment_loop_distance_to_branch_skeleton": loop_distance,
        "attachment_cap_face_indices": cap_faces,
        "attachment_cap_area": cap_area,
        "candidate_loops": detection.candidates(),
        "candidate_patches": [
            {
                "perimeter": float(patch["perimeter"]),
                "area": float(patch["area"]),
                "distance_to_branch_skeleton": float(patch["distance_to_skeleton"]),
                "centroid_global": np.asarray(patch["center"], dtype=float).tolist(),
                "n_edges": int(patch["n_edges"]),
            }
            for patch in patch_candidates
        ],
        "physical_unit": cfg.physical_unit,
        "algorithm_version": ALGORITHM_VERSION,
    }

    def save() -> None:
        atomic_export_mesh(repaired, sealed_mesh_path(record, cfg))
        _write_json_atomic(attachment_json_path(record, cfg), attachment)

    timer.run("save", save)

    n_holes_before = len(boundary_loops(mesh))
    return {
        "status": STATUS_NEEDS_REVIEW if review_reason else STATUS_SUCCESS,
        "reason": review_reason,
        "sealed_path": str(sealed_mesh_path(record, cfg)),
        "attachment_region_path": str(attachment_json_path(record, cfg)),
        "attachment_ambiguous": bool(review_reason is not None),
        "attachment_method": attachment_method,
        "n_boundary_edges_before": n_boundary_before,
        "n_boundary_edges_after": report["n_boundary_edges"],
        "n_holes_before": n_holes_before,
        "n_holes_filled": n_holes_before - len(boundary_loops(repaired)),
        "attachment_loop_perimeter": attachment["attachment_loop_perimeter"],
        "attachment_loop_area": attachment["attachment_loop_area"],
        "attachment_cap_area": cap_area,
        "n_attachment_cap_faces": int(len(cap_faces)),
        **report,
    }


# ---------------------------------------------------------------------------
# Stage 2: false spine detection
# ---------------------------------------------------------------------------


def _false_spine_eligibility(record: SpineRecord, cfg: SpinePreprocessingConfig, context: Dict[str, Any]) -> Optional[str]:
    return None if sealed_mesh_path(record, cfg).exists() else "missing_sealed_mesh"


def _false_spine_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    if not record.branch_skeleton_path.exists():
        raise FileNotFoundError(f"Missing branch skeleton: {record.branch_skeleton_path}")
    sealed_mesh = timer.run("load sealed mesh", lambda: load_trimesh(sealed_mesh_path(record, cfg), process=False))
    spine_segments = timer.run("build spine skeleton", lambda: cgal_skeleton_segments_from_trimesh(sealed_mesh))
    dendrite_segments = load_skeleton_segments_cached(record.branch_skeleton_path)

    def profile() -> np.ndarray:
        spine_path = orient_spine_path_from_dendrite(longest_path_polyline(spine_segments), dendrite_segments)
        points = sample_polyline_by_arclength(spine_path, n_points=cfg.skeleton_sample_points)
        return np.asarray([distance_to_segments(point, dendrite_segments) for point in points], dtype=float)

    distances = timer.run("distance profile", profile)
    threshold = cfg.false_spine_threshold(record.dataset)
    return {
        "status": DETECT_VALID if float(np.max(distances)) > threshold else DETECT_INVALID,
        "_state": STATUS_SUCCESS,
        "d_min": float(np.min(distances)),
        "d_max": float(np.max(distances)),
        "d_mean": float(np.mean(distances)),
        "d_median": float(np.median(distances)),
        "distance_profile": distances.tolist(),
        "threshold_nm": threshold,
    }


# ---------------------------------------------------------------------------
# Stage 3: local coordinate frame
# ---------------------------------------------------------------------------


def _orient_eligibility(record: SpineRecord, cfg: SpinePreprocessingConfig, context: Dict[str, Any]) -> Optional[str]:
    if cfg.only_valid_for_orientation and _status_lookup(context, "false_spines.parquet").get(record.key) != DETECT_VALID:
        return "not_valid_after_false_spine_detection"
    attachment_path = attachment_json_path(record, cfg)
    if not sealed_mesh_path(record, cfg).exists() or not attachment_path.exists():
        return "missing_sealed_mesh"
    attachment = _read_json(attachment_path)
    if attachment.get("attachment_ambiguous") or not attachment.get("attachment_cap_face_indices"):
        return "attachment_ambiguous"
    return None


def _make_mesh(vertices: np.ndarray, faces: np.ndarray):
    trimesh = require_trimesh()
    return trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=False)


def _orient_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    def load() -> Tuple[Any, Any, Dict[str, Any], Optional[list]]:
        segments = load_skeleton_segments_cached(record.branch_skeleton_path) if record.branch_skeleton_path.exists() else None
        return (
            load_trimesh(sealed_mesh_path(record, cfg), process=False),
            load_trimesh(record.source_path, process=False),
            _read_json(attachment_json_path(record, cfg)),
            segments,
        )

    sealed, source, attachment, segments = timer.run("load meshes", load)
    frame = timer.run(
        "compute local frame",
        lambda: orient_spines_w_holes(
            sealed,
            attachment["attachment_cap_face_indices"],
            np.asarray(attachment["attachment_loop_centroid_global"], dtype=float),
            segments,
            min_tangent_component=cfg.min_tangent_component,
        ),
    )
    g2l = frame["global_to_local"]
    local_sealed = _make_mesh(apply_homogeneous(sealed.vertices, g2l), sealed.faces)
    local_source = _make_mesh(apply_homogeneous(source.vertices, g2l), source.faces)
    centroid_local = apply_homogeneous(np.asarray(attachment["attachment_loop_centroid_global"])[None, :], g2l)[0]

    transform = {
        **record.identity,
        "origin_global": frame["origin_global"],
        "tangent": frame["tangent"],
        "radial": frame["radial"],
        "binormal": frame["binormal"],
        "global_to_local": g2l,
        "local_to_global": frame["local_to_global"],
        "orientation_fallback": frame["orientation_fallback"],
        "orientation_fallback_reasons": frame["orientation_fallback_reasons"],
        "attachment_ambiguous": bool(attachment.get("attachment_ambiguous", False)),
        "source_mesh_path": str(record.source_path),
        "sealed_mesh_path": str(sealed_mesh_path(record, cfg)),
        "local_mesh_path": str(local_mesh_path(record, cfg)),
        "local_sealed_mesh_path": str(local_sealed_mesh_path(record, cfg)),
        "branch_projection": frame["branch_projection"],
        "attachment_normal_global": frame["attachment_normal_global"],
        "axes": {"x": "tangent", "y": "radial", "z": "binormal"},
        "physical_unit": cfg.physical_unit,
        "algorithm_version": ALGORITHM_VERSION,
    }

    def save() -> None:
        atomic_export_mesh(local_sealed, local_sealed_mesh_path(record, cfg))
        atomic_export_mesh(local_source, local_mesh_path(record, cfg))
        _write_json_atomic(transform_json_path(record, cfg), transform)
        _update_json_atomic(
            attachment_json_path(record, cfg),
            {
                "attachment_loop_centroid_local": centroid_local,
                "attachment_normal_local": frame["attachment_normal_local"],
            },
        )

    timer.run("save", save)
    return {
        "status": STATUS_SUCCESS,
        "local_mesh_path": str(local_mesh_path(record, cfg)),
        "local_sealed_mesh_path": str(local_sealed_mesh_path(record, cfg)),
        "transform_path": str(transform_json_path(record, cfg)),
        "origin_global": frame["origin_global"],
        "branch_projection": frame["branch_projection"],
        "tangent": frame["tangent"],
        "radial": frame["radial"],
        "binormal": frame["binormal"],
        "orientation_fallback": frame["orientation_fallback"],
        "orientation_fallback_reasons": json.dumps(frame["orientation_fallback_reasons"]),
        "attachment_loop_perimeter": attachment.get("attachment_loop_perimeter"),
        "attachment_loop_area": attachment.get("attachment_loop_area"),
        "attachment_cap_area": attachment.get("attachment_cap_area"),
    }


# ---------------------------------------------------------------------------
# Stage 5: geometric QC of local_sealed meshes
# ---------------------------------------------------------------------------


def _qc_eligibility(record: SpineRecord, cfg: SpinePreprocessingConfig, context: Dict[str, Any]) -> Optional[str]:
    if not local_sealed_mesh_path(record, cfg).exists() or not transform_json_path(record, cfg).exists():
        return "missing_local_sealed_mesh"
    return None


def _relative_difference(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-300)


def _face_edge_lengths(mesh: Any) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=float)
    return np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)


def _qc_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    def load() -> Tuple[Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
        return (
            load_trimesh(local_sealed_mesh_path(record, cfg), process=False),
            load_trimesh(sealed_mesh_path(record, cfg), process=False),
            load_trimesh(record.source_path, process=False),
            _read_json(transform_json_path(record, cfg)),
            _read_json(attachment_json_path(record, cfg)),
        )

    local, sealed, original, transform, attachment = timer.run("load meshes", load)
    report = timer.run(
        "basic checks",
        lambda: mesh_geometry_report(
            local,
            degenerate_area_tol=cfg.degenerate_area_tol,
            check_self_intersections=cfg.check_self_intersections,
        ),
    )

    def original_self_intersections() -> Dict[str, Any]:
        # On the raw, unsealed mesh (open boundary - not watertight, so most
        # of mesh_geometry_report()'s other checks don't apply). Some
        # self-intersections predate any of our processing (e.g. raw
        # segmentation artifacts); this lets that be told apart from ones
        # introduced by hole-filling. Purely diagnostic - does not affect
        # geometry_valid/quality_status, which are about local_sealed_mesh.
        has_si, n_si, method = self_intersection_check(original)
        return {
            "original_has_self_intersections": has_si,
            "original_n_self_intersecting_pairs": n_si,
            "original_self_intersection_method": method,
        }

    original_si = (
        timer.run("original mesh self-intersections", original_self_intersections)
        if cfg.check_self_intersections
        else {
            "original_has_self_intersections": None,
            "original_n_self_intersecting_pairs": None,
            "original_self_intersection_method": "skipped",
        }
    )

    def invariance() -> Dict[str, Any]:
        same_topology = len(local.vertices) == len(sealed.vertices) and len(local.faces) == len(sealed.faces)
        cap = np.asarray(attachment.get("attachment_cap_face_indices") or [], dtype=int)
        local_cap_area = float(np.asarray(local.area_faces)[cap].sum()) if len(cap) and same_topology else None
        sealed_volume = float(sealed.volume) if sealed.is_watertight else None
        l2g = np.asarray(transform["local_to_global"], dtype=float)
        g2l = np.asarray(transform["global_to_local"], dtype=float)
        result: Dict[str, Any] = {
            "invariance_same_topology": bool(same_topology),
            "invariance_edge_length_max_abs_diff": None,
            "invariance_surface_area_rel_diff": _relative_difference(local.area, sealed.area),
            "invariance_volume_rel_diff": _relative_difference(report["volume"], sealed_volume),
            "invariance_attachment_cap_area_rel_diff": _relative_difference(
                local_cap_area, attachment.get("attachment_cap_area")
            ),
            "transform_matrix_inverse_error": float(np.max(np.abs(l2g @ g2l - np.eye(4)))),
            "transform_roundtrip_error_max": None,
        }
        if same_topology:
            result["invariance_edge_length_max_abs_diff"] = float(
                np.max(np.abs(_face_edge_lengths(local) - _face_edge_lengths(sealed)))
            )
            reconstructed = apply_homogeneous(apply_homogeneous(sealed.vertices, g2l), l2g)
            restored = apply_homogeneous(local.vertices, l2g)
            result["transform_roundtrip_error_max"] = float(
                max(
                    np.max(np.abs(reconstructed - np.asarray(sealed.vertices))),
                    np.max(np.abs(restored - np.asarray(sealed.vertices))),
                )
            )
        return result

    inv = timer.run("transform invariance", invariance)

    def within(value: Optional[float], tolerance: float) -> bool:
        return value is not None and value <= tolerance

    edge_tolerance = cfg.transform_tolerance + cfg.invariance_rtol * float(np.max(_face_edge_lengths(sealed), initial=0.0))
    invariance_ok = bool(
        inv["invariance_same_topology"]
        and within(inv["invariance_edge_length_max_abs_diff"], edge_tolerance)
        and within(inv["invariance_surface_area_rel_diff"], cfg.invariance_rtol)
        and within(inv["invariance_volume_rel_diff"], cfg.invariance_rtol)
        and (
            attachment.get("attachment_cap_area") is None
            or within(inv["invariance_attachment_cap_area_rel_diff"], cfg.invariance_rtol)
        )
    )
    roundtrip_ok = within(inv["transform_roundtrip_error_max"], cfg.transform_tolerance)

    hard_checks = {
        "check_watertight": report["is_watertight"],
        "check_manifold": report["is_manifold"],
        "check_finite_coordinates": report["has_finite_coordinates"],
        "check_degenerate_faces": report["n_degenerate_faces"] <= cfg.max_degenerate_faces,
        "check_positive_volume": report["has_positive_finite_volume"],
        "check_roundtrip": roundtrip_ok,
        "check_invariance": invariance_ok,
    }
    geometry_valid = bool(all(hard_checks.values()) and report["n_connected_components"] == 1)

    review_flags = []
    if report["has_self_intersections"]:
        review_flags.append("self_intersections")
    if report["genus"] != 0:
        review_flags.append("genus_nonzero")
    if not report["has_consistent_winding"]:
        review_flags.append("inconsistent_winding")
    if report["n_connected_components"] != 1:
        review_flags.append("multiple_connected_components")

    # Multiple components violate geometry_valid but, per the spec, lead to review
    # rather than hard exclusion; every other failed hard check means invalid.
    if not all(hard_checks.values()):
        quality_status = QUALITY_INVALID
    elif review_flags:
        quality_status = STATUS_NEEDS_REVIEW
    else:
        quality_status = QUALITY_VALID

    quality = {
        **record.identity,
        "local_sealed_mesh_path": str(local_sealed_mesh_path(record, cfg)),
        **report,
        **original_si,
        **inv,
        **hard_checks,
        "transform_tolerance": cfg.transform_tolerance,
        "geometry_valid": geometry_valid,
        "review_flags": review_flags,
        "quality_status": quality_status,
        "orientation_fallback": transform.get("orientation_fallback"),
        "algorithm_version": ALGORITHM_VERSION,
    }
    timer.run("save", lambda: _update_json_atomic(quality_json_path(record, cfg), quality))
    return {
        "status": quality_status,
        "_state": STATUS_NEEDS_REVIEW if quality_status == STATUS_NEEDS_REVIEW else STATUS_SUCCESS,
        "reason": ",".join(review_flags) or None,
        "quality_path": str(quality_json_path(record, cfg)),
        **{k: v for k, v in quality.items() if k not in record.identity and k != "review_flags"},
        "review_flags": ",".join(review_flags),
    }


# ---------------------------------------------------------------------------
# Stage 6: point clouds
# ---------------------------------------------------------------------------


def _cap_face_mask(record: SpineRecord, cfg: SpinePreprocessingConfig, n_faces: int) -> np.ndarray:
    mask = np.zeros(n_faces, dtype=bool)
    attachment = _read_json(attachment_json_path(record, cfg))
    cap = np.asarray(attachment.get("attachment_cap_face_indices") or [], dtype=int)
    mask[cap] = True
    return mask


def _pointcloud_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    mesh = timer.run("load mesh", lambda: load_trimesh(local_sealed_mesh_path(record, cfg), process=False))
    cap_mask = _cap_face_mask(record, cfg, len(mesh.faces))
    seeds: Dict[str, int] = {}

    def sample_all() -> None:
        for n_points in cfg.pointcloud_sizes:
            for variant, base_seed in enumerate(cfg.pointcloud_seeds, start=1):
                seed = stable_seed(cfg.preprocessing_version, *record.key, n_points, base_seed)
                points, normals, faces = sample_surface_area_weighted(mesh, n_points, np.random.default_rng(seed))
                _write_npz_atomic(
                    pointcloud_path(record, cfg, n_points, variant),
                    points=points.astype(np.float32),
                    normals=normals.astype(np.float32),
                    face_indices=faces.astype(np.int32),
                    is_attachment_cap=cap_mask[faces],
                    seed=np.int64(seed % (2**63)),
                    sampling_seed=np.int64(base_seed),
                )
                seeds[f"{n_points}_{variant}"] = seed

    timer.run("sample and save", sample_all)
    row: Dict[str, Any] = {"status": STATUS_SUCCESS, "n_variants": len(cfg.pointcloud_seeds)}
    for n_points in cfg.pointcloud_sizes:
        row[f"pointcloud_{n_points}_path"] = str(pointcloud_path(record, cfg, n_points, 1))
    row["seeds"] = json.dumps({key: str(value) for key, value in seeds.items()})
    return row


def _pointcloud_output(record: SpineRecord, cfg: SpinePreprocessingConfig) -> Path:
    return pointcloud_path(record, cfg, cfg.pointcloud_sizes[-1], len(cfg.pointcloud_seeds))


# ---------------------------------------------------------------------------
# Stage 7: SDF samples
# ---------------------------------------------------------------------------


def _sdf_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    mesh = timer.run("load mesh", lambda: load_trimesh(local_sealed_mesh_path(record, cfg), process=False))
    cap_mask = _cap_face_mask(record, cfg, len(mesh.faces))
    fractions = (cfg.sdf_surface_fraction, cfg.sdf_near_fraction, cfg.sdf_uniform_fraction)
    seed = stable_seed(cfg.preprocessing_version, *record.key, "sdf")

    samples = timer.run(
        "sample and compute SDF",
        lambda: build_sdf_samples(
            mesh,
            cap_mask,
            pool_size=cfg.sdf_pool_size,
            fractions=fractions,
            near_sigmas=cfg.sdf_near_sigmas,
            bbox_margin=cfg.sdf_bbox_margin,
            rng=np.random.default_rng(seed),
        ),
    )
    qc = timer.run(
        "SDF QC",
        lambda: sdf_quality(
            mesh,
            samples,
            expected_counts=split_pool(cfg.sdf_pool_size, fractions),
            surface_tolerance=cfg.sdf_surface_tolerance,
            interior_probe=cfg.sdf_interior_probe,
            bbox_margin=cfg.sdf_bbox_margin,
            rng=np.random.default_rng(seed + 1),
        ),
    )

    def save() -> None:
        _write_npz_atomic(
            sdf_samples_path(record, cfg),
            **samples,
            sample_type_codes=np.asarray(json.dumps(SAMPLE_TYPE_CODES)),
            sign_convention=np.asarray("negative_inside"),
            seed=np.int64(seed % (2**63)),
        )
        _update_json_atomic(quality_json_path(record, cfg), qc)

    timer.run("save", save)
    return {
        "status": STATUS_SUCCESS if qc["sdf_valid"] else QUALITY_INVALID,
        "_state": STATUS_SUCCESS,
        "sdf_samples_path": str(sdf_samples_path(record, cfg)),
        **qc,
    }


# ---------------------------------------------------------------------------
# Stage 8: morphometrics (src.spine_analysis.shape_metric, on local_sealed_mesh)
# ---------------------------------------------------------------------------


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
    return result if math.isfinite(result) else None


def _morphometrics_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    from src.spine_analysis.mesh.utils import v_f_to_mesh_isolated
    from src.spine_analysis.shape_metric.float_metric import (
        ConvexHullRatioSpineMetric,
        ConvexHullVolumeSpineMetric,
        VolumeSpineMetric,
    )
    from src.spine_analysis.shape_metric.histogram_metric import OldChordDistributionSpineMetric
    from src.spine_analysis.shape_metric.junction_metric import (
        AverageDistanceSpineMetric,
        CVDSpineMetric,
        LengthAreaRatioSpineMetric,
        LengthSpineMetric,
        LengthVolumeRatioSpineMetric,
        OpenAngleSpineMetric,
    )
    from src.spine_analysis.shape_metric.utils import register_attachment_center

    def load() -> Tuple[Any, Dict[str, Any]]:
        mesh = load_trimesh(local_sealed_mesh_path(record, cfg), process=False)
        attachment = _read_json(attachment_json_path(record, cfg))
        return mesh, attachment

    mesh, attachment = timer.run("load mesh", load)

    def compute_scalar_metrics() -> Dict[str, Any]:
        poly = v_f_to_mesh_isolated(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int))
        # orient_spines_w_holes() centers the local frame on the attachment
        # loop centroid, so that point is the origin here - the junction/
        # attachment-center helpers in src.spine_analysis need it registered
        # per mesh object (keyed by id()), not the original global point.
        #
        # This point is all these particular metrics actually use of the
        # attachment region: OpenAngle/CVD/AverageDistance/Length(Ratio)
        # only measure distances from `attachment_center` to every vertex
        # (src/spine_analysis/shape_metric/junction_metric.py -
        # JunctionSpineMetric._calculate uses _junction_center, i.e. exactly
        # this registered point, for _surface_vectors; _calculate_junction_center
        # returns the registered point directly rather than recomputing it).
        # None of them touch `_junction_triangles` (the library's own,
        # cruder "nearest face + 1-ring" attachment-region guess) - only
        # AreaSpineMetric does, which is why "Area" is computed separately
        # below from our own stage-1 attachment_cap_area instead of calling
        # AreaSpineMetric: on real spines the library's own 1-ring re-guess
        # of the attachment patch differed from our precise
        # attachment_cap_face_indices by 40-130% in area.
        register_attachment_center(poly, np.zeros(3, dtype=float))
        return {
            "OpenAngle": _metric_value(OpenAngleSpineMetric(poly)),
            "CVD": _metric_value(CVDSpineMetric(poly)),
            "AverageDistance": _metric_value(AverageDistanceSpineMetric(poly)),
            "LengthVolumeRatio": _metric_value(LengthVolumeRatioSpineMetric(poly)),
            "LengthAreaRatio": _metric_value(LengthAreaRatioSpineMetric(poly)),
            "Length": _metric_value(LengthSpineMetric(poly)),
            "Volume": _metric_value(VolumeSpineMetric(poly)),
            "ConvexHullVolume": _metric_value(ConvexHullVolumeSpineMetric(poly)),
            "ConvexHullRatio": _metric_value(ConvexHullRatioSpineMetric(poly)),
        }

    def compute_chord_distribution() -> Any:
        # Separate step/timing: OldChordDistributionSpineMetric sptrays 3000
        # random rays per mesh (its own fresh, unseeded random.Random()) and
        # is by far the slowest of these metrics.
        poly = v_f_to_mesh_isolated(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int))
        register_attachment_center(poly, np.zeros(3, dtype=float))
        return _metric_value(OldChordDistributionSpineMetric(poly))

    values = timer.run("compute scalar metrics", compute_scalar_metrics)
    # NOTE: OldChordDistributionSpineMetric samples its chords with an
    # unseeded `random.Random()` inside src/spine_analysis (not modified per
    # project convention) - this value is therefore not reproducible between
    # runs, even for the same mesh. Stored as a best-effort snapshot; see
    # docs/neuron-model/s-module-preprocessing.md 3.8.
    values["OldChordDistribution"] = timer.run("compute OldChordDistribution", compute_chord_distribution)
    values["JunctionArea"] = _safe_float(attachment.get("attachment_loop_area"))
    # "Area" = total surface minus the attachment region - computed from our
    # own stage-1 attachment_cap_face_indices/attachment_cap_area (exact),
    # not AreaSpineMetric's own re-derived attachment-region guess (see the
    # comment above compute_scalar_metrics). Area is rotation/translation
    # invariant, so the cap area stored in attachment_region_<mesh>.json
    # (computed on the global sealed mesh in stage 1) applies unchanged here.
    values["Area"] = _safe_float(mesh.area - attachment.get("attachment_cap_area", 0.0))

    payload = {
        **record.identity,
        "local_sealed_mesh_path": str(local_sealed_mesh_path(record, cfg)),
        **values,
        "algorithm_version": ALGORITHM_VERSION,
    }
    timer.run("save", lambda: _write_json_atomic(morphometrics_json_path(record, cfg), payload))

    row = dict(values)
    row["status"] = STATUS_SUCCESS
    row["morphometrics_path"] = str(morphometrics_json_path(record, cfg))
    return row


# ---------------------------------------------------------------------------
# Stage 9: metadata (species/health/cell_type/cell_type_binary/compartment)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _load_metadata_tables_cached(metadata_root: str) -> Dict[str, Any]:
    from .metadata import load_metadata_tables

    return load_metadata_tables(Path(metadata_root))


def _metadata_worker(record: SpineRecord, cfg: SpinePreprocessingConfig, timer: StepTimer) -> Dict[str, Any]:
    from .metadata import dataset_metadata, merge_metadata_status, merged_branch_id

    metadata_tables = timer.run(
        "load metadata tables", lambda: _load_metadata_tables_cached(str(cfg.metadata_root))
    )

    def compute() -> Dict[str, Any]:
        merged_id = merged_branch_id(record, metadata_tables)
        return {
            **record.identity,
            "original_branch_id": record.branch_id,
            "merged_branch_id": merged_id,
            **dataset_metadata(record, metadata_tables),
            "merge_metadata_status": merge_metadata_status(record, metadata_tables),
            "source_path": str(record.source_path),
            "physical_unit": cfg.physical_unit,
            "preprocessing_version": cfg.preprocessing_version,
            "algorithm_version": ALGORITHM_VERSION,
        }

    payload = timer.run("compute", compute)
    timer.run("save", lambda: _write_json_atomic(metadata_json_path(record, cfg), payload))

    row = dict(payload)
    row["status"] = STATUS_SUCCESS
    row["metadata_path"] = str(metadata_json_path(record, cfg))
    return row


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------


STAGES: Dict[str, StageSpec] = {
    STAGE_SEAL: StageSpec(
        STAGE_SEAL, "seal", "sealed_spines.parquet", _seal_worker, 7,
        lambda r, c: r.source_path, sealed_mesh_path,
    ),
    STAGE_FALSE_SPINE: StageSpec(
        STAGE_FALSE_SPINE, "false-spine", "false_spines.parquet", _false_spine_worker, 3,
        sealed_mesh_path, sealed_mesh_path, _false_spine_eligibility,
    ),
    STAGE_ORIENT: StageSpec(
        STAGE_ORIENT, "orient", "oriented_spines.parquet", _orient_worker, 3,
        sealed_mesh_path, local_sealed_mesh_path, _orient_eligibility,
    ),
    STAGE_QC: StageSpec(
        STAGE_QC, "qc", "mesh_quality.parquet", _qc_worker, 5,
        local_sealed_mesh_path, quality_json_path, _qc_eligibility,
    ),
    STAGE_POINTCLOUD: StageSpec(
        STAGE_POINTCLOUD, "pointcloud", "pointclouds.parquet", _pointcloud_worker, 2,
        local_sealed_mesh_path, _pointcloud_output, _training_eligibility,
    ),
    STAGE_SDF: StageSpec(
        STAGE_SDF, "sdf", "sdf_samples.parquet", _sdf_worker, 4,
        local_sealed_mesh_path, sdf_samples_path, _training_eligibility,
    ),
    STAGE_MORPHOMETRICS: StageSpec(
        STAGE_MORPHOMETRICS, "morphometrics", "morphometrics.parquet", _morphometrics_worker, 3,
        local_sealed_mesh_path, morphometrics_json_path, _training_eligibility,
    ),
    STAGE_METADATA: StageSpec(
        STAGE_METADATA, "metadata", "metadata.parquet", _metadata_worker, 2,
        lambda r, c: r.source_path, metadata_json_path,
    ),
}


def run_sealing_stage(config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None) -> pd.DataFrame:
    return run_stage(STAGE_SEAL, config, records)


def run_false_spine_detection_stage(
    config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None
) -> pd.DataFrame:
    return run_stage(STAGE_FALSE_SPINE, config, records)


def run_orientation_stage(config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None) -> pd.DataFrame:
    return run_stage(STAGE_ORIENT, config, records)


def run_mesh_qc_stage(config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None) -> pd.DataFrame:
    return run_stage(STAGE_QC, config, records)


def run_pointcloud_stage(config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None) -> pd.DataFrame:
    return run_stage(STAGE_POINTCLOUD, config, records)


def run_sdf_stage(config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None) -> pd.DataFrame:
    return run_stage(STAGE_SDF, config, records)


def run_morphometrics_stage(
    config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None
) -> pd.DataFrame:
    return run_stage(STAGE_MORPHOMETRICS, config, records)


def run_metadata_stage(config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None) -> pd.DataFrame:
    return run_stage(STAGE_METADATA, config, records)


# ---------------------------------------------------------------------------
# Stage 4 (logical branch merge) + metadata.json + manifest.parquet
# ---------------------------------------------------------------------------


def run_manifest_stage(config: SpinePreprocessingConfig, records: Optional[Sequence[SpineRecord]] = None) -> pd.DataFrame:
    """Aggregate every other stage's results (incl. metadata, morphometrics) into manifest.parquet.

    Stages 8 (morphometrics) and 9 (metadata) run independently, same as
    1-7; this only reads their output tables, it does not recompute
    anything itself (unlike earlier versions of this function).
    """
    cfg = config.normalized()
    records = list(discover_spines(cfg) if records is None else records)
    full_hash = config_hash(cfg)
    all_rows: List[Dict[str, Any]] = []

    groups: Dict[Path, List[SpineRecord]] = {}
    for record in records:
        groups.setdefault(preprocessed_root(record, cfg), []).append(record)

    for root, group in groups.items():
        tables = {
            name: _read_table_rows(root / table)
            for name, table in (
                ("seal", "sealed_spines.parquet"),
                ("false_spine", "false_spines.parquet"),
                ("orient", "oriented_spines.parquet"),
                ("quality", "mesh_quality.parquet"),
                ("pointcloud", "pointclouds.parquet"),
                ("sdf", "sdf_samples.parquet"),
                ("morphometrics", "morphometrics.parquet"),
                ("metadata", "metadata.parquet"),
            )
        }
        manifest_path = root / "manifest.parquet"
        rows_by_key = _read_table_rows(manifest_path)

        for record in group:
            def get(table: str, column: str) -> Any:
                return tables[table].get(record.key, {}).get(column)

            quality_status = get("quality", "status")
            allowed_quality = {QUALITY_VALID, STATUS_NEEDS_REVIEW} if cfg.allow_needs_review else {QUALITY_VALID}
            existing_path = lambda path: str(path) if path.exists() else None  # noqa: E731
            row = {
                "dataset": record.dataset,
                "neuron_id": record.neuron_id,
                "limb_id": record.limb_id,
                "branch_id": record.branch_id,
                "original_branch_id": record.branch_id,
                "merged_branch_id": get("metadata", "merged_branch_id"),
                "spine_id": record.spine_id,
                "status": get("false_spine", "status"),
                "seal_status": get("seal", "status"),
                "attachment_ambiguous": get("seal", "attachment_ambiguous"),
                "attachment_method": get("seal", "attachment_method"),
                "orient_status": get("orient", "status"),
                "quality_status": quality_status,
                "geometry_valid": get("quality", "geometry_valid"),
                "sdf_valid": get("sdf", "sdf_valid"),
                "morphometrics_status": get("morphometrics", "status"),
                "metadata_status": get("metadata", "status"),
                "merge_metadata_status": get("metadata", "merge_metadata_status"),
                "train_eligible": bool(
                    get("false_spine", "status") == DETECT_VALID
                    and quality_status in allowed_quality
                    and get("pointcloud", "status") == STATUS_SUCCESS
                    and get("sdf", "sdf_valid") is True
                ),
                "species": get("metadata", "species"),
                "health": get("metadata", "health"),
                "cell_type": get("metadata", "cell_type"),
                "cell_type_binary": get("metadata", "cell_type_binary"),
                "compartment": get("metadata", "compartment"),
                "source_path": str(record.source_path),
                "local_mesh_path": existing_path(local_mesh_path(record, cfg)),
                "sealed_mesh_path": existing_path(sealed_mesh_path(record, cfg)),
                "local_sealed_mesh_path": existing_path(local_sealed_mesh_path(record, cfg)),
                "attachment_region_path": existing_path(attachment_json_path(record, cfg)),
                "transform_path": existing_path(transform_json_path(record, cfg)),
                **{
                    f"pointcloud_{n}_path": existing_path(pointcloud_path(record, cfg, n, 1))
                    for n in cfg.pointcloud_sizes
                },
                "pointcloud_n_variants": len(cfg.pointcloud_seeds),
                "sdf_samples_path": existing_path(sdf_samples_path(record, cfg)),
                "metadata_path": existing_path(metadata_json_path(record, cfg)),
                "morphometrics_path": existing_path(morphometrics_json_path(record, cfg)),
                "quality_path": existing_path(quality_json_path(record, cfg)),
                "preprocessing_version": cfg.preprocessing_version,
                "config_hash": full_hash,
            }
            rows_by_key[record.key] = row
            all_rows.append(row)

        _write_table(
            list(rows_by_key.values()),
            manifest_path,
            artifact_name="manifest",
            config_hash=full_hash,
            input_paths={"dataset_root": group[0].dataset_root, "metadata_root": cfg.metadata_root},
        )
        _print_summary("manifest", group[0].dataset_root, [rows_by_key[record.key] for record in group])

    return pd.DataFrame(all_rows)


PIPELINE_ORDER = (
    STAGE_SEAL,
    STAGE_FALSE_SPINE,
    STAGE_ORIENT,
    STAGE_QC,
    STAGE_POINTCLOUD,
    STAGE_SDF,
    STAGE_MORPHOMETRICS,
    STAGE_METADATA,
)


def run_full_spine_preprocessing_pipeline(config: SpinePreprocessingConfig) -> Dict[str, pd.DataFrame]:
    cfg = config.normalized()
    records = discover_spines(cfg)
    results = {stage: run_stage(stage, cfg, records) for stage in PIPELINE_ORDER}
    results[STAGE_MANIFEST] = run_manifest_stage(cfg, records)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CLI_STAGES = {spec.cli_name: name for name, spec in STAGES.items()}
CLI_STAGES["manifest"] = STAGE_MANIFEST


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run neuron-model spine preprocessing stages.")
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None, help=f"YAML config (e.g. {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--output-root", type=Path, help="Default: <dataset_root>/preprocessed")
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--stage", choices=[*CLI_STAGES, "all"], default="all")
    parser.add_argument("--dataset")
    parser.add_argument("--neuron-id")
    parser.add_argument("--limb-id")
    parser.add_argument("--branch-id")
    parser.add_argument("--spine-id")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=None)
    parser.add_argument("--retry-failed", dest="retry_failed_only", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--false-spine-threshold-nm", type=float)
    parser.add_argument("--allow-needs-review", action="store_true", default=None)
    parser.add_argument(
        "--include-invalid-for-orientation", dest="only_valid_for_orientation", action="store_false", default=None
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = vars(_build_arg_parser().parse_args(argv))
    stage = args.pop("stage")
    config_path = args.pop("config")
    overrides = {key: value for key, value in args.items() if value is not None}
    if config_path is not None:
        cfg = SpinePreprocessingConfig.from_yaml(config_path, **overrides)
    else:
        cfg = SpinePreprocessingConfig(**overrides).normalized()

    if stage == "all":
        run_full_spine_preprocessing_pipeline(cfg)
    elif CLI_STAGES[stage] == STAGE_MANIFEST:
        run_manifest_stage(cfg)
    else:
        run_stage(CLI_STAGES[stage], cfg)


if __name__ == "__main__":
    main()
