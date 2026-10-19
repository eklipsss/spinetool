"""Configuration of the branch preprocessing (docs/neuron-model/tp-branch-preprocessing.md).

All thresholds live in ``data/processed/config/branch_preprocessing.yaml``; the values below are
the defaults from the specification. Lengths in nanometres.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ALGORITHM_VERSION = "branch-preprocessing-v1"

# eligibility thresholds act only on quantities that are stored in branches.parquet: changing them does NOT redo the
# geometry (``reapply_eligibility`` recomputes the flags from the tables), so they stay out of the algorithm hash
_ELIGIBILITY_FIELDS = {
    "length_min_t_nm", "length_max_t_nm", "invalid_section_fraction_max", "radius_min_nm", "radius_max_nm", "trim_max_nm",
    "unknown_fraction_max", "position_fraction_min", "spline_rms_factor", "spline_rms_floor_nm", "long_gap_nm", "twist_tolerance_per_nm",
}
# fields that change the result (enter the config hash); paths / parallelism do not
_RUNTIME_FIELDS = {
    "raw_data_root", "metadata_root", "s_preprocessed_root", "output_root", "splits",
    "neuron_ids", "limit_neurons", "workers", "io_threads", "force", "merged_branch_ids", "limit_branches", "only_s_neurons",
}


@dataclass
class BranchPreprocessingConfig:
    dataset: str = "minnie65"
    raw_data_root: Path = Path("datasets")
    metadata_root: Path = Path("datasets")
    s_preprocessed_root: Optional[Path] = None  # default <raw_data_root>/<dataset>/preprocessed
    output_root: Optional[Path] = None  # default <raw_data_root>/<dataset>/preprocessed_branches
    splits: Optional[Path] = None  # splits_*.parquet; the split of a branch is the split of its neuron
    neuron_ids: Optional[List[str]] = None
    limit_neurons: Optional[int] = None
    only_s_neurons: bool = False  # only neurons present in the S manifest (the S preprocessing may not have reached the others)
    merged_branch_ids: Optional[List[int]] = None  # debugging / small checks: only these merged branches
    limit_branches: Optional[int] = None  # debugging / small checks: first N merged branches of each neuron
    workers: int = 1
    io_threads: int = 8
    force: bool = False

    # B1 shaft-only surface
    vertex_decimals: int = 3
    # B2 centreline (initial curve, sections, QC, spline)
    initial_step_nm: float = 500.0
    tangent_half_window_nm: float = 1500.0
    slab_half_thickness_nm: float = 150.0
    max_section_radius_nm: float = 1200.0
    adaptive_max_radius_factor: float = 4.5  # second pass with max(1200, 4.5 r_med); 0 disables
    cluster_eps_nm: float = 150.0
    robust_k: float = 0.35
    min_section_points: int = 10
    large_radius_factor: float = 1.5
    kink_factor: float = 0.5
    end_distance_factor: float = 2.0
    outside_mesh_check: bool = True
    centerline_smoothing_nm: float = 1000.0  # equivalent kernel width of the centreline smoothing spline
    radius_smoothing_nm: float = 1000.0
    spline_rms_factor: float = 0.75  # QC: RMS distance of the smoothed line to the reliable centres <= max(factor * r_med, floor)
    spline_rms_floor_nm: float = 0.0  # absolute floor of that limit (thin branches: the centre noise does not scale with r)
    min_reliable_sections: int = 5
    long_gap_nm: float = 3000.0
    # grid, frame
    node_step_nm: float = 500.0
    straight_fraction: float = 0.02
    twist_tolerance_per_nm: float = 1e-5
    # member joining / soma
    join_distance_nm: float = 500.0
    soma_mismatch_fraction: float = 0.2
    # spines
    end_clamp_factor: float = 0.5
    position_outlier_factor: float = 0.75
    # eligibility
    length_min_t_nm: float = 5000.0
    length_max_t_nm: float = 255500.0
    length_min_p_nm: float = 2000.0
    invalid_section_fraction_max: float = 0.2
    radius_min_nm: float = 80.0
    radius_max_nm: float = 1500.0
    trim_max_nm: float = 2000.0
    unknown_fraction_max: float = 0.1
    position_fraction_min: float = 0.9

    algorithm_version: str = ALGORITHM_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def dataset_root(self) -> Path:
        return Path(self.raw_data_root) / self.dataset

    @property
    def s_root(self) -> Path:
        return Path(self.s_preprocessed_root) if self.s_preprocessed_root else self.dataset_root / "preprocessed"

    @property
    def out_root(self) -> Path:
        return Path(self.output_root) if self.output_root else self.dataset_root / "preprocessed_branches"

    def config_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k not in _RUNTIME_FIELDS and k not in _ELIGIBILITY_FIELDS and k != "extra"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BranchPreprocessingConfig":
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in values.items():
            (kwargs if key in known else extra)[key] = value
        for key in ("raw_data_root", "metadata_root", "s_preprocessed_root", "output_root", "splits"):
            if kwargs.get(key) is not None:
                kwargs[key] = Path(kwargs[key])
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg


def load_config(path: Optional[Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> BranchPreprocessingConfig:
    values: Dict[str, Any] = {}
    if path is not None:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for section in raw.values() if all(isinstance(v, dict) for v in raw.values()) else [raw]:
            values.update(section)
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return BranchPreprocessingConfig.from_mapping(values)
