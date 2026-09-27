from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .spine_preprocessing import SpineRecord
from .statuses import (
    MERGE_STATUS_MISSING_MAPPING,
    MERGE_STATUS_NOT_APPLICABLE,
    MERGE_STATUS_SUCCESS,
)

EXCITATORY_CELL_TYPES = {
    "23P",
    "4P",
    "5P-IT",
    "5P-NP",
    "5P-PT",
    "6P-CT",
    "6P-IT",
    "6P-U",
    "WM-P",
    "Unsure E",
}
INHIBITORY_CELL_TYPES = {"BC", "BPC", "MC", "NGC", "Unsure I"}

PREFIX_RE = re.compile(r"^(?P<neuron>.+)_limb_(?P<limb>\d+)_branch_(?P<branch>\d+)$")


def load_metadata_tables(metadata_root: Path) -> Dict[str, Any]:
    minnie_root = Path(metadata_root) / "minnie65"
    h01_root = Path(metadata_root) / "h01"
    metadata: Dict[str, Any] = {
        "merge_mapping": {},
        "cell_types": {},
        "compartments": {},
    }

    mapping_paths = {
        "minnie65": minnie_root / "merged_branches" / "single_child_merge_mapping.parquet",
        "minnie": minnie_root / "merged_branches" / "single_child_merge_mapping.parquet",
        "h01": h01_root / "merged_branches" / "h01_single_child_merge_mapping.parquet",
    }
    metadata["merge_lookup"] = {}
    for dataset, path in mapping_paths.items():
        if path.exists():
            mapping = normalize_merge_mapping(pd.read_parquet(path))
            metadata["merge_mapping"][dataset] = mapping
            metadata["merge_lookup"][dataset] = {
                (str(row.neuron_id), int(row.limb_id), int(row.original_branch_id)): int(row.merged_branch_id)
                for row in mapping.itertuples()
            }

    cell_path = minnie_root / "cell_types_with_nucleus.csv"
    if cell_path.exists():
        cell = pd.read_csv(cell_path)
        metadata["cell_types"]["minnie65"] = {
            str(row.segment_id): str(row.cell_type)
            for row in cell.itertuples()
            if pd.notna(row.segment_id) and pd.notna(row.cell_type)
        }
        metadata["cell_types"]["minnie"] = metadata["cell_types"]["minnie65"]

    compartment_path = minnie_root / "minnie65_limb_compartments_length_rule.parquet"
    if compartment_path.exists():
        compartments = pd.read_parquet(compartment_path)
        metadata["compartments"]["minnie65"] = compartment_lookup(compartments)
        metadata["compartments"]["minnie"] = metadata["compartments"]["minnie65"]

    return metadata


def normalize_merge_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["prefix"].astype(str).map(parse_branch_prefix)
    result = pd.DataFrame(parsed.tolist())
    result["merged_branch_id"] = frame["merged_branch_id"].astype(int)
    result["merged_member_count"] = frame["merged_member_count"].astype(int)
    result["was_merged"] = frame["was_merged"].astype(bool)
    if "merged_parent_branch_id" in frame.columns:
        result["merged_parent_branch_id"] = frame["merged_parent_branch_id"].astype(int)
    return result


def parse_branch_prefix(prefix: str) -> Dict[str, Any]:
    match = PREFIX_RE.match(prefix)
    if match is None:
        raise ValueError(f"Cannot parse branch prefix: {prefix}")
    return {
        "neuron_id": match.group("neuron"),
        "limb_id": int(match.group("limb")),
        "original_branch_id": int(match.group("branch")),
    }


def compartment_lookup(frame: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    result: Dict[Tuple[str, int], str] = {}
    for row in frame.itertuples():
        result[(str(row.neuron), int(row.limb_id))] = str(row.compartment)
    return result


def identity_columns(record: SpineRecord, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": record.dataset,
        "neuron_id": record.neuron_id,
        "limb_id": id_to_int(record.limb_id),
        "original_branch_id": id_to_int(record.branch_id),
        "merged_branch_id": merged_branch_id(record, metadata),
        "spine_id": id_to_int(record.spine_id),
    }


def dataset_metadata(record: SpineRecord, metadata: Dict[str, Any]) -> Dict[str, Any]:
    kind = dataset_kind(record.dataset)
    ctype = cell_type(record, metadata)
    return {
        "species": species(kind),
        "health": health(kind),
        "cell_type": ctype,
        "cell_type_binary": cell_type_binary(ctype),
        "compartment": compartment(record, metadata),
    }


def species(kind: str) -> str:
    if kind == "h01":
        return "human"
    if kind in {"minnie", "minnie65", "labid"}:
        return "mouse"
    return "unknown"


def health(kind: str) -> Optional[str]:
    if kind in {"h01", "minnie", "minnie65"}:
        return "healthy"
    if kind == "labid":
        return None
    return "unknown"


def cell_type(record: SpineRecord, metadata: Dict[str, Any]) -> Optional[str]:
    lookup = metadata.get("cell_types", {}).get(dataset_kind(record.dataset), {})
    return lookup.get(str(record.neuron_id))


def cell_type_binary(value: Optional[str]) -> Optional[str]:
    if value in EXCITATORY_CELL_TYPES:
        return "excitatory"
    if value in INHIBITORY_CELL_TYPES:
        return "inhibitory"
    return None


def compartment(record: SpineRecord, metadata: Dict[str, Any]) -> Optional[str]:
    lookup = metadata.get("compartments", {}).get(dataset_kind(record.dataset), {})
    return lookup.get((str(record.neuron_id), id_to_int(record.limb_id)))


def merged_branch_id(record: SpineRecord, metadata: Dict[str, Any]) -> Optional[int]:
    kind = dataset_kind(record.dataset)
    lookup = metadata.get("merge_lookup", {}).get(kind)
    if lookup is not None:
        key = (str(record.neuron_id), id_to_int(record.limb_id), id_to_int(record.branch_id))
        return lookup.get(key)
    mapping = metadata.get("merge_mapping", {}).get(kind)
    if mapping is None:
        return id_to_int(record.branch_id)
    local = mapping[
        (mapping.neuron_id.astype(str) == str(record.neuron_id))
        & (mapping.limb_id.astype(int) == id_to_int(record.limb_id))
        & (mapping.original_branch_id.astype(int) == id_to_int(record.branch_id))
    ]
    if len(local) == 0:
        return None
    return int(local.merged_branch_id.iloc[0])


def merge_metadata_status(record: SpineRecord, metadata: Dict[str, Any]) -> str:
    kind = dataset_kind(record.dataset)
    if kind == "labid":
        return MERGE_STATUS_NOT_APPLICABLE
    if merged_branch_id(record, metadata) is None:
        return MERGE_STATUS_MISSING_MAPPING
    missing = []
    if kind in {"minnie", "minnie65"}:
        if cell_type(record, metadata) is None:
            missing.append("cell_type")
        if compartment(record, metadata) is None:
            missing.append("compartment")
    if kind == "h01" and cell_type(record, metadata) is None:
        missing.append("cell_type")
    return MERGE_STATUS_SUCCESS if not missing else "partial_missing_" + "_".join(missing)


def dataset_kind(dataset: str) -> str:
    value = str(dataset).lower()
    if "h01" in value:
        return "h01"
    if "minnie" in value or "microns" in value:
        return "minnie65"
    if "lab" in value:
        return "labid"
    return value


def id_to_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    match = re.search(r"(\d+)$", text)
    if match is None:
        return None
    return int(match.group(1))

