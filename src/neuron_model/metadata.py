from __future__ import annotations

import re
import warnings
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

# LabID: group folder -> (health, treatment, imaging series). branch ids: labid_unit_id
LABID_GROUPS = {
    "Wt": ("healthy", None, "wt"),
    "Wt_old": ("healthy", None, "wt_old"),
    "Wt+9009": ("healthy", "9009", "wt+9009"),
    "Ab": ("disease", None, "ab"),
    "Ab+9009": ("disease", "9009", "ab+9009"),
}
_LABID_NAME = re.compile(r"^(?:ab)?(\d+)(?:-(\d+))?", re.IGNORECASE)


def labid_unit_id(group: str, dendrite: str) -> str:
    """Id of a LabID branch = ``<group>_<folder name>`` (sanitised, ``_processed`` dropped). Every folder of a group is
    a SEPARATE branch (user decision 2026-10-24: whether two folders come from one dendrite / cell is unknown); it is
    also the grouping unit of the split (``neuron_id``). The numbers in the name are kept only as metadata
    (``labid_name_numbers``)."""
    name = str(dendrite).replace("_processed", "")
    return re.sub(r"[^A-Za-z0-9+]+", "_", f"{group}_{name}").strip("_")


def labid_name_numbers(dendrite: str) -> Tuple[Optional[int], Optional[int]]:
    """The two numbers of a folder name like ``ab23-1_processed`` / ``1003-22`` (meaning unknown, metadata only)."""
    m = _LABID_NAME.match(str(dendrite).replace("_processed", ""))
    if not m:
        return None, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def labid_group(neuron_id: str) -> Optional[str]:
    """Group of a LabID cell id (longest matching ``<group>_`` prefix)."""
    for group in sorted(LABID_GROUPS, key=len, reverse=True):
        if str(neuron_id).startswith(group + "_"):
            return group
    return None


PREFIX_RE = re.compile(r"^(?P<neuron>.+)_limb_(?P<limb>\d+)_branch_(?P<branch>\d+)$")


H01_MERGED_DIRS = ("h01/merged_branches", "H01/merged_branches", "H01/H01_Seg/merged_branches")


def h01_merged_branches_dir(metadata_root: Path) -> Path:
    """First existing ``merged_branches`` folder of H01 (the segmentation ``H01_Seg`` lies one level deeper)."""
    root = Path(metadata_root)
    for rel in H01_MERGED_DIRS:
        if (root / rel).exists():
            return root / rel
    return root / H01_MERGED_DIRS[0]


def load_metadata_tables(metadata_root: Path) -> Dict[str, Any]:
    metadata_root = Path(metadata_root)
    if not metadata_root.exists():
        # A relative default (SpinePreprocessingConfig.metadata_root =
        # Path("datasets")) resolves against the process's CWD, which in a
        # notebook is often the notebook's own directory, not the repo root -
        # this then silently resolves to a nonexistent path and every
        # species/cell_type/compartment lookup below quietly returns None
        # (visible only as merge_metadata_status="partial_missing_...").
        # Pass metadata_root explicitly (e.g. REPO_ROOT / "datasets") to avoid this.
        warnings.warn(
            f"metadata_root does not exist: {metadata_root} - species/cell_type/"
            "compartment will resolve to null for every spine. Pass an explicit, "
            "absolute metadata_root instead of relying on the CWD-relative default.",
            stacklevel=2,
        )
    minnie_root = metadata_root / "minnie65"
    h01_root = metadata_root / "h01"
    metadata: Dict[str, Any] = {
        "merge_mapping": {},
        "cell_types": {},
        "compartments": {},
    }

    mapping_paths = {
        "minnie65": minnie_root / "merged_branches" / "single_child_merge_mapping.parquet",
        "minnie": minnie_root / "merged_branches" / "single_child_merge_mapping.parquet",
        "h01": h01_merged_branches_dir(metadata_root) / "h01_single_child_merge_mapping.parquet",
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


_LIMB_SUFFIX = re.compile(r"_limb_(\d+)$")


def compartment_lookup(frame: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    """``{(neuron, limb): compartment}``. In ``minnie65_limb_compartments_length_rule.parquet`` the column ``limb_id`` is a
    RUNNING row index over the whole table (0..7 for the first neuron, 8..12 for the second, ...), the limb of the neuron
    is the suffix of ``limb`` (``<neuron>_limb_003``); ``limb_id`` is used only when ``limb`` is absent."""
    result: Dict[Tuple[str, int], str] = {}
    has_limb = "limb" in frame.columns
    for row in frame.itertuples():
        limb = None
        if has_limb and isinstance(row.limb, str):
            match = _LIMB_SUFFIX.search(row.limb)
            limb = int(match.group(1)) if match else None
        if limb is None:
            limb = int(row.limb_id)
        result[(str(row.neuron), limb)] = str(row.compartment)
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
    out = {
        "species": species(kind),
        "health": health(kind),
        "cell_type": ctype,
        "cell_type_binary": cell_type_binary(ctype),
        "compartment": compartment(record, metadata),
        "treatment": None,
        "imaging_series": None,
    }
    if kind == "labid":  # health / treatment / series come from the group encoded in the cell id
        group = labid_group(record.neuron_id)
        if group is not None:
            out["health"], out["treatment"], out["imaging_series"] = LABID_GROUPS[group]
    return out


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

