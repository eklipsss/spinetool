"""Inputs of the branch preprocessing: merged-branch mapping, S-module tables, metadata, split.

Everything is read once in the main process and handed to the per-neuron workers as small
per-neuron payloads (the S tables have ~10^6 rows: only the needed columns are read).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..metadata import cell_type_binary, dataset_kind, health, load_metadata_tables, parse_branch_prefix, species
from .config import BranchPreprocessingConfig

MAPPING_FILES = {
    "minnie65": ("minnie65", "single_child_merge_mapping.parquet", "merged_branches.csv"),
    "h01": ("h01", "h01_single_child_merge_mapping.parquet", None),
}
SPINE_KEYS = ["dataset", "neuron_id", "limb_id", "branch_id", "spine_id"]


def _mapping_paths(cfg: BranchPreprocessingConfig):
    kind = dataset_kind(cfg.dataset)
    if kind not in MAPPING_FILES:
        raise ValueError(f"no merged-branch mapping known for dataset {cfg.dataset!r}")
    folder, mapping_name, summary_name = MAPPING_FILES[kind]
    base = Path(cfg.metadata_root) / folder / "merged_branches"
    if kind == "h01":
        from ..metadata import h01_merged_branches_dir

        base = h01_merged_branches_dir(Path(cfg.metadata_root))
    return base / mapping_name, (base / summary_name) if summary_name else None


def load_members(cfg: BranchPreprocessingConfig) -> pd.DataFrame:
    """One row per ORIGINAL branch: neuron_id, limb_id, original_branch_id, merged_branch_id and
    the merged-branch attributes (length in nm, depth, terminal flag, parent)."""
    mapping_path, _ = _mapping_paths(cfg)
    raw = pd.read_parquet(mapping_path)
    parsed = pd.DataFrame(raw["prefix"].astype(str).map(parse_branch_prefix).tolist())
    out = pd.DataFrame(
        {
            "neuron_id": parsed["neuron_id"].astype(str),
            "limb_id": parsed["limb_id"].astype(int),
            "original_branch_id": parsed["original_branch_id"].astype(int),
            "prefix": raw["prefix"].astype(str),
            "merged_branch_id": raw["merged_branch_id"].astype(int),
            "merged_parent_branch_id": raw.get("merged_parent_branch_id", pd.Series(-1, index=raw.index)).astype(int),
            "merged_member_count": raw["merged_member_count"].astype(int),
            "merged_length_nm": raw["merged_length_um"].astype(float) * 1000.0,
            "merged_branch_depth": raw["merged_branch_depth"].astype(int),
            "merged_is_terminal": raw["merged_is_terminal"].astype(bool),
            "merged_parent_ambiguous": raw.get("merged_parent_ambiguous", pd.Series(False, index=raw.index)).astype(bool),
        }
    )
    # H01_Seg: folders are ``neuron_<id>`` and the prefixes carry the same name, so neuron_id is kept as is
    # (the same key as in the S tables, which take the folder name)
    return out


def load_soma_midpoints(cfg: BranchPreprocessingConfig) -> Dict[str, float]:
    """``{member prefix: soma_midpoint_nm}`` from merged_branches.csv (Minnie65 only; cross-check)."""
    _, summary_path = _mapping_paths(cfg)
    if summary_path is None or not summary_path.exists():
        return {}
    table = pd.read_csv(summary_path, usecols=["member_prefixes", "soma_midpoint_um"])
    out: Dict[str, float] = {}
    for prefixes, value in zip(table["member_prefixes"].astype(str), table["soma_midpoint_um"].astype(float)):
        for prefix in prefixes.split("|"):
            out[prefix] = value * 1000.0
    return out


def _read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Main table + leftover ``<table>.parquet.parts/`` (latest wins), only ``columns`` that exist."""
    import pyarrow.parquet as pq

    files = []
    if path.exists():
        files.append(path)
    parts_dir = path.with_name(f"{path.name}.parts")
    if parts_dir.is_dir():
        files.extend(sorted(p for p in parts_dir.iterdir() if p.suffix == ".parquet"))
    frames = []
    for f in files:
        available = set(pq.read_schema(f).names)
        frames.append(pq.read_table(f, columns=[c for c in columns if c in available]).to_pandas())
    if not frames:
        return pd.DataFrame(columns=list(columns))
    frame = pd.concat(frames, ignore_index=True)
    keys = [k for k in SPINE_KEYS if k in frame.columns]
    return frame.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def load_spine_table(cfg: BranchPreprocessingConfig, neurons: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Per spine of the S preprocessing: false-spine verdict (``status``), seal status, attachment
    centre (``origin_global`` of the oriented spine = centre of the attachment loop), areas."""
    root = cfg.s_root
    manifest = _read_columns(
        root / "manifest.parquet",
        SPINE_KEYS + ["merged_branch_id", "status", "seal_status", "orient_status", "attachment_ambiguous", "attachment_method", "attachment_region_path"],
    )
    if neurons is not None:
        wanted = set(map(str, neurons))
        manifest = manifest[manifest["neuron_id"].astype(str).isin(wanted)]
    oriented = _read_columns(root / "oriented_spines.parquet", SPINE_KEYS + ["origin_global", "attachment_loop_area", "attachment_cap_area"])
    sealed = _read_columns(root / "sealed_spines.parquet", SPINE_KEYS + ["attachment_loop_area", "attachment_cap_area"])
    for frame in (manifest, oriented, sealed):
        for key in SPINE_KEYS:
            if key in frame.columns:
                frame[key] = frame[key].astype(str)
    table = manifest.merge(oriented, on=SPINE_KEYS, how="left")
    table = table.merge(sealed.rename(columns={"attachment_loop_area": "sealed_loop_area", "attachment_cap_area": "sealed_cap_area"}), on=SPINE_KEYS, how="left")
    for col, fallback in (("attachment_loop_area", "sealed_loop_area"), ("attachment_cap_area", "sealed_cap_area")):
        table[col] = table[col].where(table[col].notna(), table[fallback])
    return table.drop(columns=["sealed_loop_area", "sealed_cap_area"])


def neuron_split_lookup(splits_path: Optional[Path]) -> Dict[str, str]:
    """``{neuron_id: split}`` from a ``splits_*.parquet`` (spine_key = dataset/neuron/limb/branch/spine)."""
    if splits_path is None or not Path(splits_path).exists():
        return {}
    splits = pd.read_parquet(splits_path, columns=["spine_key", "split"])
    neuron = splits["spine_key"].astype(str).str.split("/").str[1]
    frame = pd.DataFrame({"neuron_id": neuron, "split": splits["split"].astype(str)}).drop_duplicates("neuron_id")
    return dict(zip(frame["neuron_id"], frame["split"]))


def neuron_metadata(cfg: BranchPreprocessingConfig, metadata: Dict[str, Any], neuron_id: str, limbs: Sequence[int]) -> Dict[str, Any]:
    kind = dataset_kind(cfg.dataset)
    ctype = metadata.get("cell_types", {}).get(kind, {}).get(str(neuron_id))
    comps = metadata.get("compartments", {}).get(kind, {})
    return {
        "species": species(kind),
        "health": health(kind),
        "cell_type": ctype,
        "cell_type_binary": cell_type_binary(ctype),
        "compartment_by_limb": {int(l): comps.get((str(neuron_id), int(l))) for l in limbs},
    }


def available_neurons(cfg: BranchPreprocessingConfig, members: pd.DataFrame) -> List[str]:
    """Neurons present both in the mapping and as folders of the raw dataset (one directory listing)."""
    import os

    root = cfg.dataset_root
    present = {entry.name for entry in os.scandir(root) if entry.is_dir()} if root.is_dir() else set()
    neurons = sorted(set(members["neuron_id"]) & present)
    if cfg.only_s_neurons:
        manifest = Path(cfg.s_root) / "manifest.parquet"
        in_s = set(_read_columns(manifest, ["neuron_id"])["neuron_id"].astype(str)) if manifest.exists() else set()
        neurons = [n for n in neurons if n in in_s]
    if cfg.neuron_ids:
        wanted = {str(n) for n in cfg.neuron_ids}
        neurons = [n for n in neurons if n in wanted]
    if cfg.limit_neurons:
        neurons = neurons[: int(cfg.limit_neurons)]
    return neurons


def load_metadata(cfg: BranchPreprocessingConfig) -> Dict[str, Any]:
    return load_metadata_tables(Path(cfg.metadata_root))


def as_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr if arr.size == 3 and np.all(np.isfinite(arr)) else None
