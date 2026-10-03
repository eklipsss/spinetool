"""Fixed group train/validation/test split, shared by every model.

Rules (spine_generation_technical_spec.md §31, s-module-technical-spec.md
"Общие требования"):

- never split individual spines: all spines of one group land in one split;
  group = highest available level (``animal_id`` > ``sample_id`` >
  ``neuron_id`` > ``branch_id``); Minnie/H01 metadata has ``neuron_id`` at most,
  so the default group is ``(dataset, neuron_id)``;
- fractions 80/10/10 are counted over **groups**, actual spine counts are reported;
- soft stratification by ``dataset/species/health/cell_type_binary/compartment``
  that never breaks the grouping ("no leakage > exact stratification");
- the result is saved once (``splits_v1.parquet``: ``spine_key, group_id,
  split``) and every derived representation of a spine inherits its split.

Algorithm: each group gets a stratum = its most common stratification tuple.
Groups are shuffled within their stratum (fixed seed) and get a fractional
position ``(i + 0.5) / n_stratum``; sorting all groups by that position
interleaves the strata, and cutting the sorted list at the target fractions
gives every stratum approximately the same train/val/test proportions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
GROUP_LEVELS = ("animal_id", "sample_id", "neuron_id", "branch_id")
DEFAULT_STRATIFY = ("dataset", "species", "health", "cell_type_binary", "compartment")
DEFAULT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def choose_group_level(frame: pd.DataFrame, levels: Sequence[str] = GROUP_LEVELS) -> str:
    """Highest level with a non-empty column in ``frame``."""
    for level in levels:
        if level in frame.columns and frame[level].notna().any():
            return level
    raise ValueError(f"None of the group levels {levels} is present")


def group_ids(frame: pd.DataFrame, level: str) -> pd.Series:
    """Group id including every coarser identifier, so equal ids in different datasets never collide."""
    parts = {"animal_id": ["dataset", "animal_id"], "sample_id": ["dataset", "sample_id"],
             "neuron_id": ["dataset", "neuron_id"], "branch_id": ["dataset", "neuron_id", "limb_id", "branch_id"]}[level]
    return frame[parts].astype(str).agg("/".join, axis=1)


def build_group_split(
    frame: pd.DataFrame,
    *,
    fractions: Optional[Mapping[str, float]] = None,
    seed: int = 20260929,
    group_level: Optional[str] = None,
    stratify: Sequence[str] = DEFAULT_STRATIFY,
) -> pd.DataFrame:
    """Return ``frame[["spine_key"]]`` + ``group_id``, ``stratum``, ``split``."""
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must cover {SPLITS} and sum to 1, got {fractions}")
    level = group_level or choose_group_level(frame)
    work = frame[["spine_key"]].copy()
    work["group_id"] = group_ids(frame, level).to_numpy()

    strat_cols = [c for c in stratify if c in frame.columns]
    strat_values = frame[strat_cols].astype(str).agg("|".join, axis=1) if strat_cols else "all"
    work["_strat"] = strat_values
    groups = (
        work.groupby("group_id")["_strat"]
        .agg(lambda s: s.value_counts().sort_index().idxmax())
        .rename("stratum")
        .reset_index()
        .sort_values("group_id")
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(seed)
    positions = np.empty(len(groups), dtype=float)
    for _, members in groups.groupby("stratum").groups.items():
        members = np.asarray(sorted(members))
        order = rng.permutation(len(members))
        positions[members[order]] = (np.arange(len(members)) + 0.5) / len(members)
    groups["_pos"] = positions
    groups["_tie"] = rng.random(len(groups))
    groups = groups.sort_values(["_pos", "_tie"]).reset_index(drop=True)

    n = len(groups)
    n_train = int(round(fractions["train"] * n))
    n_val = int(round(fractions["val"] * n))
    if n >= 3:  # keep every split non-empty when there are enough groups
        n_val = max(n_val, 1)
        n_train = min(max(n_train, 1), n - n_val - 1)
    labels = np.array(["test"] * n, dtype=object)
    labels[:n_train] = "train"
    labels[n_train:n_train + n_val] = "val"
    groups["split"] = labels

    result = work.drop(columns="_strat").merge(groups[["group_id", "stratum", "split"]], on="group_id", how="left")
    return result.sort_values("spine_key").reset_index(drop=True)


def assert_no_leakage(splits: pd.DataFrame) -> None:
    per_group = splits.groupby("group_id")["split"].nunique()
    leaking = per_group[per_group > 1]
    if len(leaking):
        raise AssertionError(f"Groups present in more than one split: {leaking.index[:5].tolist()}")
    if splits["spine_key"].duplicated().any():
        raise AssertionError("A spine appears more than once in the split table")


def split_summary(splits: pd.DataFrame, frame: Optional[pd.DataFrame] = None, stratify: Sequence[str] = DEFAULT_STRATIFY) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_spines": {s: int((splits["split"] == s).sum()) for s in SPLITS},
        "n_groups": {s: int(splits.loc[splits["split"] == s, "group_id"].nunique()) for s in SPLITS},
    }
    if frame is not None:
        merged = splits.merge(frame, on="spine_key", how="left")
        summary["by_column"] = {
            col: merged.groupby([col, "split"], dropna=False).size().unstack(fill_value=0).to_dict(orient="index")
            for col in stratify
            if col in merged.columns
        }
    return summary


def keys_hash(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()[:16]


def save_splits(splits: pd.DataFrame, path: Path, *, version: str, info: Mapping[str, Any]) -> None:
    """Write ``<path>`` (parquet) and ``<path>.json`` (version, seed, fractions, level, counts, hashes)."""
    assert_no_leakage(splits)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists - a split is frozen once written; bump the version instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = splits.assign(split_version=version)
    table.to_parquet(path, index=False)
    meta = {"split_version": version, "spine_keys_hash": keys_hash(splits["spine_key"].tolist()), **dict(info)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_splits(path: Path) -> pd.DataFrame:
    splits = pd.read_parquet(path)
    assert_no_leakage(splits)
    return splits
