"""Branch assembly: trunk (T) + spine placements (P) + spine meshes (S) -> one closed branch surface.

See ``assemble.py`` for the algorithm (module-interfaces-spec.md, section 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .assemble import AssemblyConfig, AssemblyResult, assemble_branch, place, rotation
from .spines import PoolSpine, SpinePool, find_mesh_files
from .trunk import DenseTrunk, densify, trunk_mesh


def load_tp_set(directory: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[int, pd.DataFrame]]:
    """A ``generate_tp.py`` output directory -> (branches table, trunk records, {branch_index: placements})."""
    directory = Path(directory)
    # every ``NpzFile[k]`` re-reads and decompresses the whole array: read each one ONCE (per-branch access was
    # quadratic and ran out of memory on 3000 branches)
    with np.load(directory / "trunks.npz") as npz:
        packed = {k: npz[k] for k in npz.files}
    offsets = packed["offsets"]
    fields = [k for k in packed if k != "offsets"]
    trunks = []
    for i in range(len(offsets) - 1):
        rec = {k: packed[k][offsets[i]:offsets[i + 1]] for k in fields}
        rec["length"] = float(rec["s"][-1])
        trunks.append(rec)
    branches = pd.read_parquet(directory / "branches.parquet")
    path = directory / "placements.parquet"
    placements = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["branch_index", "s", "theta"])
    return branches, trunks, {int(k): g.sort_values("spine_index") for k, g in placements.groupby("branch_index")}


__all__ = ["AssemblyConfig", "AssemblyResult", "DenseTrunk", "PoolSpine", "SpinePool", "assemble_branch", "densify", "find_mesh_files",
           "load_tp_set", "place", "rotation", "trunk_mesh"]
