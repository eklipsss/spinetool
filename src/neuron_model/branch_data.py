"""Training data of modules T and P from the branch preprocessing (docs/neuron-model/tp-branch-preprocessing.md, 11).

The preprocessing writes ``branches.parquet`` / ``branch_spines.parquet`` (dataset level, in its output root) and one
``branch_geometry.npz`` per merged branch (~56 000 small files on Minnie65). Reading those files one by one on every
fit is what made the S-module training NAS-bound, so the node arrays are read ONCE, with threads, into a local
**geometry pack**: a folder of a few flat ``.npy`` arrays (memory-mapped on use) plus ``index.parquet`` (key, start,
n_nodes) and ``pack_info.json`` (source root, preprocessing config hashes). A pack is rebuilt only when the branch set
or the preprocessing config changes. Everything in nm.

Geometry paths are rebuilt from the keys under the given root, not taken from the absolute ``geometry_path`` column:
an output folder copied from the Windows workstation (or moved off the NAS) works unchanged.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

BRANCH_KEYS = ["dataset", "neuron_id", "limb_id", "merged_branch_id"]
PACK_FIELDS = ("s", "radius", "kappa1", "kappa2", "gamma", "frame_tangent", "frame_e1", "frame_e2")
PACK_VERSION = "branch-pack-v1"


def _read_parquet(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    if columns is not None:
        available = set(pq.read_schema(path).names)
        columns = [c for c in columns if c in available]
    frame = pq.read_table(path, columns=columns).to_pandas()
    for key in ("neuron_id",):
        if key in frame.columns:
            frame[key] = frame[key].astype(str)
    return frame


def load_branches(root: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Dataset-level ``branches.parquet`` (only ``columns`` if given; the keys are always read)."""
    cols = None if columns is None else list(dict.fromkeys(BRANCH_KEYS + list(columns)))
    return _read_parquet(Path(root) / "branches.parquet", cols)


def load_branch_spines(root: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    cols = None if columns is None else list(dict.fromkeys(BRANCH_KEYS + list(columns)))
    path = Path(root) / "branch_spines.parquet"
    return _read_parquet(path, cols) if path.exists() else pd.DataFrame(columns=cols or BRANCH_KEYS)


def branch_key(row: Any) -> str:
    return f"{row['dataset']}/{row['neuron_id']}/{int(row['limb_id']):03d}/{int(row['merged_branch_id'])}"


def key_series(frame: pd.DataFrame) -> pd.Series:
    """``branch_key`` of every row, vectorised."""
    return (frame["dataset"].astype(str) + "/" + frame["neuron_id"].astype(str) + "/"
            + frame["limb_id"].astype(int).map("{:03d}".format) + "/" + frame["merged_branch_id"].astype(int).astype(str))


def geometry_path(root: Path, row: Any) -> Path:
    return Path(root) / str(row["neuron_id"]) / f"limb_{int(row['limb_id']):03d}" / f"merged_{int(row['merged_branch_id'])}" / "branch_geometry.npz"


def select(table: pd.DataFrame, *, eligible: Optional[str] = None, splits: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Rows with ``status == ok``, the given eligibility flag (``train_eligible_T`` / ``train_eligible_P``) and split."""
    keep = table["status"].astype(str) == "ok" if "status" in table.columns else pd.Series(True, index=table.index)
    if eligible:
        keep &= table[eligible].fillna(False).astype(bool)
    if splits is not None:
        keep &= table["split"].astype(str).isin(set(splits))
    return table[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# geometry pack
# ---------------------------------------------------------------------------


@dataclass
class GeometryPack:
    root: Path
    index: pd.DataFrame  # key, start, n_nodes
    arrays: Dict[str, np.ndarray]  # flat [sum n_nodes, ...] (memory-mapped)

    def __post_init__(self) -> None:
        self._pos = {k: i for i, k in enumerate(self.index["key"])}

    def __contains__(self, key: str) -> bool:
        return key in self._pos

    def get(self, key: str) -> Dict[str, np.ndarray]:
        i = self._pos[key]
        a, n = int(self.index["start"].iat[i]), int(self.index["n_nodes"].iat[i])
        return {name: np.asarray(arr[a:a + n]) for name, arr in self.arrays.items()}

    @classmethod
    def load(cls, folder: Path, fields: Sequence[str] = PACK_FIELDS) -> "GeometryPack":
        folder = Path(folder)
        index = pd.read_parquet(folder / "index.parquet")
        arrays = {f: np.load(folder / f"{f}.npy", mmap_mode="r") for f in fields if (folder / f"{f}.npy").exists()}
        return cls(folder, index, arrays)


def _pack_info(source_root: Path, table: pd.DataFrame) -> Dict[str, Any]:
    keys = sorted(branch_key(r) for r in table[BRANCH_KEYS].to_dict("records"))
    import hashlib

    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]
    hashes = sorted(set(table["config_hash"].astype(str))) if "config_hash" in table.columns else []
    return {"version": PACK_VERSION, "source_root": str(source_root), "n_branches": len(keys), "keys_digest": digest, "config_hashes": hashes}


def build_geometry_pack(
    source_root: Path,
    table: pd.DataFrame,
    out: Path,
    *,
    io_threads: int = 16,
    fields: Sequence[str] = PACK_FIELDS,
    force: bool = False,
    progress: bool = True,
) -> GeometryPack:
    """Read the ``branch_geometry.npz`` of every row of ``table`` (threads) into a local pack at ``out``.

    Reuses an existing pack with the same branch set and preprocessing config hashes. A missing / unreadable file is
    reported and the branch left out (the index says which ones are in)."""
    out = Path(out)
    info = _pack_info(source_root, table)
    info_path = out / "pack_info.json"
    if not force and info_path.exists():
        old = json.loads(info_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in ("version", "keys_digest", "config_hashes")} == {k: info[k] for k in ("version", "keys_digest", "config_hashes")}:
            return GeometryPack.load(out, fields)
    rows = table[BRANCH_KEYS].to_dict("records")

    def read(row):
        try:
            with np.load(geometry_path(source_root, row)) as d:
                return {f: np.asarray(d[f]) for f in fields}
        except (OSError, KeyError, ValueError) as exc:
            return exc

    results: List[Any] = [None] * len(rows)
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=len(rows), desc="geometry pack", unit="branch")
    with ThreadPoolExecutor(max_workers=max(1, int(io_threads))) as pool:
        for i, res in enumerate(pool.map(read, rows)):
            results[i] = res
            if bar is not None:
                bar.update()
    if bar is not None:
        bar.close()
    ok = [i for i, r in enumerate(results) if isinstance(r, dict)]
    failed = [(branch_key(rows[i]), repr(results[i])) for i in range(len(rows)) if not isinstance(results[i], dict)]
    out.mkdir(parents=True, exist_ok=True)
    n_nodes = np.array([len(results[i]["s"]) for i in ok], dtype=np.int64)
    starts = np.r_[0, np.cumsum(n_nodes)[:-1]].astype(np.int64)
    for f in fields:
        dtype = np.float32 if f.startswith("frame_") or f == "gamma" else np.float64  # nm coordinates: float32 ~0.1 nm
        flat = np.concatenate([np.asarray(results[i][f], dtype=dtype) for i in ok]) if ok else np.zeros(0, dtype=dtype)
        tmp = out / f"{f}.tmp.npy"
        np.save(tmp, flat)
        os.replace(tmp, out / f"{f}.npy")
    index = pd.DataFrame({"key": [branch_key(rows[i]) for i in ok], "start": starts, "n_nodes": n_nodes})
    index.to_parquet(out / "index.parquet", index=False)
    info.update(n_packed=len(ok), n_failed=len(failed), failed=failed[:50])
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    if failed:
        print(f"[geometry pack] {len(failed)} branches without a readable geometry file (first: {failed[0]})", flush=True)
    return GeometryPack.load(out, fields)


# ---------------------------------------------------------------------------
# training objects of the models
# ---------------------------------------------------------------------------


def _cond(row: Any) -> Dict[str, Any]:
    def clean(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    return {"species": clean(row.get("species")), "cell_type_binary": clean(row.get("cell_type_binary")), "compartment": clean(row.get("compartment"))}


def _mean_radius(geom: Dict[str, np.ndarray]) -> float:
    s, r = geom["s"], geom["radius"]
    return float(np.trapezoid(r, s) / max(s[-1] - s[0], 1e-9)) if len(s) > 1 else float(r[0])


def t0_training_branches(table: pd.DataFrame, pack: GeometryPack, encoder) -> List[Any]:
    """``T0Branch`` per row that is in the pack (rows should come from ``select(..., eligible='train_eligible_T')``)."""
    from .trunk_generation.conditions import group_key
    from .trunk_generation.models.t0_statistical import T0Branch

    out = []
    for row in table.to_dict("records"):
        key = branch_key(row)
        if key not in pack:
            continue
        g = pack.get(key)
        step = float(g["s"][-1] / (len(g["s"]) - 1))
        c = _cond(row)
        out.append(T0Branch(g["kappa1"][:-1].astype(float), g["kappa2"][:-1].astype(float), np.log(g["radius"].astype(float)), step,
                            encoder.row(c["species"], c["cell_type_binary"], c["compartment"], float(g["s"][-1])),
                            group_key(c["species"], c["cell_type_binary"], c["compartment"])))
    return out


def count_covariates(table: pd.DataFrame, pack: GeometryPack) -> List[Any]:
    """``BranchCovariates`` (count = ``n_spines_in_range``; rows from ``select(..., eligible='train_eligible_P')``)."""
    from .placement_generation.covariates import BranchCovariates

    out = []
    for row in table.to_dict("records"):
        key = branch_key(row)
        if key not in pack:
            continue
        g = pack.get(key)
        length = float(g["s"][-1])
        d0 = row.get("soma_distance_start_nm")
        soma_mid = None if d0 is None or not np.isfinite(d0) else float(d0) + 0.5 * length
        area = row.get("area_shaft_nm2")
        if area is None or not np.isfinite(area):
            area = float(np.trapezoid(2 * np.pi * g["radius"], g["s"]))
        c = _cond(row)
        out.append(BranchCovariates(str(row["neuron_id"]), int(row["n_spines_in_range"]), length, float(area), _mean_radius(g), soma_mid,
                                    float(row.get("tortuosity") or 1.0), bool(row.get("is_terminal")), c["species"], c["cell_type_binary"], c["compartment"]))
    return out


def position_branches(table: pd.DataFrame, spines: pd.DataFrame, pack: GeometryPack) -> List[Any]:
    """``PositionBranch`` with the observed positions s_i of the valid spines that have a position
    (``has_position``: not ``projects_to_end``, not ``position_outlier``)."""
    from .placement_generation.models.position import PositionBranch

    if len(spines):
        sp = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].fillna(False).astype(bool)]
        by_key = {k: g["s"].to_numpy(dtype=float) for k, g in sp.groupby(key_series(sp))}
    else:
        by_key = {}
    out = []
    for row in table.to_dict("records"):
        key = branch_key(row)
        if key not in pack:
            continue
        g = pack.get(key)
        d0 = row.get("soma_distance_start_nm")
        k = np.hypot(g["kappa1"], g["kappa2"]).astype(float)
        if len(k) > 1:
            k[-1] = k[-2]  # the bending of the last node is undefined (NaN in the file)
        pos = by_key.get(key)
        out.append(PositionBranch(g["s"].astype(float), g["radius"].astype(float), k, None if d0 is None or not np.isfinite(d0) else float(d0),
                                  None if pos is None else np.sort(np.clip(pos, 0.0, float(g["s"][-1])))))
    return out


def trunk_record(geom: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """A packed real branch as a trunk record of the T interface (for 'real T + generated P')."""
    return {"s": geom["s"], "gamma": geom["gamma"].astype(float), "radius": geom["radius"], "tangent": geom["frame_tangent"].astype(float),
            "normal": geom["frame_e1"].astype(float), "binormal": geom["frame_e2"].astype(float),
            "kappa1": geom["kappa1"][:-1], "kappa2": geom["kappa2"][:-1], "length": float(geom["s"][-1])}


# ---------------------------------------------------------------------------
# one call for the fit scripts
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "status", "split", "train_eligible_T", "train_eligible_P", "config_hash", "species", "health", "cell_type_binary", "compartment",
    "length_nm", "is_terminal", "tortuosity", "area_shaft_nm2", "soma_distance_start_nm", "radius_median_nm",
    "n_spines_in_range", "n_with_position", "unknown_fraction",
]
SPINE_COLUMNS = ["status", "has_position", "s", "u", "theta", "radius_at_s", "attachment_cap_area", "projects_to_end", "position_outlier"]


def prepare_training_data(branches_root: Path, pack_dir: Path, *, splits: Optional[Path] = None, io_threads: int = 16, force_pack: bool = False):
    """(branches table, spines table, geometry pack) for T and P.

    Reads only the needed columns; packs the geometry of the branches eligible for T or P (all splits) into the local
    ``pack_dir`` once. ``splits``: a ``splits_*.parquet`` used when the preprocessing ran without one (``split`` empty)."""
    branches_root = Path(branches_root)
    table = load_branches(branches_root, TABLE_COLUMNS)
    if splits is not None and ("split" not in table.columns or table["split"].isna().any()):
        from .branch_preprocessing.inputs import neuron_split_lookup

        lookup = neuron_split_lookup(Path(splits))
        filled = table["neuron_id"].map(lookup)
        table["split"] = table["split"].where(table["split"].notna(), filled) if "split" in table.columns else filled
    usable = table[(table["status"].astype(str) == "ok") & (table["train_eligible_T"].fillna(False) | table["train_eligible_P"].fillna(False))]
    pack = build_geometry_pack(branches_root, usable, Path(pack_dir), io_threads=io_threads, force=force_pack)
    spines = load_branch_spines(branches_root, SPINE_COLUMNS)
    return table, spines, pack
