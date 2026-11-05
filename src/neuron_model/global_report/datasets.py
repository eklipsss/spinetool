"""Loaders: one TIDY frame per level (spine / branch group) for every dataset, plus the dataset inventory.

Tidy columns (both levels): ``dataset`` (minnie65 / h01 / labid), ``unit_id`` = ``neuron_id`` (``<dataset>:<raw id>``; the
unit of inference: a neuron in Minnie65 / H01, a branch FOLDER in LabID, where no neuron identity exists), ``species``,
``modality``, ``health``, ``treatment``, ``group`` (LabID group), ``cell_type``, ``cell_type_binary``, ``compartment``
(+ ``limb_id``, ``branch_id`` at the spine level) and the metrics. Missing metadata = None (never silently filled).

``tidy_*`` are pure functions (testable on synthetic data); ``load_*`` read the Windows tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..compartments import layer_value
from ..metadata import LABID_GROUPS, cell_type_binary, dataset_kind, labid_group, load_metadata_tables, species

MODALITY = {"minnie65": "EM", "h01": "EM", "labid": "confocal"}
UNIT_NAME = {"minnie65": "neuron", "h01": "neuron", "labid": "branch folder"}
SPINE_KEYS = ["dataset", "neuron_id", "limb_id", "branch_id", "spine_id"]
MINNIE_INTERNEURONS = {"BC", "BPC", "MC", "NGC"}
H01_CELL_CLASS = {"PYRAMIDAL": "pyramidal", "SPINY_ATYPICAL": "other_excitatory", "INTERNEURON": "interneuron"}


def cell_class(cell_type: Any) -> Optional[str]:
    """HARMONISED cell class usable in Minnie65 and H01: pyramidal (Minnie ...P types incl. WM-P; H01 PYRAMIDAL),
    other_excitatory (H01 SPINY_ATYPICAL), interneuron (Minnie BC BPC MC NGC; H01 INTERNEURON). Unsure / OLIGO -> None."""
    if not isinstance(cell_type, str):
        return None
    if cell_type in H01_CELL_CLASS:
        return H01_CELL_CLASS[cell_type]
    if cell_type in MINNIE_INTERNEURONS:
        return "interneuron"
    if cell_type.endswith("P") or "P-" in cell_type:
        return "pyramidal"
    return None


def layer_labels(text: Any, kind: str):
    """(layer, layer_native). ``layer`` is harmonised (L1, L2/3, L4, L5, L6, WM): Minnie65 only COARSE, derived from the
    cell-type NAME (LAYER_OF_MINNIE_TYPE; excitatory cells only), H01 from h01_layer.csv with L2 and L3 merged.
    ``layer_native`` = H01 layer without merging (L1..L6, WM), None for Minnie65."""
    v = layer_value(text)
    if v is None:
        return None, None
    harmonised = "L2/3" if v in (2.0, 2.5, 3.0) else ("WM" if v == 7.0 else f"L{int(v)}")
    native = None
    if kind == "h01":
        native = "WM" if v == 7.0 else f"L{int(v)}"
    return harmonised, native


META_COLUMNS = ["dataset", "unit_id", "neuron_id", "species", "modality", "health", "treatment", "group", "cell_type",
                "cell_type_binary", "cell_class", "layer", "layer_native", "compartment"]


def _limb_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def _unit_metadata(raw_ids: pd.Series, kind: str, meta: Dict[str, Any]) -> pd.DataFrame:
    """Per-row metadata that depends on the unit only."""
    ids = raw_ids.astype(str)
    out = pd.DataFrame(index=raw_ids.index)
    out["dataset"] = kind
    out["unit_id"] = kind + ":" + ids
    out["neuron_id"] = out["unit_id"]
    out["species"] = species(kind)
    out["modality"] = MODALITY.get(kind)
    if kind == "labid":
        grp = ids.map(labid_group)
        out["group"] = grp
        out["health"] = grp.map(lambda g: LABID_GROUPS[g][0] if g in LABID_GROUPS else None)
        out["treatment"] = grp.map(lambda g: (LABID_GROUPS[g][1] or "untreated") if g in LABID_GROUPS else None)
        out["cell_type"] = None
        out["cell_type_binary"] = None
        out["cell_class"] = None
        out["layer"] = None
        out["layer_native"] = None
    else:
        out["group"] = None
        out["health"] = "healthy"
        out["treatment"] = None
        types = meta.get("cell_types", {}).get(kind, {})
        out["cell_type"] = ids.map(types)
        out["cell_type_binary"] = out["cell_type"].map(lambda t: cell_type_binary(t) if isinstance(t, str) else None)
        out["cell_class"] = out["cell_type"].map(cell_class)
        layer_src = ids.map(meta.get("layers", {}).get(kind, {})) if kind == "h01" else out["cell_type"]
        labels = layer_src.map(lambda t: layer_labels(t, kind))
        out["layer"] = labels.map(lambda x: x[0])
        out["layer_native"] = labels.map(lambda x: x[1])
    return out


def tidy_spines(raw: pd.DataFrame, kind: str, meta: Dict[str, Any], metrics: Sequence[str], *,
                exclude_groups: Sequence[str] = ("Wt_old",)) -> pd.DataFrame:
    """``raw``: one row per spine with ``neuron_id`` (+ ``limb_id``, ``branch_id``, ``spine_id`` when present) and metrics."""
    kind = dataset_kind(kind)
    raw = raw.reset_index(drop=True)
    meta_cols = _unit_metadata(raw["neuron_id"], kind, meta)
    limb = _limb_number(raw["limb_id"]) if "limb_id" in raw.columns else pd.Series(np.nan, index=raw.index)
    comps = meta.get("compartments", {}).get(kind, {})
    compartment = [comps.get((str(n), int(l))) if np.isfinite(l) else None for n, l in zip(raw["neuron_id"], limb)]
    frame = meta_cols.copy()
    frame["limb_id"] = limb
    frame["branch_id"] = pd.to_numeric(raw["branch_id"], errors="coerce") if "branch_id" in raw.columns else np.nan
    frame["spine_id"] = raw["spine_id"].astype(str) if "spine_id" in raw.columns else np.arange(len(raw)).astype(str)
    frame["compartment"] = compartment
    for m in metrics:
        frame[m] = pd.to_numeric(raw[m], errors="coerce") if m in raw.columns else np.nan
    if kind == "labid" and exclude_groups:
        frame = frame[~frame["group"].isin(list(exclude_groups))]
    return frame.reset_index(drop=True)


def load_spine_frame(kind: str, table_path: Path, metadata_root: Path, *, compartment_table: Optional[Path] = None,
                     metrics: Optional[Sequence[str]] = None, exclude_groups: Sequence[str] = ("Wt_old",)) -> pd.DataFrame:
    """Stage-8 morphometrics table (``morphometrics.parquet`` + parts) -> tidy spine frame of one dataset."""
    import pyarrow.parquet as pq

    kind = dataset_kind(kind)
    if metrics is None:
        from ..spine_morphometrics import SCALAR_METRICS
        metrics = list(SCALAR_METRICS)
    table_path = Path(table_path)
    parts = Path(f"{table_path}.parts")
    paths = [table_path] + sorted(parts.glob("*.parquet")) if parts.exists() else [table_path]
    frames = []
    for path in paths:
        names = set(pq.read_schema(path).names)
        cols = [c for c in SPINE_KEYS + ["status"] + list(metrics) if c in names]
        frames.append(pq.read_table(path, columns=cols).to_pandas())
    raw = pd.concat(frames, ignore_index=True)
    keys = [k for k in SPINE_KEYS if k in raw.columns]
    raw = raw.drop_duplicates(subset=keys, keep="last")
    if "status" in raw.columns:
        raw = raw[raw["status"].astype(str) == "success"]
    if "dataset" in raw.columns:
        raw = raw[raw["dataset"].astype(str).map(dataset_kind) == kind]
    meta = load_metadata_tables(Path(metadata_root), compartment_table)
    return tidy_spines(raw, kind, meta, metrics, exclude_groups=exclude_groups)


def tidy_branch_groups(branches: pd.DataFrame, spines: pd.DataFrame, kind: str, meta: Dict[str, Any], *,
                       min_branches: int = 10, min_pairs: int = 30, exclude_groups: Sequence[str] = ("Wt_old",)) -> pd.DataFrame:
    """Branch tables of the T/P preprocessing -> one row per (unit, compartment) with the metrics of
    ``branch_statistics.ALL_METRICS`` (trunk T, placement P). ``branches``: raw ``branches.parquet`` columns
    (``neuron_id``, ``limb_id``, ``merged_branch_id``, ``status``, eligibility flags); ``spines``: positioned valid spines.
    ``axon_like`` limbs are dropped. For LabID use ``min_branches=1`` (one branch per unit)."""
    from ..branch_statistics import group_table

    kind = dataset_kind(kind)
    b = branches.copy()
    b["neuron_id"] = b["neuron_id"].astype(str)
    b["limb_id"] = _limb_number(b["limb_id"]).fillna(0).astype(int)
    comps = meta.get("compartments", {}).get(kind, {})
    types = meta.get("cell_types", {}).get(kind, {})
    b["cell_type"] = b["neuron_id"].map(types) if kind != "labid" else None
    b["cell_type_binary"] = b["cell_type"].map(lambda t: cell_type_binary(t) if isinstance(t, str) else None)
    b["compartment"] = [comps.get((n, int(l))) for n, l in zip(b["neuron_id"], b["limb_id"])]
    b = b[b["compartment"] != "axon_like"]
    s = spines.copy()
    s["neuron_id"] = s["neuron_id"].astype(str)
    s["limb_id"] = _limb_number(s["limb_id"]).fillna(0).astype(int)
    groups = group_table(b, s, min_branches=min_branches, min_pairs=min_pairs)
    info = _unit_metadata(groups["neuron_id"], kind, meta)
    out = pd.concat([info.reset_index(drop=True), groups.drop(columns=["neuron_id", "cell_type", "cell_type_binary", "compartment"]).reset_index(drop=True)], axis=1)
    out["compartment"] = groups["compartment"].astype(object).where(groups["compartment"] != "none", None).to_numpy()
    out["cell_type"] = groups["cell_type"].to_numpy() if kind != "labid" else None
    out["cell_type_binary"] = groups["cell_type_binary"].to_numpy() if kind != "labid" else None
    if kind == "labid" and exclude_groups:
        out = out[~out["group"].isin(list(exclude_groups))]
    return out.reset_index(drop=True)


def read_table_with_parts(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """A stage table (``<name>.parquet`` + ``<name>.parquet.parts/*.parquet``), only ``columns`` that exist; later rows win
    is left to the caller (keys differ between tables)."""
    import pyarrow.parquet as pq

    path = Path(path)
    parts = Path(f"{path}.parts")
    frames = []
    for p in [path] + (sorted(parts.glob("*.parquet")) if parts.exists() else []):
        names = set(pq.read_schema(p).names)
        frames.append(pq.read_table(p, columns=[c for c in columns if c in names]).to_pandas())
    return pd.concat(frames, ignore_index=True)


def load_branch_groups(kind: str, branches_root: Path, metadata_root: Path, *, compartment_table: Optional[Path] = None,
                       min_branches: Optional[int] = None, min_pairs: int = 30, morphometrics_table: Optional[Path] = None) -> pd.DataFrame:
    """``branches.parquet`` + ``branch_spines.parquet`` of a branch-preprocessing output -> tidy group frame. With the stage-8
    ``morphometrics_table`` of the same dataset the mark metrics (neighbour similarity of volume / length, volume vs u) are
    computed too; without it they stay NaN."""
    import pyarrow.parquet as pq

    kind = dataset_kind(kind)
    root = Path(branches_root)

    def read(name, columns):
        names = set(pq.read_schema(root / name).names)
        return pq.read_table(root / name, columns=[c for c in columns if c in names]).to_pandas()

    branches = read("branches.parquet", ["dataset", "neuron_id", "limb_id", "merged_branch_id", "status", "length_nm", "radius_median_nm",
                                         "radius_proximal_nm", "radius_distal_nm", "ellipse_ratio_median", "tortuosity", "kappa_rms",
                                         "n_spines", "train_eligible_T", "train_eligible_P"])
    spines = read("branch_spines.parquet", ["neuron_id", "limb_id", "merged_branch_id", "original_branch_id", "spine_id", "status", "has_position", "s", "u"])
    spines = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].astype(bool)]
    if morphometrics_table is not None:
        from ..branch_statistics import join_spine_morphometrics

        m = read_table_with_parts(morphometrics_table, ["neuron_id", "limb_id", "branch_id", "spine_id", "status", "Volume", "Length"])
        spines = join_spine_morphometrics(spines, m)
        print(f"[global] {kind}: morphometrics joined for {spines['log_volume'].notna().mean():.1%} of the positioned spines", flush=True)
    meta = load_metadata_tables(Path(metadata_root), compartment_table)
    return tidy_branch_groups(branches, spines, kind, meta, min_branches=min_branches if min_branches is not None else (1 if kind == "labid" else 10),
                              min_pairs=min_pairs)


def _counts(frame: pd.DataFrame, by: Sequence[str], unit_col: str = "unit_id") -> pd.DataFrame:
    f = frame.copy()
    for c in by:
        f[c] = f[c].astype(object).where(f[c].notna(), "unknown")
    g = f.groupby(list(by), observed=True)
    return pd.DataFrame({"n_units": g[unit_col].nunique(), "n_rows": g.size()}).reset_index()


def dataset_inventory(spines: pd.DataFrame, branches: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Composition tables: units / limbs / branches / spines per dataset and per cell type, compartment, health, treatment,
    LabID group; share of missing metadata; (``branches`` = raw branch table with ``dataset``, ``neuron_id``, ``status``,
    ``train_eligible_T/P``, ``compartment``) eligibility counts."""
    out: Dict[str, pd.DataFrame] = {}
    rows = []
    for ds, g in spines.groupby("dataset"):
        limbs = g[["unit_id", "limb_id"]].drop_duplicates()
        br = g[["unit_id", "limb_id", "branch_id"]].drop_duplicates()
        rows.append({"dataset": ds, "species": g["species"].iloc[0], "modality": g["modality"].iloc[0], "unit": UNIT_NAME.get(ds),
                     "n_units": int(g["unit_id"].nunique()), "n_limbs": int(len(limbs)) if g["limb_id"].notna().any() else None,
                     "n_branches": int(len(br)) if g["branch_id"].notna().any() else None, "n_spines": int(len(g)),
                     "spines_per_unit_median": float(g.groupby("unit_id").size().median())})
    out["by_dataset"] = pd.DataFrame(rows)
    for col in ("cell_type_binary", "cell_type", "cell_class", "layer", "compartment", "health", "treatment", "group"):
        out[f"by_dataset_{col}"] = _counts(spines, ["dataset", col])
    miss = []
    for ds, g in spines.groupby("dataset"):
        units = g.drop_duplicates("unit_id")
        miss.append({"dataset": ds, "units_without_cell_type": float(units["cell_type"].isna().mean()),
                     "spines_without_compartment": float(g["compartment"].isna().mean()),
                     "units_without_health": float(units["health"].isna().mean())})
    out["metadata_missing"] = pd.DataFrame(miss)
    if branches is not None and len(branches):
        b = branches.copy()
        b["status_ok"] = b["status"].astype(str) == "ok"
        for flag in ("train_eligible_T", "train_eligible_P"):
            b[flag] = b[flag].fillna(False).astype(bool) if flag in b.columns else False
        g = b.groupby("dataset")
        out["branch_eligibility"] = pd.DataFrame({
            "n_units": g["neuron_id"].nunique(), "n_branches": g.size(), "n_ok": g["status_ok"].sum(),
            "n_eligible_T": g["train_eligible_T"].sum(), "n_eligible_P": g["train_eligible_P"].sum()}).reset_index()
    return out
