"""Factors and the VALID comparisons between their levels.

A comparison is valid only when the factor exists in the dataset (has values), both levels have at least
``min_neurons`` units (a unit = neuron, or a branch folder in LabID) with at least ``min_spines`` rows, and - for the
within-unit factor (compartment) - at least ``min_paired`` units carry both levels. Invalid comparisons are KEPT in the
list with a reason (the report shows what could not be tested and why). Confounded factors (species, dataset) are never
causal: they are descriptive contrasts and carry an explicit warning. The statistics are those of
``spine_generation.evaluation.morphology_by_condition`` (imported, not copied).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..spine_generation.evaluation.morphology_by_condition import neuron_medians  # noqa: F401  (re-exported for callers)

EXCLUDED_TYPES = ("Unsure", "Unsure E", "Unsure I")


@dataclass(frozen=True)
class Factor:
    name: str
    column: str
    mode: str                                  # "between" units | "within" unit (paired)
    datasets: Tuple[str, ...]                  # datasets where it is meaningful; "*" scope handled by ``cross``
    levels: Optional[Tuple[str, ...]] = None   # fixed ordered levels (pairs in this order), None = data driven
    stratify: Optional[str] = None             # test inside every value of this column
    cross: bool = False                        # compares datasets with each other (one combined frame)
    causal: bool = True
    omnibus: bool = False
    confounded_with: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    exclude_levels: Tuple[str, ...] = ()


FACTORS: Tuple[Factor, ...] = (
    # no "cell_type_binary | compartment": inhibitory cells have no apical / basal division (their compartment is undefined),
    # so the stratified contrast cannot exist; it was always "too few units" (removed 2026-11-06)
    Factor("cell_type_binary", "cell_type_binary", "between", ("minnie65", "h01"), ("inhibitory", "excitatory"),
           warnings=("inhibitory cells have no apical / basal division: the binary contrast pools all dendrites of a neuron",)),
    Factor("cell_type", "cell_type", "between", ("minnie65", "h01"), None, omnibus=True, exclude_levels=EXCLUDED_TYPES),
    Factor("compartment", "compartment", "within", ("minnie65", "h01"), ("apical", "basal", "uncertain"),
           warnings=("compartment of H01 comes from the direction rule (pia axis estimated from layers), not a ground truth",)),
    Factor("health | treatment", "health", "between", ("labid",), ("healthy", "disease"), stratify="treatment",
           warnings=("unit = branch folder: animals / slices are unknown, folders of one animal are not independent "
                     "(pseudo-replication possible)",)),
    Factor("treatment | health", "treatment", "between", ("labid",), ("untreated", "9009"), stratify="health", causal=False,
           confounded_with=("z-rescaling pipeline (the 9009 groups are not rescaled by 0.7)",),
           warnings=("treatment is confounded with the imaging / z-correction series until the laboratory confirms",)),
    Factor("cell_class", "cell_class", "between", ("minnie65", "h01"), None, omnibus=True,
           warnings=("harmonised class: pyramidal / other_excitatory / interneuron, derived from the dataset's own cell-type names",)),
    Factor("layer", "layer", "between", ("minnie65", "h01"), None, omnibus=True,
           warnings=("Minnie65 layer is COARSE and derived from the cell-type NAME (excitatory cells only): it is a function of the type, "
                     "not an independent measurement; H01 layers L2 and L3 are merged to L2/3",)),
    Factor("layer_native", "layer_native", "between", ("h01",), None, omnibus=True,
           warnings=("H01 layer from h01_layer.csv, one value per neuron",)),
    Factor("species", "species", "between", ("minnie65", "h01"), ("human", "mouse"), cross=True, causal=False,
           confounded_with=("dataset", "brain region and layer sampling", "segmentation pipeline", "age / donor"),
           warnings=("species is perfectly confounded with the dataset: descriptive contrast only, NOT a test of species",)),
    Factor("species | cell_class", "species", "between", ("minnie65", "h01"), ("human", "mouse"), stratify="cell_class", cross=True, causal=False,
           confounded_with=("dataset", "brain region and layer sampling", "segmentation pipeline", "age / donor"),
           warnings=("same cell class (e.g. pyramidal) in mouse EM vs human EM: still species = dataset, descriptive only",)),
    Factor("species | layer", "species", "between", ("minnie65", "h01"), ("human", "mouse"), stratify="layer", cross=True, causal=False,
           confounded_with=("dataset", "brain region", "segmentation pipeline", "age / donor", "coarse Minnie65 layer from the type name"),
           warnings=("same layer in mouse and human, Minnie65 layer is coarse (from the cell-type name): descriptive only",)),
    Factor("dataset", "dataset", "between", ("minnie65", "h01", "labid"), None, cross=True, causal=False,
           confounded_with=("species", "imaging modality (EM vs confocal)", "resolution and z-stretch", "segmentation"),
           warnings=("dataset contrast = species + modality + processing: descriptive only",)),
)


def resolve_min_rows(min_spines, datasets) -> int:
    """Threshold of ONE dataset: ``min_spines`` is an int or ``{dataset: int, "default": int}`` (LabID branch folders carry
    ~9 spines, Minnie65 neurons ~1000). For several datasets use ``row_thresholds`` (each unit by its own dataset); the
    largest threshold is returned here only for callers that need one number."""
    if isinstance(min_spines, dict):
        names = [datasets] if isinstance(datasets, str) else list(datasets)
        return max(int(min_spines.get(d, min_spines.get("default", 20))) for d in names)
    return int(min_spines)


def row_thresholds(datasets: pd.Series, min_spines) -> pd.Series:
    """Minimal rows of a unit, per row, by the unit's OWN dataset. Taking the largest threshold of the compared datasets
    (as before 2026-11-06) dropped almost every LabID folder from the LabID vs Minnie65 / H01 spine comparisons (20 spines
    required, ~9 per folder)."""
    if isinstance(min_spines, dict):
        default = int(min_spines.get("default", 20))
        return datasets.astype(str).map(lambda d: int(min_spines.get(d, default))).astype(int)
    return pd.Series(int(min_spines), index=datasets.index)


def _units_with_rows(frame: pd.DataFrame, column: str, min_rows) -> pd.DataFrame:
    """Per (unit, value of ``column``) row counts, units with at least the threshold of their dataset (``min_rows``: int or
    ``{dataset: int}``)."""
    f = frame[frame[column].notna()]
    keys = list(dict.fromkeys(["dataset", "neuron_id", column]))   # column may be "dataset" itself
    n = f.groupby(keys, observed=True).size().rename("n_rows").reset_index()
    return n[n["n_rows"] >= row_thresholds(n["dataset"], min_rows).to_numpy()]


def _record(factor: Factor, dataset: str, a, b, stratum, **kw) -> Dict[str, Any]:
    return {"dataset": dataset, "factor": factor.name, "column": factor.column, "mode": factor.mode, "a": a, "b": b,
            "stratum_column": factor.stratify, "stratum": stratum, "causal": factor.causal,
            "confounded_with": list(factor.confounded_with), "warnings": list(factor.warnings), "omnibus": factor.omnibus,
            "n_units_a": 0, "n_units_b": 0, "n_units_paired": None, "valid": False, "reason": None, **kw}


def _between(frame: pd.DataFrame, factor: Factor, dataset: str, stratum, min_units: int, min_rows: int) -> List[Dict[str, Any]]:
    counts = _units_with_rows(frame, factor.column, min_rows).groupby(factor.column)["neuron_id"].nunique()
    if factor.levels is not None:
        pairs = list(combinations(factor.levels, 2))
    else:
        levels = [l for l, c in counts.items() if isinstance(l, str) and l not in factor.exclude_levels and c >= min_units]
        pairs = list(combinations(sorted(levels), 2))
        records = [_record(factor, dataset, l, None, stratum, reason=f"level has {int(c)} units < {min_units}", n_units_a=int(c))
                   for l, c in counts.items() if isinstance(l, str) and l not in factor.exclude_levels and c < min_units]
    out: List[Dict[str, Any]] = [] if factor.levels is not None else records
    for a, b in pairs:
        na, nb = int(counts.get(a, 0)), int(counts.get(b, 0))
        rec = _record(factor, dataset, a, b, stratum, n_units_a=na, n_units_b=nb)
        if na == 0 and nb == 0:
            rec["reason"] = "both levels absent in this dataset / stratum"
        elif na == 0 or nb == 0:
            rec["reason"] = "level absent in this dataset / stratum"
        elif na < min_units or nb < min_units:
            rec["reason"] = f"too few units (a={na}, b={nb}, need {min_units})"
        else:
            rec["valid"] = True
        out.append(rec)
    return out


def _within(frame: pd.DataFrame, factor: Factor, dataset: str, stratum, min_units: int, min_rows: int, min_paired: int) -> List[Dict[str, Any]]:
    n = _units_with_rows(frame, factor.column, min_rows)
    present = [l for l in factor.levels if (n[factor.column] == l).any()]
    out = []
    for a, b in combinations(factor.levels, 2):
        ua = set(n.loc[n[factor.column] == a, "neuron_id"])
        ub = set(n.loc[n[factor.column] == b, "neuron_id"])
        paired = len(ua & ub)
        rec = _record(factor, dataset, a, b, stratum, n_units_a=len(ua), n_units_b=len(ub), n_units_paired=paired)
        if a not in present or b not in present:
            rec["reason"] = "level absent in this dataset" if present else "factor has no values here"
        elif paired < min_paired:
            rec["reason"] = f"too few units with both levels ({paired}, need {min_paired})"
        else:
            rec["valid"] = True
        out.append(rec)
    return out


def valid_comparisons(frame: pd.DataFrame, *, factors: Sequence[Factor] = FACTORS, min_neurons: int = 10, min_spines: int = 20,
                      min_paired: int = 5, datasets: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """All comparisons of ``factors`` for the datasets of ``frame`` (columns ``dataset``, ``neuron_id`` + factor columns),
    each a dict with ``valid`` and ``reason``. ``min_neurons``: units per level (hard guard); ``min_spines``: rows of a unit
    for it to count (1 for branch-group frames)."""
    present = sorted(frame["dataset"].unique()) if datasets is None else list(datasets)
    out: List[Dict[str, Any]] = []
    for factor in factors:
        if factor.cross:
            out += _cross(frame, factor, present, min_neurons, min_spines)
            continue
        for ds in present:
            if ds not in factor.datasets:
                out.append(_record(factor, ds, None, None, None, reason=f"factor not defined for {ds}"))
                continue
            sub = frame[frame["dataset"] == ds]
            min_spines_ds = resolve_min_rows(min_spines, ds)
            if not sub[factor.column].notna().any():
                out.append(_record(factor, ds, None, None, None, reason="metadata missing in this dataset (no values)"))
                continue
            strata = [None]
            if factor.stratify:
                strata = sorted(map(str, sub[factor.stratify].dropna().unique())) or [None]
            for st in strata:
                part = sub if st is None else sub[sub[factor.stratify].astype(str) == st]
                if factor.mode == "within":
                    out += _within(part, factor, ds, st, min_neurons, min_spines_ds, min_paired)
                else:
                    out += _between(part, factor, ds, st, min_neurons, min_spines_ds)
    for i, rec in enumerate(out):
        rec["comparison_id"] = i
    return out


def _cross(frame: pd.DataFrame, factor: Factor, present: Sequence[str], min_units: int, min_rows: int) -> List[Dict[str, Any]]:
    out = []
    if factor.name.startswith("species"):
        # human vs mouse restricted to the EM datasets (same modality): H01 vs Minnie65
        sub = frame[frame["dataset"].isin(["h01", "minnie65"]) & (frame["modality"] == "EM")]
        if not {"h01", "minnie65"} <= set(sub["dataset"].unique()):
            return [_record(factor, "h01 vs minnie65", "human", "mouse", None, reason="needs both EM datasets")]
        rows = min_rows
        if factor.stratify is None:
            return _between(sub, factor, "h01 vs minnie65", None, min_units, rows)
        if not sub[factor.stratify].notna().any():
            return [_record(factor, "h01 vs minnie65", "human", "mouse", None, reason="stratifying metadata missing")]
        sets = [set(sub.loc[sub["dataset"] == d, factor.stratify].dropna().astype(str)) for d in ("h01", "minnie65")]
        out = []
        for st in sorted(sets[0] & sets[1]):
            out += _between(sub[sub[factor.stratify].astype(str) == st], factor, "h01 vs minnie65", st, min_units, rows)
        return out or [_record(factor, "h01 vs minnie65", "human", "mouse", None, reason="no stratum present in both datasets")]
    pairs = [p for p in combinations(sorted(present), 2) if set(p) != {"h01", "minnie65"}]  # H01 vs Minnie is the 'species' row
    for a, b in pairs:
        sub = frame[frame["dataset"].isin([a, b])]
        f2 = Factor(**{**factor.__dict__, "levels": (a, b)})
        recs = _between(sub, f2, f"{a} vs {b}", None, min_units, min_rows)
        for r in recs:
            r["warnings"] = r["warnings"] + (["modality differs (EM vs confocal): size metrics depend on resolution"] if "labid" in (a, b) else [])
        out += recs
    return out


def availability_matrix(frame: pd.DataFrame, factors: Sequence[Factor] = FACTORS, **kw) -> pd.DataFrame:
    """Factor x dataset matrix: ``valid`` (>= 1 valid comparison), ``descriptive_only`` (valid but confounded),
    ``invalid:<reason>`` / ``not_defined``. For the design document and the report header."""
    comps = valid_comparisons(frame, factors=factors, **kw)
    rows = {}
    for c in comps:
        key = (c["factor"], c["dataset"])
        cell = "valid" if c["valid"] and c["causal"] else ("descriptive_only" if c["valid"] else f"invalid: {c['reason']}")
        prev = rows.get(key)
        if prev is None or cell.startswith("valid") or (cell == "descriptive_only" and prev != "valid"):
            rows[key] = cell
    mat = pd.Series(rows).unstack(fill_value="not_defined")
    return mat
