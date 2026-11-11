"""Section 4 of the global report: do GENERATED dendrites follow the rules found in the real data?

Adapters turn a generated set into the tidy frames of ``datasets.py`` (``dataset = "generated"``) next to its REAL reference
(``dataset = "real"``), and the same engine runs on both:

- **T and P (branches).** A ``generate_tp.py`` folder (``trunks.npz`` + ``branches.parquet`` + ``placements.parquet``): every
  generated branch copies the conditions and the length of a SOURCE branch (column ``source_branch`` =
  ``<dataset>/<neuron>/<limb:03d>/<merged>``). The real reference is the list of these source branches (duplicates kept), as in
  ``evaluate_assembled_branches.py``. Both sides get the SAME per-branch metrics from their trunk records
  (``branch_assembly.evaluate.trunk_functionals`` for the rms bending and the tortuosity, the radius profile for the median,
  proximal and distal radius with the definitions of the branch preprocessing) and go through the same
  ``datasets.tidy_branch_groups`` (= ``branch_statistics.group_table``): one row per (source neuron, compartment). A generated
  branch inherits the eligibility flags of its source branch (P metrics only where the real count is known). Per-spine statistics
  of the generated branches see the same measurement as the real ones: every generated spine is kept with the positioned share
  n_with_position / n_spines_in_range of its source branch (``evaluate.position_keep_mask``); counts and densities use all
  spines. The real count is ``n_spines_in_range`` (what P0 models). The ellipse ratio (T0 has no ellipse) and the mark
  correlations (no spine meshes on a T/P set) are not compared.
- **S (spines).** The table ``generated_metrics.parquet`` of the conditional report (``conditional_generation_report.py``): one
  row per generated spine with the 11 morphometrics, ``value`` (the requested condition), ``label`` (the set, e.g. a guidance
  weight), ``block`` and ``condition_<block>`` (written since 2026-11-11; older tables: ``block`` from ``run_info.json``).
  A conditional MoGen sample has NO source neuron (the condition is a type, not a neuron): when the table has no ``neuron_id``
  the generated spines of one (set, condition) are split at random into pseudo-units of ``unit_size`` spines (independent
  draws, so the blocks are independent units with the generator's own spread). The real reference is the stage-8 table of the
  source dataset (optionally only the neurons of one split).

Checks (``generated_tables``):

- **4a, reproduction of differences.** ``significance.run_comparisons`` on the real reference and on the generated set with the
  same factors (``cell_type_binary``, ``cell_type``, ``compartment`` - where both sides have values) and
  ``consistency.consistency`` with ``datasets = ("real", "generated")``: per (factor, pair, metric) the real effect and the
  generated effect with bootstrap intervals over UNITS, sign agreement, the difference generated - real with its interval and
  p (independent resampling of the two sides: conservative when they share source neurons). Verdicts: ``consistent``
  (reproduced: significant in both, same sign, difference not significant), ``same_sign_different_size``, ``opposite``,
  ``real_only`` (not reproduced), ``generated_only`` (spurious), ``neither``.
- **4b, real vs generated within a type.** Per stratum (all units / one binary type / one detailed type / one compartment) and
  metric: d = median of the generated unit medians relative to (or minus) the median of the real unit medians; 95 % interval
  and SE from a bootstrap over UNITS (paired over the source neurons when every generated unit is a real unit, independent
  otherwise); p by percentile inversion; margin = noise floor = 95 % quantile of |d| between two disjoint random halves of the
  REAL units; TOST p = Phi((|d| - margin) / SE); BH over the metrics of one (set, level, stratum); verdict as in
  ``conditional_tests`` (``differs`` / ``equivalent`` / ``indistinguishable_equivalence_not_shown``); ``min_margin`` =
  |d| + 1.645 SE = the smallest margin for which equivalence would be shown.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..metadata import cell_type_binary, dataset_kind
from ..spine_generation.evaluation.conditional_tests import _ci, _norm_cdf, _verdict_b
from ..spine_generation.evaluation.morphology_by_condition import bh_qvalues
from .consistency import consistency
from .datasets import tidy_branch_groups
from .factors import FACTORS, Factor
from .significance import run_comparisons, unit_medians

REAL, GENERATED = "real", "generated"
REPRO_FACTORS = ("cell_type_binary", "cell_type", "compartment")
CONDITION_COLUMNS = ("cell_type_binary", "cell_type", "compartment", "species")
# branch metrics a T/P set can carry (no ellipse in T0, no spine meshes for the marks)
TP_METRICS = ("radius_median_nm", "radius_ratio", "tortuosity", "kappa_rms", "length_nm", "density_per_um", "spines_per_branch",
              "share_empty", "pcf_0_500", "pcf_500_1000", "pcf_1000_2000", "pcf_2000_4000", "gap_cv", "nn_along_nm")


def generated_factors(names: Sequence[str] = REPRO_FACTORS) -> Tuple[Factor, ...]:
    """The within-dataset factors of ``factors.FACTORS`` declared for the two sides of section 4 (same levels, same mode)."""
    return tuple(dataclasses.replace(f, datasets=(REAL, GENERATED), warnings=()) for f in FACTORS if f.name in names)


# ---------------------------------------------------------------------------
# T and P: generate_tp.py set + its source branches
# ---------------------------------------------------------------------------


def parse_branch_key(key: str) -> Tuple[str, str, int, int]:
    """``<dataset>/<neuron>/<limb:03d>/<merged>`` -> (dataset, neuron, limb, merged)."""
    ds, neuron, limb, merged = str(key).split("/")
    return ds, neuron, int(limb), int(merged)


def trunk_branch_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    """Per-branch trunk metrics of one trunk record (generated or real), with the definitions of the branch preprocessing:
    median radius, mean radius over the first / last 10 % of the length, tortuosity = length / chord, rms bending (1/nm)."""
    from ..branch_assembly.evaluate import trunk_functionals

    s = np.asarray(record["s"], float)
    r = np.asarray(record["radius"], float)
    length = float(s[-1])
    f = trunk_functionals(record)
    prox, dist = r[s <= 0.1 * length], r[s >= 0.9 * length]
    return {"length_nm": length, "radius_median_nm": float(np.median(r)),
            "radius_proximal_nm": float(prox.mean()) if len(prox) else float(r[0]),
            "radius_distal_nm": float(dist.mean()) if len(dist) else float(r[-1]),
            "tortuosity": float(f["tortuosity"]), "kappa_rms": float(f["kappa_rms_per_nm"])}


def tp_branch_tables(branches: pd.DataFrame, trunks: Sequence[Dict[str, Any]], placements: Dict[int, pd.DataFrame],
                     source_rows: Dict[str, Dict[str, Any]], source_trunks: Dict[str, Dict[str, Any]],
                     source_spines: Dict[str, pd.DataFrame], *, thin: bool = True, seed: int = 0) -> Dict[str, Any]:
    """Raw branch / spine tables (``branches.parquet`` / ``branch_spines.parquet`` columns) of the generated set and of its
    paired real reference. Row i of both = request i (``merged_branch_id`` = request index, so a source branch drawn twice
    stays two branches); neuron and limb are those of the source branch (its metadata and compartment)."""
    from ..branch_assembly.evaluate import position_keep_fraction, position_keep_mask

    rng = np.random.default_rng(seed + 1)
    gb, rb, gs, rs = [], [], [], []
    kinds, keep, skipped = set(), [], 0
    src_of = dict(zip(branches["branch_index"].astype(int), branches["source_branch"].astype(str)))
    for i, t in enumerate(trunks):
        key = src_of.get(i)
        row = source_rows.get(key) if key is not None else None
        if row is None or key not in source_trunks:
            skipped += 1
            continue
        ds, neuron, limb, _ = parse_branch_key(key)
        kinds.add(dataset_kind(ds))
        flags = {"train_eligible_T": bool(row.get("train_eligible_T", True)), "train_eligible_P": bool(row.get("train_eligible_P", False))}
        base = {"neuron_id": neuron, "limb_id": limb, "merged_branch_id": i, "status": "ok", **flags}
        p = placements.get(i)
        s = p["s"].to_numpy(float) if p is not None else np.zeros(0)
        gb.append({**base, **trunk_branch_metrics(t), "n_spines": int(len(s))})
        if thin:
            f = position_keep_fraction(row.get("n_spines_in_range"), row.get("n_with_position"))
            keep.append(f)
            s = s[position_keep_mask(len(s), f, rng)]
        length = float(np.asarray(t["s"], float)[-1])
        gs.append(pd.DataFrame({"neuron_id": neuron, "limb_id": limb, "merged_branch_id": i, "s": s, "u": s / max(length, 1e-9)}))
        n_real = row.get("n_spines_in_range")
        rb.append({**base, **trunk_branch_metrics(source_trunks[key]), "n_spines": float(n_real) if n_real is not None and pd.notna(n_real) else np.nan})
        sp = source_spines.get(key)
        if sp is not None and len(sp):
            rs.append(pd.DataFrame({"neuron_id": neuron, "limb_id": limb, "merged_branch_id": i, "s": sp["s"].to_numpy(float), "u": sp["u"].to_numpy(float)}))
    empty = pd.DataFrame(columns=["neuron_id", "limb_id", "merged_branch_id", "s", "u"])
    if len(kinds) > 1:
        raise ValueError(f"a generated set mixes source datasets {sorted(kinds)}")
    return {"generated_branches": pd.DataFrame(gb), "real_branches": pd.DataFrame(rb),
            "generated_spines": pd.concat(gs, ignore_index=True) if gs else empty, "real_spines": pd.concat(rs, ignore_index=True) if rs else empty,
            "kind": next(iter(kinds)) if kinds else None, "n_skipped": skipped, "mean_keep_fraction": float(np.mean(keep)) if keep else float("nan")}


def tidy_generated_tp(raw: Dict[str, Any], meta: Dict[str, Any], *, label: str, min_branches: int = 5, min_pairs: int = 30) -> pd.DataFrame:
    """Tidy branch-group frames of both sides (``dataset`` real / generated, ``set`` = label; ``unit_id`` = the SOURCE neuron on
    both sides, so the sides are paired by neuron)."""
    parts = []
    for side, b, s in ((GENERATED, raw["generated_branches"], raw["generated_spines"]), (REAL, raw["real_branches"], raw["real_spines"])):
        if not len(b):
            continue
        g = tidy_branch_groups(b, s, raw["kind"], meta, min_branches=min_branches, min_pairs=min_pairs, exclude_groups=())
        g["source_dataset"] = raw["kind"]
        g["dataset"] = side
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out["set"] = label
    return out


def load_generated_tp(tp_dir: Path, metadata_root: Path, *, label: Optional[str] = None, compartment_table=None,
                      min_branches: int = 5, min_pairs: int = 30, thin: bool = True, seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """A ``generate_tp.py`` folder -> (tidy frame of both sides, info). The source tables and the geometry pack come from the P0
    run of the set (``generation_info.json`` -> ``p0_run`` -> ``data.branches_root`` / ``data.pack_dir``: local disk, no NAS),
    exactly as ``evaluate_assembled_branches.py``. Windows only in practice (needs the pack). ``compartment_table``: a path or
    ``{dataset kind: path}`` (the entry of the set's source dataset is used)."""
    from ..branch_assembly import load_tp_set
    from ..branch_data import key_series, prepare_training_data, trunk_record
    from ..metadata import load_metadata_tables
    from ..spine_generation.experiment.config import resolve_path
    from ..spine_generation.experiment.run import open_run

    tp_dir = Path(tp_dir)
    label = label or tp_dir.name
    gi = json.loads((tp_dir / "generation_info.json").read_text(encoding="utf-8"))
    d = open_run(Path(gi["p0_run"])).load_config()["data"]
    table, spines, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]),
                                                splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    branches, trunks, placements = load_tp_set(tp_dir)
    needed = set(branches["source_branch"].astype(str))
    keys = key_series(table)
    source_rows = {k: r for k, r in zip(keys, table.to_dict("records")) if k in needed}
    source_trunks = {k: trunk_record(pack.get(k)) for k in needed if k in pack}
    sp = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].fillna(False).astype(bool)] if len(spines) else spines
    sp = sp[key_series(sp).isin(needed)] if len(sp) else sp
    source_spines = {k: g for k, g in sp.groupby(key_series(sp))} if len(sp) else {}
    raw = tp_branch_tables(branches, trunks, placements, source_rows, source_trunks, source_spines, thin=thin, seed=seed)
    if isinstance(compartment_table, dict):
        compartment_table = compartment_table.get(raw["kind"])
    meta = load_metadata_tables(Path(metadata_root), compartment_table)
    frame = tidy_generated_tp(raw, meta, label=label, min_branches=min_branches, min_pairs=min_pairs)
    info = {"set": label, "level": "branch", "source": str(tp_dir), "source_dataset": raw["kind"], "mode": gi.get("mode"), "split": gi.get("split"),
            "p0_run": gi.get("p0_run"), "t0_run": gi.get("t0_run"), "bundle": gi.get("bundle"), "counts": gi.get("counts"),
            "n_branches": int(len(trunks)), "n_skipped": raw["n_skipped"], "position_thinning": bool(thin), "mean_keep_fraction": raw["mean_keep_fraction"],
            "unit": "source neuron", "min_branches": min_branches}
    return frame, info


# ---------------------------------------------------------------------------
# S: the generated spine table of the conditional report
# ---------------------------------------------------------------------------


def tidy_generated_spines(table: pd.DataFrame, metrics: Sequence[str], *, kind: str = "minnie65", block: Optional[str] = None,
                          label: Optional[str] = None, unit_size: int = 20, only_valid: bool = False, seed: int = 0) -> pd.DataFrame:
    """``generated_metrics.parquet`` of the conditional report -> tidy spine frame (``dataset = generated``, ``set`` = label).

    Condition of a row: ``condition_<column>`` when present, else ``value`` for ``block``. Rows of ``value == unconditional``
    are dropped (no type to compare). Unit: ``neuron_id`` of the table when present (``<kind>:<id>``, as the real frame), else
    random pseudo-units of ``unit_size`` spines within one (set, condition)."""
    t = table.copy()
    if label is not None and "label" in t.columns:
        t = t[t["label"].astype(str) == str(label)]
    if "value" in t.columns:
        t = t[t["value"].astype(str) != "unconditional"]
    if only_valid and "valid" in t.columns:
        t = t[t["valid"].fillna(False).astype(bool)]
    t = t.reset_index(drop=True)
    kind = dataset_kind(kind)
    if block is None and "block" in t.columns and t["block"].notna().any():
        block = str(t["block"].dropna().iloc[0])
    out = pd.DataFrame(index=t.index)
    for col in CONDITION_COLUMNS:
        if f"condition_{col}" in t.columns:
            out[col] = t[f"condition_{col}"].astype(object).where(t[f"condition_{col}"].notna(), None)
        elif col == block and "value" in t.columns:
            out[col] = t["value"].astype(object)
        else:
            out[col] = None
    missing = out["cell_type_binary"].isna() & out["cell_type"].notna()
    out.loc[missing, "cell_type_binary"] = out.loc[missing, "cell_type"].map(cell_type_binary)
    lab = t["label"].astype(str) if "label" in t.columns else pd.Series(str(label or "default"), index=t.index)
    if "neuron_id" in t.columns and t["neuron_id"].notna().all():
        ids = t["neuron_id"].astype(str)
        unit = ids.where(ids.str.contains(":"), kind + ":" + ids)
        mode = "source neuron"
    else:
        rng = np.random.default_rng(seed + 3)
        cond = out[["cell_type_binary", "cell_type", "compartment"]].astype(str).agg("|".join, axis=1)
        unit = pd.Series("", index=t.index, dtype=object)
        for key, idx in pd.Series(t.index, index=t.index).groupby([lab, cond]).groups.items():
            idx = np.asarray(list(idx))
            n_units = max(1, len(idx) // max(int(unit_size), 1))
            order = rng.permutation(len(idx))
            for rank, j in enumerate(order):
                unit[idx[j]] = f"generated:{key[0]}:{key[1]}:{rank % n_units:04d}"
        mode = f"pseudo-unit of {unit_size} spines"
    out["dataset"] = GENERATED
    out["unit_id"] = unit.to_numpy()
    out["neuron_id"] = out["unit_id"]
    out["source_dataset"] = kind
    out["unit_mode"] = mode
    out["set"] = lab.to_numpy()
    out["spine_id"] = t["sample_id"].astype(str).to_numpy() if "sample_id" in t.columns else np.arange(len(t)).astype(str)
    out["limb_id"] = np.nan
    out["branch_id"] = np.nan
    for m in metrics:
        out[m] = pd.to_numeric(t[m], errors="coerce") if m in t.columns else np.nan
    return out.reset_index(drop=True)


def load_generated_spines(path: Path, metrics: Sequence[str], *, kind: str = "minnie65", unit_size: int = 20, only_valid: bool = False,
                          seed: int = 0) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """``generated_metrics.parquet`` (or the conditional-report folder holding it) -> ({set label: tidy frame}, info).
    The block comes from the table (``block``) or from ``run_info.json`` next to it."""
    path = Path(path)
    table_path = path / "generated_metrics.parquet" if path.is_dir() else path
    table = pd.read_parquet(table_path)
    block = None
    info_path = table_path.parent / "run_info.json"
    if "block" not in table.columns and info_path.exists():
        block = json.loads(info_path.read_text(encoding="utf-8")).get("block")
    labels = sorted(table["label"].astype(str).unique()) if "label" in table.columns else ["default"]
    sets = {lab: tidy_generated_spines(table, metrics, kind=kind, block=block, label=lab if "label" in table.columns else None,
                                       unit_size=unit_size, only_valid=only_valid, seed=seed) for lab in labels}
    sets = {k: v for k, v in sets.items() if len(v)}
    info = {"source": str(table_path), "block": block or (str(table["block"].dropna().iloc[0]) if "block" in table.columns and table["block"].notna().any() else None),
            "labels": list(sets)}
    return sets, info


def real_reference_spines(spines: pd.DataFrame, kind: str, neurons: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Real spines of one dataset as the reference side (``dataset = real``); ``neurons``: raw ids to keep (a split)."""
    kind = dataset_kind(kind)
    r = spines[spines["dataset"] == kind].copy()
    if neurons is not None:
        keep = {f"{kind}:{n}" for n in neurons}
        r = r[r["unit_id"].isin(keep)]
    r["source_dataset"] = kind
    r["dataset"] = REAL
    return r.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4b: real vs generated within a stratum (equivalence)
# ---------------------------------------------------------------------------


def _effect(xg: np.ndarray, xr: np.ndarray, kind: str) -> float:
    if not len(xg) or not len(xr):
        return float("nan")
    mg, mr = np.median(xg), np.median(xr)
    return float(mg / mr - 1.0) if kind == "rel" and mr != 0 else float(mg - mr)


def equivalence(frame: pd.DataFrame, metrics: Sequence[str], *, level: str, set_label: str, min_rows=20,
                strata: Sequence[str] = ("cell_type_binary", "cell_type", "compartment"), include_all: bool = True,
                abs_metrics: Sequence[str] = (), min_units: int = 5, n_boot: int = 2000, n_halves: int = 1000, alpha: float = 0.05,
                seed: int = 0) -> pd.DataFrame:
    """One row per (stratum, metric): see the module docstring (4b). ``frame``: both sides (``dataset`` real / generated)."""
    rows: List[Dict[str, Any]] = []
    real, gen = frame[frame["dataset"] == REAL], frame[frame["dataset"] == GENERATED]
    plan: List[Tuple[Optional[str], Optional[str]]] = [(None, None)] if include_all else []
    for col in strata:
        if col in frame.columns:
            common = set(gen[col].dropna().astype(str)) & set(real[col].dropna().astype(str))
            plan += [(col, v) for v in sorted(common)]
    rng = np.random.default_rng(seed + 211)
    for col, value in plan:
        rs = real if col is None else real[real[col].astype(str) == value]
        gs = gen if col is None else gen[gen[col].astype(str) == value]
        ur = unit_medians(rs, metrics, min_rows)
        ug = unit_medians(gs, metrics, min_rows)
        common_units = sorted(set(ug["neuron_id"]) & set(ur["neuron_id"]))
        paired = len(ug) > 0 and len(common_units) == len(ug)
        ur_i, ug_i = ur.set_index("neuron_id"), ug.set_index("neuron_id")
        part = []
        for m in metrics:
            row = {"set": set_label, "level": level, "stratum_column": col or "all", "stratum": value or "all", "metric": m,
                   "pairing": "source neuron" if paired else "independent"}
            if paired:
                both = ur_i.loc[common_units, [m]].join(ug_i.loc[common_units, [m]], rsuffix="_g").dropna()
                xr, xg = both[m].to_numpy(float), both[f"{m}_g"].to_numpy(float)
                xr_all = ur[m].dropna().to_numpy(float)
            else:
                xr, xg = ur[m].dropna().to_numpy(float), ug[m].dropna().to_numpy(float)
                xr_all = xr
            row.update({"n_units_real": int(len(xr_all)), "n_units_generated": int(len(xg))})
            if len(xg) < min_units or len(xr_all) < max(min_units, 4):
                row["verdict"] = "not_tested"
                part.append(row)
                continue
            ref = float(np.median(xr))
            kind = "abs" if (m in abs_metrics or abs(ref) < 1e-12) else "rel"
            d = _effect(xg, xr, kind)
            boot = np.empty(n_boot)
            for b in range(n_boot):
                if paired:
                    idx = rng.integers(0, len(xr), len(xr))
                    boot[b] = _effect(xg[idx], xr[idx], kind)
                else:
                    boot[b] = _effect(xg[rng.integers(0, len(xg), len(xg))], xr[rng.integers(0, len(xr), len(xr))], kind)
            boot = boot[np.isfinite(boot)]
            half = len(xr_all) // 2
            floor = np.empty(n_halves)
            for h in range(n_halves):
                perm = rng.permutation(len(xr_all))
                floor[h] = _effect(xr_all[perm[:half]], xr_all[perm[half:2 * half]], kind)
            margin = float(np.nanquantile(np.abs(floor), 0.95))
            se = float(np.std(boot, ddof=1)) if len(boot) > 1 else float("nan")
            lo, hi = _ci(boot) if len(boot) else (float("nan"), float("nan"))
            B = len(boot)
            p_boot = min(1.0, 2 * min((1 + int((boot <= 0).sum())) / (1 + B), (1 + int((boot >= 0).sum())) / (1 + B))) if B else float("nan")
            row.update({"effect_kind": kind, "median_real": ref, "median_generated": float(np.median(xg)), "effect": d, "ci_low": lo, "ci_high": hi,
                        "se": se, "p_boot": p_boot, "margin": margin, "margin_source": "noise floor: 95 % of |effect| between halves of the real units",
                        # SE 0 (e.g. a length copied from the request): the TOST decision is exact
                        "p_tost": float(_norm_cdf((abs(d) - margin) / se)) if se > 0 else (0.0 if abs(d) < margin else 1.0) if np.isfinite(se) else float("nan"),
                        "min_margin": float(abs(d) + 1.645 * se) if np.isfinite(se) else float("nan"),
                        "within_floor": bool(abs(d) <= margin)})
            row["min_margin_over_margin"] = row["min_margin"] / margin if margin > 0 else float("nan")
            part.append(row)
        tested = [r for r in part if r.get("verdict") != "not_tested"]
        if tested:
            q = bh_qvalues(np.array([r["p_boot"] for r in tested], float))
            qt = bh_qvalues(np.array([r["p_tost"] for r in tested], float))
            for r, a, b in zip(tested, q, qt):
                r["q_boot"], r["q_tost"] = float(a), float(b)
                r["verdict"] = _verdict_b(float(a), float(b))
        rows += part
    return pd.DataFrame(rows)


def equivalence_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or not len(table):
        return pd.DataFrame()
    keys = ["set", "level", "stratum_column", "stratum"] + (["module"] if "module" in table.columns else [])
    out = []
    for key, g in table.groupby(keys, dropna=False, sort=False):
        rec = dict(zip(keys, key))
        v = g["verdict"].value_counts()
        rec.update({"n_metrics": int(len(g)), "n_tested": int((g["verdict"] != "not_tested").sum()), "n_equivalent": int(v.get("equivalent", 0)),
                    "n_differs": int(v.get("differs", 0)), "n_not_shown": int(v.get("indistinguishable_equivalence_not_shown", 0)),
                    "n_units_real": int(g["n_units_real"].max()), "n_units_generated": int(g["n_units_generated"].max())})
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# all section-4 tables of one generated set
# ---------------------------------------------------------------------------


def reproduction_summary(pairs: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Adds the share of the significant REAL effects reproduced with the same sign and the number of spurious effects."""
    if summary is None or not len(summary):
        return pd.DataFrame()
    s = summary.copy()
    s["n_reproduced_same_sign"] = [int(((pairs["set"] == r["set"]) & (pairs["level"] == r["level"]) & (pairs["factor"] == r["factor"]) & (pairs["a"] == r["a"])
                                        & (pairs["b"] == r["b"]) & pairs["present_real"] & pairs["present_generated"] & pairs["same_sign"]).sum())
                                   for r in s.to_dict("records")]
    s["share_reproduced"] = [r["n_reproduced_same_sign"] / r["n_present_real"] if r["n_present_real"] else np.nan for r in s.to_dict("records")]
    s["n_spurious"] = s["n_generated_only"]
    return s


def generated_tables(frame: pd.DataFrame, metrics: Sequence[str], *, level: str, set_label: str, modules: Optional[Dict[str, Sequence[str]]] = None,
                     abs_metrics: Sequence[str] = (), min_neurons: int = 5, min_rows=20, n_permutations: int = 10000, n_boot: int = 500,
                     n_boot_difference: int = 2000, n_halves: int = 1000, strata: Sequence[str] = ("cell_type_binary", "cell_type", "compartment"),
                     include_all: bool = True, seed: int = 0, log=print) -> Dict[str, pd.DataFrame]:
    """4a + 4b for ONE set. ``frame``: both sides; ``level``: spine / branch; ``modules``: {level name: metrics} for the
    reproduction summary per module (branch: trunk / placement), default one group named ``level``."""
    metrics = [m for m in metrics if m in frame.columns and frame.loc[frame["dataset"] == GENERATED, m].notna().any()
               and frame.loc[frame["dataset"] == REAL, m].notna().any()]
    factors = generated_factors()
    res = run_comparisons(frame, metrics, level=level, factors=factors, min_neurons=min_neurons, min_spines=min_rows, n_permutations=n_permutations,
                          seed=seed, abs_metrics=abs_metrics, n_boot=n_boot, log=log)
    for k in ("results", "comparisons", "omnibus"):
        if len(res[k]):
            res[k].insert(0, "set", set_label)
    pairs, summ = [], []
    for name, mm in (modules or {level: metrics}).items():
        mm = [m for m in metrics if m in mm]
        if not mm:
            continue
        cons = consistency(frame, mm, level=name, factors=REPRO_FACTORS, datasets=(REAL, GENERATED), results=res["results"], min_neurons=min_neurons,
                           min_spines=min_rows, abs_metrics=abs_metrics, n_boot=n_boot_difference, seed=seed, factor_defs=factors)
        for k, lst in (("pairs", pairs), ("summary", summ)):
            if len(cons[k]):
                lst.append(cons[k].assign(set=set_label))
    pairs_t = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    summ_t = reproduction_summary(pairs_t, pd.concat(summ, ignore_index=True)) if summ else pd.DataFrame()
    eq = equivalence(frame, metrics, level=level, set_label=set_label, min_rows=min_rows, strata=strata, include_all=include_all, abs_metrics=abs_metrics,
                     min_units=min_neurons, n_boot=n_boot_difference, n_halves=n_halves, seed=seed)
    if modules and len(eq):
        mod_of = {m: name for name, mm in modules.items() for m in mm}
        eq["module"] = eq["metric"].map(mod_of)
    log(f"[global/generated] {set_label} ({level}): {len(res['results'])} tests, reproduction rows {len(pairs_t)}, equivalence rows {len(eq)}")
    return {"results": res["results"], "comparisons": res["comparisons"], "reproduction": pairs_t, "reproduction_summary": summ_t,
            "equivalence": eq, "equivalence_summary": equivalence_summary(eq)}


def set_inventory(frame: pd.DataFrame, info: Dict[str, Any], level: str) -> List[Dict[str, Any]]:
    """Rows of ``generated_sets``: one per side."""
    out = []
    for side in (REAL, GENERATED):
        f = frame[frame["dataset"] == side]
        out.append({**{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in info.items()}, "level": level, "side": side,
                    "n_units": int(f["unit_id"].nunique()) if len(f) else 0, "n_rows": int(len(f)),
                    "n_inhibitory_units": int(f.loc[f["cell_type_binary"] == "inhibitory", "unit_id"].nunique()) if len(f) else 0})
    return out


def merge_tables(per_set: Sequence[Dict[str, pd.DataFrame]], suffix: str) -> Dict[str, pd.DataFrame]:
    """{``generated_<name>_<suffix>``: concatenation over the sets}."""
    out: Dict[str, pd.DataFrame] = {}
    for name in ("results", "comparisons", "reproduction", "reproduction_summary", "equivalence", "equivalence_summary"):
        parts = [t[name] for t in per_set if name in t and t[name] is not None and len(t[name])]
        if parts:
            df = pd.concat(parts, ignore_index=True)
            for col in df.columns:   # list cells (comparisons) -> JSON text, CSV-safe
                if df[col].map(lambda v: isinstance(v, (list, tuple))).any():
                    df[col] = df[col].map(lambda v: json.dumps(list(v)) if isinstance(v, (list, tuple)) else v)
            out[f"generated_{name}_{suffix}"] = df
    return out


__all__ = ["GENERATED", "REAL", "REPRO_FACTORS", "TP_METRICS", "equivalence", "equivalence_summary", "generated_factors", "generated_tables",
           "load_generated_spines", "load_generated_tp", "merge_tables", "real_reference_spines", "set_inventory", "tidy_generated_spines",
           "tidy_generated_tp", "tp_branch_tables", "trunk_branch_metrics"]
