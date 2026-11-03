"""Driver: run every valid (dataset, factor, level) comparison and return tidy tables.

Per comparison and metric: the neuron-level permutation test of ``morphology_by_condition`` (between units: labels
permuted between units; within unit: sign flips of the paired differences), the effect (relative difference of the
medians, or absolute for metrics in ``abs_metrics`` / near-zero medians) with a bootstrap interval over UNITS, the minimal
detectable effect (MDE) at the available number of units, and a label:

- ``significant``       q_bh < alpha within the family (dataset, factor, level, stratum; metrics x pairs);
- ``underpowered``      not significant and MDE > the relevance margin (the data cannot exclude a relevant effect);
- ``not_significant``   not significant and MDE <= margin (a relevant effect would have been detected);
- ``not_tested``        too few units for the test of this metric.
``equivalent`` = not significant and the whole interval lies inside +-margin ("same within the margin", never "p > 0.05").
Margin: ``margin_rel`` (0.10) of the reference median for relative effects; for absolute effects ``margins[metric]`` or
``margin_sd`` x robust between-unit SD. MDE (asymptotic, two-sided alpha, ``power``): (z_{1-alpha/2} + z_power) *
sqrt(se_a^2 + se_b^2), se of a median = 1.2533 s / sqrt(n), s = robust SD of the unit values.
Cross-dataset comparisons (species, dataset) are labelled ``descriptive_only``: their ``causal`` flag is False and the
confounders are listed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..spine_generation.evaluation.morphology_by_condition import (
    _perm_omnibus, bh_qvalues, compare_neuron_groups, compare_within_neuron, neuron_medians)
from .factors import FACTORS, Factor, resolve_min_rows, valid_comparisons

FAMILY = ["dataset", "factor", "level", "stratum"]


def _robust_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _mde_abs(se_diff: float, alpha: float, power: float) -> float:
    from scipy.stats import norm

    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se_diff)


def _boot_ci(xa: np.ndarray, xb: Optional[np.ndarray], kind: str, n_boot: int, rng: np.random.Generator):
    """Bootstrap 95 % interval of the effect. ``xb`` None = paired (``xa`` holds the per-unit differences, abs only)."""
    stats = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(xa, len(xa))
        if xb is None:
            stats[i] = np.median(ra)
            continue
        rb = rng.choice(xb, len(xb))
        ma, mb = np.median(ra), np.median(rb)
        stats[i] = ma / mb - 1.0 if kind == "rel" and mb != 0 else ma - mb
    lo, hi = np.nanquantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def _decorate(row: Dict[str, Any], comp: Dict[str, Any], level: str) -> Dict[str, Any]:
    keep = {k: comp[k] for k in ("comparison_id", "dataset", "factor", "mode", "stratum", "causal") if k in comp}
    keep["level"] = level
    keep["confounded_with"] = json.dumps(comp["confounded_with"]) if comp["confounded_with"] else ""
    keep["warnings"] = " | ".join(comp["warnings"])
    keep["interpretation"] = "within-dataset contrast" if comp["causal"] else "descriptive_only (confounded)"
    return {**keep, **{k: v for k, v in row.items() if k not in ("factor",)}}


def run_comparisons(frame: pd.DataFrame, metrics: Sequence[str], *, level: str = "spine", comparisons: Optional[List[Dict[str, Any]]] = None,
                    factors: Sequence[Factor] = FACTORS, min_neurons: int = 10, min_spines: int = 20, n_permutations: int = 10000,
                    seed: int = 0, margin_rel: float = 0.10, margin_sd: float = 0.5, margins: Optional[Dict[str, float]] = None,
                    abs_metrics: Sequence[str] = (), n_boot: int = 500, alpha: float = 0.05, power: float = 0.8,
                    min_paired: int = 5, log=print) -> Dict[str, pd.DataFrame]:
    """Returns ``results`` (one row per comparison x metric), ``omnibus`` (factors with >= 3 levels), ``comparisons``
    (the full list incl. invalid ones with the reason). ``frame``: tidy frame (see datasets.py); ``min_spines`` = min rows of a
    unit (1 for branch-group frames)."""
    comps = comparisons if comparisons is not None else valid_comparisons(
        frame, factors=factors, min_neurons=min_neurons, min_spines=min_spines, min_paired=min_paired)
    margins = margins or {}
    metrics = [m for m in metrics if m in frame.columns and frame[m].notna().any()]
    rng = np.random.default_rng(seed + 7)
    rows: List[Dict[str, Any]] = []
    for comp in comps:
        if not comp["valid"]:
            continue
        sub = frame[frame["dataset"].isin(comp["dataset"].split(" vs "))]
        if comp["stratum"] is not None:
            sub = sub[sub[comp["stratum_column"]].astype(str) == comp["stratum"]]
        col, a, b = comp["column"], comp["a"], comp["b"]
        min_rows = resolve_min_rows(min_spines, comp["dataset"].split(" vs "))
        sd_seed = seed + 31 * int(comp["comparison_id"])
        if comp["mode"] == "within":
            res = compare_within_neuron(sub, col, a, b, metrics, min_spines=min_rows, n_permutations=n_permutations, seed=sd_seed)
            med = neuron_medians(sub[sub[col].isin([a, b])], metrics, ["neuron_id", col], min_rows).pivot(index="neuron_id", columns=col)
        else:
            unit = neuron_medians(sub, metrics, ["neuron_id"], min_rows).merge(
                sub.groupby("neuron_id")[[col]].first(), on="neuron_id")
            res = compare_neuron_groups(unit, col, a, b, metrics, n_permutations=n_permutations, seed=sd_seed)
        for r in res:
            m = r["metric"]
            r = _effect_columns(r, m, comp, sub, med if comp["mode"] == "within" else unit, col, a, b, abs_metrics, margin_rel,
                                margin_sd, margins, n_boot, alpha, power, rng)
            rows.append(_decorate(r, comp, level))
    results = pd.DataFrame(rows)
    if len(results) and "p" in results.columns:
        results["q_bh"] = np.nan
        for _, idx in results.groupby(FAMILY, dropna=False).groups.items():
            results.loc[idx, "q_bh"] = bh_qvalues(results.loc[idx, "p"].to_numpy(float))
        results["significant"] = results["q_bh"] < alpha
        results["label"] = [_label(r, alpha) for r in results.to_dict("records")]
        results["equivalent"] = [(not r["significant"]) and bool(r.get("within_margin", False)) for r in results.to_dict("records")]
    omni = _omnibus(frame, metrics, comps, min_spines, n_permutations, seed, level)
    log(f"[global] {len(results)} tests, {int(results['significant'].sum()) if 'significant' in results else 0} significant; "
        f"{sum(c['valid'] for c in comps)} of {len(comps)} comparisons valid")
    return {"results": results, "omnibus": omni, "comparisons": pd.DataFrame(comps)}


def _effect_columns(r, m, comp, sub, tab, col, a, b, abs_metrics, margin_rel, margin_sd, margins, n_boot, alpha, power, rng):
    r = dict(r)
    if "p" not in r:
        return r
    if comp["mode"] == "within":
        va, vb = tab[(m, a)].dropna().to_numpy(float), tab[(m, b)].dropna().to_numpy(float)
        idx = tab[(m, a)].dropna().index.intersection(tab[(m, b)].dropna().index)
        va, vb = tab.loc[idx, (m, a)].to_numpy(float), tab.loc[idx, (m, b)].to_numpy(float)
        d = va - vb
        ref = float(np.median(vb))
        kind = "abs" if (m in abs_metrics or abs(ref) < 1e-12) else "rel"
        s = _robust_sd(d)
        mde = _mde_abs(1.2533 * s / np.sqrt(len(d)), alpha, power) if np.isfinite(s) else float("nan")
        lo, hi = _boot_ci(d, None, "abs", n_boot, rng)
        eff = float(np.median(d))
        if kind == "rel":
            eff, lo, hi, mde = eff / abs(ref), lo / abs(ref), hi / abs(ref), mde / abs(ref)
        margin = margin_rel if kind == "rel" else margins.get(m, margin_sd * s)
        sd_ref = s
    else:
        t = tab[tab[col].isin([a, b])]
        xa = t.loc[t[col] == a, m].dropna().to_numpy(float)
        xb = t.loc[t[col] == b, m].dropna().to_numpy(float)
        ref = float(np.median(xb))
        kind = "abs" if (m in abs_metrics or abs(ref) < 1e-12) else "rel"
        sa, sb = _robust_sd(xa), _robust_sd(xb)
        sp = float(np.sqrt(np.nanmean([sa ** 2, sb ** 2]))) if np.isfinite(sa) or np.isfinite(sb) else float("nan")
        se = float(np.sqrt((1.2533 * sa) ** 2 / len(xa) + (1.2533 * sb) ** 2 / len(xb))) if np.isfinite(sa) and np.isfinite(sb) else float("nan")
        mde = _mde_abs(se, alpha, power) if np.isfinite(se) else float("nan")
        lo, hi = _boot_ci(xa, xb, kind, n_boot, rng)
        eff = float(np.median(xa) - np.median(xb))
        if kind == "rel":
            eff = float(np.median(xa) / ref - 1.0)
            mde = mde / abs(ref)
        margin = margin_rel if kind == "rel" else margins.get(m, margin_sd * sp)
        sd_ref = sp
    r.update({"effect_kind": kind, "effect": eff, "ci_low": lo, "ci_high": hi, "mde": mde, "margin": float(margin),
              "unit_robust_sd": float(sd_ref), "within_margin": bool(np.isfinite(margin) and lo > -margin and hi < margin)})
    return r


def _label(r: Dict[str, Any], alpha: float) -> str:
    p, q = r.get("p"), r.get("q_bh")
    if p is None or not np.isfinite(p):
        return "not_tested"
    if np.isfinite(q) and q < alpha:
        return "significant"
    mde, margin = r.get("mde"), r.get("margin")
    if mde is None or margin is None or not np.isfinite(mde) or not np.isfinite(margin) or mde > margin:
        return "underpowered"
    return "not_significant"


def _omnibus(frame, metrics, comps, min_spines, n_permutations, seed, level) -> pd.DataFrame:
    rows = []
    done = set()
    for comp in comps:
        if not comp["valid"] or comp["mode"] != "between" or comp["factor"] != "cell_type":
            continue
        key = (comp["dataset"], comp["factor"])
        if key in done:
            continue
        done.add(key)
        levels = sorted({c[k] for c in comps if (c["dataset"], c["factor"]) == key and c["valid"] for k in ("a", "b")})
        if len(levels) < 3:
            continue
        sub = frame[frame["dataset"] == comp["dataset"]]
        unit = neuron_medians(sub, metrics, ["neuron_id"], resolve_min_rows(min_spines, comp["dataset"])).merge(sub.groupby("neuron_id")[["cell_type"]].first(), on="neuron_id")
        unit = unit[unit["cell_type"].isin(levels)]
        rng = np.random.default_rng(seed + 1)
        for m in metrics:
            x = unit[m].to_numpy(float)
            ok = np.isfinite(x)
            if ok.sum() >= 10:
                rows.append({"dataset": comp["dataset"], "factor": "cell_type", "level": level, "metric": m, "n_types": len(levels),
                             "n_neurons": int(ok.sum()), "p": _perm_omnibus(x[ok], unit["cell_type"].to_numpy()[ok], n_permutations, rng)})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = np.nan
        for _, idx in out.groupby("dataset").groups.items():
            out.loc[idx, "q_bh"] = bh_qvalues(out.loc[idx, "p"].to_numpy(float))
        out["significant"] = out["q_bh"] < 0.05
    return out


def to_records(table: pd.DataFrame) -> List[Dict[str, Any]]:
    """JSON-safe records (NaN / inf -> None, numpy scalars -> Python)."""
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, np.ndarray):
            return [clean(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    return [{k: clean(v) for k, v in rec.items()} for rec in table.to_dict("records")]


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """``<name>.csv`` and ``<name>.json`` (list of records) for every table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        if t is None or not len(t):
            continue
        t.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.json").write_text(json.dumps(to_records(t), ensure_ascii=False), encoding="utf-8")
