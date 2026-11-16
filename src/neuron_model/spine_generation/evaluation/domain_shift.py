"""Decomposition of the shift between an EM dataset and a confocal one with the confocal forward model.

Three sets of spines, all measured by the SAME procedure (meshes in the generator layout -> 11 morphometrics, base by
``DownFacingCapFinder``; make_real_controls.py writes them): the EM reference ``R`` (Minnie65 meshes as they are, control
``asis``), the EM spines through the imaging model ``C`` (control ``confocal``, one set per parameter choice) and the confocal
target ``T`` (LabID healthy spines, control ``asis`` on the LabID manifest). For every metric, on the log scale,

    total  = med log T - med log R        (everything that differs between the domains)
    optics = med log C - med log R        (what the imaging alone does to EM spines)
    rest   = med log T - med log C        (what remains: biology - brain region, age, cell types - and model error)

with intervals from a cluster bootstrap: units (source neurons of R / C, branch folders of T) are resampled with replacement
on each side independently and all spines of a drawn unit enter. ``share`` = optics / total when both have the same sign.
Base-dependent metrics (base found by the heuristic on very different surfaces) are flagged and left out of the summary.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

BASE_INDEPENDENT = ("Volume", "ConvexHullVolume", "ConvexHullRatio")


def _unit_groups(values: np.ndarray, units: np.ndarray):
    order = np.argsort(units, kind="stable")
    u, start = np.unique(units[order], return_index=True)
    return [values[order][a:b] for a, b in zip(start, list(start[1:]) + [len(order)])]


def _boot_medians(groups, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    k = len(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        out[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return out


def log_values(frame: pd.DataFrame, metric: str, unit_col: str = "unit") -> Optional[tuple]:
    x = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & (x > 0)
    if ok.sum() < 5:
        return None
    return np.log(x[ok]), frame[unit_col].astype(str).to_numpy()[ok]


def shift_table(sets: Dict[str, pd.DataFrame], reference: str, target: str, metrics: Sequence[str], *, n_boot: int = 2000,
                seed: int = 0, unit_col: str = "unit") -> pd.DataFrame:
    """One row per (imaged set, metric): total / optics / rest with 95 % cluster-bootstrap intervals and the explained share.
    ``sets[name]`` has one row per spine with ``unit_col`` and the metric columns."""
    rng = np.random.default_rng(seed)
    rows = []
    boot: Dict[tuple, np.ndarray] = {}
    med: Dict[tuple, float] = {}
    for name, frame in sets.items():
        for m in metrics:
            lv = log_values(frame, m, unit_col)
            if lv is None:
                continue
            groups = _unit_groups(*lv)
            med[(name, m)] = float(np.median(lv[0]))
            boot[(name, m)] = _boot_medians(groups, n_boot, rng)
    imaged = [n for n in sets if n not in (reference, target)]
    for name in imaged:
        for m in metrics:
            if not all((k, m) in med for k in (reference, target, name)):
                continue
            row = {"set": name, "metric": m, "base_independent": m in BASE_INDEPENDENT}
            for label, (a, b) in {"total": (reference, target), "optics": (reference, name), "rest": (name, target)}.items():
                est = med[(b, m)] - med[(a, m)]
                d = boot[(b, m)] - boot[(a, m)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                row.update({label: est, f"{label}_lo": lo, f"{label}_hi": hi, f"{label}_sig": bool(lo > 0 or hi < 0)})
            row["share"] = row["optics"] / row["total"] if abs(row["total"]) > 1e-6 and np.sign(row["optics"]) == np.sign(row["total"]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, sets: Dict[str, pd.DataFrame], reference: str) -> pd.DataFrame:
    """Per imaged set: mean |total| and |rest| over the base-independent metrics in units of the reference sd of the log
    metric (how much of the gap the imaging closes) + the number of metrics whose rest is still significant."""
    out = []
    ref = sets[reference]
    for name, g in table.groupby("set"):
        gi = g[g["base_independent"]]
        sd = {m: float(np.std(np.log(pd.to_numeric(ref[m], errors="coerce").pipe(lambda s: s[s > 0])))) for m in gi["metric"]}
        tot = np.mean([abs(r.total) / sd[r.metric] for r in gi.itertuples()]) if len(gi) else np.nan
        rest = np.mean([abs(r.rest) / sd[r.metric] for r in gi.itertuples()]) if len(gi) else np.nan
        out.append({"set": name, "mean_abs_total_sd": tot, "mean_abs_rest_sd": rest, "closed_fraction": 1 - rest / tot if tot else np.nan,
                    "n_rest_significant_all": int(g["rest_sig"].sum()), "n_metrics": int(len(g))})
    return pd.DataFrame(out).sort_values("mean_abs_rest_sd")
