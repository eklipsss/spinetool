"""Differences of spine morphometrics between the LabID groups (health, treatment) with the SOURCE IMAGE as the unit.

LabID (Rakovskaya et al. 2023, docs/neuron-model/labid-data-provenance.md): cultured hippocampal neurons, Wt / Ab (amyloid
oligomers) / Wt+9009 / Ab+9009 (treatment). Every branch folder is a crop of one source image = one cell, so the spines of
one image are not independent: the unit is the image (``metadata.labid_source_id``). Acquisition differs between groups
(section 2g of the provenance note): Wt and Ab are deconvolved and imaged with NA 1.40, the 9009 groups are NOT
deconvolved and about half of their images use NA 1.25 (February 2024). Hence:

* inferential comparisons only between groups imaged and processed the same way (same NA, same deconvolution status);
* treatment contrasts (9009 vs untreated) mix the drug with deconvolution / objective and are reported descriptively.

Statistic per metric: every unit is summarised by the median of log(metric) over its spines (units with fewer than
``min_spines`` spines dropped); the effect is the difference of the MEANS of unit medians between the groups, reported as
a ratio exp(effect) - 1. p: permutation of the group labels among units (two-sided); interval: bootstrap of units within
each group; Benjamini-Hochberg over the metrics of one comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

METRICS = ("OpenAngle", "CVD", "AverageDistance", "LengthVolumeRatio", "LengthAreaRatio", "Length", "JunctionArea",
           "Area", "Volume", "ConvexHullVolume", "ConvexHullRatio")


@dataclass
class Comparison:
    name: str
    group_a: str
    group_b: str
    unit_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None  # on the unit table (acquisition columns)
    inferential: bool = True
    note: str = ""


def _same_na(na: float):
    return lambda u: np.isclose(u["na"].astype(float), na)


def _deconvolved(flag: bool):
    return lambda u: (u["deconvolved_crops"] >= u["n_crops"]) if flag else (u["deconvolved_crops"] == 0)


def default_comparisons() -> List[Comparison]:
    both = lambda *fs: (lambda u: np.logical_and.reduce([f(u) for f in fs]))  # noqa: E731
    return [
        Comparison("Ab против Wt: NA 1,40, деконволюция", "Ab", "Wt", both(_same_na(1.4), _deconvolved(True)),
                   note="основное сравнение эффекта амилоида"),
        Comparison("Ab против Wt: все снимки", "Ab", "Wt", None, note="чувствительность (включает снимок Ab 18 без деконволюции)"),
        Comparison("Ab+9009 против Wt+9009: NA 1,40", "Ab+9009", "Wt+9009", _same_na(1.4),
                   note="эффект амилоида при лечении, одна оптика, без деконволюции у обеих групп"),
        Comparison("Ab+9009 против Wt+9009: NA 1,25", "Ab+9009", "Wt+9009", _same_na(1.25),
                   note="то же в февральской сессии с объективом 1,25"),
        Comparison("Ab+9009 против Ab", "Ab+9009", "Ab", None, inferential=False,
                   note="описательно: лечение смешано с деконволюцией и частично с объективом"),
        Comparison("Wt+9009 против Wt", "Wt+9009", "Wt", None, inferential=False,
                   note="описательно: лечение смешано с деконволюцией и частично с объективом"),
    ]


def unit_table(morph: pd.DataFrame, acquisition: Optional[pd.DataFrame], metrics: Sequence[str] = METRICS,
               min_spines: int = 3) -> pd.DataFrame:
    """One row per source image: group, n spines, median log of every metric, acquisition columns (na, deconvolution)."""
    from .metadata import labid_group, labid_source_id

    m = morph.copy()
    m["unit"] = m["neuron_id"].map(labid_source_id)
    m["group"] = m["neuron_id"].map(labid_group)
    rows = []
    for unit, g in m.groupby("unit"):
        row = {"unit": unit, "group": g["group"].iloc[0], "n_spines": len(g)}
        for k in metrics:
            v = g[k].to_numpy(float)
            v = v[np.isfinite(v) & (v > 0)]
            row[k] = float(np.median(np.log(v))) if len(v) >= min_spines else np.nan
        rows.append(row)
    u = pd.DataFrame(rows)
    u = u[u["n_spines"] >= min_spines]
    if acquisition is not None:
        a = acquisition.copy()
        a["unit"] = a["group"].astype(str) + "_" + a["source"].astype(str)
        u = u.merge(a[["unit", "na", "deconvolved_crops", "n_crops", "start_time"]], on="unit", how="left")
    return u.reset_index(drop=True)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def compare(units: pd.DataFrame, c: Comparison, metrics: Sequence[str] = METRICS, n_perm: int = 10000, n_boot: int = 5000,
            seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = units
    if c.unit_filter is not None and "na" in u.columns:
        keep = pd.Series(np.asarray(c.unit_filter(u)), index=u.index).fillna(False).astype(bool)
        u = u[keep]
    a, b = u[u["group"] == c.group_a], u[u["group"] == c.group_b]
    rows = []
    for k in metrics:
        xa, xb = a[k].dropna().to_numpy(), b[k].dropna().to_numpy()
        row = {"comparison": c.name, "a": c.group_a, "b": c.group_b, "metric": k, "n_units_a": len(xa), "n_units_b": len(xb),
               "n_spines_a": int(a.loc[a[k].notna(), "n_spines"].sum()), "n_spines_b": int(b.loc[b[k].notna(), "n_spines"].sum()),
               "inferential": c.inferential, "note": c.note}
        if len(xa) < 3 or len(xb) < 3:
            rows.append({**row, "effect": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan})
            continue
        eff = xa.mean() - xb.mean()
        pooled = np.r_[xa, xb]
        perm = np.empty(n_perm)
        for i in range(n_perm):
            s = rng.permutation(pooled)
            perm[i] = s[: len(xa)].mean() - s[len(xa):].mean()
        p = (1 + np.sum(np.abs(perm) >= abs(eff) - 1e-12)) / (n_perm + 1)
        boots = np.array([rng.choice(xa, len(xa)).mean() - rng.choice(xb, len(xb)).mean() for _ in range(n_boot)])
        rows.append({**row, "effect": eff, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "p": float(p)})
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out["q"] = np.nan
    if ok.any():
        out.loc[ok, "q"] = _bh(out.loc[ok, "p"].to_numpy())
    out["significant"] = out["inferential"] & (out["q"] < 0.05)
    for col in ("effect", "lo", "hi"):
        out[col + "_pct"] = 100 * np.expm1(out[col])
    return out


def run(morph: pd.DataFrame, acquisition: Optional[pd.DataFrame], comparisons: Optional[List[Comparison]] = None, **kw) -> pd.DataFrame:
    units = unit_table(morph, acquisition)
    return pd.concat([compare(units, c, **kw) for c in (comparisons or default_comparisons())], ignore_index=True)
