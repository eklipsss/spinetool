"""Per-group distribution summaries of every metric (quantiles, mean, sd, n, histogram bins, share of variance between
units). Works on a spine frame and on a branch-group frame alike (``level`` is only a label)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def icc1(values: np.ndarray, units: np.ndarray) -> float:
    """One-way ICC(1): share of the variance that lies BETWEEN units (neuron "style"); near 1 = spines of one neuron are
    nearly interchangeable, so spines are not independent samples. NaN if < 2 units or no replicates."""
    ok = np.isfinite(values)
    v, u = values[ok], units[ok]
    codes, inv = np.unique(u, return_inverse=True)
    k, n = len(codes), len(v)
    if k < 2 or n <= k:
        return float("nan")
    sizes = np.bincount(inv).astype(float)
    means = np.bincount(inv, weights=v) / sizes
    grand = v.mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    msw = float(((v - means[inv]) ** 2).sum() / (n - k))
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else float("nan")


def summarize_metrics(frame: pd.DataFrame, metrics: Sequence[str], *, by: Sequence[str] = ("dataset",), level: str = "spine",
                      n_bins: int = 30, unit_col: str = "neuron_id") -> pd.DataFrame:
    """One row per (group of ``by``, metric): n, n_units, n_missing, mean, sd, min, q01..q99, max, ``icc``, histogram
    (``hist_edges`` shared by all groups of a metric, over the pooled 1-99 % range; ``hist_counts`` per group, outliers
    clipped into the edge bins). Groups with a missing ``by`` value are kept as "unknown"."""
    f = frame.copy()
    for c in by:
        f[c] = f[c].astype(object).where(f[c].notna(), "unknown")
    rows: List[Dict] = []
    for m in metrics:
        x_all = pd.to_numeric(f[m], errors="coerce").to_numpy(float)
        fin = x_all[np.isfinite(x_all)]
        if len(fin) == 0:
            continue
        lo, hi = np.quantile(fin, [0.01, 0.99])
        if hi <= lo:
            lo, hi = float(fin.min()), float(fin.max()) + 1e-9
        edges = np.linspace(lo, hi, n_bins + 1)
        for key, g in f.groupby(list(by), observed=True):
            key = key if isinstance(key, tuple) else (key,)
            x = pd.to_numeric(g[m], errors="coerce").to_numpy(float)
            ok = np.isfinite(x)
            row = dict(zip(by, key))
            row.update({"level": level, "metric": m, "n": int(ok.sum()), "n_missing": int((~ok).sum()),
                        "n_units": int(g.loc[ok, unit_col].nunique()) if ok.any() else 0})
            if ok.any():
                xs = x[ok]
                q = np.quantile(xs, QUANTILES)
                row.update({"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else float("nan"),
                            "min": float(xs.min()), "max": float(xs.max()),
                            **{f"q{int(round(p * 100)):02d}": float(v) for p, v in zip(QUANTILES, q)},
                            "icc": icc1(xs, g.loc[ok, unit_col].to_numpy()),
                            "hist_edges": [float(e) for e in edges],
                            "hist_counts": np.histogram(np.clip(xs, lo, hi), bins=edges)[0].astype(int).tolist()})
            rows.append(row)
    return pd.DataFrame(rows)


def unit_summary(frame: pd.DataFrame, metrics: Sequence[str], *, min_rows: int = 1,
                 keep: Sequence[str] = ("dataset", "cell_type", "cell_type_binary", "health", "treatment", "group")) -> pd.DataFrame:
    """Per-unit medians (the values every between-unit test works on) + the unit's metadata; for the report's plots."""
    g = frame.groupby("neuron_id", observed=True)
    med = g[list(metrics)].median()
    med["n_rows"] = g.size()
    info = g[[c for c in keep if c in frame.columns]].first()
    return med.join(info)[lambda d: d["n_rows"] >= min_rows].reset_index()
