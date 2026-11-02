"""Branch-level statistics for the T and P modules, aggregated per neuron (or neuron x compartment), for the tests of
``spine_generation/evaluation/morphology_by_condition.py`` (does the trunk / the spine placement differ between cell
types and compartments?).

Input: the tables of the branch preprocessing (``branches.parquet``, ``branch_spines.parquet``) and, for the marks,
the stage-8 morphometrics table of the S preprocessing. Output: one row per GROUP = (neuron, compartment) with

- trunk (module T; branches eligible for T): medians over the branches of the median radius, the ratio of the proximal
  and distal radius, the flattening of the cross-section (ellipse b/a), the tortuosity, the rms bending, the length;
- simple placement (module P; branches eligible for P): median density (spines per um), median number of spines, the
  share of branches without spines;
- spatial placement (positioned spines of the P branches, POOLED over the branches of the group: a branch carries too
  few spines for a stable estimate):
    * pair correlation function g(r) of the spine positions s along the branch for r in 0-0.5 / 0.5-1 / 1-2 / 2-4 um
      (observed pairs / pairs expected for independent uniform positions on a segment of the branch's length, exact
      edge correction: P(|U1 - U2| in [a, b)) = ((L-a)^2 - (L-b)^2) / L^2); g < 1 = repulsion, g > 1 = clustering;
    * coefficient of variation of the gaps between consecutive spines (1 for a Poisson process, < 1 = more regular);
    * median distance to the nearest neighbour along the branch (nm);
- marks (morphometrics of the spines): correlation between a spine and its nearest neighbour of log Volume and of
  Length, both centred by the branch mean (only the LOCAL similarity counts, not that a thick branch carries big
  spines), reported as the EXCESS over the same statistic with the marks permuted within each branch (``n_null``
  permutations): centring by the mean of n spines alone gives about -1/(n-1) for independent marks, so the raw value
  is negative on branches with few spines (inhibitory neurons) without any neighbour effect; 0 = no local similarity.
  Spearman correlation of log Volume with the position u (proximal -> distal gradient).

Groups with fewer than ``min_branches`` branches get NaN for the branch medians; spatial / mark statistics need at
least ``min_pairs`` pairs. Units: nm (density per um). ``"diff": "abs"`` in METRIC_INFO: the report shows the absolute difference (shares, g(r),
correlations: a ratio of two values near 0 is meaningless).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PCF_BINS_NM = ((0.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0))
TRUNK_METRICS = ("radius_median_nm", "radius_ratio", "ellipse_ratio", "tortuosity", "kappa_rms", "length_nm")
PLACEMENT_METRICS = ("density_per_um", "spines_per_branch", "share_empty")
SPATIAL_METRICS = tuple(f"pcf_{int(a)}_{int(b)}" for a, b in PCF_BINS_NM) + ("gap_cv", "nn_along_nm")
MARK_METRICS = ("nn_corr_log_volume", "nn_corr_length", "volume_vs_u")
ALL_METRICS = TRUNK_METRICS + PLACEMENT_METRICS + SPATIAL_METRICS + MARK_METRICS

METRIC_INFO: Dict[str, Dict[str, Any]] = {
    "radius_median_nm": {"name": "Медианный радиус", "unit": "мкм", "factor": 1e-3, "about": "медиана радиуса ствола вдоль ветви", "module": "T"},
    "radius_ratio": {"name": "Радиус прокс. / дист.", "unit": "", "factor": 1.0, "about": "сужение ветви к дистальному концу", "module": "T"},
    "ellipse_ratio": {"name": "Сплюснутость сечения b/a", "unit": "", "factor": 1.0, "about": "медианное отношение полуосей эллипса сечения", "module": "T"},
    "tortuosity": {"name": "Извилистость", "unit": "", "factor": 1.0, "about": "длина линии / расстояние между концами", "module": "T"},
    "kappa_rms": {"name": "Среднеквадратичный изгиб", "unit": "мкм⁻¹", "factor": 1e3, "about": "rms кривизны центральной линии", "module": "T"},
    "length_nm": {"name": "Длина ветви", "unit": "мкм", "factor": 1e-3, "about": "длина центральной линии объединённой ветви", "module": "T"},
    "density_per_um": {"name": "Плотность шипиков", "unit": "мкм⁻¹", "factor": 1.0, "about": "шипиков на микрометр длины", "module": "P"},
    "spines_per_branch": {"name": "Число шипиков на ветвь", "unit": "", "factor": 1.0, "about": "медиана по ветвям", "module": "P"},
    "share_empty": {"diff": "abs", "name": "Доля ветвей без шипиков", "unit": "", "factor": 1.0, "about": "доля ветвей группы без единого шипика", "module": "P"},
    "pcf_0_500": {"diff": "abs", "name": "g(r), 0–0,5 мкм", "unit": "", "factor": 1.0, "about": "парная корреляция положений вдоль ветви; < 1 — отталкивание", "module": "P"},
    "pcf_500_1000": {"diff": "abs", "name": "g(r), 0,5–1 мкм", "unit": "", "factor": 1.0, "about": "парная корреляция положений вдоль ветви", "module": "P"},
    "pcf_1000_2000": {"diff": "abs", "name": "g(r), 1–2 мкм", "unit": "", "factor": 1.0, "about": "парная корреляция положений вдоль ветви", "module": "P"},
    "pcf_2000_4000": {"diff": "abs", "name": "g(r), 2–4 мкм", "unit": "", "factor": 1.0, "about": "парная корреляция положений вдоль ветви", "module": "P"},
    "gap_cv": {"diff": "abs", "name": "Вариация интервалов (CV)", "unit": "", "factor": 1.0, "about": "1 — случайно, < 1 — регулярнее случайного", "module": "P"},
    "nn_along_nm": {"name": "До соседа вдоль ветви", "unit": "мкм", "factor": 1e-3, "about": "медиана по шипикам группы", "module": "P"},
    "nn_corr_log_volume": {"diff": "abs", "name": "Корреляция объёма соседей", "unit": "", "factor": 1.0, "about": "соседние шипики похожи по объёму (за вычетом среднего по ветви; превышение над перестановкой объёмов внутри ветви, 0 — нет сходства)", "module": "P"},
    "nn_corr_length": {"diff": "abs", "name": "Корреляция длины соседей", "unit": "", "factor": 1.0, "about": "соседние шипики похожи по длине (за вычетом среднего по ветви; превышение над перестановкой длин внутри ветви, 0 — нет сходства)", "module": "P"},
    "volume_vs_u": {"diff": "abs", "name": "Объём и положение u", "unit": "", "factor": 1.0, "about": "корреляция Спирмена log объёма с u (проксимально → дистально)", "module": "P"},
}


def expected_pairs(lengths: np.ndarray, counts: np.ndarray, a: float, b: float) -> float:
    """Expected number of ordered pairs with |s_i - s_j| in [a, b) for ``counts`` independent uniform points on segments
    of ``lengths``: n (n - 1) P(|U1 - U2| in [a, b)), P = ((L - a)^2 - (L - b)^2) / L^2 (clipped at L)."""
    L = np.asarray(lengths, float)
    n = np.asarray(counts, float)
    ok = L > 0
    a_, b_ = np.minimum(a, L), np.minimum(b, L)
    p = np.where(ok, ((L - a_) ** 2 - (L - b_) ** 2) / np.where(ok, L, 1.0) ** 2, 0.0)
    return float((n * (n - 1) * p).sum())


def _corr(x: np.ndarray, y: np.ndarray, min_pairs: int) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_pairs or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def group_spatial_stats(spines: pd.DataFrame, lengths: Dict[Any, float], *, min_pairs: int = 30, n_null: int = 20,
                        seed: int = 0) -> Dict[str, float]:
    """Pooled spatial / mark statistics of one group. ``spines``: positioned spines with ``branch`` (key into
    ``lengths``, nm), ``s`` (nm), ``u`` and optional marks ``log_volume``, ``length``."""
    out = {m: float("nan") for m in SPATIAL_METRICS + MARK_METRICS}
    if spines.empty:
        return out
    obs = np.zeros(len(PCF_BINS_NM))
    gaps, nn = [], []
    pair_a = {"log_volume": [], "length": []}
    pair_b = {"log_volume": [], "length": []}
    null_a = {"log_volume": [], "length": []}
    null_b = {"log_volume": [], "length": []}
    rng = np.random.default_rng(seed)
    lens, cnts = [], []
    for key, g in spines.groupby("branch", sort=False):
        s = np.sort(g["s"].to_numpy(float))
        n = len(s)
        lens.append(lengths.get(key, np.nan))
        cnts.append(n)
        if n >= 2:
            d = np.abs(s[:, None] - s[None, :])[np.triu_indices(n, 1)]
            for k, (a, b) in enumerate(PCF_BINS_NM):
                obs[k] += 2 * int(((d >= a) & (d < b)).sum())  # ordered pairs
            gaps.extend(np.diff(s).tolist())
            order = np.argsort(g["s"].to_numpy(float))
            gs = g.iloc[order]
            ss = gs["s"].to_numpy(float)
            dist = np.abs(ss[:, None] - ss[None, :])
            np.fill_diagonal(dist, np.inf)
            j = dist.argmin(axis=1)
            nn.extend(dist[np.arange(n), j].tolist())
            for mark in pair_a:
                if mark in gs.columns:
                    v = gs[mark].to_numpy(float)
                    v = v - np.nanmean(v)  # centred by the branch: local similarity only
                    pair_a[mark].extend(v.tolist())
                    pair_b[mark].extend(v[j].tolist())
                    for _ in range(n_null):  # the same pairs with the marks shuffled along the branch
                        w = v[rng.permutation(n)]
                        null_a[mark].extend(w.tolist())
                        null_b[mark].extend(w[j].tolist())
    lens, cnts = np.asarray(lens, float), np.asarray(cnts, float)
    good = np.isfinite(lens)
    total_pairs = float((cnts * (cnts - 1)).sum())
    if total_pairs >= min_pairs:
        for k, (a, b) in enumerate(PCF_BINS_NM):
            exp = expected_pairs(lens[good], cnts[good], a, b)
            out[f"pcf_{int(a)}_{int(b)}"] = obs[k] / exp if exp > 0 else float("nan")
    if len(gaps) >= min_pairs:
        gaps = np.asarray(gaps)
        out["gap_cv"] = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else float("nan")
    if len(nn) >= min_pairs:
        out["nn_along_nm"] = float(np.median(nn))
    for mark, key in (("log_volume", "nn_corr_log_volume"), ("length", "nn_corr_length")):
        observed = _corr(np.asarray(pair_a[mark]), np.asarray(pair_b[mark]), min_pairs)
        null = _corr(np.asarray(null_a[mark]), np.asarray(null_b[mark]), min_pairs) if n_null else 0.0
        out[key] = observed - null
    if "log_volume" in spines.columns:
        from scipy.stats import spearmanr

        ok = spines[["u", "log_volume"]].dropna()
        if len(ok) >= min_pairs:
            out["volume_vs_u"] = float(spearmanr(ok["u"], ok["log_volume"]).correlation)
    return out


def branch_metric_table(branches: pd.DataFrame) -> pd.DataFrame:
    """Per-branch trunk / placement metrics (status ok)."""
    b = branches[branches["status"].astype(str) == "ok"].copy()
    b["radius_ratio"] = b["radius_proximal_nm"] / b["radius_distal_nm"]
    b["ellipse_ratio"] = b.get("ellipse_ratio_median", pd.Series(np.nan, index=b.index))
    b["density_per_um"] = b["n_spines"] / (b["length_nm"] / 1000.0)
    b["empty"] = (b["n_spines"] == 0).astype(float)
    return b


def group_table(branches: pd.DataFrame, spines: pd.DataFrame, *, min_branches: int = 10, min_pairs: int = 30,
                group_by: Sequence[str] = ("neuron_id", "compartment")) -> pd.DataFrame:
    """One row per group with every metric of ``ALL_METRICS``, ``n_spines_group`` = positioned spines, ``n_branches``.
    ``branches`` must carry ``cell_type``, ``cell_type_binary``, ``compartment`` and the eligibility flags; ``spines``
    the positioned spines with ``neuron_id``, ``limb_id``, ``merged_branch_id``, ``s``, ``u`` (+ marks)."""
    b = branch_metric_table(branches)
    b["compartment"] = b["compartment"].fillna("none")
    keys = ["neuron_id", "limb_id", "merged_branch_id"]
    b["branch"] = list(zip(*[b[k] for k in keys]))
    lengths = dict(zip(b["branch"], b["length_nm"]))
    sp = spines.copy()
    sp["branch"] = list(zip(*[sp[k] for k in keys]))
    p_branches = set(b.loc[b["train_eligible_P"].astype(bool), "branch"])
    sp = sp[sp["branch"].isin(p_branches)]
    comp_of = dict(zip(b["branch"], b["compartment"]))
    sp["compartment"] = sp["branch"].map(comp_of)
    rows: List[Dict[str, Any]] = []
    for group, g in b.groupby(list(group_by), sort=False):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(group_by, group))
        row["cell_type"] = g["cell_type"].iloc[0]
        row["cell_type_binary"] = g["cell_type_binary"].iloc[0]
        gt = g[g["train_eligible_T"].astype(bool)]
        gp = g[g["train_eligible_P"].astype(bool)]
        row["n_branches"] = int(len(g))
        for m, col in (("radius_median_nm", "radius_median_nm"), ("radius_ratio", "radius_ratio"), ("ellipse_ratio", "ellipse_ratio"),
                       ("tortuosity", "tortuosity"), ("kappa_rms", "kappa_rms"), ("length_nm", "length_nm")):
            row[m] = float(gt[col].median()) if len(gt) >= min_branches else float("nan")
        row["density_per_um"] = float(gp["density_per_um"].median()) if len(gp) >= min_branches else float("nan")
        row["spines_per_branch"] = float(gp["n_spines"].median()) if len(gp) >= min_branches else float("nan")
        row["share_empty"] = float(gp["empty"].mean()) if len(gp) >= min_branches else float("nan")
        mask = np.ones(len(sp), bool)
        for k, v in zip(group_by, group):
            mask &= (sp[k] == v).to_numpy()
        gs = sp[mask]
        row["n_spines_group"] = int(len(gs))
        row.update(group_spatial_stats(gs, lengths, min_pairs=min_pairs))
        rows.append(row)
    return pd.DataFrame(rows)
