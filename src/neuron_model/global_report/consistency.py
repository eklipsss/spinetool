"""Is a within-dataset effect the SAME in Minnie65 (mouse) and H01 (human)?

The significance tables answer "does factor X change metric m" separately in every dataset. This module puts the two answers
side by side for the factors that exist in both EM datasets (binary cell type, harmonised cell class, compartment): per metric
the effect in each dataset with a bootstrap interval over UNITS (neurons), whether the signs agree, and the DIFFERENCE of
the two effects with its interval and p (the datasets are independent, so one bootstrap replicate resamples the neurons of
each dataset independently; p by the shift method, BH over the metrics of one contrast). One replicate resamples the neurons
once for ALL metrics, so the summary over metrics (Spearman correlation of the two effect vectors) gets an interval too.

Effects use the definitions of ``significance._effect_columns``: between units median_a / median_b - 1 (relative) or
median_a - median_b (absolute); within a unit (compartment) median(d) / |median(b)| or median(d), d = per-neuron difference.
The difference of two RELATIVE effects is in percentage points. Species is confounded with the dataset (modality is the
same, but region, donor, segmentation are not): a difference between the effects is "the factor acts differently in the two
datasets", not a proven species interaction.

Verdict per metric (``present`` = significant in that dataset by the main tables when given, else the 95 % interval excludes 0):
``consistent`` (present in both, same sign, difference not significant), ``same_sign_different_size``, ``opposite``,
``minnie65_only`` / ``h01_only``, ``neither``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..spine_generation.evaluation.morphology_by_condition import bh_qvalues, neuron_medians
from .factors import FACTORS, Factor, resolve_min_rows, valid_comparisons
from .significance import _with_level, unit_medians

DEFAULT_FACTORS = ("cell_type_binary", "cell_class", "compartment")
PAIR = ("minnie65", "h01")


def _unit_table(sub: pd.DataFrame, comp: Dict[str, Any], metrics: Sequence[str], min_spines) -> Dict[str, Any]:
    """Unit-level arrays of one dataset for one contrast: between -> (xa [na, M], xb [nb, M]); within -> (va, vb [n, M])."""
    col, a, b = comp["column"], comp["a"], comp["b"]
    if comp["mode"] == "within":
        rows = resolve_min_rows(min_spines, comp["dataset"])
        med = neuron_medians(sub[sub[col].isin([a, b])], metrics, ["neuron_id", col], rows)
        pa = med[med[col] == a].set_index("neuron_id")[list(metrics)]
        pb = med[med[col] == b].set_index("neuron_id")[list(metrics)]
        idx = pa.index.intersection(pb.index)
        return {"mode": "within", "a": pa.loc[idx].to_numpy(float), "b": pb.loc[idx].to_numpy(float)}
    unit = _with_level(unit_medians(sub, metrics, min_spines), sub, col)
    return {"mode": "between", "a": unit.loc[unit[col] == a, list(metrics)].to_numpy(float),
            "b": unit.loc[unit[col] == b, list(metrics)].to_numpy(float)}


def _effects(t: Dict[str, Any], kinds: Sequence[str], ia=None, ib=None) -> np.ndarray:
    """Effect per metric; ``ia`` / ``ib`` row indices of a bootstrap replicate (within: ia used for both)."""
    import warnings

    xa = t["a"] if ia is None else t["a"][ia]
    if t["mode"] == "within":
        xb = t["b"] if ia is None else t["b"][ia]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmedian(xa - xb, axis=0)
            ref = np.nanmedian(np.where(np.isfinite(xa - xb), xb, np.nan), axis=0)
        return np.array([d[j] / abs(ref[j]) if k == "rel" and ref[j] != 0 else d[j] for j, k in enumerate(kinds)])
    xb = t["b"] if ib is None else t["b"][ib]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ma, mb = np.nanmedian(xa, axis=0), np.nanmedian(xb, axis=0)
    return np.array([ma[j] / mb[j] - 1.0 if k == "rel" and mb[j] != 0 else ma[j] - mb[j] for j, k in enumerate(kinds)])


def _boot(t: Dict[str, Any], kinds: Sequence[str], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n_boot, len(kinds)))
    na, nb = len(t["a"]), len(t["b"])
    for i in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = None if t["mode"] == "within" else rng.integers(0, nb, nb)
        out[i] = _effects(t, kinds, ia, ib)
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def _verdict(present_m: bool, present_h: bool, em: float, eh: float, diff_sig: bool, names=PAIR) -> str:
    if present_m and present_h:
        if np.sign(em) != np.sign(eh):
            return "opposite"
        return "same_sign_different_size" if diff_sig else "consistent"
    if present_m:
        return f"{names[0]}_only"
    if present_h:
        return f"{names[1]}_only"
    return "neither"


def consistency(frame: pd.DataFrame, metrics: Sequence[str], *, level: str = "spine", factors: Sequence[str] = DEFAULT_FACTORS,
                datasets: Sequence[str] = PAIR, results: Optional[pd.DataFrame] = None, min_neurons: int = 10, min_spines=20,
                min_paired: int = 5, abs_metrics: Sequence[str] = (), n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                factor_defs: Sequence[Factor] = FACTORS) -> Dict[str, pd.DataFrame]:
    """``pairs``: one row per (factor, a, b, metric); ``summary``: one row per (factor, a, b) over the metrics.
    ``datasets`` may be any two values of the ``dataset`` column (section 4: ``("real", "generated")``) when ``factor_defs``
    declares the factors for them (``generated.generated_factors``)."""
    dm, dh = datasets
    frame = frame[frame["dataset"].isin(datasets)]
    metrics = [m for m in metrics if m in frame.columns and frame[m].notna().any()]
    chosen = [f for f in factor_defs if f.name in factors and not f.cross and f.stratify is None]
    comps = [c for c in valid_comparisons(frame, factors=chosen, min_neurons=min_neurons, min_spines=min_spines, min_paired=min_paired)
             if c["valid"] and c["stratum"] is None]
    by_key: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for c in comps:
        by_key.setdefault((c["factor"], c["a"], c["b"]), {})[c["dataset"]] = c
    label = {}
    if results is not None and len(results):
        r = results[results["stratum"].isna()] if "stratum" in results.columns else results
        label = {(x["dataset"], x["factor"], x["a"], x["b"], x["metric"]): x["label"] for x in r.to_dict("records")}
    rng = np.random.default_rng(seed + 101)
    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for (factor, a, b), per in by_key.items():
        if not {dm, dh} <= set(per):
            continue
        tabs = {d: _unit_table(frame[frame["dataset"] == d], per[d], metrics, min_spines) for d in (dm, dh)}
        if min(len(tabs[d]["a"]) for d in tabs) < 3 or min(len(tabs[d]["b"]) for d in tabs) < 3:
            continue
        # relative effect unless the metric is absolute or a reference median is ~0 in either dataset
        kinds = []
        for j, m in enumerate(metrics):
            refs = [np.nanmedian(tabs[d]["b"][:, j]) if np.isfinite(tabs[d]["b"][:, j]).any() else np.nan for d in (dm, dh)]
            kinds.append("abs" if m in abs_metrics or not all(np.isfinite(r) and abs(r) > 1e-12 for r in refs) else "rel")
        eff = {d: _effects(tabs[d], kinds) for d in (dm, dh)}
        boot = {d: _boot(tabs[d], kinds, n_boot, rng) for d in (dm, dh)}
        diff, diff_boot = eff[dm] - eff[dh], boot[dm] - boot[dh]
        part = []
        for j, m in enumerate(metrics):
            if not (np.isfinite(eff[dm][j]) and np.isfinite(eff[dh][j])):
                continue
            ci = {d: np.nanquantile(boot[d][:, j], [0.025, 0.975]) for d in (dm, dh)}
            db = diff_boot[:, j][np.isfinite(diff_boot[:, j])]
            p = (1 + np.sum(np.abs(db - diff[j]) >= abs(diff[j]))) / (1 + len(db)) if len(db) else np.nan
            dlo, dhi = np.quantile(db, [0.025, 0.975]) if len(db) else (np.nan, np.nan)
            present = {}
            for d in (dm, dh):
                lab = label.get((d, factor, a, b, m))
                present[d] = (lab == "significant") if lab is not None else bool(ci[d][0] > 0 or ci[d][1] < 0)
            part.append({"level": level, "factor": factor, "a": a, "b": b, "metric": m, "effect_kind": kinds[j],
                         f"effect_{dm}": eff[dm][j], f"ci_low_{dm}": ci[dm][0], f"ci_high_{dm}": ci[dm][1], f"present_{dm}": present[dm],
                         f"effect_{dh}": eff[dh][j], f"ci_low_{dh}": ci[dh][0], f"ci_high_{dh}": ci[dh][1], f"present_{dh}": present[dh],
                         f"n_units_{dm}": len(tabs[dm]["a"]) + (0 if tabs[dm]["mode"] == "within" else len(tabs[dm]["b"])),
                         f"n_units_{dh}": len(tabs[dh]["a"]) + (0 if tabs[dh]["mode"] == "within" else len(tabs[dh]["b"])),
                         "same_sign": bool(np.sign(eff[dm][j]) == np.sign(eff[dh][j])),
                         "difference": diff[j], "difference_ci_low": dlo, "difference_ci_high": dhi, "p_difference": p})
        if not part:
            continue
        q = bh_qvalues(np.array([r["p_difference"] for r in part], float))
        for r, qq in zip(part, q):
            r["q_difference"] = float(qq)
            r["difference_significant"] = bool(qq < alpha)
            r["verdict"] = _verdict(r[f"present_{dm}"], r[f"present_{dh}"], r[f"effect_{dm}"], r[f"effect_{dh}"], r["difference_significant"], (dm, dh))
        rows += part
        # over metrics: rank correlation of the two effect vectors (scale-free, mixes relative and absolute effects) + interval
        rho = _spearman(eff[dm], eff[dh])
        rho_b = np.array([_spearman(boot[dm][i], boot[dh][i]) for i in range(n_boot)])
        rho_b = rho_b[np.isfinite(rho_b)]
        both = [r for r in part if r[f"present_{dm}"] and r[f"present_{dh}"]]
        summary.append({"level": level, "factor": factor, "a": a, "b": b, "n_metrics": len(part),
                        f"n_present_{dm}": sum(r[f"present_{dm}"] for r in part), f"n_present_{dh}": sum(r[f"present_{dh}"] for r in part),
                        "n_present_both": len(both), "n_same_sign_of_both": sum(r["same_sign"] for r in both),
                        "share_same_sign_all": float(np.mean([r["same_sign"] for r in part])),
                        "n_difference_significant": sum(r["difference_significant"] for r in part),
                        "spearman": rho, "spearman_ci_low": float(np.quantile(rho_b, 0.025)) if len(rho_b) else np.nan,
                        "spearman_ci_high": float(np.quantile(rho_b, 0.975)) if len(rho_b) else np.nan,
                        **{f"n_{v}": sum(r["verdict"] == v for r in part) for v in
                           ("consistent", "same_sign_different_size", "opposite", f"{dm}_only", f"{dh}_only", "neither")}})
    return {"pairs": pd.DataFrame(rows), "summary": pd.DataFrame(summary)}
