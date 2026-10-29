"""Generated (or assembled) branches vs real ones (module-interfaces-spec.md, section 10: evaluation of the assembled branch).

The real reference of a generated set is the list of SOURCE branches of its requests (``branches.parquet`` column
``source_branch`` of ``generate_tp.py``): the same conditions and lengths, so the comparison is not confounded by the
condition mix. Compared quantities (generated vs real, W1 / std of real, KS, medians):

- per branch: number of spines, density per um, share of branches without spines (overall and per length class),
  trunk functionals of T (``trunk_generation.fit.functionals``);
- per spine: position along the branch u, distance to the nearest neighbour on the trunk surface
  sqrt(ds^2 + (r dtheta)^2), gap to the next spine along s, angle around the trunk (mean resultant length of
  e^{i theta} and e^{2 i theta}: 0 = uniform);
- assembly (when given): spines placed / dropped, immersion depth.

Noise floor: the same comparison between two halves of the real reference (by neuron when it has >= 2 neurons,
otherwise by branch - optimistic). Real spine positions are those with ``has_position`` (about 7 % of the valid spines
have none), so the real nearest-neighbour distances are slightly too long. Units: nm (density per um).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import branch_geometry as bg
from ..trunk_generation.fit import FUNCTIONALS, compare, functionals

LENGTH_CLASSES = (("< 20 um", 0.0, 20000.0), ("20-50 um", 20000.0, 50000.0), (">= 50 um", 50000.0, np.inf))


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def spine_geometry_stats(s: np.ndarray, theta: np.ndarray, radius: np.ndarray, length: float) -> Dict[str, np.ndarray]:
    """Per spine of ONE branch: nearest-neighbour surface distance, gap to the next spine along s, u."""
    s, theta, radius = (np.asarray(x, dtype=float) for x in (s, theta, radius))
    n = len(s)
    out = {"u": s / max(length, 1e-9), "theta": _wrap(theta), "nn_surface_nm": np.full(n, np.nan), "gap_nm": np.zeros(0)}
    if n >= 2:
        r = 0.5 * (radius[:, None] + radius[None, :])
        d = np.sqrt((s[:, None] - s[None, :]) ** 2 + (r * _wrap(theta[:, None] - theta[None, :])) ** 2)
        np.fill_diagonal(d, np.inf)
        out["nn_surface_nm"] = d.min(axis=1)
        out["gap_nm"] = np.diff(np.sort(s))
    return out


def trunk_functionals(record: Dict[str, Any]) -> Dict[str, float]:
    """T functionals of a trunk record (s, gamma, radius, tangent, normal, binormal [, kappa1, kappa2])."""
    s = np.asarray(record["s"], dtype=float)
    step = float(s[-1] / (len(s) - 1))
    if "kappa1" in record and record["kappa1"] is not None and len(record["kappa1"]) >= len(s) - 1:
        k1, k2 = np.asarray(record["kappa1"], float)[: len(s) - 1], np.asarray(record["kappa2"], float)[: len(s) - 1]
    else:
        k1, k2, _ = bg.bending_from_frames(np.asarray(record["tangent"], float), np.asarray(record["normal"], float),
                                           np.asarray(record["binormal"], float), step)
    return functionals(k1, k2, np.log(np.asarray(record["radius"], float)), step, np.asarray(record["gamma"], float))


def circular_moments(theta: np.ndarray) -> Dict[str, float]:
    theta = np.asarray(theta, dtype=float)
    theta = theta[np.isfinite(theta)]
    if not len(theta):
        return {"R1": float("nan"), "R2": float("nan"), "n": 0}
    return {"R1": float(np.abs(np.mean(np.exp(1j * theta)))), "R2": float(np.abs(np.mean(np.exp(2j * theta)))), "n": int(len(theta))}


RADIUS_CLASSES_EVAL = (("r < 200 nm", 0.0, 200.0), ("200-300 nm", 200.0, 300.0), ("300-400 nm", 300.0, 400.0), ("r >= 400 nm", 400.0, 1e12))


def branch_level(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """rows: dicts with length_nm, n_spines (+ functionals) -> frame with density and length class."""
    frame = pd.DataFrame(list(rows))
    frame["density_per_um"] = frame["n_spines"] / (frame["length_nm"] / 1000.0)
    frame["no_spines"] = (frame["n_spines"] == 0).astype(float).where(frame["n_spines"].notna())  # NaN: count unknown (real, not eligible for P)
    frame["length_class"] = pd.cut(frame["length_nm"], [lo for _, lo, _ in LENGTH_CLASSES] + [np.inf], labels=[n for n, _, _ in LENGTH_CLASSES], right=False)
    return frame


def _with_ci(res: Dict[str, Any], gen: pd.DataFrame, real: pd.DataFrame, col: str, n_boot: int) -> Dict[str, Any]:
    """Adds the cluster-bootstrap difference of the medians (clusters = neurons) and the plain KS p-value to ``res``."""
    from ..significance import cluster_bootstrap_diff, ks_pvalue

    if "neuron" in gen.columns and "neuron" in real.columns and col in gen.columns and col in real.columns:
        res["ci"] = cluster_bootstrap_diff(gen[col].to_numpy(float), gen["neuron"], real[col].to_numpy(float), real["neuron"], n_boot=n_boot)
        res["ks_p_naive"] = ks_pvalue(gen[col].to_numpy(float), real[col].to_numpy(float))
    return res


def compare_sets(gen_b: pd.DataFrame, real_b: pd.DataFrame, gen_s: pd.DataFrame, real_s: pd.DataFrame, *, n_boot: int = 0) -> Dict[str, Any]:
    """All comparisons of one generated set against one real set (the noise floor uses the same function).
    ``n_boot`` > 0 (and a ``neuron`` column in the frames): cluster-bootstrap interval of the median difference per quantity."""
    out: Dict[str, Any] = {"branch": {}, "spine": {}, "by_length": {}, "by_radius": {}}
    for col in ["n_spines", "density_per_um", *[f for f in FUNCTIONALS if f in gen_b.columns and f in real_b.columns]]:
        out["branch"][col] = compare(gen_b[col].to_numpy(float), real_b[col].to_numpy(float))
        if n_boot:
            _with_ci(out["branch"][col], gen_b, real_b, col, n_boot)
    out["branch"]["no_spines_share"] = {"gen": float(gen_b["no_spines"].mean()), "real": float(real_b["no_spines"].mean())}
    for name, _, _ in LENGTH_CLASSES:
        g, r = gen_b[gen_b["length_class"] == name], real_b[real_b["length_class"] == name]
        if len(g) and len(r):
            out["by_length"][name] = {"n_gen": int(len(g)), "n_real": int(len(r)),
                                      "density_gen": float(g["density_per_um"].median()), "density_real": float(r["density_per_um"].median()),
                                      "no_spines_gen": float(g["no_spines"].mean()), "no_spines_real": float(r["no_spines"].mean())}
    if "mean_radius_nm" in gen_b.columns and "mean_radius_nm" in real_b.columns:
        for name, lo, hi in RADIUS_CLASSES_EVAL:  # each side by ITS OWN radius (the generated radius is drawn, it is not the real one)
            g, r = gen_b[(gen_b["mean_radius_nm"] >= lo) & (gen_b["mean_radius_nm"] < hi)], real_b[(real_b["mean_radius_nm"] >= lo) & (real_b["mean_radius_nm"] < hi)]
            if len(g) and len(r):
                out["by_radius"][name] = {"n_gen": int(len(g)), "n_real": int(len(r)),
                                          "n_spines_gen": float(g["n_spines"].median()), "n_spines_real": float(r["n_spines"].median()),
                                          "density_gen": float(g["density_per_um"].median()), "density_real": float(r["density_per_um"].median()),
                                          "no_spines_gen": float(g["no_spines"].mean()), "no_spines_real": float(r["no_spines"].mean())}
    for col in ("u", "nn_surface_nm", "gap_nm"):
        a = gen_s[col].dropna().to_numpy(float) if col in gen_s.columns else np.zeros(0)
        b = real_s[col].dropna().to_numpy(float) if col in real_s.columns else np.zeros(0)
        out["spine"][col] = compare(a, b)
        if n_boot and "neuron" in gen_s.columns and "neuron" in real_s.columns and col in gen_s.columns and col in real_s.columns:
            from ..significance import cluster_bootstrap_diff, ks_pvalue

            ga, gb = gen_s[gen_s[col].notna()], real_s[real_s[col].notna()]
            out["spine"][col]["ci"] = cluster_bootstrap_diff(ga[col].to_numpy(float), ga["neuron"], gb[col].to_numpy(float), gb["neuron"], n_boot=n_boot)
            out["spine"][col]["ks_p_naive"] = ks_pvalue(ga[col].to_numpy(float), gb[col].to_numpy(float))
    out["spine"]["theta"] = {"gen": circular_moments(gen_s.get("theta", pd.Series(dtype=float))), "real": circular_moments(real_s.get("theta", pd.Series(dtype=float)))}
    return out


def collect(branch_rows: List[Dict[str, Any]], spine_stats: List[Dict[str, np.ndarray]], neurons: Optional[Sequence[str]] = None) -> tuple:
    """(branch frame, spine frame) from per-branch rows and per-branch spine statistics (aligned lists); ``neurons``: the neuron of
    every branch (column ``neuron`` in both frames, for the cluster bootstrap)."""
    b = branch_level(branch_rows)
    if neurons is not None:
        b["neuron"] = [str(n) for n in neurons]
    parts, gaps = [], []
    for i, st in enumerate(spine_stats):
        n = len(st["u"])
        if n:
            parts.append(pd.DataFrame({"branch": i, "u": st["u"], "theta": st["theta"], "nn_surface_nm": st["nn_surface_nm"]}))
        if len(st["gap_nm"]):
            gaps.append(pd.DataFrame({"branch": i, "gap_nm": st["gap_nm"]}))
    s = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["branch", "u", "theta", "nn_surface_nm"])
    g = pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame(columns=["branch", "gap_nm"])
    s = pd.concat([s, g], ignore_index=True)  # gap rows carry NaN in the other columns (dropna per column)
    if neurons is not None and len(s):
        s["neuron"] = [str(neurons[int(i)]) for i in s["branch"]]
    return b, s


def split_halves(groups: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of the first half: by group (neuron) when there are >= 2 groups, else by element."""
    groups = np.asarray(groups)
    names = np.unique(groups)
    if len(names) >= 2:
        first = set(rng.permutation(names)[: len(names) // 2])
        return np.array([g in first for g in groups])
    return rng.permutation(len(groups)) < len(groups) // 2


def _ci_text(m: Dict[str, Any]) -> str:
    ci = m.get("ci")
    if not ci or not np.isfinite(ci.get("diff", np.nan)):
        return ""
    star = " *" if ci["significant"] else ""
    return f"   diff of medians {ci['diff']:+.3g} [95 % CI {ci['lo']:+.3g}, {ci['hi']:+.3g}]{star}; KS p (naive) {m.get('ks_p_naive', float('nan')):.2g}"


def summarize(result: Dict[str, Any], floor: Optional[Dict[str, Any]] = None) -> List[str]:
    """Human-readable lines."""
    lines = []
    fl = lambda part, col: (floor or {}).get(part, {}).get(col, {}).get("w1_rel", float("nan"))
    lines.append("per branch (median generated | real, W1/std, noise floor; diff of medians with a 95 % interval from resampling NEURONS, * = interval excludes 0):")
    for col, m in result["branch"].items():
        if col == "no_spines_share":
            lines.append(f"  {'share without spines':26s} {m['gen']:.2f} | {m['real']:.2f}")
            continue
        lines.append(f"  {col:26s} {m['median_a']:.4g} | {m['median_b']:.4g}   W1/std {m['w1_rel']:.2f}  (floor {fl('branch', col):.2f})" + _ci_text(m))
    lines.append("per length class (density median per um, share without spines; generated | real):")
    for name, m in result["by_length"].items():
        lines.append(f"  {name:10s} n {m['n_gen']:3d} | {m['n_real']:3d}   density {m['density_gen']:.3g} | {m['density_real']:.3g}   no spines {m['no_spines_gen']:.2f} | {m['no_spines_real']:.2f}")
    if result.get("by_radius"):
        lines.append("per radius class (each side by its own mean radius; spines per branch and density, medians; share without spines; generated | real):")
        for name, m in result["by_radius"].items():
            lines.append(f"  {name:10s} n {m['n_gen']:3d} | {m['n_real']:3d}   spines {m['n_spines_gen']:.3g} | {m['n_spines_real']:.3g}   density {m['density_gen']:.3g} | {m['density_real']:.3g}"
                         f"   no spines {m['no_spines_gen']:.2f} | {m['no_spines_real']:.2f}")
    lines.append("per spine (median generated | real, W1/std, noise floor):")
    for col in ("u", "nn_surface_nm", "gap_nm"):
        m = result["spine"][col]
        lines.append(f"  {col:26s} {m['median_a']:.4g} | {m['median_b']:.4g}   W1/std {m['w1_rel']:.2f}  KS {m['ks']:.2f}  (floor {fl('spine', col):.2f})" + _ci_text(m))
    t = result["spine"]["theta"]
    lines.append(f"  angle around the trunk: |mean e^(i theta)| {t['gen']['R1']:.3f} | {t['real']['R1']:.3f}, |mean e^(2i theta)| {t['gen']['R2']:.3f} | {t['real']['R2']:.3f}"
                 f"  (uniform ~ {1.0 / np.sqrt(max(t['gen']['n'], 1)):.3f} | {1.0 / np.sqrt(max(t['real']['n'], 1)):.3f})")
    return lines
