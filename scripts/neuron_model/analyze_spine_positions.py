"""Where do the spines sit relative to the trunk, and which branches have none? (input for ``position_outlier_*``)

Runs on the output of ``run_branch_preprocessing.py`` only (tables + ``branch_geometry.npz``, local disk, no NAS):

1. spines per branch by cell type and radius class (raw segments and spines after the false-spine verdict) — is
   "61 % of the ok branches without spines" a property of the data (aspiny interneurons) or a loss in our processing?
2. the signed offset of the attachment centre from the trunk surface, d = |v_perp| - r(s), per radius class, in nm and
   in units of r — for the circle r(s) and for the section ellipse in the direction of the spine (radius of the ellipse
   at angle theta - phi(s)). The current outlier rule |d| > 0.75 r is symmetric; a systematic offset makes it wrong.
3. outlier rate and the share of branches that pass the "spines" check (n_with_position >= position_fraction_min *
   n_in_range) for candidate asymmetric rules  -lo_factor r <= d <= max(hi_factor r, hi_floor).

    python scripts/neuron_model/analyze_spine_positions.py --root C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches

Writes ``<root>/audit/spine_positions.csv`` (one row per positioned spine). Nothing else is changed.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RADIUS_CLASSES = (("r < 120 nm", 0.0, 120.0), ("120-200 nm", 120.0, 200.0), ("r >= 200 nm", 200.0, 1e9))
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def radius_class(r: pd.Series) -> pd.Series:
    out = pd.Series("n/a", index=r.index, dtype=object)
    for name, lo, hi in RADIUS_CLASSES:
        out[(r >= lo) & (r < hi)] = name
    return out


def ellipse_radius(geometry: Path, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Radius of the section ellipse in the direction theta (RMF angle, as the projection of the spines), NaN without a profile."""
    try:
        g = np.load(geometry)
    except OSError:
        return np.full(len(s), np.nan)
    if "ellipse_a" not in g.files:
        return np.full(len(s), np.nan)
    nodes = g["s"]
    a, b = np.interp(s, nodes, g["ellipse_a"]), np.interp(s, nodes, g["ellipse_b"])
    # the angle has period pi: interpolate the doubled angle
    ang2 = np.unwrap(2.0 * g["ellipse_angle"])
    phi = 0.5 * np.interp(s, nodes, ang2)
    psi = theta - phi
    return a * b / np.sqrt((b * np.cos(psi)) ** 2 + (a * np.sin(psi)) ** 2)


def describe(name: str, x: pd.Series, unit: str) -> str:
    q = x.quantile(list(QUANTILES))
    fmt = "{:+.0f}" if unit == "nm" else "{:+.2f}"
    return f"    {name:22s} " + "  ".join(fmt.format(v) for v in q.values) + f"   IQR {q[0.75] - q[0.25]:.3g} {unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--position-fraction-min", type=float, default=0.9, help="as in branch_preprocessing.yaml")
    args = parser.parse_args()
    pd.set_option("display.width", 220)

    b = pd.read_parquet(args.root / "branches.parquet")
    sp = pd.read_parquet(args.root / "branch_spines.parquet")
    ok = b[b["status"] == "ok"].copy()
    ok["radius_class"] = radius_class(ok["radius_median_nm"].astype(float))
    ok["cell_type"] = ok["cell_type"].fillna("unknown").astype(str)
    ok["compartment"] = ok["compartment"].fillna("unknown").astype(str) if "compartment" in ok.columns else "unknown"
    ok["length_um"] = ok["length_nm"].astype(float) / 1000.0

    # ---- 1. which branches have spines ------------------------------------------------------
    print(f"== 1. status-ok branches: {len(ok)}; spine segments per branch (raw = all segments in spines/, spines = after the false-spine verdict) ==")
    ok["has_raw"] = ok["n_segments_total"].fillna(0) > 0
    ok["has_spines"] = ok["n_spines"].fillna(0) > 0
    ok["density_per_um"] = ok["n_spines"].fillna(0) / ok["length_um"]
    agg = dict(branches=("merged_branch_id", "size"), neurons=("neuron_id", "nunique"), share_with_raw_segments=("has_raw", "mean"),
               share_with_spines=("has_spines", "mean"), density_median_per_um=("density_per_um", "median"), length_median_um=("length_um", "median"),
               eligible_T=("train_eligible_T", "sum"), eligible_P=("train_eligible_P", "sum"))
    for by in (["compartment", "radius_class"], ["cell_type", "radius_class"]):
        print(f"\n  by {' x '.join(by)}:")
        print(ok.groupby(by).agg(**agg).round(2).to_string())
    per_neuron = ok.groupby("neuron_id").agg(cell_type=("cell_type", "first"), branches=("merged_branch_id", "size"), r_median_nm=("radius_median_nm", "median"),
                                             share_with_spines=("has_spines", "mean"), spines=("n_spines", "sum"), density_per_um=("density_per_um", "median"))
    print("\n  per neuron:")
    print(per_neuron.round(2).to_string())
    if "dist_perp_minus_radius" not in sp.columns:
        print("\nno spine positions in branch_spines.parquet (no S verdicts?): parts 2-3 skipped")
        return

    # ---- 2. signed offset of the attachment centre ------------------------------------------
    keys = ["neuron_id", "limb_id", "merged_branch_id"]
    s = sp[(sp["status"] == "valid") & sp["dist_perp_minus_radius"].notna() & ~sp["projects_to_end"].fillna(False).astype(bool)].copy()
    s = s.merge(ok[keys + ["radius_median_nm", "radius_class", "geometry_path", "cell_type"]], on=keys, how="inner")
    s["d"] = s["dist_perp_minus_radius"].astype(float)
    s["r"] = s["radius_at_s"].astype(float)
    r_ell = np.full(len(s), np.nan)
    for (n, l, m), idx in s.groupby(keys).indices.items():
        path = args.root / str(n) / f"limb_{int(l):03d}" / f"merged_{int(m)}" / "branch_geometry.npz"
        r_ell[idx] = ellipse_radius(path, s["s"].to_numpy(float)[idx], s["theta"].to_numpy(float)[idx])
    s["r_ell"] = r_ell
    s["d_ell"] = s["dist_perp"].astype(float) - s["r_ell"]
    print(f"\n== 2. signed offset d = |v_perp| - r of {len(s)} valid spines (not beyond the ends); > 0: the attachment centre is OUTSIDE the trunk surface ==")
    print("    quantiles:            " + "  ".join(f"{int(q * 100):>4d}%" for q in QUANTILES))
    for name, _, _ in RADIUS_CLASSES:
        x = s[s["radius_class"] == name]
        if not len(x):
            continue
        print(f"  {name}: {len(x)} spines, r(s) median {x['r'].median():.0f} nm")
        print(describe("circle, nm", x["d"], "nm"))
        print(describe("circle, / r", x["d"] / x["r"], "r"))
        e = x[x["r_ell"].notna()]
        if len(e):
            print(describe("ellipse, nm", e["d_ell"], "nm"))
            print(describe("ellipse, / r", e["d_ell"] / e["r"], "r"))

    # ---- 3. candidate rules -----------------------------------------------------------------
    print(f"\n== 3. outlier rules: share of spines without a position / share of status-ok branches with spines that fail 'spines' (position_fraction_min {args.position_fraction_min}) ==")
    in_range = s.groupby(keys).size().rename("n")

    def evaluate(outlier: pd.Series) -> str:
        pos = (~outlier).groupby([s[k] for k in keys]).sum()
        frac = (pos / in_range).fillna(1.0)
        parts = []
        for name, _, _ in RADIUS_CLASSES:
            m = s["radius_class"] == name
            if m.any():
                parts.append(f"{name} {outlier[m].mean():.0%}")
        return f"spines out: {outlier.mean():.0%} ({', '.join(parts)}); branches failing: {(frac < args.position_fraction_min).mean():.0%}"

    rules = [("now: |d| > 0.75 r", s["d"].abs() > 0.75 * s["r"])]
    for lo in (0.5,):
        for hi, floor in ((1.0, 0.0), (1.0, 300.0), (1.5, 300.0), (1.5, 400.0), (2.0, 400.0), (2.5, 500.0)):
            rules.append((f"d < -{lo} r or d > max({hi} r, {floor:.0f})", (s["d"] < -lo * s["r"]) | (s["d"] > np.maximum(hi * s["r"], floor))))
    if s["r_ell"].notna().any():
        de = s["d_ell"].fillna(s["d"])
        rules.append(("ellipse: |d_ell| > 0.75 r", de.abs() > 0.75 * s["r"]))
        rules.append(("ellipse: d_ell < -0.5 r or > max(1.5 r, 300)", (de < -0.5 * s["r"]) | (de > np.maximum(1.5 * s["r"], 300.0))))
    for name, outlier in rules:
        print(f"  {name:46s} {evaluate(outlier)}")

    out = args.root / "audit" / "spine_positions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    s[keys + ["spine_id", "cell_type", "radius_class", "s", "u", "theta", "r", "r_ell", "dist_perp", "d", "d_ell"]].to_csv(out, index=False)
    print(f"\nper spine: {out}")


if __name__ == "__main__":
    main()
