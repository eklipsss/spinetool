"""Look at the section centres of branches (reads ``branch_sections.npz`` / ``branch_geometry.npz``, no mesh): is the jitter
noise or shape?

    python scripts/neuron_model/plot_branch_sections.py --root C:/.../preprocessed_branches --neuron-id <id> [--branches 3 3]
    python scripts/neuron_model/plot_branch_sections.py --root ... --neuron-id <id> --merged-branch-id 10067 --merged-branch-id 12

Without --merged-branch-id: the ``--branches`` N_thin N_thick longest status-ok branches with r < 200 nm / r >= 200 nm.
One PNG per run (``<root>/<neuron>/section_plots.png``), one row per branch, three panels:

1. offsets of the section centres from their own 5-point moving average, along the two directions perpendicular to the
   branch, vs the position along it (nm). Red = flagged (any flag), the lines are +-0.5 r and +-150 nm. White noise looks like
   independent scatter around 0; real shape looks like smooth waves and steps that last for several sections.
2. radius of the sections vs position; r_med and 1.5 r_med (the large-radius limit).
3. the deviation from the neighbours' midpoint (the kink statistic) with the limits 0.5 r and 150 nm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FLAG_NAMES = {1: "few_points", 2: "large_radius", 4: "kink", 8: "outside_mesh", 16: "end_section"}


def _load(d: Path, row) -> dict:
    z = np.load(d / f"limb_{int(row['limb_id']):03d}" / f"merged_{int(row['merged_branch_id'])}" / "branch_sections.npz")
    return {k: np.asarray(z[k]) for k in ("points", "tangents", "centers", "radii", "flags")}


def _perp_basis(t: np.ndarray):
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    ref = np.where((np.abs(t[:, [0]]) < 0.9), np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    a = np.cross(t, ref)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = np.cross(t, a)
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--neuron-id", required=True)
    parser.add_argument("--merged-branch-id", type=int, action="append", dest="ids")
    parser.add_argument("--branches", type=int, nargs=2, default=[3, 3], metavar=("N_THIN", "N_THICK"))
    parser.add_argument("--floor-nm", type=float, default=150.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = args.root / args.neuron_id
    b = pd.read_parquet(d / "branches.parquet")
    b = b[b["status"] == "ok"]
    if args.ids:
        chosen = b[b["merged_branch_id"].isin(args.ids)]
    else:
        thin = b[b["radius_median_nm"] < 200].sort_values("length_nm", ascending=False).head(args.branches[0])
        thick = b[b["radius_median_nm"] >= 200].sort_values("length_nm", ascending=False).head(args.branches[1])
        chosen = pd.concat([thin, thick])
    if chosen.empty:
        raise SystemExit("no branches to plot")
    fig, axes = plt.subplots(len(chosen), 3, figsize=(18, 3.2 * len(chosen)), squeeze=False)
    for row_axes, row in zip(axes, chosen.to_dict("records")):
        sec = _load(d, row)
        fitted = np.isfinite(sec["radii"])
        idx = np.flatnonzero(fitted)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sec["points"], axis=0), axis=1))] / 1000.0  # um
        c, r, flags = sec["centers"][idx], sec["radii"][idx], sec["flags"][idx]
        r_med = float(row["radius_median_nm"])
        # panel 1: offsets from the 5-point moving average, in the plane perpendicular to the branch
        k = 2
        smooth = np.full_like(c, np.nan)
        for j in range(len(c)):
            lo, hi = max(0, j - k), min(len(c), j + k + 1)
            smooth[j] = c[lo:hi].mean(axis=0)
        a, bb = _perp_basis(sec["tangents"][idx])
        off_a, off_b = ((c - smooth) * a).sum(1), ((c - smooth) * bb).sum(1)
        ax = row_axes[0]
        ax.plot(s[idx], off_a, "-o", ms=3, lw=0.7, label="perp. 1")
        ax.plot(s[idx], off_b, "-o", ms=3, lw=0.7, label="perp. 2")
        bad = flags != 0
        ax.scatter(s[idx][bad], off_a[bad], s=22, facecolors="none", edgecolors="red", label="flagged")
        for lim, style in ((0.5 * r_med, "--"), (args.floor_nm, ":")):
            ax.axhline(lim, color="k", ls=style, lw=0.8)
            ax.axhline(-lim, color="k", ls=style, lw=0.8)
        ax.set(title=f"limb {int(row['limb_id'])} merged {int(row['merged_branch_id'])}: L {row['length_nm'] / 1000:.1f} um, r_med {r_med:.0f} nm  (dashed 0.5 r, dotted {args.floor_nm:.0f} nm)", xlabel="position, um", ylabel="centre - moving average, nm")
        ax.legend(fontsize=7, loc="upper right")
        # panel 2: radius
        ax = row_axes[1]
        ax.plot(s[idx], r, "-o", ms=3, lw=0.7)
        ax.axhline(r_med, color="k", lw=0.8)
        ax.axhline(1.5 * r_med, color="r", ls="--", lw=0.8)
        ax.set(title="section radius (red dashed: 1.5 r_med, 'large_radius')", xlabel="position, um", ylabel="nm")
        # panel 3: kink statistic
        ax = row_axes[2]
        dev = np.linalg.norm(c[1:-1] - 0.5 * (c[:-2] + c[2:]), axis=1) if len(c) > 2 else np.zeros(0)
        ax.bar(s[idx][1:-1], dev, width=0.35, color=["tab:red" if (f & 4) else "tab:gray" for f in flags[1:-1]])
        ax.axhline(0.5 * r_med, color="k", ls="--", lw=0.8)
        ax.axhline(args.floor_nm, color="k", ls=":", lw=0.8)
        ax.set(title="deviation from the neighbours' midpoint (red = flagged kink)", xlabel="position, um", ylabel="nm")
    fig.tight_layout()
    out = args.out or d / "section_plots.png"
    fig.savefig(out, dpi=100)
    print(f"[plot] {len(chosen)} branches -> {out}")


if __name__ == "__main__":
    main()
