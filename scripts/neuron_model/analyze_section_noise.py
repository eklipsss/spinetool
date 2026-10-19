"""Is the 'kink' of the section centres real geometry or measurement noise? (reads ``branch_sections.npz`` only, no mesh)

    python scripts/neuron_model/analyze_section_noise.py --root C:/.../preprocessed_branches --neuron-id <id>

For every ok branch the deviation of each fitted centre from the midpoint of its two neighbours (the quantity the kink
flag thresholds at ``kink_factor * r_med``) is collected, per radius class:

* ``dev`` in nm and relative to r: if it is ~constant in nm and does not shrink with r, the flags of thin branches are
  an artefact of a threshold proportional to r;
* implied radius of curvature ``R = (2 step)^2 / (8 dev)``: a dev of 60 nm at the 500 nm step means R ~ 2 um, much
  sharper than a dendrite bends - noise (or a real kink) rather than smooth shape;
* lag-1 correlation of consecutive second-difference vectors: white noise of the centres gives -2/3, a smooth curve +,
  a single isolated kink -1/2..0. Values near -2/3 say the centres are noisy, not the branch crooked;
* radii relative to r_med (the 'large_radius' flag: > 1.5 r_med) and their spread.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def second_differences(centers: np.ndarray, radii: np.ndarray):
    idx = np.flatnonzero(np.isfinite(radii))
    if len(idx) < 3:
        return np.zeros((0, 3)), idx
    c = centers[idx]
    return c[1:-1] - 0.5 * (c[:-2] + c[2:]), idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--neuron-id", required=True)
    parser.add_argument("--step-nm", type=float, default=500.0, help="spacing of the sections (initial_step_nm)")
    args = parser.parse_args()
    d = args.root / args.neuron_id
    b = pd.read_parquet(d / "branches.parquet")
    b = b[(b["status"] == "ok") | (b["status"] == "no_centerline")]
    classes = {"r < 120 nm": (0, 120), "120-200 nm": (120, 200), "r >= 200 nm": (200, 1e9)}
    acc = {k: {"dev": [], "dev_over_r": [], "corr_num": 0.0, "corr_den": 0.0, "rad_ratio": [], "n_branches": 0} for k in classes}
    for row in b.to_dict("records"):
        r_med = row.get("radius_median_nm")
        path = d / f"limb_{int(row['limb_id']):03d}" / f"merged_{int(row['merged_branch_id'])}" / "branch_sections.npz"
        if r_med is None or not np.isfinite(r_med) or not path.exists():
            continue
        z = np.load(path)
        dev_vec, idx = second_differences(z["centers"], z["radii"])
        if not len(dev_vec):
            continue
        for name, (lo, hi) in classes.items():
            if lo <= r_med < hi:
                a = acc[name]
                dev = np.linalg.norm(dev_vec, axis=1)
                a["dev"].extend(dev.tolist())
                a["dev_over_r"].extend((dev / r_med).tolist())
                a["rad_ratio"].extend((z["radii"][idx] / r_med).tolist())
                if len(dev_vec) > 1:
                    a["corr_num"] += float((dev_vec[:-1] * dev_vec[1:]).sum())
                    a["corr_den"] += float((dev_vec ** 2).sum())
                a["n_branches"] += 1
    print(f"neuron {args.neuron_id}; section spacing {args.step_nm:.0f} nm; threshold of the kink flag: 0.5 r\n")
    for name, a in acc.items():
        if not a["dev"]:
            continue
        dev, rel, rad = np.array(a["dev"]), np.array(a["dev_over_r"]), np.array(a["rad_ratio"])
        med = float(np.median(dev))
        R = (2 * args.step_nm) ** 2 / (8 * med) / 1000.0 if med > 0 else float("inf")
        corr = a["corr_num"] / a["corr_den"] if a["corr_den"] > 0 else float("nan")
        print(f"{name:12s} {a['n_branches']:3d} branches, {len(dev):5d} centres")
        print(f"  deviation from the neighbours' midpoint: median {med:.0f} nm (10-90 %: {np.percentile(dev, 10):.0f}-{np.percentile(dev, 90):.0f}), relative to r: median {np.median(rel):.2f}; flagged (> 0.5 r): {np.mean(rel > 0.5):.0%}")
        print(f"  implied radius of curvature at the median deviation: {R:.1f} um;  lag-1 correlation of consecutive second differences: {corr:+.2f} (white noise: -0.67)")
        print(f"  radius / r_med: median {np.median(rad):.2f}, 10-90 %: {np.percentile(rad, 10):.2f}-{np.percentile(rad, 90):.2f}; > 1.5: {np.mean(rad > 1.5):.0%}")
        for floor in (60.0, 100.0, 150.0):
            r_of_centre = dev / np.maximum(rel, 1e-9)  # r_med of the branch the centre belongs to
            print(f"  with an absolute floor of {floor:.0f} nm under the limit max(0.5 r, floor): flagged {np.mean(dev > np.maximum(0.5 * r_of_centre, floor)):.0%}")
        print()


if __name__ == "__main__":
    main()
