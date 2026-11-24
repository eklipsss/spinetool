"""Spine morphometrics of the LabID groups (health, treatment), source image as the unit (``src/neuron_model/labid_group_effects.py``).

    python scripts/neuron_model/labid_group_effects.py --morphometrics <labid preprocessed>/morphometrics.parquet \
        --acquisition data/interim/neuron-model/labid_acquisition.csv --out <dir>

``--morphometrics``: the stage-8 table of the LabID S preprocessing (conversion WITHOUT the z factor 0.7, i.e. labid_raw_z1*),
``--acquisition``: per-image table of ``scripts/neuron_model/labid_acquisition_metadata.py``. Wt_old has no acquisition
metadata and is not compared. Writes ``effects.csv``, ``units.csv`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.labid_group_effects import METRICS, run, unit_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morphometrics", type=Path, required=True)
    ap.add_argument("--acquisition", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    morph = pd.read_parquet(args.morphometrics)
    if "status" in morph.columns:
        morph = morph[morph["status"] == "success"]
    acq = pd.read_csv(args.acquisition)
    units = unit_table(morph, acq)
    table = run(morph, acq, n_perm=args.n_perm, n_boot=args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.out / "units.csv", index=False)
    table.to_csv(args.out / "effects.csv", index=False)

    lines = ["# LabID: differences between groups (unit = source image)", "",
             "Effect = mean over images of the median log metric, group a minus group b, as a ratio − 1 (%). "
             "p: permutation of group labels among images, interval: bootstrap of images, q: Benjamini–Hochberg over 11 metrics. "
             "Descriptive comparisons (treatment mixed with deconvolution / objective) carry no significance mark.", ""]
    for name, g in table.groupby("comparison", sort=False):
        r0 = g.iloc[0]
        lines += [f"## {name}", "", f"{r0.a}: {r0.n_units_a} images, {r0.n_spines_a} spines; {r0.b}: {r0.n_units_b} images, {r0.n_spines_b} spines. {r0.note}.", "",
                  "| metric | effect, % | 95 % interval | p | q |", "| --- | --- | --- | --- | --- |"]
        for _, r in g.iterrows():
            if not np.isfinite(r.effect):
                lines.append(f"| {r.metric} | — | — | — | — |")
                continue
            star = "*" if r.significant else ""
            lines.append(f"| {r.metric} | {r.effect_pct:+.1f}{star} | [{r.lo_pct:+.1f}; {r.hi_pct:+.1f}] | {r.p:.4f} | {r.q:.3f} |")
        n_sig = int(g["significant"].sum())
        lines += ["", f"Significant (q < 0.05): {n_sig} of {int(g['effect'].notna().sum())}" + ("" if r0.inferential else " (descriptive, not tested)"), ""]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
