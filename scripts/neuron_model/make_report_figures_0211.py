"""Figures 1 and 4 of legacy/02.11-08.11.md (only reads tables; matplotlib on CPU, seconds).

Figure 1: distribution of cos(initial limb direction, direction to the pia) for apical / basal / axon-like limbs.
    python scripts/neuron_model/make_report_figures_0211.py fig1 --out legacy/02.11-08.11_figures \
        --limbs "Minnie65=<...apical-by-direction-minnie65>/limbs.csv" "H01=<...apical-by-direction-h01>/limbs.csv"

Figure 4: AUC "real vs set" by feature group for the generators and the processing controls.
    python scripts/neuron_model/make_report_figures_0211.py fig4 --out legacy/02.11-08.11_figures \
        --eval-run runs/analysis/neuron-model/2026-11-06_112409_generator-evaluation-untouched
  reads <eval-run>/classifier_ablation/postprocessed/summary_real.csv (classifier_ablation.py --eval-run <eval-run>);
  --models limits/orders the sets (default: all rows of the table).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PINK, GREEN, VIOLET, GREY, INK = "#d6418f", "#3f8a2a", "#8a6bd1", "#8e9196", "#1d1d1f"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#c9cbd0", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ececef", "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "axes.unicode_minus": False,
})

CLASSES = [("apical", "апикальные", PINK), ("basal", "базальные", GREEN), ("axon_like", "аксоноподобные", VIOLET)]
GROUPS = [("auc_all", "все признаки", INK), ("auc_no_attachment", "без признаков основания", GREEN),
          ("auc_attachment_only", "только признаки основания", PINK)]


def fig1(args) -> None:
    items = [s.split("=", 1) for s in args.limbs]
    fig, axes = plt.subplots(1, len(items), figsize=(5.6 * len(items), 4), sharey=False, squeeze=False)
    bins = np.linspace(-1, 1, 41)
    for ax, (name, path) in zip(axes[0], items):
        d = pd.read_csv(path)
        label = d["compartment_dir"].astype(str)
        for key, ru, color in CLASSES:
            if key == "axon_like":
                sel = d[d["axon_like"].astype(bool)] if "axon_like" in d else d[label == key]
            else:
                sel = d[(label == key) & ~d.get("axon_like", pd.Series(False, index=d.index)).astype(bool)]
            sel = sel["cos_pia"].dropna()
            if len(sel):
                ax.hist(sel, bins=bins, color=color, alpha=0.65, label=f"{ru} (n = {len(sel)})")
        ax.axvline(0.6, color=GREY, ls="--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("косинус направления лимба к поверхности коры")
        ax.set_ylabel("число лимбов")
        ax.legend()
    _save(fig, args.out, "figure1_limb_direction_cosine.png")


def fig4(args) -> None:
    path = args.eval_run / "classifier_ablation" / args.kind / "summary_real.csv"
    d = pd.read_csv(path)
    if args.models:
        d = d.set_index("model").loc[args.models].reset_index()
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(d) + 3), 4.6))
    x = np.arange(len(d))
    w = 0.8 / len(GROUPS)
    for i, (col, ru, color) in enumerate(GROUPS):
        if col in d:
            ax.bar(x + (i - (len(GROUPS) - 1) / 2) * w, d[col], w, color=color, label=ru)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], rotation=30, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUC «реальные против набора»")
    ax.legend(ncol=len(GROUPS), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    _save(fig, args.out, "figure4_auc_by_feature_group.png")


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {out / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fig1")
    a.add_argument("--limbs", nargs="+", required=True, help="NAME=limbs.csv")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("fig4")
    b.add_argument("--eval-run", type=Path, required=True)
    b.add_argument("--kind", default="postprocessed")
    b.add_argument("--models", nargs="*")
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"fig1": fig1, "fig4": fig4}[args.cmd](args)


if __name__ == "__main__":
    main()
