"""What happened to the branches: statuses, why branches are not eligible for T / P, and what the failures look like.

    python scripts/neuron_model/summarize_branches.py --root C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches [--neuron-id <id>]

Reads ``<root>/branches.parquet`` (or the per-neuron table with ``--neuron-id``). Nothing is changed. Paste the output into the chat.
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


def _bins(frame: pd.DataFrame, column: str, edges, label: str) -> None:
    if column not in frame.columns or frame[column].notna().sum() == 0:
        return
    cut = pd.cut(frame[column], edges, include_lowest=True)
    print(f"  {label}: " + ", ".join(f"{k}: {v}" for k, v in cut.value_counts().sort_index().items() if v))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--neuron-id")
    args = parser.parse_args()
    path = (args.root / args.neuron_id / "branches.parquet") if args.neuron_id else args.root / "branches.parquet"
    b = pd.read_parquet(path)
    print(f"{len(b)} merged branches from {path}\n")
    print("status:", b["status"].value_counts().to_dict())
    for flag in ("train_eligible_T", "train_eligible_P"):
        if flag in b.columns:
            print(f"{flag}: {int(b[flag].fillna(False).sum())} of {len(b)} ({int((b['status'] == 'ok').sum())} with status ok)")

    ok = b[b["status"] == "ok"]
    print(f"\n== status ok: {len(ok)} branches; why NOT eligible for T (a branch can fail several checks) ==")
    reasons = Counter()
    for r in ok.loc[~ok["train_eligible_T"].fillna(False).astype(bool), "qc_reason"].fillna(""):
        reasons.update([x for x in str(r).split(";") if x])
    # length bounds are not in qc_reason: report them separately
    if "length_nm" in ok.columns:
        short = ((ok["length_nm"] < 5000.0) & ~ok["train_eligible_T"].fillna(False).astype(bool)).sum()
        reasons["length < 5 um (T only)"] = int(short)
    for k, v in reasons.most_common():
        print(f"  {k:28s} {v}")
    only = ok[~ok["train_eligible_T"].fillna(False).astype(bool)]
    if len(only) and "qc_reason" in only.columns:
        single = only["qc_reason"].fillna("").map(lambda s: s.split(";")[0] if s and ";" not in s else None).dropna().value_counts()
        if len(single):
            print("  fails exactly ONE check:", single.to_dict())

    print("\n== status ok: eligible for P but not for T, and vice versa ==")
    print("  P only:", int((ok["train_eligible_P"].fillna(False) & ~ok["train_eligible_T"].fillna(False)).sum()), " T only:", int((ok["train_eligible_T"].fillna(False) & ~ok["train_eligible_P"].fillna(False)).sum()))

    print("\n== lengths, nm ==")
    _bins(b, "table_length_nm", [0, 2000, 5000, 10000, 20000, 50000, 1e9], "all branches, table length")
    for status in ("no_centerline", "too_short"):
        sub = b[b["status"] == status]
        if len(sub):
            _bins(sub, "table_length_nm", [0, 2000, 5000, 10000, 20000, 50000, 1e9], f"{status} ({len(sub)})")

    nc = b[b["status"] == "no_centerline"]
    if len(nc):
        print(f"\n== no_centerline ({len(nc)}): the fit found fewer than 5 reliable sections ==")
        for col in ("n_sections", "n_reliable_sections", "radius_median_nm", "section_max_radius_nm", "n_mesh_vertices", "n_spine_vertices_removed"):
            if col in nc.columns and nc[col].notna().any():
                v = nc[col].astype(float)
                print(f"  {col:26s} median {v.median():.0f}, min {v.min():.0f}, max {v.max():.0f}")
        flag_cols = [c for c in nc.columns if c.startswith("sections_")]
        if flag_cols:
            print("  section flags (sum over these branches):", {c[len('sections_'):]: int(nc[c].fillna(0).sum()) for c in flag_cols if nc[c].fillna(0).sum() > 0})
        print("  by members:", nc["member_count"].value_counts().to_dict() if "member_count" in nc.columns else "n/a")
        cols = [c for c in ("limb_id", "merged_branch_id", "table_length_nm", "n_sections", "n_reliable_sections", "radius_median_nm") if c in nc.columns]
        print(nc[cols].sort_values("table_length_nm", ascending=False).head(10).to_string(index=False))

    if len(ok):
        print("\n== status ok: the quantities behind the checks (median, 10-90 %) ==")
        for col, note in (("length_nm", "nm"), ("radius_median_nm", "nm"), ("section_invalid_fraction", "limit 0.2"), ("trim_start_nm", "limit 2000"), ("trim_end_nm", "limit 2000"),
                          ("longest_gap_nm", "limit 3000"), ("centerline_rms_to_centers_nm", ""), ("max_r_kappa", "limit 1"), ("twist_max_per_nm", "limit 1e-5"), ("unknown_fraction", "limit 0.1")):
            if col in ok.columns and ok[col].notna().any():
                v = ok[col].astype(float)
                print(f"  {col:30s} {v.median():.4g}  ({v.quantile(0.1):.4g} .. {v.quantile(0.9):.4g})  {note}")
        if "n_spines_in_range" in ok.columns:
            print(f"  spines per branch: median {ok['n_spines_in_range'].median():.0f}, branches without spines {(ok['n_spines_in_range'] == 0).mean():.0%}")
        if "tube_self_contact" in ok.columns:
            print(f"  tube self-contact: {ok['tube_self_contact'].mean():.0%}; skeleton topology not ok: {(~ok['skeleton_topology_ok'].fillna(True).astype(bool)).mean():.0%}" if "skeleton_topology_ok" in ok.columns else "")
        if "soma_distance_mismatch" in ok.columns:
            print(f"  soma distance mismatch with the table (> 20 %): {ok['soma_distance_mismatch'].fillna(False).astype(bool).mean():.0%}")


if __name__ == "__main__":
    main()
