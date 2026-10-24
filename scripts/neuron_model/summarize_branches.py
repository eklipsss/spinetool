"""Text summary of a branch preprocessing output (reads only branches.parquet / branch_spines.parquet): statuses,
eligibility, QC reasons and medians of the main quantities, overall and per group (default: ``labid_group`` for LabID,
``cell_type_binary`` otherwise). Paste the output into the chat instead of sending files.

    python scripts/neuron_model/summarize_branches.py --root C:\\...\\preprocessed_branches [--group-by compartment]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# kappa_rms is printed in 1/um
COLUMNS = ["length_nm", "radius_median_nm", "section_invalid_fraction", "centerline_inside_fraction", "tortuosity", "kappa_rms", "step_nm", "n_spines", "n_with_position"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group-by", default=None)
    args = parser.parse_args()
    branches = pd.read_parquet(args.root / "branches.parquet")
    spines_path = args.root / "branch_spines.parquet"
    spines = pd.read_parquet(spines_path) if spines_path.exists() else pd.DataFrame()
    group = args.group_by or ("labid_group" if "labid_group" in branches.columns else "cell_type_binary")
    if group not in branches.columns:
        group = None
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"branches {len(branches)}, neurons {branches['neuron_id'].nunique()}, spine rows {len(spines)}")
    print("status:", branches["status"].value_counts().to_dict())
    print(f"train_eligible_T {int(branches['train_eligible_T'].sum())}, train_eligible_P {int(branches['train_eligible_P'].sum())}")
    reasons = branches["qc_reason"].dropna().str.split(";").explode().value_counts()
    print("qc reasons (branches failing each):", reasons.to_dict())
    if "kappa_rms" in branches.columns:
        branches["kappa_rms"] = branches["kappa_rms"] * 1000.0  # 1/nm -> 1/um
    cols = [c for c in COLUMNS if c in branches.columns]
    print("\nmedians, all branches with status ok:")
    print(branches.loc[branches["status"] == "ok", cols].median().round(3).to_string())
    if group:
        ok = branches[branches["status"] == "ok"]
        table = ok.groupby(group, dropna=False)[cols].median().round(3)
        table.insert(0, "n_ok", ok.groupby(group, dropna=False).size())
        table.insert(1, "n_all", branches.groupby(group, dropna=False).size())
        table.insert(2, "elig_T", branches.groupby(group, dropna=False)["train_eligible_T"].sum())
        table.insert(3, "elig_P", branches.groupby(group, dropna=False)["train_eligible_P"].sum())
        print(f"\nper {group}:")
        print(table.to_string())
    if len(spines):
        print("\nspines: has_position", round(float(spines["has_position"].mean()), 3), "| status", spines["status"].value_counts().to_dict())
        if "attachment_method" in spines.columns:
            print("attachment_method:", spines["attachment_method"].value_counts().to_dict())
        if "dist_perp_minus_radius" in spines.columns:
            print("attachment offset from the trunk surface d - r (nm), quantiles 10/50/90:", spines["dist_perp_minus_radius"].quantile([0.1, 0.5, 0.9]).round(0).tolist())
    failed = branches[branches["status"] == "failed"]
    if len(failed):
        print("\nfailed branches, first 5 errors:")
        print(failed[["neuron_id", "error"]].head(5).to_string())


if __name__ == "__main__":
    main()
