"""Reasons behind failed / invalid / needs_review spines of an S preprocessing output (any dataset), from the stage tables.

    python scripts/neuron_model/summarize_s_tables.py --preprocessed-root C:\\...\\h01\\preprocessed\\H01_Seg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_preprocessing import _read_table_frame  # noqa: E402

TABLES = ["sealed_spines", "false_spines", "oriented_spines", "mesh_quality", "sdf_samples", "pointclouds", "morphometrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    args = parser.parse_args()
    for name in TABLES:
        path = args.preprocessed_root / f"{name}.parquet"
        frame = _read_table_frame(path)
        if not len(frame):
            continue
        print(f"\n== {name}: {len(frame)} rows; columns: {list(frame.columns)[:14]}")
        status = frame["status"].astype(str)
        print(status.value_counts().to_string())
        for col in ("reason", "error", "review_reasons", "quality_reasons"):
            if col in frame.columns:
                bad = frame[status.isin(["failed", "invalid", "needs_review"])]
                if len(bad):
                    print(f"-- {col} among non-success:")
                    print(bad[col].astype(str).str.slice(0, 90).value_counts().head(8).to_string())
        if name == "sdf_samples":  # which of the SDF checks failed (spine_sampling.sdf_quality), and the link to the mesh QC
            bad = frame[status == "invalid"]
            checks = {
                "not all finite": ~bad.get("sdf_all_finite", pd.Series(True, index=bad.index)).astype(bool),
                "no positive": ~bad.get("sdf_has_positive", pd.Series(True, index=bad.index)).astype(bool),
                "no negative": ~bad.get("sdf_has_negative", pd.Series(True, index=bad.index)).astype(bool),
                "surface |sdf| > tol": bad.get("sdf_surface_max_abs", pd.Series(0.0, index=bad.index)).astype(float) > 1e-3,
                "exterior point not outside": bad.get("sdf_exterior_check", pd.Series(1.0, index=bad.index)).astype(float) <= 0,
                "interior probe < 0.9": bad.get("sdf_interior_fraction", pd.Series(1.0, index=bad.index)).astype(float) < 0.9,
                "origin outside the query box": ~bad.get("sdf_in_local_coordinates", pd.Series(True, index=bad.index)).astype(bool),
            }
            print("-- failed SDF checks among invalid (a spine may fail several):")
            for k, v in checks.items():
                print(f"   {k:32s} {int(v.sum())}")
            if "sdf_interior_fraction" in bad.columns:
                print("-- interior fraction of invalid, quantiles:", bad["sdf_interior_fraction"].astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist())
            qc = _read_table_frame(args.preprocessed_root / "mesh_quality.parquet")
            if len(qc):
                keys = ["dataset", "neuron_id", "limb_id", "branch_id", "spine_id"]
                m = frame[keys + ["status"]].merge(qc[keys + ["status", "reason"]].rename(columns={"status": "qc_status", "reason": "qc_reason"}), on=keys, how="left")
                print("-- SDF status by mesh-QC reason:")
                print(pd.crosstab(m["qc_reason"].fillna("valid"), m["status"]).to_string())
        if name == "false_spines":
            for col in frame.columns:
                if "d_max" in col or "max_distance" in col:
                    print(f"-- {col} (nm) quantiles:", frame[col].astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(0).tolist())


if __name__ == "__main__":
    main()
