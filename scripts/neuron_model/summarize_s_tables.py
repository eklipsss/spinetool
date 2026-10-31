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
        if name == "false_spines":
            for col in frame.columns:
                if "d_max" in col or "max_distance" in col:
                    print(f"-- {col} (nm) quantiles:", frame[col].astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(0).tolist())


if __name__ == "__main__":
    main()
