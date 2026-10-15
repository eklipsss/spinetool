"""Reverse-reconstruction audit of the branch preprocessing (round cross-section only).

Run after ``run_branch_preprocessing.py`` with the same paths:

    python scripts/neuron_model/audit_branch_reconstruction.py --raw-data-root O:/Datasets --metadata-root datasets --per-stratum 25

Writes ``<output_root>/audit/reconstruction_circle.parquet`` (per branch) and ``..._summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.neuron_model.branch_preprocessing import load_config
from src.neuron_model.branch_preprocessing.audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "data/processed/config/branch_preprocessing.yaml")
    parser.add_argument("--dataset", default="minnie65")
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--s-preprocessed-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--per-stratum", type=int, default=25, help="branches per length quartile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-test", action="store_true", help="also sample test-split neurons (not recommended)")
    args = parser.parse_args()
    cfg = load_config(
        args.config,
        {
            "dataset": args.dataset,
            "raw_data_root": args.raw_data_root.resolve(),
            "metadata_root": args.metadata_root.resolve(),
            "s_preprocessed_root": args.s_preprocessed_root.resolve() if args.s_preprocessed_root else None,
            "output_root": args.output_root.resolve() if args.output_root else None,
        },
    )
    print(json.dumps(run_audit(cfg, per_stratum=args.per_stratum, seed=args.seed, exclude_test=not args.include_test), indent=2))


if __name__ == "__main__":
    main()
