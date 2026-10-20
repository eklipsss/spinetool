"""Change the eligibility thresholds of finished branches WITHOUT redoing the geometry.

The geometry (centreline, radius, frame, positions) does not depend on the eligibility thresholds, the flags
``train_eligible_T/P`` and ``qc_*`` are functions of quantities stored in ``branches.parquet``. This recomputes them in
minutes for the whole dataset (no mesh is read).

    # what would change? (nothing is written)
    python scripts/neuron_model/reapply_eligibility.py --output-root C:/.../preprocessed_branches --dry-run \
        --set invalid_section_fraction_max=0.5 --set spline_rms_floor_nm=200

    # apply (rewrites branches.parquet of every finished neuron and the dataset-level tables)
    python scripts/neuron_model/reapply_eligibility.py --output-root C:/.../preprocessed_branches --set invalid_section_fraction_max=0.5

Thresholds: length_min_t_nm, length_max_t_nm, invalid_section_fraction_max, radius_min_nm, radius_max_nm, trim_max_nm,
unknown_fraction_max, position_fraction_min, spline_rms_factor, spline_rms_floor_nm, long_gap_nm, twist_tolerance_per_nm,
centerline_inside_min, twist_relative_max, position_outlier_factor, position_outlier_floor_nm, position_inside_factor
(the spine position rule is re-applied to the stored offsets: has_position / n_with_position in branch_spines.parquet).

``--refresh-metadata`` re-reads the compartment of every limb (needs ``--metadata-root``; the lookup was fixed on
2026-10-20: before, only the first neuron of the compartment table got its compartments).
Defaults: data/processed/config/branch_preprocessing.yaml. Note the values you applied: they are NOT stored in the tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.branch_preprocessing import load_config, reapply_eligibility
from src.neuron_model.branch_preprocessing.config import _ELIGIBILITY_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "data/processed/config/branch_preprocessing.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="threshold override (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="only count, write nothing")
    parser.add_argument("--refresh-metadata", action="store_true", help="re-read the compartments from the metadata tables")
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        name, value = item.split("=", 1)
        if name not in _ELIGIBILITY_FIELDS:
            parser.error(f"{name!r} is not an eligibility threshold ({sorted(_ELIGIBILITY_FIELDS)}); other parameters change the geometry")
        overrides[name] = float(value)
    cfg = load_config(args.config, {"output_root": args.output_root.resolve(), "metadata_root": args.metadata_root.resolve(), **overrides})
    print("thresholds:", {k: getattr(cfg, k) for k in sorted(_ELIGIBILITY_FIELDS)})
    result = reapply_eligibility(cfg, write=not args.dry_run, refresh_metadata=args.refresh_metadata)
    print(json.dumps(result, indent=2, default=str))
    if args.dry_run:
        print("(dry run: nothing written)")


if __name__ == "__main__":
    main()
