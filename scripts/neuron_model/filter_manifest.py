"""Copy of a manifest (or any spine table) without the rows whose column matches a regular expression.

    python scripts/neuron_model/filter_manifest.py --manifest <in.parquet> --exclude "neuron_id~^Wt_old" --out <out.parquet>

Used to build the LabID split without the group Wt_old (no source images, not analysed), see
configs/neuron-model/data/splits_labid_v1.yaml. ``--keep COLUMN~REGEX`` keeps only matching rows instead (repeatable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--exclude", action="append", default=[], metavar="COLUMN~REGEX")
    ap.add_argument("--keep", action="append", default=[], metavar="COLUMN~REGEX")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frame = pd.read_parquet(args.manifest)
    n0 = len(frame)
    for item, keep in [(i, False) for i in args.exclude] + [(i, True) for i in args.keep]:
        column, sep, pattern = item.partition("~")
        if not sep or column not in frame.columns:
            ap.error(f"{item!r}: expected COLUMN~REGEX with a column of the table")
        hit = frame[column].astype(str).str.contains(pattern, regex=True)
        frame = frame[hit if keep else ~hit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"{n0} -> {len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    main()
