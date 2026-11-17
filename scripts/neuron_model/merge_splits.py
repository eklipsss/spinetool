"""Merge the group splits of several datasets into one split file (e.g. splits_v1 of Minnie65 + splits_h01_v1 of H01).

    python scripts/neuron_model/merge_splits.py data/processed/neuron-model/splits/splits_v1.parquet ^
        data/processed/neuron-model/splits/splits_h01_v1.parquet --out data/processed/neuron-model/splits/splits_v1_h01.parquet

Every input keeps its own assignment (nothing is re-split, so the untouched test parts stay what they were); the groups
carry the dataset (``<dataset>/<neuron>``), so two datasets cannot share a group. The merged file is checked for leakage
(a group in two parts, a spine twice) and is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.neuron_model.spine_generation.data.splits import assert_no_leakage, load_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists (a split is frozen once written)")
    frames = [load_splits(p).assign(source_split=p.stem) for p in args.splits]
    merged = pd.concat(frames, ignore_index=True)
    assert_no_leakage(merged)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    dataset = merged["spine_key"].astype(str).str.split("/").str[0]
    summary = merged.assign(dataset=dataset).groupby(["dataset", "split"]).agg(spines=("spine_key", "size"), groups=("group_id", "nunique"))
    args.out.with_suffix(".json").write_text(json.dumps({"inputs": [str(p) for p in args.splits],
                                                         "summary": {f"{d}/{s}": {k: int(v) for k, v in row.items()} for (d, s), row in summary.iterrows()}},
                                                        indent=2), encoding="utf-8")
    print(summary.to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
