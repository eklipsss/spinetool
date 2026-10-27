"""Does a stage-8 morphometrics table cover the spines of a manifest + split (condition_effects.py, evaluation)?

    python scripts/neuron_model/check_morphometrics_table.py --table cache/morphometrics_minnie65_full.parquet \
        --manifest cache/minnie65_dev/manifest.parquet --splits data/processed/neuron-model/splits/splits_dev_100n.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.data.index import load_spine_index  # noqa: E402
from src.neuron_model.spine_generation.data.splits import load_splits  # noqa: E402
from src.neuron_model.spine_generation.evaluation.morphometrics import _read_stage_table  # noqa: E402
from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--path-mode", default="relative")
    args = parser.parse_args()
    table = _read_stage_table(args.table)
    print(f"table: {len(table)} spines, columns present: {[m for m in SCALAR_METRICS if m in table.columns]}")
    missing_cols = [m for m in SCALAR_METRICS if m not in table.columns]
    print(f"missing metric columns: {missing_cols or 'none'}")
    index = load_spine_index([args.manifest], path_mode=args.path_mode, check_files=())
    index = index.with_split(load_splits(args.splits), split=args.split, drop_unassigned=True)
    keys = set(index.frame["spine_key"])
    sub = table[table["spine_key"].isin(keys)]
    print(f"split '{args.split}': {len(keys)} spines, found in the table: {len(sub)} ({len(sub) / max(len(keys), 1):.1%})")
    for m in SCALAR_METRICS:
        if m in sub.columns:
            print(f"  {m:24s} non-null {sub[m].notna().mean():.1%}")


if __name__ == "__main__":
    main()
