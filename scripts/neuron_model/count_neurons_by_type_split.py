"""How many neurons of each detailed cell type are in each part of a split (train / val / test)?

A neuron counts when it has at least ``--min-spines`` spines in the stage-8 table (as the conditional report: a pool needs >= 6
such neurons). Quick (seconds, CPU), no model is involved.

    python scripts/neuron_model/count_neurons_by_type_split.py ^
        --morphometrics-table C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet ^
        --splits data\\processed\\neuron-model\\splits\\splits_v1.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402


def count(frame: pd.DataFrame, min_spines: int) -> pd.DataFrame:
    """Neurons with >= ``min_spines`` spines per detailed type (rows) and split (columns), with spine counts."""
    per = frame.groupby(["neuron_id", "split", "cell_type"], dropna=False).size().rename("n_spines").reset_index()
    per = per[per["n_spines"] >= min_spines]
    neurons = per.pivot_table(index="cell_type", columns="split", values="neuron_id", aggfunc="nunique", fill_value=0)
    spines = per.pivot_table(index="cell_type", columns="split", values="n_spines", aggfunc="sum", fill_value=0)
    out = neurons.add_prefix("neurons_").join(spines.add_prefix("spines_"))
    cols = [c for c in ("neurons_train", "neurons_val", "neurons_test") if c in out.columns]
    return out.sort_values(cols[0] if cols else out.columns[0], ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morphometrics-table", type=Path, nargs="+", required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["minnie65"], choices=("minnie65", "h01"))
    ap.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    ap.add_argument("--min-spines", type=int, default=20)
    ap.add_argument("--min-neurons", type=int, default=6, help="pool size the conditional report needs")
    ap.add_argument("--out", type=Path, help="csv")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("morphology_by_condition", REPO_ROOT / "scripts/neuron_model/morphology_by_condition.py")
    mbc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mbc)
    frame = mbc.load_spines(args.morphometrics_table, args.metadata_root.resolve(), args.splits, None, datasets=args.datasets)
    table = count(frame, args.min_spines)
    if "neurons_val" in table.columns:
        table["val_enough"] = table["neurons_val"] >= args.min_neurons
    print(table.to_string())
    if "val_enough" in table.columns:
        print("\ntypes with >= %d validation neurons: %s" % (args.min_neurons, ", ".join(str(t) for t in table.index[table["val_enough"]])))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
