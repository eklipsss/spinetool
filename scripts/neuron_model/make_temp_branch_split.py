"""A TEMPORARY neuron-level split for trying T0 / P0 on a partial branch preprocessing (e.g. the 10-neuron pilot).

Not ``splits_v1``: the frozen split is built after the S preprocessing is finished. This one only lets ``fit_t0`` /
``fit_p0`` / ``generate_tp`` run on the neurons that are already preprocessed (needs no NAS).

    python scripts/neuron_model/make_temp_branch_split.py --root C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches

Writes ``<root>/splits_tmp.parquet`` (format of ``splits_*.parquet``: ``spine_key`` = dataset/neuron/-/-/-, ``split``);
pass it as ``data.splits=<root>/splits_tmp.parquet``. At least one neuron goes to val and one to test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="output root of run_branch_preprocessing.py")
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default <root>/splits_tmp.parquet")
    args = parser.parse_args()
    table = pd.read_parquet(args.root / "branches.parquet", columns=["dataset", "neuron_id"]).drop_duplicates()
    neurons = sorted(table["neuron_id"].astype(str))
    if len(neurons) < 3:
        parser.error(f"need at least 3 neurons, found {len(neurons)}")
    dataset = str(table["dataset"].iloc[0])
    order = np.random.default_rng(args.seed).permutation(len(neurons))
    n_val = max(1, int(round(args.val * len(neurons))))
    n_test = max(1, int(round(args.test * len(neurons))))
    split = np.array(["train"] * len(neurons), dtype=object)
    split[order[:n_val]] = "val"
    split[order[n_val:n_val + n_test]] = "test"
    out = args.out or args.root / "splits_tmp.parquet"
    frame = pd.DataFrame({"spine_key": [f"{dataset}/{n}/-/-/-" for n in neurons], "neuron_id": neurons, "split": split})
    frame.to_parquet(out, index=False)
    print(frame["split"].value_counts().to_dict(), "->", out)
    print(frame[["neuron_id", "split"]].to_string(index=False))


if __name__ == "__main__":
    main()
