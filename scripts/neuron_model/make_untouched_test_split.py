"""Split file for a CLEAN final evaluation of the dev-trained generators.

train / val are those of the dev split (``--dev``, e.g. splits_dev_100n: what the generators were trained and tuned on), the dev
TEST part is dropped (it was already used to choose between models), and the new test part is the TEST part of ``--final``
(e.g. splits_v1) minus every neuron of the dev split: neurons that no model, selection or earlier evaluation has touched.
The same untouched test as ``tune_flow_prior.py --test-splits`` (there it is built on the fly).

    python scripts/neuron_model/make_untouched_test_split.py \
        --dev data/processed/neuron-model/splits/splits_dev_100n.parquet \
        --final data/processed/neuron-model/splits/splits_v1.parquet \
        --out data/processed/neuron-model/splits/splits_dev_100n_untouched_test.parquet

Then ``evaluate_generators.py ... data.splits=<out> data.drop_unassigned=true``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.neuron_model.spine_generation.data.splits import load_splits, save_splits


def untouched_split(dev: pd.DataFrame, final: pd.DataFrame, *, final_part: str = "test", train_source: str = "dev") -> pd.DataFrame:
    """dev train/val + (final ``final_part`` minus all dev neurons); columns ``spine_key, group_id, split``.

    ``train_source="final"`` (comparison of a dev-trained model with a final-trained one on final VAL neurons that the dev
    model never saw, 2026-11-17): train = the train part of ``final`` instead of dev train/val."""
    dev_groups = set(dev["group_id"])
    if train_source == "final":
        keep_dev = final[final["split"] == "train"][["spine_key", "group_id", "split"]]
    else:
        keep_dev = dev[dev["split"].isin(["train", "val"])][["spine_key", "group_id", "split"]]
    test = final[(final["split"] == final_part) & ~final["group_id"].isin(dev_groups)][["spine_key", "group_id"]].assign(split="test")
    test = test[~test["spine_key"].isin(set(keep_dev["spine_key"]))]
    return pd.concat([keep_dev, test], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, required=True, help="split the generators were trained on (its test part is dropped)")
    ap.add_argument("--final", type=Path, required=True, help="split whose test neurons (minus the dev neurons) form the new test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final-part", default="test", help="part of --final that (minus the dev neurons) becomes the test, e.g. val")
    ap.add_argument("--train-source", default="dev", choices=("dev", "final"), help="train part: dev train/val (default) or the train part of --final")
    args = ap.parse_args()

    dev, final = load_splits(args.dev), load_splits(args.final)
    table = untouched_split(dev, final, final_part=args.final_part, train_source=args.train_source)
    counts = {s: {"spines": int((table.split == s).sum()), "neurons": int(table.loc[table.split == s, "group_id"].nunique())} for s in ("train", "val", "test")}
    dropped = dev[dev["split"] == "test"]
    print(f"dev test dropped: {len(dropped)} spines of {dropped['group_id'].nunique()} neurons")
    for s, c in counts.items():
        print(f"{s:5s}: {c['spines']:7d} spines of {c['neurons']:4d} neurons")
    if counts["test"]["spines"] == 0:
        raise SystemExit("no untouched test neurons")
    save_splits(table, args.out, version=args.out.stem,
                info={"built_from": {"dev": str(args.dev), "final": str(args.final)}, "counts": counts,
                      "rule": f"train: {args.train_source}; test = {args.final_part} of final minus every neuron of dev"})
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
