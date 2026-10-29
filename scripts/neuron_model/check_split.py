"""What a frozen split looks like: spines / neurons per split, the stratification (cell type, compartment) per split and,
optionally, whether a smaller split (e.g. the dev split) is consistent with it (a neuron has the same split in both).

    python scripts/neuron_model/check_split.py --splits data/processed/neuron-model/splits/splits_v1.parquet \
        --manifest cache/manifest_minnie65_meta.parquet --other data/processed/neuron-model/splits/splits_dev_100n.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def neuron_of(keys: pd.Series) -> pd.Series:
    """spine_key = dataset/neuron_id/limb/branch/spine."""
    return keys.astype(str).str.split("/").str[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="manifest with cell_type_binary / compartment (refresh_manifest_metadata.py output)")
    parser.add_argument("--other", type=Path, help="another split to compare (neurons shared with --splits)")
    args = parser.parse_args()
    s = pd.read_parquet(args.splits)
    s["neuron_id"] = neuron_of(s["spine_key"])
    per_neuron = s.drop_duplicates("neuron_id")[["neuron_id", "split", "stratum"]]
    print(f"{len(s)} spines, {len(per_neuron)} neurons; neurons per split: {per_neuron['split'].value_counts().to_dict()}; spines per split: {s['split'].value_counts().to_dict()}")
    print("strata (distinct):", per_neuron["stratum"].nunique(), "| most common:", per_neuron["stratum"].value_counts().head(4).to_dict())
    m = pd.read_parquet(args.manifest, columns=["neuron_id", "cell_type_binary", "compartment"])
    m["neuron_id"] = m["neuron_id"].astype(str)
    ct = m.drop_duplicates("neuron_id").set_index("neuron_id")["cell_type_binary"].fillna("unknown")
    comp = m.groupby("neuron_id")["compartment"].agg(lambda x: x.fillna("unknown").value_counts().idxmax())
    per_neuron = per_neuron.assign(cell_type=per_neuron["neuron_id"].map(ct), main_compartment=per_neuron["neuron_id"].map(comp))
    print("\ncell type x split (neurons):")
    print(pd.crosstab(per_neuron["cell_type"], per_neuron["split"]))
    print("\nmain compartment x split (neurons):")
    print(pd.crosstab(per_neuron["main_compartment"], per_neuron["split"]))
    if args.other:
        o = pd.read_parquet(args.other)
        o["neuron_id"] = neuron_of(o["spine_key"])
        on = o.drop_duplicates("neuron_id")[["neuron_id", "split"]].rename(columns={"split": "other_split"})
        joined = on.merge(per_neuron[["neuron_id", "split"]], on="neuron_id", how="left")
        print(f"\nother split: {len(on)} neurons, {on['other_split'].value_counts().to_dict()}; present in the first split: {int(joined['split'].notna().sum())}")
        print(pd.crosstab(joined["other_split"], joined["split"].fillna("absent")))


if __name__ == "__main__":
    main()
