"""Check the inputs of the branch preprocessing BEFORE a long run (reads tables only, no meshes).

Same path flags as run_branch_preprocessing.py. Reports, per neuron to be processed:
  1. S verdicts: is the neuron in the S manifest, share of spines with a false-spine verdict
     (valid/invalid) and with an attachment centre (origin_global). Without them the branch
     is not eligible for P (T is not affected).
  2. metadata: cell type and compartment found (the compartment is read here from the
     compartment table with the fixed lookup, not from the S stage 9).
  3. output: config hash now vs the hash in existing _done.json markers (a change of
     branch_preprocessing.yaml after a run makes every neuron be redone).

    python scripts/neuron_model/check_branch_inputs.py --raw-data-root O:\\Datasets ^
        --metadata-root <repo>\\datasets --output-root C:\\...\\preprocessed_branches
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.branch_preprocessing.config import load_config  # noqa: E402
from src.neuron_model.branch_preprocessing.inputs import available_neurons, load_members, load_metadata, load_spine_table  # noqa: E402
from src.neuron_model.metadata import dataset_kind  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "data/processed/config/branch_preprocessing.yaml")
    parser.add_argument("--dataset", default="minnie65")
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--s-preprocessed-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--only-s-neurons", action="store_true")
    parser.add_argument("--neuron-id", action="append", dest="neuron_ids")
    parser.add_argument("--min-verdict", type=float, default=0.9, help="a neuron with a smaller share of verdicts is listed")
    parser.add_argument("--csv", type=Path, help="write the per-neuron table here")
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {
            "dataset": args.dataset,
            "raw_data_root": args.raw_data_root.resolve(),
            "metadata_root": args.metadata_root.resolve(),
            "s_preprocessed_root": args.s_preprocessed_root.resolve() if args.s_preprocessed_root else None,
            "output_root": args.output_root.resolve() if args.output_root else None,
            "neuron_ids": args.neuron_ids,
            "only_s_neurons": args.only_s_neurons or None,
        },
    )
    members = load_members(cfg)
    neurons = available_neurons(cfg, members)
    print(f"neurons: {members['neuron_id'].nunique()} in the merge mapping, {len(neurons)} also as folders in {cfg.dataset_root}")

    # 1. S verdicts
    manifest = cfg.s_root / "manifest.parquet"
    print(f"\n[1] S tables: {cfg.s_root}  (manifest exists: {manifest.exists()})")
    spines = load_spine_table(cfg, neurons) if manifest.exists() else pd.DataFrame(columns=["neuron_id", "status", "origin_global"])
    rows = []
    groups = {n: g for n, g in spines.groupby("neuron_id")} if len(spines) else {}
    for n in neurons:
        g = groups.get(n)
        if g is None or not len(g):
            rows.append({"neuron_id": n, "n_spines": 0, "verdict": 0.0, "origin": 0.0})
            continue
        rows.append({
            "neuron_id": n,
            "n_spines": len(g),
            "verdict": float(g["status"].isin(["valid", "invalid"]).mean()),
            "origin": float(g["origin_global"].notna().mean()) if "origin_global" in g else 0.0,
        })
    table = pd.DataFrame(rows)
    absent = table[table["n_spines"] == 0]
    low = table[(table["n_spines"] > 0) & (table["verdict"] < args.min_verdict)]
    ok = table[(table["n_spines"] > 0) & (table["verdict"] >= args.min_verdict)]
    print(f"    in the S manifest: {len(table) - len(absent)} of {len(table)} neurons; verdict >= {args.min_verdict:.0%}: {len(ok)}; below: {len(low)}")
    if len(spines):
        print(f"    spine rows {len(spines)}: false-spine verdict {spines['status'].isin(['valid', 'invalid']).mean():.1%}, "
              f"attachment centre {spines['origin_global'].notna().mean():.1%}; status counts {dict(Counter(spines['status'].astype(str)).most_common(6))}")
    if len(absent):
        print(f"    NOT in the S manifest (their branches will not be eligible for P): {len(absent)}, first {absent['neuron_id'].head(5).tolist()}")
    if len(low):
        print(f"    low verdict share: {low.sort_values('verdict').head(5).to_dict('records')}")

    # 2. metadata
    meta = load_metadata(cfg)
    kind = dataset_kind(cfg.dataset)
    ctypes = meta.get("cell_types", {}).get(kind, {})
    comps = meta.get("compartments", {}).get(kind, {})
    comp_neurons = {k[0] for k in comps}
    table["cell_type"] = table["neuron_id"].map(lambda n: ctypes.get(str(n)))
    table["has_compartments"] = table["neuron_id"].isin(comp_neurons)
    limbs = members[members["neuron_id"].isin(set(neurons))][["neuron_id", "limb_id"]].drop_duplicates()
    found = limbs.apply(lambda r: comps.get((str(r["neuron_id"]), int(r["limb_id"]))), axis=1)
    print(f"\n[2] metadata: {cfg.metadata_root}")
    print(f"    cell type known: {table['cell_type'].notna().sum()} of {len(table)} neurons {dict(Counter(table['cell_type'].dropna()).most_common(6))}")
    print(f"    compartment known: {table['has_compartments'].sum()} of {len(table)} neurons, {found.notna().mean():.1%} of their limbs; "
          f"values {dict(Counter(found.dropna()).most_common(6))}")
    if table["has_compartments"].sum() <= 1 < len(table):
        print("    WARNING: compartments found for at most one neuron: the old lookup bug, or a wrong --metadata-root")

    # 3. output and config hash
    h = cfg.config_hash()
    print(f"\n[3] output: {cfg.out_root}  (config hash now {h})")
    hashes = Counter()
    if cfg.out_root.is_dir():
        for marker in cfg.out_root.glob("*/_done.json"):
            try:
                info = json.loads(marker.read_text(encoding="utf-8"))
                hashes[(info.get("config_hash"), bool(info.get("partial", False)))] += 1
            except (OSError, ValueError):
                hashes[("unreadable", False)] += 1
    if not hashes:
        print("    no finished neurons yet")
    for (hh, partial), count in hashes.items():
        note = "same config, will be skipped" if hh == h and not partial else ("partial run, will be redone" if partial else "OTHER config, will be redone")
        print(f"    {count} neurons with hash {hh}: {note}")
    if str(cfg.out_root)[:2].upper() in ("O:",):
        print("    WARNING: the output is on the NAS; put it on the local disk (--output-root C:\\...)")

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nper-neuron table: {args.csv}")


if __name__ == "__main__":
    main()
