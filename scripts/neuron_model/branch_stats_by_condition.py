"""Do real branches differ between cell types and compartments in the metrics of the T and P modules?
(src/neuron_model/branch_statistics.py for the metrics, spine_generation/evaluation/morphology_by_condition.py for the
neuron-level tests, reports/morphology_report.py for the page)

Unlike the training of T0 / P0, ALL branches are used here (inhibitory neurons included: their limbs have no compartment
in the Minnie65 table, they form the group "none"); axon_like limbs are left out. Metrics per group (neuron x
compartment): trunk medians over the T-eligible branches, density / spines per branch / share of empty branches over the
P-eligible branches, pooled pair correlation function along the branch, CV of the gaps, nearest-neighbour distance and,
with --morphometrics-table, the similarity of neighbouring spines in volume / length and the volume gradient along the
branch.

    python scripts/neuron_model/branch_stats_by_condition.py \
        --branches-root C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches \
        --morphometrics-table C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

Writes ``runs/analysis/neuron-model/<time>_branch-stats-by-condition/{<table>.csv, groups.csv, summary.md, report.html}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.branch_statistics import ALL_METRICS, group_table  # noqa: E402
from src.neuron_model.metadata import cell_type_binary, dataset_kind, load_metadata_tables  # noqa: E402
from src.neuron_model.spine_generation.evaluation.morphology_by_condition import morphology_by_condition, neuron_medians  # noqa: E402

BRANCH_COLUMNS = ["dataset", "neuron_id", "limb_id", "merged_branch_id", "status", "length_nm", "radius_median_nm", "radius_proximal_nm",
                  "radius_distal_nm", "ellipse_ratio_median", "tortuosity", "kappa_rms", "n_spines", "train_eligible_T", "train_eligible_P"]
SPINE_COLUMNS = ["neuron_id", "limb_id", "merged_branch_id", "original_branch_id", "spine_id", "status", "has_position", "s", "u"]


def _read(path: Path, columns):
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    return pq.read_table(path, columns=[c for c in columns if c in names]).to_pandas()


def load(args):
    root = args.branches_root
    branches = _read(root / "branches.parquet", BRANCH_COLUMNS)
    spines = _read(root / "branch_spines.parquet", SPINE_COLUMNS)
    for frame in (branches, spines):
        frame["neuron_id"] = frame["neuron_id"].astype(str)
        frame["limb_id"] = pd.to_numeric(frame["limb_id"]).astype(int)
        frame["merged_branch_id"] = pd.to_numeric(frame["merged_branch_id"]).astype(int)
    if args.splits is not None:
        s = pd.read_parquet(args.splits)
        split_of = dict(zip(s["spine_key"].astype(str).str.split("/").str[1], s["split"].astype(str)))
        keep = branches["neuron_id"].map(split_of)
        branches = branches[keep == args.split] if args.split else branches[keep.notna()]
    kind = dataset_kind(str(branches["dataset"].iloc[0]))
    meta = load_metadata_tables(args.metadata_root.resolve())
    types = meta.get("cell_types", {}).get(kind, {})
    comps = meta.get("compartments", {}).get(kind, {})
    branches["cell_type"] = branches["neuron_id"].map(types)
    branches["cell_type_binary"] = branches["cell_type"].map(lambda t: cell_type_binary(t) if isinstance(t, str) else None)
    branches["compartment"] = [comps.get((n, int(l))) for n, l in zip(branches["neuron_id"], branches["limb_id"])]
    branches = branches[branches["compartment"] != "axon_like"]
    spines = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].astype(bool)].copy()
    if args.morphometrics_table is not None:
        m = _read(args.morphometrics_table, ["neuron_id", "limb_id", "branch_id", "spine_id", "status", "Volume", "Length"])
        if "status" in m.columns:
            m = m[m["status"].astype(str) == "success"]
        m["neuron_id"] = m["neuron_id"].astype(str)
        m["limb_id"] = pd.to_numeric(m["limb_id"]).astype(int)
        m["original_branch_id"] = pd.to_numeric(m["branch_id"]).astype(int)
        m["spine_id"] = m["spine_id"].astype(str)
        spines["spine_id"] = spines["spine_id"].astype(str)
        spines = spines.merge(m[["neuron_id", "limb_id", "original_branch_id", "spine_id", "Volume", "Length"]],
                              on=["neuron_id", "limb_id", "original_branch_id", "spine_id"], how="left")
        spines["log_volume"] = np.log(pd.to_numeric(spines["Volume"], errors="coerce").where(lambda v: v > 0))
        spines["length"] = pd.to_numeric(spines["Length"], errors="coerce")
        print(f"[branch-stats] morphometrics joined for {spines['log_volume'].notna().mean():.1%} of the positioned spines", flush=True)
    return branches, spines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--morphometrics-table", type=Path)
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--min-branches", type=int, default=10, help="per group, for the branch medians")
    parser.add_argument("--min-pairs", type=int, default=30, help="per group, for the pooled spatial / mark statistics")
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    t0 = time.time()
    branches, spines = load(args)
    print(f"[branch-stats] {len(branches)} branches, {branches['neuron_id'].nunique()} neurons, {len(spines)} positioned spines; "
          f"cell types (neurons) {branches.drop_duplicates('neuron_id')['cell_type_binary'].value_counts(dropna=False).to_dict()}; "
          f"compartments (branches) {branches['compartment'].value_counts(dropna=False).to_dict()}", flush=True)
    groups = group_table(branches, spines, min_branches=args.min_branches, min_pairs=args.min_pairs)
    print(f"[branch-stats] {len(groups)} groups (neuron x compartment) in {time.time() - t0:.0f} s", flush=True)
    metrics = [m for m in ALL_METRICS if m in groups.columns and groups[m].notna().any()]
    tables = morphology_by_condition(groups, metrics, min_neurons=args.min_neurons, min_spines=1, n_permutations=args.n_permutations,
                                     seed=args.seed, log=lambda s: print(s, flush=True))
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_branch-stats-by-condition"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    groups.to_csv(out / "groups.csv", index=False)
    info = groups.groupby("neuron_id")[["cell_type", "cell_type_binary"]].first()
    neuron_medians(groups, metrics, ["neuron_id"], 1).merge(info, on="neuron_id").to_csv(out / "neuron_medians.csv", index=False)
    cm = groups[groups["compartment"].isin(["apical", "basal", "uncertain"])]
    neuron_medians(cm, metrics, ["neuron_id", "compartment"], 1).merge(info, on="neuron_id").to_csv(out / "neuron_compartment_medians.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "profile": "branches", "branches_root": str(args.branches_root), "morphometrics_table": str(args.morphometrics_table),
        "min_branches": args.min_branches, "min_pairs": args.min_pairs, "min_spines": args.min_branches, "n_permutations": args.n_permutations,
        "n_branches": int(len(branches)), "n_neurons": int(branches["neuron_id"].nunique()), "n_positioned_spines": int(len(spines)),
        "created": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, table in tables.items():
        if not len(table) or "q_bh" not in table.columns:
            continue
        sig = table[table["significant"]]
        print(f"\n[branch-stats] {name}: {len(sig)} of {int(table['p'].notna().sum())} tests significant (q < 0.05)")
        diff = "rel_diff_a_vs_b" if "rel_diff_a_vs_b" in table.columns else ("median_paired_rel_diff" if "median_paired_rel_diff" in table.columns else None)
        for _, r in sig.sort_values("q_bh").head(40).iterrows():
            pair = f"{r['a']} vs {r['b']}" if isinstance(r.get("a"), str) else ""
            where = f"[{r['compartment']}] " if name.endswith("within_compartment") else ""
            d = f" rel. diff {r[diff]:+.1%}" if diff and np.isfinite(r.get(diff, np.nan)) else ""
            print(f"   {where}{pair:28s} {r['metric']:20s}{d}  q {r['q_bh']:.3g}")
    from src.neuron_model.reports.morphology_report import build_report

    print(f"[branch-stats] report -> {build_report(out)}")
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
