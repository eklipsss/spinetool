"""Tables of the global report: inventory, descriptive statistics and significance tests for every dataset given.
(src/neuron_model/global_report; design: docs/neuron-model/global-report-design.md). CPU, minutes; run on Windows.

    python scripts/neuron_model/build_global_tables.py ^
        --spines minnie65=C:\\...\\cache\\morphometrics_minnie65_full.parquet h01=C:\\...\\h01\\preprocessed_v2\\H01_Seg\\morphometrics.parquet ^
                 labid=C:\\...\\labid\\labid_raw_z07\\labid\\preprocessed\\morphometrics.parquet ^
        --branches minnie65=C:\\...\\minnie65\\preprocessed_branches h01=C:\\...\\h01\\preprocessed_branches labid=C:\\...\\labid\\preprocessed_branches_z07 ^
        --compartment-table h01=<run>\\limb_compartments_direction.parquet

Writes ``runs/analysis/neuron-model/<time>_global-tables/{inventory_*, descriptive_spine, descriptive_branch, results_spine,
results_branch, omnibus_*, comparisons_*, availability_*}.{csv,json}`` + ``run_info.json``. Any dataset may be omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.branch_statistics import ALL_METRICS, METRIC_INFO  # noqa: E402
from src.neuron_model.global_report.datasets import dataset_inventory, load_branch_groups, load_spine_frame  # noqa: E402
from src.neuron_model.global_report.descriptive import summarize_metrics  # noqa: E402
from src.neuron_model.global_report.factors import availability_matrix  # noqa: E402
from src.neuron_model.global_report.significance import run_comparisons, save_tables  # noqa: E402
from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402


def _pairs(items):
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        out[key] = Path(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spines", nargs="*", help="<dataset>=<stage-8 morphometrics table>")
    ap.add_argument("--branches", nargs="*", help="<dataset>=<branch preprocessing output root>")
    ap.add_argument("--compartment-table", nargs="*", help="<dataset>=<limb compartments parquet> instead of the default")
    ap.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    ap.add_argument("--min-neurons", type=int, default=10)
    ap.add_argument("--min-spines", type=int, default=20)
    ap.add_argument("--min-spines-labid", type=int, default=5, help="LabID: spines per branch folder (~9 on average)")
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.time()
    comp = _pairs(args.compartment_table)
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_global-tables"
    spines = pd.concat([load_spine_frame(k, p, args.metadata_root, compartment_table=comp.get(k), metrics=list(SCALAR_METRICS))
                        for k, p in _pairs(args.spines).items()], ignore_index=True) if args.spines else None
    groups = pd.concat([load_branch_groups(k, p, args.metadata_root, compartment_table=comp.get(k))
                        for k, p in _pairs(args.branches).items()], ignore_index=True) if args.branches else None
    tables = {}
    ms = {"default": args.min_spines, "labid": args.min_spines_labid}
    if spines is not None:
        tables.update({f"inventory_{k}": v for k, v in dataset_inventory(spines).items()})
        tables["descriptive_spine"] = summarize_metrics(spines, list(SCALAR_METRICS), by=["dataset"], level="spine")
        tables["descriptive_spine_by_cell_type"] = summarize_metrics(spines, list(SCALAR_METRICS), by=["dataset", "cell_type_binary"], level="spine")
        res = run_comparisons(spines, list(SCALAR_METRICS), level="spine", min_neurons=args.min_neurons, min_spines=ms,
                              n_permutations=args.n_permutations, seed=args.seed)
        tables.update({f"{k}_spine": v for k, v in res.items()})
        tables["availability_spine"] = availability_matrix(spines, min_neurons=args.min_neurons, min_spines=ms).reset_index()
    if groups is not None:
        metrics = [m for m in ALL_METRICS if m in groups.columns]
        tables["descriptive_branch"] = summarize_metrics(groups, metrics, by=["dataset"], level="branch")
        abs_metrics = [m for m in metrics if METRIC_INFO.get(m, {}).get("diff") == "abs"]
        res = run_comparisons(groups, metrics, level="branch", min_neurons=args.min_neurons, min_spines=1, n_permutations=args.n_permutations,
                              seed=args.seed, abs_metrics=abs_metrics)
        tables.update({f"{k}_branch": v for k, v in res.items()})
        tables["availability_branch"] = availability_matrix(groups, min_neurons=args.min_neurons, min_spines=1).reset_index()
    save_tables(tables, out)
    (out / "run_info.json").write_text(json.dumps({"spines": {k: str(v) for k, v in _pairs(args.spines).items()},
        "branches": {k: str(v) for k, v in _pairs(args.branches).items()}, "compartment_tables": {k: str(v) for k, v in comp.items()},
        "min_neurons": args.min_neurons, "min_spines": args.min_spines, "n_permutations": args.n_permutations,
        "created": time.strftime("%Y-%m-%d %H:%M")}, indent=2), encoding="utf-8")
    print(f"-> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
