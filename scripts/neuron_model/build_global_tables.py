"""Tables of the global report: inventory, descriptive statistics and significance tests for every dataset given.
(src/neuron_model/global_report; design: docs/neuron-model/global-report-design.md). CPU, minutes; run on Windows.

    python scripts/neuron_model/build_global_tables.py ^
        --spines minnie65=C:\\...\\cache\\morphometrics_minnie65_full.parquet h01=C:\\...\\h01\\preprocessed_v2\\H01_Seg\\morphometrics.parquet ^
                 labid=C:\\...\\labid\\labid_raw_z07\\labid\\preprocessed\\morphometrics.parquet ^
        --branches minnie65=C:\\...\\minnie65\\preprocessed_branches h01=C:\\...\\h01\\preprocessed_branches labid=C:\\...\\labid\\preprocessed_branches_z07 ^
        --compartment-table h01=<run>\\limb_compartments_direction.parquet

Writes ``runs/analysis/neuron-model/<time>_global-tables/{inventory_*, descriptive_spine, descriptive_branch, results_spine,
results_branch, omnibus_*, comparisons_*, availability_*, consistency_pairs_*, consistency_summary_*}.{csv,json}`` + ``run_info.json``.
Any dataset may be omitted. A dataset given in BOTH --spines and --branches gets the neighbour / position correlations of the
spine marks (its stage-8 table is joined to the positioned spines). ``consistency_*``: the same contrast (binary type, cell
class, compartment) in Minnie65 and in H01 side by side, with the difference of the effects (global_report/consistency.py).
Each unit is checked against the minimal number of spines of ITS dataset (--min-spines, --min-spines-labid).

Section 4 (generated dendrites, global_report/generated.py): ``--generated-tp <generate_tp.py folder>[@<name>]`` (repeatable; T and P,
real reference = the source branches, read through the P0 run of the set as evaluate_assembled_branches.py does) and
``--generated-spines <conditional-report folder | generated_metrics.parquet>[@<name>]`` (repeatable; S, real reference = the
--spines table of --generated-reference, optionally only the neurons of --generated-reference-split). ``--add-to <global-tables
folder>`` writes ONLY the section-4 tables into an existing folder (inputs from its run_info.json), without recomputing sections
0-3. Tables: generated_sets, generated_{results,comparisons,reproduction,reproduction_summary,equivalence,equivalence_summary}_{spine,branch}.

    python scripts/neuron_model/build_global_tables.py --add-to runs\analysis\neuron-model\<time>_global-tables ^
        --generated-tp runs\analysis\neuron-model\tp_generation\<time>_generated_seed1@T0+P1b2 ^
        --generated-spines runs\analysis\neuron-model\<time>_conditional-report@MoGen-v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32" and any(a.startswith("--generated-tp") for a in sys.argv):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch  # noqa: F401  (FIRST on Windows: torch after pandas / pyarrow / scipy fails with WinError 127; the T0 / P0 loaders use it)

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.branch_statistics import ALL_METRICS, METRIC_INFO  # noqa: E402
from src.neuron_model.global_report.datasets import (attached_spine_flags, dataset_inventory, keep_attached_spines,  # noqa: E402
                                                     load_branch_groups, load_spine_frame)
from src.neuron_model.global_report.consistency import consistency  # noqa: E402
from src.neuron_model.global_report.descriptive import summarize_metrics  # noqa: E402
from src.neuron_model.global_report.factors import availability_matrix  # noqa: E402
from src.neuron_model.global_report import generated as gen  # noqa: E402
from src.neuron_model.global_report.significance import run_comparisons, save_tables  # noqa: E402
from src.neuron_model.metadata import dataset_kind  # noqa: E402
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
    ap.add_argument("--from-run", type=Path, help="earlier global-tables folder: take --spines / --branches / --compartment-table from its "
                    "run_info.json (arguments given explicitly win)")
    ap.add_argument("--attached-only", action="store_true",
                    help="SENSITIVITY: spine level only with spines whose attachment centre lies on the trunk surface (not "
                         "position_outlier in branch_spines.parquet of --branches), for every dataset that has branch tables; "
                         "H01 head-only spines (neck left in the trunk) are removed. Output folder suffix _attached-only")
    ap.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    ap.add_argument("--min-neurons", type=int, default=10)
    ap.add_argument("--min-spines", type=int, default=20)
    ap.add_argument("--min-spines-labid", type=int, default=5, help="LabID: spines per branch folder (~9 on average)")
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--n-boot-consistency", type=int, default=2000, help="bootstrap replicates of the Minnie65 / H01 consistency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    g = ap.add_argument_group("section 4: generated sets")
    g.add_argument("--generated-tp", action="append", default=[], metavar="DIR[@NAME]", help="generate_tp.py output folder (repeatable)")
    g.add_argument("--generated-spines", action="append", default=[], metavar="TABLE[@NAME]",
                   help="conditional-report folder or its generated_metrics.parquet (repeatable; every label inside = one set)")
    g.add_argument("--generated-reference", default="minnie65", help="dataset of the real reference of the generated spines (must be in --spines)")
    g.add_argument("--generated-reference-splits", type=Path, help="splits_*.parquet: real reference spines only of the neurons of --generated-reference-split")
    g.add_argument("--generated-reference-split", default=None, help="e.g. test (with --generated-reference-splits)")
    g.add_argument("--generated-unit-size", type=int, default=20, help="spines per pseudo-unit when the generated table has no neuron_id")
    g.add_argument("--generated-only-valid", action="store_true", help="generated spines: valid meshes only")
    g.add_argument("--generated-min-neurons", type=int, default=5, help="units per level in the section-4 comparisons (a val / test split has ~8 inhibitory neurons)")
    g.add_argument("--generated-min-branches", type=int, default=5, help="T/P: branches of a (source neuron, compartment) group for its medians, both sides")
    g.add_argument("--generated-n-boot", type=int, default=2000, help="bootstrap replicates of the effect differences (4a) and of the equivalence (4b)")
    g.add_argument("--generated-n-halves", type=int, default=1000, help="random halves of the real units for the noise-floor margin (4b)")
    g.add_argument("--no-position-thinning", action="store_true", help="T/P: per-spine statistics on ALL generated spines (biased, see evaluate.py)")
    g.add_argument("--add-to", type=Path, help="existing global-tables folder: write only the section-4 tables there (inputs from its run_info.json)")
    args = ap.parse_args()
    if args.add_to is not None:
        args.from_run = args.from_run or args.add_to
        args.out = args.add_to
    if args.from_run is not None:
        prev = json.loads((args.from_run / "run_info.json").read_text(encoding="utf-8"))
        for arg, key in (("spines", "spines"), ("branches", "branches"), ("compartment_table", "compartment_tables")):
            if not getattr(args, arg) and prev.get(key):
                setattr(args, arg, [f"{k}={v}" for k, v in prev[key].items()])
        print(f"from {args.from_run}: spines {args.spines}, branches {args.branches}, compartments {args.compartment_table}", flush=True)
    t0 = time.time()
    comp = _pairs(args.compartment_table)
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / (f"{time.strftime('%Y-%m-%d_%H%M%S')}_global-tables" + ("_attached-only" if args.attached_only else ""))
    spine_tables = _pairs(args.spines)
    need_spines = args.add_to is None or bool(args.generated_spines)
    spines = pd.concat([load_spine_frame(k, p, args.metadata_root, compartment_table=comp.get(k), metrics=list(SCALAR_METRICS))
                        for k, p in spine_tables.items() if args.add_to is None or gen_kind(k) == gen_kind(args.generated_reference)],
                       ignore_index=True) if args.spines and need_spines else None
    if args.add_to is not None:
        tables = generated_section(args, spines, comp)
        save_tables(tables, out)
        info_path = out / "run_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        info["generated"] = generated_info(args)
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"-> {out} (section 4 only, {time.time() - t0:.0f} s)")
        return
    groups = pd.concat([load_branch_groups(k, p, args.metadata_root, compartment_table=comp.get(k), morphometrics_table=spine_tables.get(k))
                        for k, p in _pairs(args.branches).items()], ignore_index=True) if args.branches else None
    tables = {}
    if spines is not None and args.attached_only:
        branch_roots = _pairs(args.branches)
        flags = pd.concat([attached_spine_flags(p, k) for k, p in branch_roots.items()], ignore_index=True)
        spines = spines[spines["dataset"].isin([dataset_kind(k) for k in branch_roots])]
        spines, kept = keep_attached_spines(spines, flags)
        tables["attached_only_filter"] = kept
        print("[global] attached-only filter (share of spines kept):")
        print(kept.to_string(index=False), flush=True)
    ms = {"default": args.min_spines, "labid": args.min_spines_labid}
    if spines is not None:
        tables.update({f"inventory_{k}": v for k, v in dataset_inventory(spines).items()})
        tables["descriptive_spine"] = summarize_metrics(spines, list(SCALAR_METRICS), by=["dataset"], level="spine")
        tables["descriptive_spine_by_cell_type"] = summarize_metrics(spines, list(SCALAR_METRICS), by=["dataset", "cell_type_binary"], level="spine")
        res = run_comparisons(spines, list(SCALAR_METRICS), level="spine", min_neurons=args.min_neurons, min_spines=ms,
                              n_permutations=args.n_permutations, seed=args.seed)
        tables.update({f"{k}_spine": v for k, v in res.items()})
        cons = consistency(spines, list(SCALAR_METRICS), level="spine", results=res["results"], min_neurons=args.min_neurons, min_spines=ms,
                           n_boot=args.n_boot_consistency, seed=args.seed)
        tables.update({f"consistency_{k}_spine": v for k, v in cons.items()})
        tables["availability_spine"] = availability_matrix(spines, min_neurons=args.min_neurons, min_spines=ms).reset_index()
    if groups is not None:
        metrics = [m for m in ALL_METRICS if m in groups.columns]
        tables["descriptive_branch"] = summarize_metrics(groups, metrics, by=["dataset"], level="branch")
        abs_metrics = [m for m in metrics if METRIC_INFO.get(m, {}).get("diff") == "abs"]
        res = run_comparisons(groups, metrics, level="branch", min_neurons=args.min_neurons, min_spines=1, n_permutations=args.n_permutations,
                              seed=args.seed, abs_metrics=abs_metrics)
        tables.update({f"{k}_branch": v for k, v in res.items()})
        # trunk (T) and placement (P) separately: the summary over metrics (sign agreement, rank correlation) is per module
        parts = {"pairs": [], "summary": []}
        for name, module in (("trunk", "T"), ("placement", "P")):
            mm = [m for m in metrics if METRIC_INFO.get(m, {}).get("module") == module]
            cons = consistency(groups, mm, level=name, results=res["results"], min_neurons=args.min_neurons, min_spines=1,
                               abs_metrics=abs_metrics, n_boot=args.n_boot_consistency, seed=args.seed)
            for k in parts:
                parts[k].append(cons[k])
        tables.update({f"consistency_{k}_branch": pd.concat(v, ignore_index=True) for k, v in parts.items()})
        tables["availability_branch"] = availability_matrix(groups, min_neurons=args.min_neurons, min_spines=1).reset_index()
    if args.generated_tp or args.generated_spines:
        tables.update(generated_section(args, spines, comp))
    save_tables(tables, out)
    (out / "run_info.json").write_text(json.dumps({"generated": generated_info(args), "spines": {k: str(v) for k, v in _pairs(args.spines).items()},
        "branches": {k: str(v) for k, v in _pairs(args.branches).items()}, "compartment_tables": {k: str(v) for k, v in comp.items()},
        "min_neurons": args.min_neurons, "min_spines": args.min_spines, "min_spines_labid": args.min_spines_labid, "n_permutations": args.n_permutations,
        "n_boot_consistency": args.n_boot_consistency, "attached_only": bool(args.attached_only),
        "created": time.strftime("%Y-%m-%d %H:%M")}, indent=2), encoding="utf-8")
    print(f"-> {out} ({time.time() - t0:.0f} s)")


def gen_kind(name: str) -> str:
    return dataset_kind(name)


def _named(item: str):
    """``<path>[@<name>]`` (the name may not contain @; a path with @ is not supported)."""
    path, _, name = item.rpartition("@") if "@" in item else (item, "", "")
    return Path(path), (name or None)


def generated_info(args) -> dict:
    return {"tp": args.generated_tp, "spines": args.generated_spines, "reference": args.generated_reference,
            "reference_splits": str(args.generated_reference_splits) if args.generated_reference_splits else None,
            "reference_split": args.generated_reference_split, "unit_size": args.generated_unit_size, "only_valid": bool(args.generated_only_valid),
            "min_neurons": args.generated_min_neurons, "min_branches": args.generated_min_branches, "n_boot": args.generated_n_boot,
            "n_halves": args.generated_n_halves, "position_thinning": not args.no_position_thinning}


def generated_section(args, spines, comp) -> dict:
    """Section-4 tables of every generated set (global_report/generated.py)."""
    sets_rows, spine_parts, branch_parts = [], [], []
    ms = {"default": args.min_spines, "labid": args.min_spines_labid, gen.GENERATED: min(args.min_spines, args.generated_unit_size)}
    common = dict(min_neurons=args.generated_min_neurons, n_permutations=args.n_permutations, n_boot_difference=args.generated_n_boot,
                  n_halves=args.generated_n_halves, seed=args.seed)
    for item in args.generated_spines:
        path, name = _named(item)
        if spines is None:
            raise SystemExit("--generated-spines needs --spines of the reference dataset (or --add-to a folder made with it)")
        real = gen.real_reference_spines(spines, args.generated_reference, neurons=_split_neurons(args))
        sets, sinfo = gen.load_generated_spines(path, list(SCALAR_METRICS), kind=args.generated_reference, unit_size=args.generated_unit_size,
                                                only_valid=args.generated_only_valid, seed=args.seed)
        for label, g in sets.items():
            set_name = label if name is None else (name if len(sets) == 1 else f"{name}/{label}")
            g = g.assign(set=set_name)
            frame = pd.concat([real.assign(set=set_name), g], ignore_index=True)
            print(f"[global/generated] spines {set_name}: {g['unit_id'].nunique()} generated units ({g['unit_mode'].iloc[0]}), "
                  f"real reference {real['unit_id'].nunique()} neurons", flush=True)
            spine_parts.append(gen.generated_tables(frame, list(SCALAR_METRICS), level="spine", set_label=set_name, min_rows=ms, include_all=False, **common))
            sets_rows += gen.set_inventory(frame, {"set": set_name, "source": sinfo["source"], "source_dataset": gen_kind(args.generated_reference),
                                                   "block": sinfo["block"], "label": label, "unit": g["unit_mode"].iloc[0],
                                                   "reference_split": args.generated_reference_split}, "spine")
    for item in args.generated_tp:
        path, name = _named(item)
        # the compartment table of the source dataset (--compartment-table), as for the real groups of section 3
        frame, info = gen.load_generated_tp(path, args.metadata_root, label=name or path.name, compartment_table={gen_kind(k): v for k, v in comp.items()},
                                            min_branches=args.generated_min_branches, thin=not args.no_position_thinning, seed=args.seed)
        metrics = [m for m in gen.TP_METRICS if m in frame.columns]
        abs_metrics = [m for m in metrics if METRIC_INFO.get(m, {}).get("diff") == "abs"]
        modules = {lv: [m for m in metrics if METRIC_INFO.get(m, {}).get("module") == mod] for lv, mod in (("trunk", "T"), ("placement", "P"))}
        print(f"[global/generated] T/P {info['set']}: {info['n_branches']} branches ({info['n_skipped']} without a source), "
              f"mean positioned share {info['mean_keep_fraction']:.3f}", flush=True)
        branch_parts.append(gen.generated_tables(frame, metrics, level="branch", set_label=info["set"], modules=modules, abs_metrics=abs_metrics,
                                                 min_rows=1, include_all=True, **common))
        sets_rows += gen.set_inventory(frame, info, "branch")
    tables = {}
    tables.update(gen.merge_tables(spine_parts, "spine"))
    tables.update(gen.merge_tables(branch_parts, "branch"))
    if sets_rows:
        tables["generated_sets"] = pd.DataFrame(sets_rows)
    return tables


def _split_neurons(args):
    if args.generated_reference_splits is None or not args.generated_reference_split:
        return None
    from src.neuron_model.branch_preprocessing.inputs import neuron_split_lookup

    lookup = neuron_split_lookup(args.generated_reference_splits)
    return [n for n, s in lookup.items() if s == args.generated_reference_split]


if __name__ == "__main__":
    main()
