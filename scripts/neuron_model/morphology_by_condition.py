"""Real spine morphology by cell type and compartment: is each morphometric significantly different?
(src/neuron_model/spine_generation/evaluation/morphology_by_condition.py; tests at the level of NEURONS)

Input: the stage-8 morphometrics table of the full Minnie65 (keys + metrics, one row per spine; only spines that passed
stages 1-8). Cell types and compartments from the repository metadata tables (fixed compartment lookup). CPU, minutes.

    python scripts/neuron_model/morphology_by_condition.py \
        --morphometrics-table C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

Options: --splits/--split restrict the neurons (default: all, this is a description of the data, no model is fitted);
--min-neurons (detailed cell types with fewer neurons are left out, default 10), --min-spines per neuron (20).
Writes ``runs/analysis/neuron-model/<time>_morphology-by-condition/{<table>.csv, neuron_medians.csv,
neuron_compartment_medians.csv, summary.md, <date>_<what>.html}`` (the report alone again: build_morphology_report.py --dir <folder>).
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

from src.neuron_model.metadata import cell_type_binary, dataset_kind, load_metadata_tables  # noqa: E402
from src.neuron_model.spine_generation.evaluation.morphology_by_condition import KEYS, morphology_by_condition, neuron_medians  # noqa: E402
from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402


def load_spines(table_path: Path, metadata_root: Path, splits=None, split=None, compartment_table=None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    paths = [table_path] + sorted(Path(f"{table_path}.parts").glob("*.parquet")) if Path(f"{table_path}.parts").exists() else [table_path]
    frames = []
    for path in paths:
        names = set(pq.read_schema(path).names)
        cols = [c for c in KEYS + ["status"] + list(SCALAR_METRICS) if c in names]
        frames.append(pq.read_table(path, columns=cols).to_pandas())
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEYS, keep="last")
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str) == "success"]
    frame = frame[frame["dataset"].astype(str).map(dataset_kind) == "minnie65"].copy()
    for key in ("neuron_id", "limb_id"):
        frame[key] = frame[key].astype(str)
    if splits is not None:
        s = pd.read_parquet(splits)
        neuron_split = dict(zip(s["spine_key"].astype(str).str.split("/").str[1], s["split"].astype(str)))
        frame["split"] = frame["neuron_id"].map(neuron_split)
        frame = frame[frame["split"] == split] if split else frame[frame["split"].notna()]
    meta = load_metadata_tables(metadata_root, compartment_table)
    types = meta["cell_types"].get("minnie65", {})
    comps = meta["compartments"].get("minnie65", {})
    frame["cell_type"] = frame["neuron_id"].map(types)
    frame["cell_type_binary"] = frame["cell_type"].map(lambda t: cell_type_binary(t) if isinstance(t, str) else None)
    limb = pd.to_numeric(frame["limb_id"], errors="coerce")
    frame["compartment"] = [comps.get((n, int(l))) if np.isfinite(l) else None for n, l in zip(frame["neuron_id"], limb)]
    for m in SCALAR_METRICS:
        frame[m] = pd.to_numeric(frame[m], errors="coerce")
    return frame


def _fmt(v, digits=3):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{digits}g}" if isinstance(v, float) else str(v))


def summarize(tables, frame, args) -> str:
    lines = ["# Морфология шипиков по типу клетки и компартменту", "",
             f"Шипиков {len(frame)}, нейронов {frame['neuron_id'].nunique()}; с типом клетки {int(frame['cell_type'].notna().sum())} шипиков, "
             f"с компартментом {int(frame['compartment'].notna().sum())}. Единица вывода — нейрон (медиана метрики по его шипикам, "
             f">= {args.min_spines} шипиков); перестановок {args.n_permutations}; q — поправка Бенджамини–Хохберга внутри таблицы; * = q < 0,05.", ""]
    for name, table in tables.items():
        lines += [f"## {name}", ""]
        if not len(table):
            lines += ["(нет проверяемых пар)", ""]
            continue
        cols = [c for c in ["compartment", "a", "b", "metric", "n_neurons_a", "n_neurons_b", "n_neurons_paired", "n_types", "n_neurons",
                            "median_a", "median_b", "rel_diff_a_vs_b", "median_paired_rel_diff", "cliffs_delta", "share_neurons_a_gt_b", "p", "q_bh"] if c in table.columns]
        lines.append("| " + " | ".join(cols) + " | |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for _, r in table.iterrows():
            lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + f" | {'*' if r.get('significant') else ''} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--morphometrics-table", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--compartment-table", type=Path, help="limb compartments instead of the default table")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--min-spines", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    t0 = time.time()
    frame = load_spines(args.morphometrics_table, args.metadata_root.resolve(), args.splits, args.split, args.compartment_table)
    print(f"[morph] {len(frame)} spines, {frame['neuron_id'].nunique()} neurons ({time.time() - t0:.0f} s); "
          f"cell types {frame.drop_duplicates('neuron_id')['cell_type'].value_counts(dropna=False).to_dict()}; "
          f"compartments (spines) {frame['compartment'].value_counts(dropna=False).to_dict()}", flush=True)
    tables = morphology_by_condition(frame, list(SCALAR_METRICS), min_neurons=args.min_neurons, min_spines=args.min_spines,
                                     n_permutations=args.n_permutations, seed=args.seed, log=lambda s: print(s, flush=True))
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_morphology-by-condition"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.md").write_text(summarize(tables, frame, args), encoding="utf-8")
    # neuron-level medians (the values every test works on) for the plots of the report
    info = frame.groupby("neuron_id")[["cell_type", "cell_type_binary"]].first()
    neuron_medians(frame, list(SCALAR_METRICS), ["neuron_id"], args.min_spines).merge(info, on="neuron_id").to_csv(out / "neuron_medians.csv", index=False)
    neuron_medians(frame[frame["compartment"].notna()], list(SCALAR_METRICS), ["neuron_id", "compartment"], args.min_spines).merge(
        info, on="neuron_id").to_csv(out / "neuron_compartment_medians.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "morphometrics_table": str(args.morphometrics_table), "splits": str(args.splits) if args.splits else None, "split": args.split,
        "compartment_table": str(args.compartment_table) if args.compartment_table else "default",
        "min_neurons": args.min_neurons, "min_spines": args.min_spines, "n_permutations": args.n_permutations,
        "n_spines": int(len(frame)), "n_neurons": int(frame["neuron_id"].nunique()), "created": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=2), encoding="utf-8")
    from src.neuron_model.reports.morphology_report import build_report

    report = build_report(out)
    print(f"[morph] report -> {report}")

    for name, table in tables.items():
        if not len(table) or "q_bh" not in table.columns:
            continue
        sig = table[table["significant"]]
        print(f"\n[morph] {name}: {len(sig)} of {len(table)} tests significant (q < 0.05)")
        diff = "rel_diff_a_vs_b" if "rel_diff_a_vs_b" in table.columns else ("median_paired_rel_diff" if "median_paired_rel_diff" in table.columns else None)
        for _, r in sig.sort_values("q_bh").head(40).iterrows():
            where = f"[{r['compartment']}] " if "compartment" in r and isinstance(r.get("compartment"), str) and name.endswith("compartment") else ""
            pair = f"{r['a']} vs {r['b']}" if "a" in r and isinstance(r.get("a"), str) else ""
            d = f" rel. diff {r[diff]:+.1%}" if diff and np.isfinite(r.get(diff, np.nan)) else ""
            print(f"   {where}{pair:28s} {r['metric']:20s}{d}  q {r['q_bh']:.3g}")
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
