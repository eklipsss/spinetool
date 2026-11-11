"""Strict statistical report on a CONDITIONAL spine generator: HTML page + tables
(src/neuron_model/spine_generation/evaluation/conditional_tests.py, page: src/neuron_model/reports/conditional_report.py).

Generated sets: ``--gen <value>[@<label>]=<set folder | generated_metrics.parquet | analysis folder>`` (repeatable; the value
is the condition, ``unconditional`` for a set without a condition; the label groups sets of one guidance weight, e.g.
``inhibitory@g1.25`` and ``excitatory@g1.25``). A set folder without ``generated_metrics.parquet`` is measured on the spot
(meshes -> 11 metrics, base by DownFacingCapFinder, as compare_conditional_generation.py). Real spines: the stage-8 table +
the metadata tables (as morphology_by_condition.py); the neuron is the unit of inference.

    python scripts/neuron_model/conditional_generation_report.py ^
        --gen excitatory@g1.25=<run>\\meshes\\cond_cell_type_binary-excitatory_g1.25_seed_1 ^
        --gen inhibitory@g1.25=<run>\\meshes\\cond_cell_type_binary-inhibitory_g1.25_seed_1 ^
        --gen unconditional=<run>\\meshes\\seed_1 ^
        --morphometrics-table cache\\morphometrics_minnie65_full.parquet

CPU, minutes (permutation tests, bootstraps, reference distributions). Writes ``runs/analysis/neuron-model/<time>_conditional-report/``
{results.json, generated_metrics.parquet, run_info.json, *.csv, <date>_conditional-generation-report.html}.
``generated_metrics.parquet`` (one row per generated spine: metrics, value, label, block, condition_<block>) is read by the
global report, section 4 (build_global_tables.py --generated-spines <this folder>).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.spine_generation.evaluation import conditional_tests as ct  # noqa: E402
from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402


def load_set(value: str, label: str, source: Path, mesh_kind: str) -> tuple:
    """``(frame, info)`` of one generated set: a parquet / analysis folder (``generated_metrics.parquet``) or a set folder to measure."""
    source = Path(source)
    parquet = source if source.suffix == ".parquet" else source / "generated_metrics.parquet"
    info = {"source": str(source)}
    if parquet.exists():
        frame = pd.read_parquet(parquet)
        if "value" in frame.columns and "label" in frame.columns:
            frame = frame[(frame["value"] == value) & (frame["label"] == label)]
        if not len(frame):
            raise SystemExit(f"{parquet}: no rows of set {value}@{label}")
        folder = Path(str(frame["folder"].iloc[0])) if "folder" in frame.columns else None
    else:
        from src.neuron_model.spine_generation.evaluation.morphometrics import generated_set_metrics

        print(f"[cond-report] measuring {source} ...", flush=True)
        frame = generated_set_metrics(source, mesh_kind)
        folder = source
    summary = folder / "generation_summary.json" if folder is not None else None
    if summary is not None and summary.exists():
        cfg = json.loads(summary.read_text(encoding="utf-8")).get("inference_config", {})
        cond = cfg.get("cond") if isinstance(cfg.get("cond"), dict) else {}
        info.update({"checkpoint": cfg.get("checkpoint_id"), "guidance": cond.get("guidance"), "seed": cfg.get("generation_seed"), "steps": cfg.get("n_steps")})
    return frame.reset_index(drop=True), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", action="append", required=True, metavar="VALUE[@LABEL]=SOURCE", help="a generated set (repeatable)")
    parser.add_argument("--block", default="cell_type_binary", choices=("cell_type_binary", "cell_type"), help="condition block: binary cell type or detailed cell types (sets <type>@<label>)")
    parser.add_argument("--morphometrics-table", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--splits", type=Path, help="real spines only of the neurons of --split (default: all neurons)")
    parser.add_argument("--split")
    parser.add_argument("--mesh-kind", default="postprocessed", choices=("raw", "postprocessed"))
    parser.add_argument("--min-spines", type=int, default=20, help="spines per neuron needed for the neuron to take part")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-reference", type=int, default=1000, help="real half-vs-half repetitions (noise floor of test B)")
    parser.add_argument("--n-reference-classifier", type=int, default=100, help="repetitions of the real half-vs-half classifier (test D)")
    parser.add_argument("--neuron-weighting", default="equal", choices=("equal", "spines"),
                        help="real draws: every neuron equally likely (balanced per-neuron draws) or proportional to its spines")
    parser.add_argument("--tost-margin-sd", type=float, default=ct.TOST_MARGIN_SD, help="equivalence margin of the median, in pooled sd of the log metric")
    parser.add_argument("--classifiers", default="logistic", help="comma list: logistic, gbm (non-linear, slower) for the real-vs-generated test D and the binary condition classifier")
    parser.add_argument("--n-real-draw", type=int, default=5000, help="real spines drawn (balanced over neurons) per bootstrap replicate of test B")
    parser.add_argument("--only-valid", action="store_true", help="statistics on valid meshes only (the validity rate is reported either way)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    t0 = time.time()
    spec = importlib.util.spec_from_file_location("morphology_by_condition", REPO_ROOT / "scripts/neuron_model/morphology_by_condition.py")
    mbc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mbc)
    real = mbc.load_spines(args.morphometrics_table, args.metadata_root.resolve(), args.splits, args.split)
    known = real[real[args.block].notna()]
    print(f"[cond-report] real: {len(real)} spines, {real['neuron_id'].nunique()} neurons; {args.block} known for {len(known)} spines of {known['neuron_id'].nunique()} neurons", flush=True)

    gen_sets, infos = {}, {}
    for item in args.gen:
        key, source = item.split("=", 1)
        value, _, label = key.partition("@")
        label = label or "default"
        frame, info = load_set(value, label, Path(source), args.mesh_kind)
        gen_sets[(value, label)] = frame
        infos[f"{value}@{label}"] = {**info, "n": int(len(frame))}
        print(f"[cond-report] {value}@{label}: {len(frame)} samples ({source})", flush=True)

    results = ct.run_conditional_tests(
        real, gen_sets, args.block, metrics=[m for m in SCALAR_METRICS if m in real.columns], min_spines=args.min_spines, n_permutations=args.n_permutations,
        n_boot=args.n_boot, n_reference=args.n_reference, n_reference_classifier=args.n_reference_classifier, weighting=args.neuron_weighting,
        only_valid=args.only_valid, classifiers=[c.strip() for c in args.classifiers.split(",") if c.strip()], seed=args.seed,
        tost_margin_sd=args.tost_margin_sd, n_real_draw=args.n_real_draw, log=lambda s: print(s, flush=True))

    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_conditional-report"
    out.mkdir(parents=True, exist_ok=True)
    # contract with the global report (section 4, global_report/generated.py): one row per generated spine, the 11 metrics,
    # value / label, the condition block and condition_<block> (None for an unconditional set); no neuron_id: a conditional
    # sample has no source neuron
    pd.concat([f.assign(value=v, label=lab, block=args.block, **{f"condition_{args.block}": (None if v == "unconditional" else v)})
               for (v, lab), f in gen_sets.items()], ignore_index=True).to_parquet(out / "generated_metrics.parquet", index=False)
    safe = lambda text: re.sub(r"[^0-9A-Za-z._@+-]+", "_", str(text))  # noqa: E731  (file names on Windows)
    for name, table in results.get("real_effects", {}).items():
        table.to_csv(out / f"real_effect_{safe(name)}.csv", index=False)
    for name in ("validity", "real_effect", "dose"):
        if name in results and len(results[name]):
            results[name].to_csv(out / f"{name}.csv", index=False)
    for lab, e in results.get("effects", {}).items():
        e["concordance"].to_csv(out / f"concordance_{safe(lab)}.csv", index=False)
        e["generated"].to_csv(out / f"generated_effect_{safe(lab)}.csv", index=False)
    for key, s in results["same_type"].items():
        for pool, r in s["by_pool"].items():
            r["table"].to_csv(out / f"same_type_{safe(key)}_vs_{safe(pool)}.csv", index=False)
    info = {"created": time.strftime("%Y-%m-%d %H:%M"), "block": args.block, "morphometrics_table": str(args.morphometrics_table), "splits": str(args.splits) if args.splits else None,
            "split": args.split, "mesh_kind": args.mesh_kind, "gen": args.gen, "sets": infos, "n_real_spines": int(len(known)), "n_real_neurons": int(known["neuron_id"].nunique()),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("gen",)}}
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(ct.to_jsonable(results), ensure_ascii=False), encoding="utf-8")

    from src.neuron_model.reports.conditional_report import build_report

    report = build_report(out)
    print(f"[cond-report] report -> {report}")
    for lab, e in results.get("effects", {}).items():
        print(f"[cond-report] A @ {lab}: concordant {e['n_concordant']}/{e['n_metrics']} (reproduced {e['n_reproduced']} of {e['n_real_significant']} real effects), "
              f"underpowered {e['n_underpowered']}, discordant {e['n_discordant']}")
    for key, s in results["same_type"].items():
        t = s["by_pool"][s["own"]]["table"]
        print(f"[cond-report] B {key} vs real {s['own']}: equivalent {int((t['verdict'] == 'equivalent').sum())}, differs {int((t['verdict'] == 'differs').sum())} of {len(t)}")
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
