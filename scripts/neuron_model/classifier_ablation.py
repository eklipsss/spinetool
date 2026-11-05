"""Feature ablation of the real-vs-generated classifier, on the tables an ``evaluate_generators.py`` run already wrote (CPU, seconds).

    python scripts/neuron_model/classifier_ablation.py --eval-run runs/analysis/neuron-model/<id>_generator-evaluation
    python scripts/neuron_model/classifier_ablation.py --eval-run <id> --reference real_poisson --models mogen_a

Reads ``real_test_morphometrics.parquet`` and ``<kind>/morphometrics_<model>_seed<s>.parquet`` (all seeds pooled). The reference
class is the real test set (default) or another set of the same run, e.g. a real control from ``make_real_controls.py``
(``--reference real_poisson``: generated MoGen meshes against REAL spines that went through the same Poisson chain, so the
classifier cannot use the surface pipeline). Writes ``<eval-run>/classifier_ablation/<kind>/<reference>_vs_<model>_*.csv``
and ``summary.csv`` (one line per model: full / no area / no attachment / attachment-free AUC, and the top features).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.neuron_model.spine_generation.evaluation.classifier_ablation import classifier_ablation


def load_sets(run: Path, kind: str):
    sets = {}
    for path in sorted((run / kind).glob("morphometrics_*_seed*.parquet")):
        m = re.fullmatch(r"morphometrics_(.+)_seed(\d+)\.parquet", path.name)
        if m:
            sets.setdefault(m.group(1), []).append(pd.read_parquet(path))
    return {k: pd.concat(v, ignore_index=True) for k, v in sets.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-run", type=Path, required=True)
    ap.add_argument("--kind", default="postprocessed", choices=["raw", "postprocessed"])
    ap.add_argument("--models", nargs="*", help="default: every generated set of the run (except the reference)")
    ap.add_argument("--reference", default="real", help="'real' (real test spines) or the name of a set of the run")
    ap.add_argument("--model", default="logistic", choices=["logistic", "gbm"], help="classifier (the report uses logistic)")
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets = load_sets(args.eval_run, args.kind)
    if args.reference == "real":
        ref = pd.read_parquet(args.eval_run / "real_test_morphometrics.parquet")
    elif args.reference in sets:
        ref = sets[args.reference]
    else:
        ap.error(f"reference {args.reference!r} not found; sets: {sorted(sets)}")
    models = args.models or [m for m in sets if m != args.reference]
    out = args.eval_run / "classifier_ablation" / args.kind
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for name in models:
        if name not in sets:
            print(f"skip {name}: not in {sorted(sets)}")
            continue
        tables = classifier_ablation(ref, sets[name], model=args.model, n_repeats=args.n_repeats, seed=args.seed)
        tag = f"{args.reference}_vs_{name}"
        for key, table in tables.items():
            table.to_csv(out / f"{tag}_{key}.csv", index=False)
        sub = tables["subsets"]
        groups = sub[sub.kind == "group"].set_index("name")
        single = sub[sub.kind == "single"].sort_values("roc_auc", ascending=False)
        drop = sub[sub.kind == "drop_one"].sort_values("auc_minus_all")
        row = {"reference": args.reference, "model": name, "n_per_class": int(groups.loc["all", "n_per_class"])}
        for g in ("all", "no_area", "no_attachment", "attachment_only", "no_area_no_attachment", "area_only"):
            if g in groups.index:
                row[f"auc_{g}"] = groups.loc[g, "roc_auc"]
                row[f"acc_{g}"] = groups.loc[g, "accuracy"]
        row["best_single"] = ", ".join(f"{r['name']} {r['roc_auc']:.2f}" for _, r in single.head(3).iterrows())
        row["largest_drop"] = ", ".join(f"{r['name'].replace('without ', '')} {r['auc_minus_all']:+.3f}" for _, r in drop.head(3).iterrows())
        summary.append(row)
        print(f"\n== {tag} ({row['n_per_class']} per class, {args.model}, {args.n_repeats} repeats)")
        for g in groups.index:
            print(f"  {g:24s} AUC {groups.loc[g, 'roc_auc']:.3f} +- {groups.loc[g, 'roc_auc_sd']:.3f}   accuracy {groups.loc[g, 'accuracy']:.3f}")
        print(f"  best single features: {row['best_single']}")
        print(f"  largest AUC loss when dropped: {row['largest_drop']}")
        shifts = tables["shifts"].reindex(tables["shifts"]["median_shift_sd"].abs().sort_values(ascending=False).index)
        print("  median shift (generated - reference) / SD:", ", ".join(f"{r.feature} {r.median_shift_sd:+.2f}" for r in shifts.head(5).itertuples()))
    if summary:
        path = out / f"summary_{args.reference}.csv"
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
