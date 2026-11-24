"""Density-ratio rejection sampling of generated spines on morphometrics: fit on one (real, generated) pair, check on another.

Two modes.

1. Proper (Windows): fit on TRAINING real spines and one generation seed, evaluate on held-out real neurons and another seed:
       python scripts/neuron_model/density_ratio_selection.py --real-fit <train morphometrics.parquet> \
           --gen-fit <morphometrics_mogen_A_seed1.parquet> --real-eval <real_test_morphometrics.parquet> \
           --gen-eval <morphometrics_mogen_A_seed2.parquet> [--floor <morphometrics_real_poisson_seed2.parquet>] --out <dir>
2. Prototype on one real table (mac): ``--split-neurons`` splits the neurons of --real-eval into two disjoint halves,
   the first for fitting, the second for evaluation (--real-fit is then not needed).

Real tables need ``spine_key`` (neuron = its first two parts) and the 11 metrics; generated tables the 11 metrics.
Writes ``per_metric.csv`` (d = ln median ratio generated / real with a neuron-level bootstrap interval, BH q, W1/sd) for
the generated set before and after selection, a random subset of the same size (control: fewer samples = less power),
(and the floor, if given), ``summary.md`` and ``selector.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.evaluation.density_ratio import DensityRatioSelector, per_metric_shift  # noqa: E402
from src.neuron_model.spine_generation.evaluation.paired_bootstrap import METRICS, cv_auc, log_complete, neuron_of  # noqa: E402


def read(path: Path, real: bool) -> pd.DataFrame:
    cols = (["spine_key"] if real else ["sample_id"]) + list(METRICS)
    have = set(pd.read_parquet(path, columns=None).columns) if not real else None
    if have is not None and "sample_id" not in have:
        cols = list(METRICS)
    return pd.read_parquet(path, columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-fit", type=Path)
    ap.add_argument("--gen-fit", type=Path, required=True)
    ap.add_argument("--real-eval", type=Path, required=True)
    ap.add_argument("--gen-eval", type=Path, required=True)
    ap.add_argument("--floor", type=Path, help="real spines through the same reconstruction chain (descriptive reference)")
    ap.add_argument("--real-fit-splits", type=Path, help="splits_v*.parquet: keep only --real-fit-split spines of --real-fit")
    ap.add_argument("--real-fit-split", default="train")
    ap.add_argument("--real-fit-max", type=int, default=30000, help="random subsample of the real fitting spines (0 = all)")
    ap.add_argument("--gen-eval-dir", type=Path, help="set folder (sample_*/) of --gen-eval; with --export, the kept samples are copied")
    ap.add_argument("--export", type=Path, help="write the kept samples as a new set <export>/<name of --gen-eval-dir>")
    ap.add_argument("--split-neurons", action="store_true")
    ap.add_argument("--model", default="gbm", choices=["gbm", "logistic"])
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    real_eval = read(args.real_eval, True)
    if args.split_neurons:
        neurons = real_eval["spine_key"].map(neuron_of)
        uniq = np.array(sorted(neurons.unique()))
        rng.shuffle(uniq)
        half = set(uniq[: len(uniq) // 2])
        real_fit = real_eval[neurons.isin(half)].reset_index(drop=True)
        real_eval = real_eval[~neurons.isin(half)].reset_index(drop=True)
        print(f"split: {len(half)} neurons for fitting ({len(real_fit)} spines), {len(uniq) - len(half)} for evaluation ({len(real_eval)} spines)")
    else:
        if args.real_fit is None:
            ap.error("--real-fit is required without --split-neurons")
        real_fit = read(args.real_fit, True)
        if args.real_fit_splits is not None:
            sp = pd.read_parquet(args.real_fit_splits, columns=["spine_key", "split"])
            keys = set(sp.loc[sp["split"] == args.real_fit_split, "spine_key"])
            real_fit = real_fit[real_fit["spine_key"].isin(keys)].reset_index(drop=True)
            print(f"real fit: {len(real_fit)} {args.real_fit_split} spines of {real_fit['spine_key'].map(neuron_of).nunique()} neurons")
        if args.real_fit_max and len(real_fit) > args.real_fit_max:
            real_fit = real_fit.iloc[rng.choice(len(real_fit), args.real_fit_max, replace=False)].reset_index(drop=True)
            print(f"real fit subsampled to {len(real_fit)} spines")
    gen_fit, gen_eval = read(args.gen_fit, False), read(args.gen_eval, False)

    sel = DensityRatioSelector(model=args.model, quantile=args.quantile, seed=args.seed).fit(real_fit, gen_fit)
    keep = sel.accept(gen_eval, rng)
    selected = gen_eval[keep].reset_index(drop=True)
    print(f"selector {args.model}: M = {sel.M_:.3g}, mean acceptance on the fitting seed {sel.mean_accept_:.3f}; "
          f"kept {keep.sum()} of {len(gen_eval)} evaluation samples")

    random_same_n = gen_eval.iloc[rng.choice(len(gen_eval), int(keep.sum()), replace=False)].reset_index(drop=True)
    if args.export is not None:
        import shutil

        if args.gen_eval_dir is None or "sample_id" not in gen_eval.columns:
            ap.error("--export needs --gen-eval-dir and a sample_id column in --gen-eval")
        target = args.export / args.gen_eval_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for sid in gen_eval.loc[keep, "sample_id"]:
            src = args.gen_eval_dir / str(sid)
            if src.exists() and not (target / str(sid)).exists():
                shutil.copytree(src, target / str(sid))
        print(f"exported {int(keep.sum())} kept samples -> {target}")
    sets = {"generated": gen_eval, "generated_selected": selected, "generated_random_same_n": random_same_n}
    if args.floor:
        sets["floor"] = read(args.floor, False)
    table = per_metric_shift(real_eval, sets, n_boot=args.n_boot, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "per_metric.csv", index=False)

    xr, _ = log_complete(real_eval, METRICS)
    aucs = {}
    for name, frame in sets.items():
        xg, _ = log_complete(frame, METRICS)
        mu, sd = xr.mean(0), xr.std(0)
        aucs[name] = cv_auc((xr[rng.choice(len(xr), min(3000, len(xr)), replace=False)] - mu) / sd, (xg - mu) / sd, rng)

    lines = ["# Density-ratio selection of generated spines", "",
             f"Selector: {args.model}, M = {sel.M_:.3g} ({args.quantile:g} quantile), kept {int(keep.sum())} of {len(gen_eval)} "
             f"({keep.mean():.1%}). Evaluation real: {real_eval['spine_key'].map(neuron_of).nunique()} neurons, {len(real_eval)} spines.", "",
             "| set | n | significant metrics (BH q < 0.05) | median abs d, % | mean W1/sd | AUC vs real |", "| --- | --- | --- | --- | --- | --- |"]
    for name, g in table.groupby("set", sort=False):
        lines.append(f"| {name} | {g['n'].iloc[0]} | {int(g['significant'].sum())} of {len(g)} | {100 * np.expm1(g['d'].abs()).median():.1f} | "
                     f"{g['w1_sd'].mean():.3f} | {aucs[name]:.3f} |")
    lines += ["", "| metric | " + " | ".join(f"{n}: d %" for n in sets) + " |", "| --- |" + " --- |" * len(sets)]
    for m in METRICS:
        cells = []
        for n in sets:
            r = table[(table["set"] == n) & (table["metric"] == m)].iloc[0]
            cells.append(f"{100 * np.expm1(r.d):+.0f}{'*' if r.significant else ''}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("d = median generated / median real - 1; * = significant after BH (interval: real resampled by neuron).")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "selector.json").write_text(json.dumps({"model": args.model, "quantile": args.quantile, "M": sel.M_,
                                                        "mean_accept_fit": sel.mean_accept_, "kept": int(keep.sum()),
                                                        "n_eval": len(gen_eval), "auc": aucs}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
