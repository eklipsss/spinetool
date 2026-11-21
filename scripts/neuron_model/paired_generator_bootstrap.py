"""Paired neuron-level bootstrap of generator differences from a finished ``evaluate_generators.py`` run (tables only).

Reads ``<eval-run>/real_test_morphometrics.parquet`` and ``<eval-run>/<view>/morphometrics_<model>_seed<s>.parquet``,
compares every model with the SAME resampled real neurons (see ``evaluation/paired_bootstrap.py``) and writes
``<eval-run>/paired_bootstrap_<view>/{points.csv, replicates.parquet, differences.csv, summary.md}``.

    python scripts/neuron_model/paired_generator_bootstrap.py --eval-run <run> --models mogen_A mogen_v3 real_poisson \
        --pairs mogen_A:mogen_v3 mogen_A:real_poisson mogen_v3:real_poisson --n-boot 500
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.evaluation.paired_bootstrap import METRICS, pair_differences, paired_bootstrap  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-run", type=Path, required=True)
    p.add_argument("--view", default="postprocessed")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="a:b (difference a - b)")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--n-real", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-auc", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    real = pd.read_parquet(args.eval_run / "real_test_morphometrics.parquet", columns=["spine_key", *METRICS])
    gens = {}
    for m in args.models:
        files = sorted((args.eval_run / args.view).glob(f"morphometrics_{m}_seed*.parquet"))
        if not files:
            raise SystemExit(f"no morphometrics_{m}_seed*.parquet in {args.eval_run / args.view}")
        parts = []
        for f in files:
            frame = pd.read_parquet(f)
            frame["generation_seed"] = int(re.search(r"_seed(\d+)\.parquet$", f.name).group(1))
            parts.append(frame)
        gens[m] = pd.concat(parts, ignore_index=True)
    pairs = [tuple(s.split(":")) for s in args.pairs]
    points, reps = paired_bootstrap(real, gens, n_boot=args.n_boot, n_real=args.n_real, seed=args.seed, auc=not args.no_auc)
    diffs = pair_differences(points, reps, pairs)
    out = args.out or (args.eval_run / f"paired_bootstrap_{args.view}")
    out.mkdir(parents=True, exist_ok=True)
    points.to_csv(out / "points.csv", index=False)
    reps.to_parquet(out / "replicates.parquet", index=False)
    diffs.to_csv(out / "differences.csv", index=False)
    main_stats = ["auc", "energy", "w1_mean", "w1_Volume", "w1_Area", "w1_JunctionArea", "w1_LengthAreaRatio", "w1_Length"]
    lines = ["# Paired neuron-level bootstrap", "",
             f"Eval run `{args.eval_run.name}`, view {args.view}, {args.n_boot} replicates (real neurons resampled, "
             f"generated samples resampled within seed), {args.n_real} real spines per replicate. Lower = closer to real.", "",
             "| a - b | statistic | a | b | diff | 95 % CI | p | share a better |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in diffs[diffs["statistic"].isin(main_stats)].iterrows():
        lines.append(f"| {r.a} - {r.b} | {r.statistic} | {r.a_value:.4f} | {r.b_value:.4f} | {r['diff']:+.4f} | [{r.lo:+.4f}; {r.hi:+.4f}] | {r.p:.3f} | {r.share_a_better:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
