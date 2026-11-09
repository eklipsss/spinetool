"""Tails of the morphometrics of generated sets against the real test spines of one finished evaluation.

Why: W1/sd of Volume can be large while the classifier's median shift is small (chain R: MoGen B after the screened
Poisson chain has Volume W1/sd 3.1-3.2 at a median shift of +0.28 sd) - a few samples with a huge volume. This prints the
quantiles of each metric per set relative to the real ones, the share of samples beyond the real 99th percentile and the
worst samples (their folders can then be opened and looked at).

    python scripts/neuron_model/morphometric_tails.py --eval-run runs/analysis/neuron-model/<date>_mogen-poisson-val \
        --models B_d8_pw8 A_d8_pw8 real_d8_pw8 --metrics Volume Area Length
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-run", type=Path, required=True)
    ap.add_argument("--mesh-kind", default="postprocessed", choices=["raw", "postprocessed"])
    ap.add_argument("--models", nargs="*", help="default: every morphometrics_<model>_seed<k>.parquet of the run")
    ap.add_argument("--metrics", nargs="+", default=["Volume", "Area", "Length", "ConvexHullVolume"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    real = pd.read_parquet(args.eval_run / "real_test_morphometrics.parquet")
    folder = args.eval_run / args.mesh_kind
    models = args.models or sorted(p.name[len("morphometrics_"):-len(f"_seed{args.seed}.parquet")]
                                   for p in folder.glob(f"morphometrics_*_seed{args.seed}.parquet"))
    q = [0.5, 0.9, 0.99]
    for metric in args.metrics:
        r = real[metric].dropna().to_numpy(dtype=float)
        rq = np.quantile(r, q)
        print(f"\n=== {metric}: real test n={len(r)}  median {rq[0]:.4g}  p90 {rq[1]:.4g}  p99 {rq[2]:.4g}  max {r.max():.4g}")
        print(f"{'set':<22}{'n':>5}{'p50/real':>10}{'p90/real':>10}{'p99/real':>10}{'max/real p99':>14}{'> real p99':>12}{'> 3x real p99':>15}")
        for m in models:
            path = folder / f"morphometrics_{m}_seed{args.seed}.parquet"
            if not path.exists():
                print(f"{m:<22} (no table)")
                continue
            g = pd.read_parquet(path)
            v = g[metric].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if not ok.any():
                print(f"{m:<22} (no values)")
                continue
            gq = np.quantile(v[ok], q)
            print(f"{m:<22}{ok.sum():>5}{gq[0] / rq[0]:>10.2f}{gq[1] / rq[1]:>10.2f}{gq[2] / rq[2]:>10.2f}"
                  f"{v[ok].max() / rq[2]:>14.1f}{(v[ok] > rq[2]).mean():>12.3f}{(v[ok] > 3 * rq[2]).mean():>15.3f}")
            worst = g.loc[ok].nlargest(args.top, metric)
            cols = [c for c in ["sample_id", metric, "Area", "Length", "attachment_found"] if c in worst.columns]
            cols = list(dict.fromkeys(cols))
            print("    worst: " + "; ".join(", ".join(f"{c}={row[c]:.4g}" if isinstance(row[c], float) else f"{c}={row[c]}"
                                                    for c in cols) for _, row in worst.iterrows()))


if __name__ == "__main__":
    main()
