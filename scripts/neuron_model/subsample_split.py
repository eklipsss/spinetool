"""Split file with a random subsample of one part (the others kept as they are, the rest unassigned).

    python scripts/neuron_model/subsample_split.py --splits <splits.parquet> --part train --n 20000 --keep test \
        [--part-from <other splits.parquet> ...] --out <new.parquet>

Use (2026-11-24): the Chamfer memorization of evaluate_generators.py compares the distance of every generated and every
held-out real cloud to its nearest TRAINING cloud; the training clouds are read once into a local cache. With ~600 000
training spines on the NAS that cache was never built (chains AF / AI ran with real.chamfer_memorization=false). A random
subsample of the training part keeps the test unbiased: generated and held-out real clouds are measured against the SAME
subsample, so the ratio of their median distances and the near-copy rate keep their meaning (a copy of a training spine
outside the subsample is not caught: the near-copy rate is a lower bound, scaled by n / n_train). Sampling is by spine.
``--part-from``: the sampled part is the UNION of that part in these files (e.g. the training parts of splits_v1 and of the
dev split, when the compared models were trained on different splits); rows already in a kept part are excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--part", default="train")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--keep", nargs="*", default=["test"], help="parts copied unchanged")
    ap.add_argument("--part-from", type=Path, nargs="*", default=[], help="more split files whose --part is pooled with that of --splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    s = pd.read_parquet(args.splits)
    kept = s[s["split"].isin(args.keep)]
    sources = [s] + [pd.read_parquet(p) for p in args.part_from]
    part = pd.concat([f[f["split"] == args.part] for f in sources], ignore_index=True).drop_duplicates("spine_key")
    part = part[~part["spine_key"].isin(set(kept["spine_key"]))].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    pick = part.iloc[np.sort(rng.choice(len(part), size=min(args.n, len(part)), replace=False))]
    out = pd.concat([pick, kept], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{args.part}: {len(pick)} of {len(part)} spines ({pick['group_id'].nunique() if 'group_id' in pick.columns else '?'} groups); "
          + ", ".join(f"{k}: {int((out['split'] == k).sum())}" for k in args.keep) + f" -> {args.out}")


if __name__ == "__main__":
    main()
