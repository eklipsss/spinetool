"""Training curves of a finished run: every periodic checkpoint evaluated with ONE estimator (train and validation).

The run must have been trained with ``train.keep_every=<N>`` (and ``train.checkpoint_every`` dividing N), which writes
``checkpoints/step_<N>.pt``; the final ``latest.pt`` is added as the last point. For each checkpoint, in eval mode,
on a FIXED list of validation spines and a FIXED random subset of training spines of the same size, with fixed
query points / (t, noise) draws (src/neuron_model/spine_generation/evaluation/training_curves.py):

  VAE    sdf, surface, kl, far, eikonal, normal; total with the loss weights AT THE STEP (KL warm-up) and with the FINAL
         weights; z = mu
  MoGen  flow-matching loss (+ with the null condition for conditional runs), --n-draws independent (t, noise) draws

Mean and standard error over batches per checkpoint. Separate process on purpose: validation inside the VAE loop slows
the Windows training ~10x (validate_every=100000 in the recipe).

Windows (GPU), after training with train.keep_every=1000 train.checkpoint_every=1000:

    python scripts/neuron_model/eval_training_curves.py --run runs/training/neuron-model/<run> [--weights ema|raw|both] \
        [--n-spines 1000] [--n-draws 3] [--batch-size 16]

Writes ``<run>/training_curves/{curves.csv, curves.json, curves.png, curves.html, train_log.csv, val_inloop.csv}``;
an interrupted evaluation continues from curves.json (same settings), ``--no-reuse`` recomputes everything.
``--manifest/--splits/--path-mode`` override the run's data (e.g. on a copy of the cache).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":  # torch + MKL OpenMP runtimes (see generate_vae.py)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401  (first: on Windows torch imported after pandas/pyarrow can fail to load shm.dll)

from src.neuron_model.spine_generation.evaluation.training_curves import run_curves
from src.neuron_model.spine_generation.evaluation.training_curves_report import write_html, write_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", choices=["ema", "raw", "both"], default="ema")
    parser.add_argument("--n-spines", type=int, default=1000, help="spines per split (validation and train subset), capped by the split size")
    parser.add_argument("--n-draws", type=int, default=3, help="MoGen: independent fixed (t, noise) draws per batch")
    parser.add_argument("--batch-size", type=int, default=None, help="default: the run's data.batch_size")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--final-step", type=int, default=None, help="step of the 'final' loss weights (default train.max_steps)")
    parser.add_argument("--steps", type=int, nargs="*", help="only these checkpoint steps")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--manifest", type=Path, nargs="*")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--path-mode", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = run_curves(
        args.run, weights_kinds=("ema", "raw") if args.weights == "both" else (args.weights,), n_spines=args.n_spines, batch_size=args.batch_size,
        n_draws=args.n_draws, seed=args.seed, device=args.device, num_workers=args.num_workers, final_step=args.final_step, steps=args.steps,
        reuse=not args.no_reuse, manifests=args.manifest, splits=args.splits, path_mode=args.path_mode, out_dir=args.out,
    )
    out_dir = out["out_dir"]
    result = out["result"]
    write_png(out_dir / "curves.png", out["kind"], out["curves"], out["logs"], result["summary"])
    write_html(out_dir / "curves.html", out["kind"], out["curves"], out["logs"], result["summary"], out["cfg"], result["run"], result["settings"])
    for which, info in result["summary"].items():
        if isinstance(info, dict):
            trend = info.get("gap_trend", {})
            print(f"[{which}] best validation checkpoint: step {info['best_step']} ({info['best_val']:.5g} +- {info['best_val_se']:.2g}); "
                  f"gap val - train at the last step {info.get('gap_last', float('nan')):.4g} +- {info.get('gap_last_se', float('nan')):.2g}; "
                  f"trend {trend.get('slope_per_1000', float('nan')):.3g} +- {trend.get('slope_se', float('nan')):.2g} per 1000 steps "
                  f"({'growing' if trend.get('growing') else 'not growing'})")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
