"""Branch preprocessing for modules T and P (docs/neuron-model/tp-branch-preprocessing.md).

Windows, full Minnie65. The output is small (a few GB) -> write it to the LOCAL disk (fast small-file writes; the fit
scripts then read it locally); the input (~160 GB of meshes) is read from the NAS once. No --splits: the split is
attached at fit time (``data.splits=...``), the frozen splits_v1 needs the finished S preprocessing.

    python scripts/neuron_model/run_branch_preprocessing.py --raw-data-root O:/Datasets --metadata-root O:/Datasets \
        --output-root C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches --workers 8 --io-threads 8

One neuron first (pilot, with timing):

    python scripts/neuron_model/run_branch_preprocessing.py --raw-data-root O:/Datasets --metadata-root O:/Datasets \
        --output-root C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches --neuron-id 864691134886335738

mac (local 3-neuron sample; a few branches only - project rule):

    python scripts/neuron_model/run_branch_preprocessing.py --raw-data-root datasets --metadata-root datasets \
        --s-preprocessed-root data/interim/neuron-model/smoke-test/minnie65 --output-root data/interim/neuron-model/branches-smoke \
        --neuron-id 864691134886335738

Resumable: rerun the same command; finished neurons (``_done.json`` with the same config hash) are skipped.
``--consolidate-only`` rebuilds the dataset-level tables from the finished neurons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

# Windows conda: torch (libomp) and MKL (libiomp5md) both load an OpenMP runtime; the first use of both (e.g. mesh
# extraction after training, SVD) aborts the process with "OMP: Error #15". Standard workaround, Windows only.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.neuron_model.branch_preprocessing import consolidate, load_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "data/processed/config/branch_preprocessing.yaml")
    parser.add_argument("--dataset", default="minnie65")
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True, help="folder with <dataset>/merged_branches, cell types, compartments")
    parser.add_argument("--s-preprocessed-root", type=Path, help="S-module preprocessed root (default <raw>/<dataset>/preprocessed)")
    parser.add_argument("--output-root", type=Path, help="default <raw>/<dataset>/preprocessed_branches")
    parser.add_argument("--splits", type=Path, help="splits_*.parquet: adds the split of each branch (by neuron)")
    parser.add_argument("--neuron-id", action="append", dest="neuron_ids")
    parser.add_argument("--limit-neurons", type=int)
    parser.add_argument("--merged-branch-id", action="append", type=int, dest="merged_branch_ids", help="small checks: only these merged branches")
    parser.add_argument("--limit-branches", type=int, help="small checks: first N merged branches of each neuron")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--io-threads", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="recompute finished neurons too")
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {
            "dataset": args.dataset,
            "raw_data_root": args.raw_data_root.resolve(),
            "metadata_root": args.metadata_root.resolve(),
            "s_preprocessed_root": args.s_preprocessed_root.resolve() if args.s_preprocessed_root else None,
            "output_root": args.output_root.resolve() if args.output_root else None,
            "splits": args.splits.resolve() if args.splits else None,
            "neuron_ids": args.neuron_ids,
            "limit_neurons": args.limit_neurons,
            "merged_branch_ids": args.merged_branch_ids,
            "limit_branches": args.limit_branches,
            "workers": args.workers,
            "io_threads": args.io_threads,
            "force": args.force,
        },
    )
    if args.consolidate_only:
        print(consolidate(cfg))
    else:
        run(cfg)


if __name__ == "__main__":
    main()
