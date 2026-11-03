"""Build the model bundle of the web application (docs/neuron-model/webapp-architecture.md): one self-contained folder.

Windows (CPU; the spine morphometrics take ~0.5 s per spine - a library of a few thousand spines is under an hour):

    python scripts/neuron_model/build_webapp_bundle.py \
        --t0-run runs/training/neuron-model/<id>_t0 --p0-run runs/training/neuron-model/<id>_p0 \
        --spines any=runs/training/neuron-model/<mogen_run>/meshes/cond_cell_type_binary-excitatory_g2_seed_1 \
        --spines mouse/excitatory=runs/training/neuron-model/<mogen_run>/meshes/cond_cell_type_binary-excitatory_g2_seed_1 \
        --spines mouse/inhibitory=runs/training/neuron-model/<mogen_run>/meshes/cond_cell_type_binary-inhibitory_g2_seed_1 \
        --context-spines C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches/branch_spines.parquet \
        --branches-reference C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches/branches.parquet \
        --max-spines-per-key 500

``--spines KEY=FOLDER`` (repeatable): spine meshes in the local spine frame (``**/generated_mesh_postprocessed.off`` of the
generators, else any *.off / *.ply / *.obj). Keys: ``any`` (required, the last fallback), ``<species>``,
``<species>/<cell_type_binary>``, ``<species>/<cell_type_binary>/<detailed cell type>`` (e.g. ``mouse/excitatory/BPC``).
``--context-spines``: REAL spines of the branch preprocessing -> the pool spine is matched to the trunk radius (default:
uniform choice). ``--branches-reference``: real branches -> quantiles per condition for the comparison tab.
Writes ``models/webapp/<bundle-id>/`` (git-ignored); copy the folder to the serving machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch FIRST, before pandas / pyarrow / scipy: imported after them on Windows it fails with
# "OSError: [WinError 127] ... torch\lib\shm.dll" (the P0 count model imports torch)
import torch  # noqa: F401,E402

from src.neuron_model.spine_generation.experiment.config import load_config
from src.neuron_model.webapp.bundle import DEFAULT_LIMITS, build_bundle


def _key_value(text: str):
    key, sep, value = text.partition("=")
    if not sep or not key or not value:
        raise argparse.ArgumentTypeError(f"expected KEY=FOLDER, got {text!r}")
    return key, Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t0-run", type=Path, required=True)
    parser.add_argument("--t0-model", default=None, help="parabola / gam / parabola_mix (default: the one chosen in report.json)")
    parser.add_argument("--p0-run", type=Path, required=True)
    parser.add_argument("--spines", type=_key_value, action="append", required=True, metavar="KEY=FOLDER")
    parser.add_argument("--context-spines", type=Path, default=None, help="branch_spines.parquet of REAL spines (base size given the trunk radius)")
    parser.add_argument("--branches-reference", type=Path, default=None, help="branches.parquet (or its folder) of the branch preprocessing")
    parser.add_argument("--assembly-config", type=Path, default=REPO_ROOT / "configs/neuron-model/assembly/default.yaml")
    parser.add_argument("--max-spines-per-key", type=int, default=0, help="at most this many meshes per library key (0 = all)")
    parser.add_argument("--no-morphometrics", action="store_true", help="do not precompute the spine morphometrics (CGAL; ~0.5 s per spine)")
    parser.add_argument("--max-length-um", type=float, default=DEFAULT_LIMITS["max_length_nm"] / 1000.0)
    parser.add_argument("--max-spines", type=int, default=DEFAULT_LIMITS["max_spines"], help="cap on spines per generated branch (server load)")
    parser.add_argument("--all-categories", action="store_true",
                        help="offer every species / cell type / compartment, also those without a trained T0 group (pooled parameters, with a warning)")
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "models/webapp")
    args = parser.parse_args()
    spines = dict(args.spines)
    if len(spines) != len(args.spines):
        parser.error("a --spines key is given twice")
    raw = load_config(args.assembly_config, []) if args.assembly_config.exists() else {}
    root = build_bundle(args.out_root, t0_run=args.t0_run, p0_run=args.p0_run, spines=spines, bundle_id=args.bundle_id, t0_model=args.t0_model,
                        context_spines=args.context_spines, branches_reference=args.branches_reference, assembly=raw.get("assembly", raw),
                        compute_morphometrics=not args.no_morphometrics, max_spines_per_key=args.max_spines_per_key,
                        limits={"max_length_nm": args.max_length_um * 1000.0, "max_spines": args.max_spines}, all_categories=args.all_categories)
    print(f"[webapp-bundle] done: python scripts/neuron_model/run_webapp.py --bundle {root}", flush=True)


if __name__ == "__main__":
    main()
