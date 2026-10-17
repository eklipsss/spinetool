"""Fit and validate T0 (statistical trunk model) on the preprocessed branches.

Windows (after run_branch_preprocessing.py; CPU only, minutes):

    python scripts/neuron_model/fit_t0.py --config configs/neuron-model/trunk/t0.yaml \
        data.branches_root=C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches

The first call packs the node arrays of the eligible branches into ``data.pack_dir`` (local disk, threads; reused
afterwards). Writes ``runs/training/neuron-model/<id>_t0/`` with ``t0_<mean>.json``, ``encoder.json``, ``limits.json``
and ``report.json`` (acceptance, functionals vs validation, noise floor).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.neuron_model.branch_data import _cond, prepare_training_data, select, t0_training_branches
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.run import create_run
from src.neuron_model.trunk_generation.conditions import DesignEncoder
from src.neuron_model.trunk_generation.fit import fit_t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/neuron-model/trunk/t0.yaml")
    parser.add_argument("--force-pack", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    d, m = cfg["data"], cfg["model"]
    if not d.get("branches_root"):
        parser.error("set data.branches_root=<output root of run_branch_preprocessing.py>")
    table, _, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None,
                                           io_threads=int(d["io_threads"]), force_pack=args.force_pack)
    tr = select(table, eligible="train_eligible_T", splits=d["train_splits"])
    va = select(table, eligible="train_eligible_T", splits=d["val_splits"])
    if len(va) > int(d["max_val_branches"]):
        va = va.sample(int(d["max_val_branches"]), random_state=int(cfg["seed"])).reset_index(drop=True)
    if not len(tr) or not len(va):
        raise SystemExit(f"no train ({len(tr)}) or validation ({len(va)}) branches: is the split filled? (data.splits=...)")
    encoder = DesignEncoder.fit(tr["length_nm"].astype(float).tolist())
    train, val = t0_training_branches(tr, pack, encoder), t0_training_branches(va, pack, encoder)
    print(f"[t0] train {len(train)} branches, validation {len(val)}", flush=True)
    rows = va.to_dict("records")
    result = fit_t0(train, val, [_cond(r) for r in rows], [str(r["neuron_id"]) for r in rows], encoder=encoder, mean_models=m["mean_models"],
                    shrinkage_std=float(m["shrinkage_std"]), max_iter=int(m["max_iter"]), length_bounds=tuple(m["length_bounds_nm"]), seed=int(cfg["seed"]))
    run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_train": len(train), "n_val": len(val), "branches_root": str(d["branches_root"])})
    for name, model in result["models"].items():
        model.save(run.root / f"t0_{name}.json")
    (run.root / "encoder.json").write_text(json.dumps(encoder.to_dict()), encoding="utf-8")
    (run.root / "limits.json").write_text(json.dumps(result["limits"].to_dict(), indent=2), encoding="utf-8")
    (run.root / "report.json").write_text(json.dumps(result["report"], indent=2, default=float), encoding="utf-8")
    print(f"[t0] chosen: {result['chosen']}; noise floor W1/std {result['report']['noise_floor_mean_w1_rel']:.3f}; run -> {run.root}", flush=True)


if __name__ == "__main__":
    main()
