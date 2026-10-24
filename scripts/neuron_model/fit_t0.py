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
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch FIRST, before pandas / pyarrow / scipy: imported after them on Windows it fails with
# "OSError: [WinError 127] ... torch\lib\shm.dll" (a DLL of the same name is already loaded)
import torch  # noqa: F401,E402

from src.neuron_model.branch_data import _cond, branch_key, prepare_training_data, select, t0_training_branches
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
    tr = select(table, eligible="train_eligible_T", splits=d["train_splits"], exclude_compartments=d.get("exclude_compartments"))
    va = select(table, eligible="train_eligible_T", splits=d["val_splits"], exclude_compartments=d.get("exclude_compartments"))
    if len(va) > int(d["max_val_branches"]):
        va = va.sample(int(d["max_val_branches"]), random_state=int(cfg["seed"])).reset_index(drop=True)
    # keep only rows with geometry in the pack: the conditions below are matched to the branches by position
    tr = tr[[branch_key(r) in pack for r in tr.to_dict("records")]].reset_index(drop=True)
    va = va[[branch_key(r) in pack for r in va.to_dict("records")]].reset_index(drop=True)
    if not len(tr) or not len(va):
        raise SystemExit(f"no train ({len(tr)}) or validation ({len(va)}) branches: is the split filled? (data.splits=...)")
    encoder = DesignEncoder.fit(tr["length_nm"].astype(float).tolist())
    train, val = t0_training_branches(tr, pack, encoder), t0_training_branches(va, pack, encoder)
    print(f"[t0] train {len(train)} branches, validation {len(val)}", flush=True)
    rows = va.to_dict("records")
    result = fit_t0(train, val, [_cond(r) for r in rows], [str(r["neuron_id"]) for r in rows], encoder=encoder, mean_models=m["mean_models"],
                    shrinkage_std=float(m["shrinkage_std"]), max_iter=int(m["max_iter"]), length_bounds=tuple(m["length_bounds_nm"]), seed=int(cfg["seed"]),
                    direction=str(m.get("direction", "tilt")), radius_residual=str(m.get("radius_residual", "branch")),
                    train_conditions=[_cond(r) for r in tr.to_dict("records")])
    run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_train": len(train), "n_val": len(val), "branches_root": str(d["branches_root"])})
    for name, model in result["models"].items():
        model.save(run.root / f"t0_{name}.json")
    (run.root / "encoder.json").write_text(json.dumps(encoder.to_dict()), encoding="utf-8")
    (run.root / "limits.json").write_text(json.dumps(result["limits"].to_dict(), indent=2), encoding="utf-8")
    (run.root / "report.json").write_text(json.dumps(result["report"], indent=2, default=float), encoding="utf-8")
    if result["report"].get("direction"):
        p = result["report"]["direction"]["pooled"]
        a = result["report"]["direction"].get("amplitude") or {}
        rest = "inf (no restoring)" if p["restore_nm"] is None else f"{p['restore_nm'] / 1000.0:g} um"
        print(f"[t0] direction: restoring length L_d {rest}, turning-rate memory {p['drive_length_nm'] / 1000.0:.3g} um; "
              f"turning amplitude of a branch with r = 300 nm {p['drive_std'] * 1000.0:.3g} /um, x r^{a.get('beta', 0.0):.2f}, "
              f"spread between branches x/: {math.exp(a.get('tau', 0.0)):.2f}; group offsets {({k: round(v, 2) for k, v in a.get('groups', {}).items()})}", flush=True)
    if result["report"].get("radius_residual"):
        q = result["report"]["radius_residual"]
        print(f"[t0] radius residual: memory {q['length_nm'] / 1000.0:.3g} um; amplitude (sd of log r) of a branch with r = 300 nm, L = 20 um "
              f"{math.exp(q['alpha']):.3f}, x r^{q['beta_r']:.2f} x L^{q['beta_l']:.2f}, spread between branches x/: {math.exp(q['tau']):.2f}; "
              f"single pooled amplitude would be {q['info'].get('pooled_amplitude', float('nan')):.3f}, median branch {q['info'].get('median_branch_amplitude', float('nan')):.3f}", flush=True)
    print(f"[t0] chosen: {result['chosen']}; noise floor W1/std {result['report']['noise_floor_mean_w1_rel']:.3f}; run -> {run.root}", flush=True)


if __name__ == "__main__":
    main()
