"""Fit and validate P0 (spine count + positions) on the preprocessed branches.

Windows (after run_branch_preprocessing.py; CPU only, minutes):

    python scripts/neuron_model/fit_p0.py --config configs/neuron-model/placement/p0.yaml \
        data.branches_root=C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches

Count models of the grid family x exposure x neuron effect are compared by the validation NLL; the positional
density with each covariate dropped in turn; d_min from the observed attachment points. Writes
``runs/training/neuron-model/<id>_p0/`` with ``p0_bundle.json`` (design + chosen count model + position model + d_min)
and ``report.json``.
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

from src.neuron_model.branch_data import count_covariates, key_series, position_branches, prepare_training_data, select
from src.neuron_model.placement_generation.fit import estimate_d_min, fit_p0
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.run import create_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/neuron-model/placement/p0.yaml")
    parser.add_argument("--force-pack", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    d, m = cfg["data"], cfg["model"]
    if not d.get("branches_root"):
        parser.error("set data.branches_root=<output root of run_branch_preprocessing.py>")
    table, spines, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None,
                                                io_threads=int(d["io_threads"]), force_pack=args.force_pack)
    tr = select(table, eligible="train_eligible_P", splits=d["train_splits"])
    va = select(table, eligible="train_eligible_P", splits=d["val_splits"])
    if not len(tr) or not len(va):
        raise SystemExit(f"no train ({len(tr)}) or validation ({len(va)}) branches: is the split filled? (data.splits=...)")
    train_cov, val_cov = count_covariates(tr, pack), count_covariates(va, pack)
    train_pos, val_pos = position_branches(tr, spines, pack), position_branches(va, spines, pack)
    print(f"[p0] train {len(train_cov)} branches, validation {len(val_cov)}", flush=True)
    result = fit_p0(train_cov, val_cov, train_pos, val_pos, families=m["families"], exposures=m["exposures"], random_effects=m["random_effects"],
                    position_use=m["position_use"], n_basis=int(m["n_basis"]), smooth_penalty=float(m["smooth_penalty"]), max_iter=int(m["max_iter"]), seed=int(cfg["seed"]))
    train_spines = spines[key_series(spines).isin(set(key_series(tr)))] if len(spines) else spines
    d_min = estimate_d_min(train_spines, float(m["d_min_quantile_percent"]))
    result["report"]["d_min"] = d_min
    run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_train": len(train_cov), "n_val": len(val_cov), "branches_root": str(d["branches_root"])})
    bundle = {"design": result["design"].to_dict(), "count_model": result["count_model"].to_dict(), "position_model": result["position_model"].to_dict(), "d_min_nm": d_min["d_min_nm"]}
    (run.root / "p0_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    (run.root / "report.json").write_text(json.dumps(result["report"], indent=2, default=float), encoding="utf-8")
    c = result["report"]["count_chosen"]
    print(f"[p0] chosen count model: {c['family']} / {c['exposure']} / neuron effect {c['random_effect']}; d_min {d_min['d_min_nm']:.0f} nm; run -> {run.root}", flush=True)


if __name__ == "__main__":
    main()
