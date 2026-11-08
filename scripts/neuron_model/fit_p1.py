"""Fit P1 (spine positions with interaction, p-module-technical-spec.md section 7) on top of a fitted P0 and compare
P0 / P1a (renewal along the branch) / P1b (Strauss on the trunk surface) on the validation branches.

Why: the assembled branches place spines less regularly than real ones (2026-10-29: nearest neighbour -94 nm, intervals
-200 nm, significant over neurons; real-T + generated-P test: intervals 1 188 vs 1 370 nm), and real spines repel each
other at short range (g(0-0.5 um) = 0.76, 2026-11-02). P0 places the N spines of a branch independently.

    python scripts/neuron_model/fit_p1.py --p0-run runs/training/neuron-model/<id>_p0 [--n-eval 400 --n-sim 19]

Both models keep the P0 count model and the P0 longitudinal density (the same ``p0_bundle.json``); only the dependence
between points changes. Validation:
  * P1a: renewal log-likelihood on the validation branches at the fitted k against k = 1 (P0);
  * all three: Monte Carlo goodness of fit - for --n-eval random validation branches, --n-sim simulated configurations at
    the observed N of every branch; g(r) along the branch (0-0.5 / 0.5-1 / 1-2 / 2-4 um), median nearest-neighbour
    interval, CV of the intervals, median and 10 % quantile of the nearest-neighbour SURFACE distance; z of the observed
    value and whether it lies outside the 2.5-97.5 % envelope. ``chosen`` = the smallest mean |z|.

Writes ``runs/training/neuron-model/<date>_p1/`` with ``p1a_bundle.json``, ``p1b_bundle.json``, ``p0_bundle.json`` (= the
chosen one, so ``generate_tp.py --p0-run <this run>`` uses it; ``--bundle p1a_bundle.json`` picks another) and
``report.json``. CPU; P1b simulation dominates (minutes to tens of minutes).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402,F401  (torch first: Windows DLL order, see generate_tp.py)
import numpy as np  # noqa: E402


def main() -> None:
    from src.neuron_model.branch_data import position_branches, prepare_training_data, select
    from src.neuron_model.placement_generation.diagnostics import simulation_comparison
    from src.neuron_model.placement_generation.models.interaction import RenewalModel, StraussModel
    from src.neuron_model.placement_generation.models.position import PositionModel
    from src.neuron_model.spine_generation.experiment.config import resolve_path
    from src.neuron_model.spine_generation.experiment.run import create_run, open_run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0-run", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=400, help="validation branches for the simulation check")
    ap.add_argument("--n-sim", type=int, default=19, help="simulated configurations per model (19 -> 5-95 %% envelope; 39 -> 2.5-97.5 %%)")
    ap.add_argument("--strauss-branches", type=int, default=3000, help="training branches for the pseudo-likelihood")
    ap.add_argument("--strauss-radii", type=float, nargs="*", help="radius grid, nm (default: d_min ... 90 %% quantile of the nearest-neighbour surface distance)")
    ap.add_argument("--n-angles", type=int, default=32)
    ap.add_argument("--burn-factor", type=int, default=50, help="Metropolis-Hastings moves per point")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    started = time.time()

    p0 = open_run(args.p0_run)
    cfg = p0.load_config()
    d = cfg["data"]
    bundle = json.loads((p0.root / "p0_bundle.json").read_text(encoding="utf-8"))
    base = PositionModel.from_dict(bundle["position_model"])
    table, spines, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]),
                                                splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    tr = select(table, eligible="train_eligible_P", splits=d["train_splits"], exclude_compartments=d.get("exclude_compartments"))
    va = select(table, eligible="train_eligible_P", splits=d["val_splits"], exclude_compartments=d.get("exclude_compartments"))
    train_pos, val_pos = position_branches(tr, spines, pack), position_branches(va, spines, pack)
    print(f"[p1] P0 {args.p0_run.name}: train {len(train_pos)} branches, validation {len(val_pos)}", flush=True)

    renewal = RenewalModel(base).fit(train_pos)
    val_ll, val_ll_1 = renewal.loglik(val_pos), renewal.loglik(val_pos, 1.0)
    renewal.fit_info.update(val_loglik=val_ll, val_loglik_poisson=val_ll_1, val_gain=val_ll - val_ll_1)
    print(f"[p1a] Weibull k {renewal.k:.3f} (1 = P0; > 1 regular); train LR {renewal.fit_info['lr_statistic']:.1f} on "
          f"{renewal.fit_info['n_interior_gaps']} intervals; validation log-lik gain over k = 1: {val_ll - val_ll_1:+.1f} ({time.time() - started:.0f} s)", flush=True)

    strauss = StraussModel(base, burn_factor=args.burn_factor).fit(train_pos, radii=args.strauss_radii, d_min_nm=float(bundle["d_min_nm"]),
                                                                    n_angles=args.n_angles, max_branches=args.strauss_branches, seed=args.seed)
    fi = strauss.fit_info
    print(f"[p1b] R {strauss.radius_nm:.0f} nm, gamma {strauss.gamma:.3f} on {fi['n_branches']} branches / {fi['n_points']} spines"
          f"{' (R at the edge of the grid!)' if fi['radius_at_grid_edge'] else ''} ({time.time() - started:.0f} s)", flush=True)
    for r in fi["profile"]:
        print(f"    R {r['radius_nm']:7.0f} nm: log PL {r['log_pl']:.1f} (gain over gamma = 1 {r['log_pl_gain_over_poisson']:+.1f}), gamma {r['gamma']:.3f}, "
              f"neighbours within R {r['mean_neighbours_within_R']:.2f}", flush=True)

    rng = np.random.default_rng(args.seed)
    eval_pool = [b for b in val_pos if b.positions is not None and b.angles is not None and len(b.positions) >= 1]
    eval_set = [eval_pool[i] for i in rng.choice(len(eval_pool), min(args.n_eval, len(eval_pool)), replace=False)]
    check = simulation_comparison({"P0": base, "P1a": renewal, "P1b": strauss}, eval_set, n_sim=args.n_sim, seed=args.seed)
    scores = {k: v["score_mean_abs_z"] for k, v in check["models"].items()}
    chosen = min(scores, key=scores.get)
    keys = list(check["observed"])
    print(f"\n[p1] simulation check on {check['n_branches']} validation branches ({check['n_points']} spines), {args.n_sim} simulations; "
          f"observed / simulated mean (z, * = outside the envelope):", flush=True)
    print("  statistic".ljust(26) + "observed".rjust(10) + "".join(m.rjust(22) for m in check["models"]), flush=True)
    for k in keys:
        row = f"  {k}".ljust(26) + f"{check['observed'][k]:10.3f}"
        for m, v in check["models"].items():
            t = v["statistics"][k]
            row += f"{t['sim_mean']:12.3f} ({t['z']:+5.1f}){'*' if t['outside'] else ' '}".rjust(22)
        print(row, flush=True)
    print("  mean |z|".ljust(36) + "".join(f"{scores[m]:22.2f}" for m in check["models"]), flush=True)

    out_cfg = copy.deepcopy(cfg)
    out_cfg["name"] = "p1"
    out_cfg["p1"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run = create_run(out_cfg, runs_root=resolve_path(cfg.get("runs_root", "runs/training/neuron-model")),
                     extra_info={"p0_run": str(args.p0_run), "n_train": len(train_pos), "n_val": len(val_pos)})
    bundles = {"P1a": {**bundle, "interaction": renewal.to_dict()}, "P1b": {**bundle, "interaction": strauss.to_dict()}, "P0": bundle}
    (run.root / "p1a_bundle.json").write_text(json.dumps(bundles["P1a"]), encoding="utf-8")
    (run.root / "p1b_bundle.json").write_text(json.dumps(bundles["P1b"]), encoding="utf-8")
    (run.root / "p0_bundle.json").write_text(json.dumps(bundles[chosen]), encoding="utf-8")
    report = {"p0_run": str(args.p0_run), "renewal": renewal.fit_info, "strauss": strauss.fit_info, "simulation_check": check, "scores": scores,
              "chosen": chosen, "seconds": time.time() - started}
    (run.root / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n[p1] chosen {chosen} (smallest mean |z|); p0_bundle.json = {chosen}; run -> {run.root}", flush=True)


if __name__ == "__main__":
    main()
