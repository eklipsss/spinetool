"""Does the T0 DIRECTION model reproduce the per-branch functionals when it is given the real radius and length of each
branch? (models/direction_functional.py)

Separates the direction model from the rest of the generation: for every train (and val) branch one bending chain is drawn
with the branch's own node step, number of nodes, group and REAL median radius, and log kappa rms, log(tortuosity - 1) and
the kappa autocorrelation at 1 um are compared with the real branch, per length x radius stratum (quantiles 10 / 50 / 90 %
and the quantile loss of the functional fit). ``--refit`` additionally refits the direction model of the run to the train
functionals (the fit of ``model.direction_options.fit_target=functionals``) and evaluates the refit model on val, so the
new fit can be judged in minutes-hours without refitting the whole T0.

    python scripts\\neuron_model\\diagnose_t0_direction.py --t0-run runs\\training\\neuron-model\\2026-11-15_044148_t0 ^
        data.branches_root=C:\\Users\\Student\\LipsEkaterina\\data\\minnie65\\preprocessed_branches data.splits=data\\processed\\neuron-model\\splits\\splits_v1.parquet [--refit]

Writes ``<t0-run>/direction_diagnosis/{table_<model>_<split>.csv, summary.json[, direction_refit.json]}``. CPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  (Windows: torch before pandas / scipy, see CLAUDE.md)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.branch_data import branch_key, prepare_training_data, select, t0_training_branches  # noqa: E402
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path  # noqa: E402
from src.neuron_model.trunk_generation.conditions import DesignEncoder  # noqa: E402
from src.neuron_model.trunk_generation.models.direction_functional import (FunctionalSimulator, comparison_table, fit_to_functionals,  # noqa: E402
                                                                           quantile_loss, quantile_targets, real_functionals, stratum_of)
from src.neuron_model.trunk_generation.models.t0_statistical import T0Model  # noqa: E402


def evaluate(direction, branches, seed: int):
    radius = np.array([float(np.median(np.exp(b.log_radius))) for b in branches])
    real = np.array([real_functionals(b.kappa1, b.kappa2, b.step) for b in branches])
    strata = np.array([stratum_of(b.length, r) for b, r in zip(branches, radius)])
    neurons = np.array([str(b.neuron) for b in branches])
    req = [{"n_nodes": b.n_nodes, "step": b.step, "group": b.group, "radius_nm": r} for b, r in zip(branches, radius)]
    sim = FunctionalSimulator(req, seed=seed).simulate(direction.theta, direction.amplitude)
    targets = quantile_targets(real, strata, neurons, seed=seed)
    return comparison_table(real, strata, sim, strata), quantile_loss(sim, strata, targets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0-run", type=Path, required=True)
    ap.add_argument("--model", default=None, help="t0_<mean>.json of the run (default: the chosen variant of report.json)")
    ap.add_argument("--max-branches", type=int, default=3000, help="per split")
    ap.add_argument("--refit", action="store_true", help="refit the direction to the train functionals and evaluate on val")
    ap.add_argument("--maxfev", type=int, default=400)
    ap.add_argument("--n-simulated", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    run = args.t0_run
    cfg = load_config(run / "config.yaml" if (run / "config.yaml").exists() else REPO_ROOT / "configs/neuron-model/trunk/t0.yaml", args.overrides)
    d = cfg["data"]
    chosen = args.model or f"t0_{json.loads((run / 'report.json').read_text(encoding='utf-8'))['chosen']}.json"
    model = T0Model.load(run / chosen)
    direction = model.direction
    if direction is None or getattr(direction, "kind", "") != "multiscale":
        raise SystemExit(f"{chosen}: direction model is not multiscale")
    table, _, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None,
                                           io_threads=int(d["io_threads"]))
    encoder = DesignEncoder.from_dict(json.loads((run / "encoder.json").read_text(encoding="utf-8")))
    out = run / "direction_diagnosis"
    out.mkdir(exist_ok=True)
    summary = {"model": chosen}
    sets = {}
    for split, names in (("train", d["train_splits"]), ("val", d["val_splits"])):
        t = select(table, eligible="train_eligible_T", splits=names, exclude_compartments=d.get("exclude_compartments"))
        t = t[[branch_key(r) in pack for r in t.to_dict("records")]]
        if len(t) > args.max_branches:
            t = t.sample(args.max_branches, random_state=args.seed)
        sets[split] = t0_training_branches(t.reset_index(drop=True), pack, encoder)
        print(f"[diag] {split}: {len(sets[split])} branches", flush=True)

    def report(tag, direc):
        for split, branches in sets.items():
            rows, loss = evaluate(direc, branches, args.seed + 1)
            pd.DataFrame(rows).to_csv(out / f"table_{tag}_{split}.csv", index=False)
            summary[f"{tag}_{split}_loss"] = loss
            print(f"\n[diag] {tag}, {split}: quantile loss {loss:.1f} (real | direction model with the real radius and length: p10 / median / p90)", flush=True)
            for r in rows:
                print(f"  {r['stratum']:16s} {r['functional']:22s} n {r['n_real']:4d}  real {r['real_p10']:+.3f} / {r['real_median']:+.3f} / {r['real_p90']:+.3f}"
                      f"  | model {r['sim_p10']:+.3f} / {r['sim_median']:+.3f} / {r['sim_p90']:+.3f}", flush=True)

    report("current", direction)
    if args.refit:
        new = copy.deepcopy(direction)
        info = fit_to_functionals(new, sets["train"], n_branches=args.n_simulated, maxfev=args.maxfev, seed=args.seed, log=lambda s: print(s, flush=True))
        (out / "direction_refit.json").write_text(json.dumps({"direction": new.to_dict(), "fit": info}, indent=2, default=float), encoding="utf-8")
        report("refit", new)
        a = new.amplitude
        print(f"\n[diag] refit: fast wiggle sd {new.sd_fast:.3g} / {new.length_fast_nm / 1000:.3g} um, node noise {new.noise_sd:.3g}, slow turning L_d {new.restore_nm / 1000:.3g} um, "
              f"memory {new.drive_length_nm / 1000:.3g} um, amplitude {new.drive_std * 1000:.3g} /um; branch amplitude x exp({a.alpha:.2f}) x r^{a.beta:.2f}, spread x/ {np.exp(a.tau):.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
