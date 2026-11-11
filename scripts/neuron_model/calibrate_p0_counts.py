"""Is the P0 count model off for a cell type on the TRAINING neurons too, or only on the held-out ones?

    python scripts/neuron_model/calibrate_p0_counts.py --p0-run runs/training/neuron-model/<id>_p0 [--splits train val]

For each split: per neuron observed spines vs the model mean (neuron effect 0 and integrated out), empty branches vs
simulated, rough neuron effect log((obs + 0.5) / (E0 + 0.5)); per (split, cell_type_binary) and per (split, detailed
cell type) the totals and the spread of the neuron effect against the single sigma_b of the model
(``placement_generation/calibration.py``). Reads the data of the P0 run (local tables and geometry pack), CPU, minutes.
Also: is a held-out group (e.g. the 8 val inhibitory neurons) unusual against random subsets of as many training
neurons of its group (``held_out_rarity``). Output: ``<p0-run>/calibration/{neurons, groups, groups_cell_type,
held_out_rarity, branches}.csv``.
Branch level (2026-11-11, ``branch_calibration``): per covariate class (radius, length, distance to the soma, terminal,
compartment, cell type) observed density, share of empty branches and spine-weighted density against simulations with
fresh neuron effects and with each neuron's effect drawn from its posterior given its own counts: off under the
posterior = the branch-level part of the count model (covariate shape, dispersion), off only with fresh effects = the
neuron level. ``*`` = observed outside the 5-95 % band of the posterior simulations.
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

import torch  # noqa: F401,E402  (first on Windows, see generate_tp.py)

import pandas as pd

from src.neuron_model.branch_data import count_covariates, load_branches, prepare_training_data, select
from src.neuron_model.placement_generation.calibration import branch_calibration, group_calibration, held_out_rarity, neuron_calibration
from src.neuron_model.placement_generation.fit import load_p0_bundle
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run


def _branch_table(b: pd.DataFrame) -> pd.DataFrame:
    star = lambda kind: lambda r: "*" if not (r[f"posterior_{kind}_p05"] <= r[f"obs_{kind}"] <= r[f"posterior_{kind}_p95"]) else ""
    out = b[["split", "factor", "level", "n_neurons", "n_branches"]].copy()
    for kind, label in (("density", "density"), ("empty", "empty"), ("sw_density", "sw")):
        flag = b.apply(star(kind), axis=1)
        out[label] = [f"{o:.3f}{f} | {fr:.3f} | {po:.3f}" for o, f, fr, po in zip(b[f"obs_{kind}"], flag, b[f"fresh_{kind}"], b[f"posterior_{kind}"])]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p0-run", type=Path, required=True, help="a P0 run (or a P1 run: its p0_bundle.json carries the same count model)")
    parser.add_argument("--bundle", default="p0_bundle.json")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--n-sim", type=int, default=200)
    parser.add_argument("--n-sim-branch", type=int, default=100, help="simulations of the branch-level table (0 = skip)")
    args = parser.parse_args()
    run = open_run(args.p0_run)
    d = run.load_config()["data"]
    bundle = load_p0_bundle(run.root / args.bundle)
    model, design = bundle["count_model"], bundle["design"]
    root = resolve_path(d["branches_root"])
    table, _, pack = prepare_training_data(root, resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    detailed = load_branches(root, ["cell_type"])
    cell_type = detailed.drop_duplicates("neuron_id").set_index("neuron_id")["cell_type"].astype(str).to_dict() if "cell_type" in detailed.columns else None

    cov = {s: count_covariates(select(table, eligible="train_eligible_P", splits=[s], exclude_compartments=d.get("exclude_compartments")), pack) for s in args.splits}
    neurons = pd.concat([neuron_calibration(model, design, cov[s], split=s, cell_type=cell_type, n_sim=args.n_sim) for s in args.splits], ignore_index=True)
    groups = group_calibration(neurons)
    by_type = group_calibration(neurons, by=("split", "cell_type_binary", "cell_type"))
    out = run.root / "calibration"
    out.mkdir(exist_ok=True)
    neurons.to_csv(out / "neurons.csv", index=False)
    groups.to_csv(out / "groups.csv", index=False)
    by_type.to_csv(out / "groups_cell_type.csv", index=False)
    rarity = held_out_rarity(neurons)
    rarity.to_csv(out / "held_out_rarity.csv", index=False)
    branches = pd.concat([branch_calibration(model, design, cov[s], split=s, n_sim=args.n_sim_branch) for s in args.splits], ignore_index=True) if args.n_sim_branch > 0 else pd.DataFrame()
    if len(branches):
        branches.to_csv(out / "branches.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(f"[calibrate-p0] {model.family} / {model.exposure}; sigma_b by group {model.fit_info.get('sigma_b_by_group', model.sigma_b)}")
        print(groups.to_string(index=False))
        print(by_type[by_type["n_neurons"] >= 2].to_string(index=False))
        print("held-out groups against random subsets of the same number of training neurons (obs / model of the totals):")
        print(rarity.to_string(index=False))
        if len(branches):
            print("branch level (observed | fresh neuron effects | posterior neuron effects; sw = spine-weighted density, "
                  "* = observed outside the 5-95 % band of the posterior simulations):")
            print(_branch_table(branches).to_string(index=False))
    print(f"[calibrate-p0] -> {out}", flush=True)


if __name__ == "__main__":
    main()
