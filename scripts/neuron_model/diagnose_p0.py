"""Diagnostics of the spine placement against a fitted P0 (p-module-technical-spec.md, section 8).

    python scripts/neuron_model/diagnose_p0.py --p0-run runs/training/neuron-model/<id>_p0 [--split val] [--n-sim 99]

Uses the branches / pack of the run's config. Writes ``<run>/diagnostics_<split>.json`` and ``.png``: K(h) and G(h)
with envelopes under the fitted P0-position (at the observed N of every branch), circular tests of theta (all spines
and per compartment), angle differences of longitudinal neighbours, dispersion index of the counts. Reading: K / G
inside the envelopes -> no evidence for P1a/P1b/P2; a deficit of short intervals -> repulsion (P1a/P1b); an excess
-> clustering (P2). CPU, minutes.
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

import numpy as np

from src.neuron_model.branch_data import key_series, position_branches, prepare_training_data, select
from src.neuron_model.placement_generation import diagnostics as dg
from src.neuron_model.placement_generation.fit import dispersion_index, load_p0_bundle
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p0-run", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-sim", type=int, default=99)
    parser.add_argument("--h-max-nm", type=float, default=5000.0)
    args = parser.parse_args()
    run = open_run(args.p0_run)
    cfg = run.load_config()
    d = cfg["data"]
    bundle = load_p0_bundle(run.root / "p0_bundle.json")
    table, spines, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    rows = select(table, eligible="train_eligible_P", splits=[args.split], exclude_compartments=d.get("exclude_compartments"))
    branches = position_branches(rows, spines, pack)
    h = np.linspace(50.0, args.h_max_nm, 40)
    env = dg.envelopes(bundle["position_model"], branches, h, n_sim=args.n_sim, seed=int(cfg["seed"]))
    sp = spines[key_series(spines).isin(set(key_series(rows)))] if len(spines) else spines
    sp = sp[(sp["status"].astype(str) == "valid") & sp["has_position"].fillna(False).astype(bool)]
    comp = rows.set_index(key_series(rows))["compartment"].astype(str)
    sp_comp = key_series(sp).map(comp)
    circ = {"all": dg.circular_summary(sp["theta"].to_numpy(dtype=float))}
    for c in sorted(set(sp_comp.dropna())):
        circ[c] = dg.circular_summary(sp.loc[(sp_comp == c).to_numpy(), "theta"].to_numpy(dtype=float))
    groups = [(g["theta"].to_numpy(dtype=float), g["s"].to_numpy(dtype=float)) for _, g in sp.groupby(key_series(sp))]
    diffs = dg.nearest_angle_differences([t for t, _ in groups], [s for _, s in groups])
    out = {"split": args.split, "longitudinal": env, "angles": circ, "neighbour_angle_differences": dg.circular_summary(diffs),
           "dispersion_index": dispersion_index(rows["n_spines_in_range"].to_numpy(dtype=float), rows["length_nm"].to_numpy(dtype=float))}
    path = run.root / f"diagnostics_{args.split}.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        hh = np.array(env["h_nm"])
        ax[0].fill_between(hh, np.array(env["K_lo"]) - 2 * hh, np.array(env["K_hi"]) - 2 * hh, alpha=0.3, label="P0 envelope")
        ax[0].plot(hh, np.array(env["K_obs"]) - 2 * hh, "k", label="data")
        ax[0].set(xlabel="h, nm", ylabel="K(h) - 2h, nm", title="K function"); ax[0].legend()
        ax[1].fill_between(hh, env["G_lo"], env["G_hi"], alpha=0.3); ax[1].plot(hh, env["G_obs"], "k")
        ax[1].set(xlabel="h, nm", ylabel="G(h)", title="nearest-neighbour interval")
        ax[2].hist(sp["theta"].to_numpy(dtype=float), bins=36, density=True, alpha=0.7); ax[2].axhline(1 / (2 * np.pi), color="k")
        ax[2].set(xlabel="theta, rad", title="angle (0 = bending direction)")
        fig.tight_layout(); fig.savefig(path.with_suffix(".png"), dpi=120); plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"[diagnose-p0] no plot: {exc!r}")
    a = circ["all"]
    print(f"[diagnose-p0] {env['n_branches']} branches: K outside envelope {env['K_outside_fraction']:.0%}, G outside {env['G_outside_fraction']:.0%}, "
          f"short-interval deficit {env['short_interval_deficit']}; theta: Rayleigh p {a.get('rayleigh', {}).get('p')}, Kuiper p {a.get('kuiper', {}).get('p')} -> {path}", flush=True)


if __name__ == "__main__":
    main()
