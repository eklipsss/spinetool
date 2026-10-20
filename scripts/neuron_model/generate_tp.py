"""Generate branch trunks (T0) with spine placements (P0): the T + P part of the branch generator.

    python scripts/neuron_model/generate_tp.py --t0-run runs/training/neuron-model/<id>_t0 --p0-run runs/training/neuron-model/<id>_p0 \
        --n 1000 --seed 1 [--mode generated|real_t] [--split val]

The requests (conditions, length, is_terminal, distance to the soma) are copied from random ``--split`` branches, so
the generated set has the condition mix of real data. ``--mode real_t`` keeps the real trunk of each request and
generates only P (the 'real T + generated P' level of the evaluation, p-module-technical-spec.md section 8).
Output (one folder, a few files - no file per branch): ``runs/analysis/neuron-model/tp_generation/<id>/``
``trunks.npz`` (flat node arrays + offsets), ``branches.parquet`` (one row per branch: request, checks, N,
attempts), ``placements.parquet`` (one row per spine: s, u, theta, anchor, normal), ``generation_info.json``. CPU.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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

import numpy as np
import pandas as pd

from src.neuron_model.branch_data import _cond, branch_key, prepare_training_data, select, trunk_record
from src.neuron_model.placement_generation.fit import load_p0_bundle
from src.neuron_model.placement_generation.generate import covariates_from_trunk, generate_placements, pack_records
from src.neuron_model.placement_generation.models.count import build_count_data
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run
from src.neuron_model.trunk_generation.constraints import TrunkLimits
from src.neuron_model.trunk_generation.generate import generate_trunks
from src.neuron_model.trunk_generation.models.t0_statistical import T0Model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t0-run", type=Path)
    parser.add_argument("--t0-model", default=None, help="parabola / gam (default: the one chosen in report.json)")
    parser.add_argument("--p0-run", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["generated", "real_t"], default="generated")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs/analysis/neuron-model/tp_generation")
    args = parser.parse_args()
    if args.mode == "generated" and args.t0_run is None:
        parser.error("--t0-run is required for --mode generated")
    p_run = open_run(args.p0_run)
    d = p_run.load_config()["data"]
    bundle = load_p0_bundle(p_run.root / "p0_bundle.json")
    table, _, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    eligible = "train_eligible_T" if args.mode == "generated" else "train_eligible_P"
    pool = select(table, eligible=eligible, splits=[args.split], exclude_compartments=d.get("exclude_compartments"))
    pool = pool[[branch_key(r) in pack for r in pool.to_dict("records")]].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    picked = pool.iloc[rng.integers(0, len(pool), args.n)].reset_index(drop=True)
    requests = [dict(_cond(r), length_nm=float(r["length_nm"]), is_terminal=bool(r["is_terminal"]),
                     step_nm=None if pd.isna(r.get("step_nm")) else float(r["step_nm"]),
                     soma_distance_start_nm=None if pd.isna(r.get("soma_distance_start_nm")) else float(r["soma_distance_start_nm"]), source_branch=branch_key(r))
                for r in picked.to_dict("records")]

    info = {"mode": args.mode, "seed": args.seed, "n": args.n, "split": args.split, "p0_run": str(args.p0_run), "t0_run": str(args.t0_run) if args.t0_run else None}
    if args.mode == "generated":
        t_run = open_run(args.t0_run)
        chosen = args.t0_model or json.loads((t_run.root / "report.json").read_text(encoding="utf-8"))["chosen"]
        model = T0Model.load(t_run.root / f"t0_{chosen}.json")
        limits = TrunkLimits(**json.loads((t_run.root / "limits.json").read_text(encoding="utf-8")))
        t_req = [{k: q[k] for k in ("species", "cell_type_binary", "compartment", "length_nm")} for q in requests]
        trunks, t_summary = generate_trunks(model, t_req, limits, seed=args.seed, model_id=f"t0-{chosen}")
        info.update(t0_model=chosen, trunk_summary=t_summary)
    else:
        trunks = [trunk_record(pack.get(q["source_branch"])) for q in requests]
    keep = [i for i, t in enumerate(trunks) if t is not None]
    trunks = [trunks[i] for i in keep]
    requests = [requests[i] for i in keep]
    # one neuron per generated branch (independent neuron effects); real_t: the real neuron of the request
    neurons = [q["source_branch"].split("/")[1] if args.mode == "real_t" else f"gen{i}" for i, q in enumerate(requests)]
    cov = [covariates_from_trunk(t, q, n) for t, q, n in zip(trunks, requests, neurons)]
    data = build_count_data(bundle["design"], cov, bundle["count_model"].exposure)
    counts = bundle["count_model"].sample(data, rng)
    placements = generate_placements(trunks, counts, bundle["position_model"], d_min_nm=bundle["d_min_nm"],
                                     soma_start_nm=[q["soma_distance_start_nm"] for q in requests], seed=args.seed)

    out = Path(args.out_root) / f"{_dt.datetime.now():%Y-%m-%d_%H%M%S}_{args.mode}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trunks.npz", **pack_records(trunks))
    brows, prows = [], []
    for i, (t, q, p) in enumerate(zip(trunks, requests, placements)):
        brows.append({"branch_index": i, **{k: q[k] for k in ("species", "cell_type_binary", "compartment", "is_terminal", "soma_distance_start_nm", "source_branch")},
                      "length_nm": float(t["length"]), "n_spines": p["n_spines"], "placement_attempts": p["attempts"], "d_min_satisfied": p["d_min_satisfied"],
                      "min_distance_nm": p["min_distance_nm"], "trunk_attempts": t.get("attempts")})
        for j in range(p["n_spines"]):
            prows.append({"branch_index": i, "spine_index": j, "s": p["s"][j], "u": p["u"][j], "theta": p["theta"][j],
                          "anchor_x": p["anchor"][j, 0], "anchor_y": p["anchor"][j, 1], "anchor_z": p["anchor"][j, 2],
                          "normal_x": p["normal"][j, 0], "normal_y": p["normal"][j, 1], "normal_z": p["normal"][j, 2]})
    pd.DataFrame(brows).to_parquet(out / "branches.parquet", index=False)
    pd.DataFrame(prows).to_parquet(out / "placements.parquet", index=False)
    info.update(n_branches=len(trunks), n_spines=int(sum(p["n_spines"] for p in placements)), d_min_nm=bundle["d_min_nm"],
                d_min_unsatisfied=int(sum(not p["d_min_satisfied"] for p in placements)))
    (out / "generation_info.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    print(f"[generate-tp] {len(trunks)} branches, {info['n_spines']} spines -> {out}", flush=True)


if __name__ == "__main__":
    main()
