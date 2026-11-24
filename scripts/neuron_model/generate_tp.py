"""Generate branch trunks (T0) with spine placements (P0): the T + P part of the branch generator.

    python scripts/neuron_model/generate_tp.py --t0-run runs/training/neuron-model/<id>_t0 --p0-run runs/training/neuron-model/<id>_p0 \
        --n 1000 --seed 1 [--mode generated|real_t] [--split val]

The requests (conditions, length, is_terminal, distance to the soma) are copied from random ``--split`` branches, so
the generated set has the condition mix of real data. ``--mode real_t`` keeps the real trunk of each request and
generates only P (the 'real T + generated P' level of the evaluation, p-module-technical-spec.md section 8).
``--counts`` separates the count model from the position model (diagnostic): ``model`` (default) samples N from the
P0 count model; ``real`` takes N = ``n_spines_in_range`` of the source branch (the count the evaluation compares
with); ``real_positioned`` takes N = ``n_with_position`` (the spines that have a position, the ones the real
nearest-neighbour distances are computed on and the ones the P1 Monte Carlo check of fit_p1.py conditions on).
With a real count the pool is restricted to ``train_eligible_P`` branches (elsewhere the count is not reliable).
``model_neuron`` (2026-11-11) samples N from the count model with the effect of each source neuron drawn from its
posterior given the observed counts of ALL its ``train_eligible_P`` branches of ``--split`` (``CountModel.
posterior_neuron_effects``): the ladder model -> model_neuron -> real separates the neuron level (the held-out neurons
are richer / poorer than the population) from the branch level (covariate shape, dispersion) of the count model.
``--t0-neuron`` (2026-11-24, T0 with ``parabola_mix_neuron``): the neuron effect of the trunk radius profile is
``independent`` (a fresh draw per branch, the marginal of the model without it; default), ``per_neuron`` (one fresh draw
per source neuron, shared by its requests: a NEW neuron with the conditions of the real one) or ``posterior`` (a draw
from the posterior of the source neuron given its OTHER ``train_eligible_T`` branches of ``--split``, the source branch
itself left out: the diagnostic counterpart of ``--counts model_neuron`` for the trunk).
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

from src.neuron_model.branch_data import _cond, branch_key, count_covariates, prepare_training_data, select, t0_training_branches, trunk_record
from src.neuron_model.placement_generation.fit import load_p0_bundle
from src.neuron_model.placement_generation.generate import covariates_from_trunk, generate_placements, pack_records
from src.neuron_model.placement_generation.models.count import build_count_data
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run
from src.neuron_model.trunk_generation.constraints import TrunkLimits
from src.neuron_model.trunk_generation.generate import generate_trunks
from src.neuron_model.trunk_generation.models.t0_statistical import T0Model


REAL_COUNT_COLUMNS = {"real": "n_spines_in_range", "real_positioned": "n_with_position"}


def real_counts(rows: pd.DataFrame, counts: str) -> np.ndarray:
    """N of each source branch for ``--counts real|real_positioned`` (rows from a ``train_eligible_P`` pool)."""
    values = pd.to_numeric(rows[REAL_COUNT_COLUMNS[counts]], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum())} source branches without {REAL_COUNT_COLUMNS[counts]}")
    return values.to_numpy(dtype=np.int64)


def posterior_counts(model, design, cov, observed_cov, rng: np.random.Generator):
    """``--counts model_neuron``: N of each request from the count model, the effect of its neuron drawn from the
    posterior given the observed counts ``observed_cov`` of that neuron (fresh from the prior when the neuron has no
    observed branch). Returns (counts, summary)."""
    data = build_count_data(design, cov, model.exposure)
    obs_data = build_count_data(design, observed_cov, model.exposure)
    observed_names = pd.unique(np.array([b.neuron for b in observed_cov], dtype=object))  # = neuron index order of obs_data
    effect_of = dict(zip(observed_names, model.posterior_neuron_effects(obs_data, rng))) if len(observed_cov) else {}
    order = pd.unique(np.array([b.neuron for b in cov], dtype=object))  # = neuron index order of data
    sig = model.neuron_sigmas(data)
    effects = np.array([effect_of[n] if n in effect_of else rng.normal(0.0, sig[i]) for i, n in enumerate(order)])
    n_fresh = int(sum(n not in effect_of for n in order))
    summary = {"n_posterior": len(order) - n_fresh, "n_fresh": n_fresh,
               "mean": float(np.mean(effects)) if len(effects) else float("nan"), "sd": float(np.std(effects)) if len(effects) else float("nan")}
    return model.sample(data, rng, neuron_effects=effects), summary


def neuron_effects_for_requests(model, requests, mode: str, table: pd.DataFrame, pack, split: str, exclude_compartments, rng: np.random.Generator):
    """``--t0-neuron per_neuron|posterior``: the T0 neuron effect of every request (aligned list) + a summary."""
    names = [q["source_branch"].split("/")[1] for q in requests]
    if mode == "per_neuron":
        fresh = {n: model.fresh_neuron_effect(rng) for n in pd.unique(np.array(names, dtype=object))}
        effects = [fresh[n] for n in names]
        n_post, n_others = 0, []
    else:
        obs = select(table, eligible="train_eligible_T", splits=[split], exclude_compartments=exclude_compartments)
        obs = obs[obs["neuron_id"].astype(str).isin(set(names))]
        obs = obs[[branch_key(r) in pack for r in obs.to_dict("records")]].reset_index(drop=True)
        keys = [branch_key(r) for r in obs.to_dict("records")]
        branches = t0_training_branches(obs, pack, model.encoder)  # one per row (all rows are in the pack)
        by_neuron = {}
        for key, b in zip(keys, branches):
            by_neuron.setdefault(b.neuron, []).append((key, b))
        effects, n_others = [], []
        for q, n in zip(requests, names):
            others = [b for key, b in by_neuron.get(n, []) if key != q["source_branch"]]  # the source branch itself left out
            n_others.append(len(others))
            effects.append(model.posterior_neuron_effect(others, rng))
        n_post = int(sum(k > 0 for k in n_others))
    level = np.array([e[0] for e in effects])
    return {"effects": [np.asarray(e).tolist() for e in effects], "n_requests": len(requests), "n_neurons": int(len(set(names))),
            "n_with_other_branches": n_post, "median_other_branches": float(np.median(n_others)) if len(n_others) else 0.0,
            "level_mean": float(level.mean()), "level_sd": float(level.std()),
            "prior_sd_level": float(np.sqrt(model.mean.neuron_cov[0, 0]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t0-run", type=Path)
    parser.add_argument("--t0-model", default=None, help="parabola / gam (default: the one chosen in report.json)")
    parser.add_argument("--p0-run", type=Path, required=True, help="a P0 run (fit_p0.py) or a P1 run (fit_p1.py)")
    parser.add_argument("--bundle", default="p0_bundle.json", help="bundle file inside --p0-run (P1 runs: p1a_bundle.json / p1b_bundle.json; "
                        "p0_bundle.json of a P1 run = the chosen one)")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["generated", "real_t"], default="generated")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset", default=None, help="only source branches of this dataset kind (minnie65 / h01) when P0 was fitted on several")
    parser.add_argument("--counts", choices=list(REAL_COUNT_COLUMNS) + ["model", "model_neuron"], default="model",
                        help="N per branch: from the P0 count model, or the real count of the source branch (diagnostic, see above)")
    parser.add_argument("--t0-neuron", choices=["independent", "per_neuron", "posterior"], default="independent",
                        help="neuron effect of the T0 radius profile (needs a T0 variant with it, e.g. --t0-model parabola_mix_neuron)")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs/analysis/neuron-model/tp_generation")
    args = parser.parse_args()
    if args.mode == "generated" and args.t0_run is None:
        parser.error("--t0-run is required for --mode generated")
    p_run = open_run(args.p0_run)
    d = p_run.load_config()["data"]
    bundle = load_p0_bundle(p_run.root / args.bundle)
    table, _, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    eligible = "train_eligible_T" if args.mode == "generated" else "train_eligible_P"
    pool = select(table, eligible=eligible, splits=[args.split], exclude_compartments=d.get("exclude_compartments"))
    if args.dataset:
        from src.neuron_model.metadata import dataset_kind

        pool = pool[pool["dataset"].astype(str).map(dataset_kind) == dataset_kind(args.dataset)]
    if args.counts in REAL_COUNT_COLUMNS and eligible != "train_eligible_P":
        pool = pool[pool["train_eligible_P"].fillna(False).astype(bool)]
    pool = pool[[branch_key(r) in pack for r in pool.to_dict("records")]].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    picked = pool.iloc[rng.integers(0, len(pool), args.n)].reset_index(drop=True)
    requests = [dict(_cond(r), length_nm=float(r["length_nm"]), is_terminal=bool(r["is_terminal"]),
                     step_nm=None if pd.isna(r.get("step_nm")) else float(r["step_nm"]),
                     soma_distance_start_nm=None if pd.isna(r.get("soma_distance_start_nm")) else float(r["soma_distance_start_nm"]), source_branch=branch_key(r))
                for r in picked.to_dict("records")]

    info = {"mode": args.mode, "seed": args.seed, "n": args.n, "split": args.split, "p0_run": str(args.p0_run), "bundle": args.bundle,
            "counts": args.counts, "dataset": args.dataset, "t0_run": str(args.t0_run) if args.t0_run else None}
    if args.mode == "generated":
        t_run = open_run(args.t0_run)
        chosen = args.t0_model or json.loads((t_run.root / "report.json").read_text(encoding="utf-8"))["chosen"]
        model = T0Model.load(t_run.root / f"t0_{chosen}.json")
        limits = TrunkLimits(**json.loads((t_run.root / "limits.json").read_text(encoding="utf-8")))
        t_req = [{k: q[k] for k in ("species", "cell_type_binary", "compartment", "length_nm", "step_nm", "soma_distance_start_nm", "is_terminal")} for q in requests]
        if args.t0_neuron != "independent":
            if not model.has_neuron_effect:
                parser.error(f"--t0-neuron {args.t0_neuron}: T0 variant {chosen} has no neuron effect (use --t0-model parabola_mix_neuron)")
            ne_info = neuron_effects_for_requests(model, requests, args.t0_neuron, table, pack, args.split, d.get("exclude_compartments"),
                                                  np.random.default_rng(args.seed + 101))
            for q, e in zip(t_req, ne_info.pop("effects")):
                q["t0_neuron_effect"] = e
            info.update(t0_neuron=args.t0_neuron, t0_neuron_effects=ne_info)
            print(f"[generate-tp] T0 neuron effects ({args.t0_neuron}): {ne_info}", flush=True)
        trunks, t_summary = generate_trunks(model, t_req, limits, seed=args.seed, model_id=f"t0-{chosen}")
        info.update(t0_model=chosen, trunk_summary=t_summary)
    else:
        trunks = [trunk_record(pack.get(q["source_branch"])) for q in requests]
    keep = [i for i, t in enumerate(trunks) if t is not None]
    trunks = [trunks[i] for i in keep]
    requests = [requests[i] for i in keep]
    if args.counts == "model":
        # one neuron per generated branch (independent neuron effects); real_t: the real neuron of the request
        neurons = [q["source_branch"].split("/")[1] if args.mode == "real_t" else f"gen{i}" for i, q in enumerate(requests)]
        cov = [covariates_from_trunk(t, q, n) for t, q, n in zip(trunks, requests, neurons)]
        data = build_count_data(bundle["design"], cov, bundle["count_model"].exposure)
        counts = bundle["count_model"].sample(data, rng)
    elif args.counts == "model_neuron":
        names = [q["source_branch"].split("/")[1] for q in requests]
        cov = [covariates_from_trunk(t, q, n) for t, q, n in zip(trunks, requests, names)]
        observed = select(table, eligible="train_eligible_P", splits=[args.split], exclude_compartments=d.get("exclude_compartments"))
        observed = observed[observed["neuron_id"].astype(str).isin(set(names))]
        counts, effects_info = posterior_counts(bundle["count_model"], bundle["design"], cov, count_covariates(observed, pack), rng)
        info.update(neuron_effects=effects_info)
        print(f"[generate-tp] neuron effects from the posterior: {effects_info['n_posterior']} neurons (fresh for {effects_info['n_fresh']}), "
              f"mean {effects_info['mean']:+.3f}, sd {effects_info['sd']:.3f}", flush=True)
    else:
        counts = real_counts(picked.iloc[keep], args.counts)
    placements = generate_placements(trunks, counts, bundle["position_model"], d_min_nm=bundle["d_min_nm"],
                                     soma_start_nm=[q["soma_distance_start_nm"] for q in requests], seed=args.seed)

    suffix = ("" if args.counts == "model" else f"_counts-{args.counts}") + ("" if args.t0_neuron == "independent" else f"_t0n-{args.t0_neuron}")
    out = Path(args.out_root) / f"{_dt.datetime.now():%Y-%m-%d_%H%M%S}_{args.mode}{suffix}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trunks.npz", **pack_records(trunks))
    brows, prows = [], []
    for i, (t, q, p) in enumerate(zip(trunks, requests, placements)):
        brows.append({"branch_index": i, **{k: q.get(k) for k in ("species", "cell_type_binary", "cell_type", "compartment", "is_terminal", "soma_distance_start_nm", "source_branch")},
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
