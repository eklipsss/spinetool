"""Cross-validation of P0 (spine count) and T0 (trunk) by NEURON on the train + val neurons of a split (2026-11-14).

    python scripts/neuron_model/cv_tp_neurons.py --branches-root C:/.../preprocessed_branches \
        --splits data/processed/neuron-model/splits/splits_v1.parquet [--k 5] [--repeats 3] [--models p0 t0]

Why: the 8 validation inhibitory neurons of ``splits_v1`` are much poorer in spines than the training ones (P0 obs /
model 0.44, rarity 0.001), and one fixed validation set of 8 neurons gives neither an interval nor an answer to "is
it those neurons or the split?". Every train + val neuron is held out exactly once per repetition (K folds,
stratified by ``cell_type_binary``); the TEST neurons of the split are never read (``src/neuron_model/neuron_cv.py``).

P0 (CPU, ~1 min per fit): the count model variants of ``--p0-variants`` (family:exposure:re|nore:sg1|sg2, the first one
is the primary) are fitted on K-1 folds and scored on the held-out fold (NLL per branch); the primary one is also
calibrated per held-out neuron (``calibration.neuron_calibration``). Outputs:
- ``p0_folds.csv``: held-out NLL per variant, repetition, fold; obs / model per fold and cell type;
- ``p0_neurons.csv``: every held-out neuron (repetition, fold, ``v1_split`` = its part of the split, observed vs model);
- ``p0_groups.csv`` / ``p0_groups_cell_type.csv``: pooled over the held-out folds (every neuron once per repetition);
- ``p0_v1val_vs_rest.csv``: the validation neurons of the split against the other held-out neurons of their cell
  type (both held out, so a fair comparison): ratio of totals, rarity, difference of the neuron effect, permutation p.
T0 (CPU, ~10-20 min per fold, first repetition only): the variant ``--t0-mean`` fitted on K-1 folds, generation for
the held-out fold, functionals W1/std and median differences with neuron-bootstrap intervals (``trunk_generation.fit
.fit_t0``) -> ``t0_folds.csv``: does the fold of the original validation neurons look like the others?
Always: ``neuron_descriptors.csv`` and ``descriptors_v1val_vs_rest.csv`` (what is different about the validation
neurons: eligible share, spines without an S verdict, share with a position, radius, density).
Writes ``runs/analysis/neuron-model/<date>_tp-neuron-cv/`` (the last line prints the folder).
"""

from __future__ import annotations

import argparse
import datetime as _dt
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

import torch  # noqa: F401,E402  (first on Windows, see generate_tp.py)

import numpy as np
import pandas as pd

from src.neuron_model.branch_data import _cond, branch_key, count_covariates, load_branches, prepare_training_data, select, t0_training_branches
from src.neuron_model.neuron_cv import compare_descriptors, flagged_vs_rest, neuron_descriptors, neuron_folds
from src.neuron_model.placement_generation.calibration import group_calibration, neuron_calibration
from src.neuron_model.placement_generation.covariates import CountDesign
from src.neuron_model.placement_generation.models.count import CountModel, build_count_data
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path

DESCRIPTOR_COLUMNS = ["status", "split", "cell_type", "cell_type_binary", "compartment", "length_nm", "n_spines", "n_spines_in_range", "n_unknown",
                      "n_with_position", "radius_median_nm", "train_eligible_T", "train_eligible_P"]


def parse_variant(text: str) -> dict:
    family, exposure, re, sg = text.split(":")
    if re not in ("re", "nore") or sg not in ("sg1", "sg2"):
        raise ValueError(f"variant {text!r}: family:exposure:re|nore:sg1|sg2")
    return {"family": family, "exposure": exposure, "random_effect": re == "re", "sigma_groups": 2 if (sg == "sg2" and re == "re") else 1, "name": text}


def run_p0(args, cfg, table, pack, folds_by_rep, cell_type, out, log):
    d, m = cfg["data"], cfg["model"]
    rows = select(table, eligible="train_eligible_P", splits=args.cv_splits, exclude_compartments=d.get("exclude_compartments"))
    cov_all = count_covariates(rows, pack)
    neuron_of = np.array([b.neuron for b in cov_all])
    v1 = rows.drop_duplicates("neuron_id").set_index("neuron_id")["split"].astype(str).to_dict()
    variants = [parse_variant(v) for v in args.p0_variants]
    fold_rows, neuron_frames = [], []
    for rep, folds in folds_by_rep.items():
        fold_of = pd.Series(neuron_of).map(folds).to_numpy()
        for f in range(args.k):
            tr = [b for b, x in zip(cov_all, fold_of) if x != f]
            te = [b for b, x in zip(cov_all, fold_of) if x == f]
            design = CountDesign(int(m["n_basis"])).fit(tr)
            for vi, v in enumerate(variants):
                started = time.perf_counter()
                model = CountModel(v["family"], v["exposure"], v["random_effect"], n_basis=int(m["n_basis"]), smooth_penalty=float(m["smooth_penalty"]),
                                   sigma_groups=v["sigma_groups"]).fit(build_count_data(design, tr, v["exposure"]), max_iter=int(m["max_iter"]))
                nll = model.neg_loglik(build_count_data(design, te, v["exposure"])) / max(len(te), 1)
                sig = model.fit_info.get("sigma_b_by_group", {}) if v["random_effect"] else {}
                fold_rows.append({"repeat": rep, "fold": f, "variant": v["name"], "n_train_branches": len(tr), "n_heldout_branches": len(te),
                                  "n_heldout_neurons": len({b.neuron for b in te}), "heldout_nll_per_branch": float(nll),
                                  **{f"sigma_b_{k}": float(s) for k, s in sig.items()}, "seconds": round(time.perf_counter() - started, 1)})
                log(f"[cv-p0] repeat {rep} fold {f} {v['name']:24s} held-out NLL/branch {nll:.4f}  sigma_b {({k: round(s, 3) for k, s in sig.items()})}")
                if vi == 0:
                    nc = neuron_calibration(model, design, te, split=f"r{rep}f{f}", cell_type=cell_type, n_sim=args.n_sim, seed=rep * 100 + f)
                    nc["repeat"], nc["fold"] = rep, f
                    nc["v1_split"] = nc["neuron"].map(v1)
                    neuron_frames.append(nc)
    folds_t = pd.DataFrame(fold_rows)
    neurons = pd.concat(neuron_frames, ignore_index=True)
    neurons["is_v1_val"] = neurons["v1_split"] == "val"
    per_fold = group_calibration(neurons, by=("repeat", "fold", "cell_type_binary"))
    pooled = group_calibration(neurons.assign(split="cv"), by=("repeat", "cell_type_binary"))
    pooled_type = group_calibration(neurons.assign(split="cv"), by=("repeat", "cell_type_binary", "cell_type"))
    by_v1 = group_calibration(neurons, by=("repeat", "v1_split", "cell_type_binary"))
    flagged = flagged_vs_rest(neurons, "is_v1_val", n_perm=args.n_perm)
    folds_t.to_csv(out / "p0_folds.csv", index=False)
    per_fold.to_csv(out / "p0_groups_by_fold.csv", index=False)
    neurons.to_csv(out / "p0_neurons.csv", index=False)
    pooled.to_csv(out / "p0_groups.csv", index=False)
    pooled_type.to_csv(out / "p0_groups_cell_type.csv", index=False)
    by_v1.to_csv(out / "p0_groups_by_v1_split.csv", index=False)
    flagged.to_csv(out / "p0_v1val_vs_rest.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        nll = folds_t.groupby("variant")["heldout_nll_per_branch"].agg(["mean", "std", "count"])
        log("[cv-p0] held-out NLL per branch over all folds and repetitions (mean, sd over folds):\n" + nll.to_string())
        if len(variants) > 1:  # paired: the same folds for every variant
            wide = folds_t.pivot_table(index=["repeat", "fold"], columns="variant", values="heldout_nll_per_branch")
            base = variants[0]["name"]
            for v in variants[1:]:
                dd = (wide[v["name"]] - wide[base]).to_numpy()
                log(f"[cv-p0] {v['name']} - {base}: mean {dd.mean():+.4f}, folds better {int((dd < 0).sum())} of {len(dd)}")
        log("[cv-p0] obs / model per fold and cell type (primary variant):\n"
            + per_fold.pivot_table(index=["repeat", "fold"], columns="cell_type_binary", values="obs_over_model").to_string())
        log("[cv-p0] pooled over the held-out folds:\n" + pooled.drop(columns=["split"], errors="ignore").to_string(index=False))
        log("[cv-p0] pooled, detailed cell types with >= 3 neurons:\n" + pooled_type[pooled_type["n_neurons"] >= 3].drop(columns=["split"], errors="ignore").to_string(index=False))
        log("[cv-p0] by the part of the original split (all held out here):\n" + by_v1.to_string(index=False))
        log("[cv-p0] original validation neurons against the other held-out neurons of their cell type:\n" + flagged.to_string(index=False))


def run_t0(args, table, pack, folds, out, log):
    from src.neuron_model.trunk_generation.conditions import DesignEncoder
    from src.neuron_model.trunk_generation.fit import fit_t0

    cfg = load_config(args.t0_config, [f"data.branches_root={args.branches_root}"] + args.t0_overrides)
    d, m = cfg["data"], cfg["model"]
    rows = select(table, eligible="train_eligible_T", splits=args.cv_splits, exclude_compartments=d.get("exclude_compartments"))
    rows = rows[[branch_key(r) in pack for r in rows.to_dict("records")]].reset_index(drop=True)
    rows["fold"] = rows["neuron_id"].astype(str).map(folds)
    out_rows = []
    for f in range(args.k):
        started = time.perf_counter()
        tr = rows[rows["fold"] != f].reset_index(drop=True)
        va = rows[rows["fold"] == f].reset_index(drop=True)
        if len(va) > int(d["max_val_branches"]):
            va = va.sample(int(d["max_val_branches"]), random_state=int(cfg["seed"])).reset_index(drop=True)
        soma_mid = (tr["soma_distance_start_nm"].astype(float) + 0.5 * tr["length_nm"].astype(float)).tolist() if bool(m.get("use_soma_distance", False)) else None
        encoder = DesignEncoder.fit(tr["length_nm"].astype(float).tolist(), soma_mid)
        train, val = t0_training_branches(tr, pack, encoder), t0_training_branches(va, pack, encoder)
        recs = va.to_dict("records")
        log(f"[cv-t0] fold {f}: train {len(train)} branches, held out {len(val)} ({va['neuron_id'].nunique()} neurons, of them original val "
            f"{va.loc[va['split'] == 'val', 'neuron_id'].nunique()})")
        res = fit_t0(train, val, [_cond(r) for r in recs], [str(r["neuron_id"]) for r in recs], encoder=encoder, mean_models=[args.t0_mean],
                     shrinkage_std=float(m["shrinkage_std"]), max_iter=int(m["max_iter"]), length_bounds=tuple(m["length_bounds_nm"]), seed=int(cfg["seed"]),
                     direction=str(m.get("direction", "multiscale")), radius_residual=str(m.get("radius_residual", "multiscale")), train_conditions=None,
                     log=lambda *a, **k: None)
        rep = res["report"]
        var = rep["variants"][args.t0_mean]
        row = {"fold": f, "n_train": len(train), "n_heldout": len(val), "n_heldout_neurons": int(va["neuron_id"].nunique()),
               "n_v1_val_neurons": int(va.loc[va["split"] == "val", "neuron_id"].nunique()),
               "acceptance": var["generation"].get("acceptance_rate"), "mean_w1_rel": var["mean_w1_rel"],
               "noise_floor_mean_w1_rel": rep["noise_floor_mean_w1_rel"], "real_rejection_rate": rep["real_val_rejection_rate"],
               "seconds": round(time.perf_counter() - started, 1)}
        for fn, mm in var["functionals"].items():
            ci = mm.get("ci", {})
            row.update({f"{fn}_median_gen": mm["median_a"], f"{fn}_median_real": mm["median_b"], f"{fn}_w1_rel": mm["w1_rel"],
                        f"{fn}_floor": rep["noise_floor"][fn]["w1_rel"], f"{fn}_diff": ci.get("diff"), f"{fn}_lo": ci.get("lo"), f"{fn}_hi": ci.get("hi"),
                        f"{fn}_significant": ci.get("significant")})
        out_rows.append(row)
        sig = [fn for fn, mm in var["functionals"].items() if mm.get("ci", {}).get("significant")]
        log(f"[cv-t0] fold {f}: acceptance {row['acceptance']:.3f}, mean W1/std {row['mean_w1_rel']:.3f} (floor {row['noise_floor_mean_w1_rel']:.3f}), "
            f"significant: {sig or 'none'} ({row['seconds']:.0f} s)")
        pd.DataFrame(out_rows).to_csv(out / "t0_folds.csv", index=False)  # after every fold: a crash keeps the finished ones
    t = pd.DataFrame(out_rows)
    with pd.option_context("display.width", 240, "display.max_columns", 60, "display.float_format", "{:.4g}".format):
        cols = ["fold", "n_heldout", "n_heldout_neurons", "n_v1_val_neurons", "acceptance", "mean_w1_rel", "noise_floor_mean_w1_rel"]
        log("[cv-t0] per fold:\n" + t[cols].to_string(index=False))
        fns = [c[: -len("_median_gen")] for c in t.columns if c.endswith("_median_gen")]
        summary = pd.DataFrame([{"functional": fn, "w1_rel_mean": t[f"{fn}_w1_rel"].mean(), "w1_rel_min": t[f"{fn}_w1_rel"].min(), "w1_rel_max": t[f"{fn}_w1_rel"].max(),
                                 "floor_mean": t[f"{fn}_floor"].mean(), "diff_mean": t[f"{fn}_diff"].astype(float).mean(),
                                 "folds_significant": int(t[f"{fn}_significant"].fillna(False).astype(bool).sum())} for fn in fns])
        summary.to_csv(out / "t0_summary.csv", index=False)
        log("[cv-t0] functionals over the folds:\n" + summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True, help="splits_*.parquet; only the --cv-splits neurons are used (the test part is never read)")
    parser.add_argument("--cv-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--models", nargs="+", default=["p0", "t0"], choices=["p0", "t0"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3, help="P0: repetitions with different fold assignments (T0 uses the first only)")
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--p0-config", type=Path, default=REPO_ROOT / "configs/neuron-model/placement/p0.yaml")
    parser.add_argument("--p0-variants", nargs="+", default=["zinb:length:re:sg2", "zinb:length:re:sg1"],
                        help="family:exposure:re|nore:sg1|sg2; the first is calibrated per neuron (default: the model chosen in chain V, and one sigma_b)")
    parser.add_argument("--n-sim", type=int, default=200)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--t0-config", type=Path, default=REPO_ROOT / "configs/neuron-model/trunk/t0.yaml")
    parser.add_argument("--t0-mean", default="parabola_mix")
    parser.add_argument("--t0-overrides", nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{_dt.datetime.now():%Y-%m-%d_%H%M%S}_tp-neuron-cv"
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "log.txt").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    cfg = load_config(args.p0_config, [f"data.branches_root={args.branches_root}", f"data.splits={args.splits}"])
    d = cfg["data"]
    table, _, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]), io_threads=int(d["io_threads"]))
    if "split" not in table.columns or table["split"].isna().all():
        raise SystemExit("no split in the branch table: check --splits")

    # descriptors over ALL branches of the train + val neurons (test neurons are not looked at)
    desc_table = load_branches(resolve_path(d["branches_root"]), DESCRIPTOR_COLUMNS)
    desc_table["split"] = desc_table["neuron_id"].map(table.drop_duplicates("neuron_id").set_index("neuron_id")["split"])
    desc_table = desc_table[desc_table["split"].astype(str).isin(set(args.cv_splits))]
    cell_type = desc_table.drop_duplicates("neuron_id").set_index("neuron_id")["cell_type"].astype(str).to_dict() if "cell_type" in desc_table.columns else None
    desc = neuron_descriptors(desc_table)
    desc["v1_split"] = desc["neuron_id"].map(desc_table.drop_duplicates("neuron_id").set_index("neuron_id")["split"].astype(str))
    desc["is_v1_val"] = desc["v1_split"] == "val"
    desc.to_csv(out / "neuron_descriptors.csv", index=False)
    cmp_desc = compare_descriptors(desc, "is_v1_val")
    cmp_desc.to_csv(out / "descriptors_v1val_vs_rest.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        log("[cv] neuron descriptors, original validation neurons against the other train + val neurons of their cell type "
            "(medians, rank-sum p, descriptive):\n" + cmp_desc.to_string(index=False))

    # folds: stratified by cell_type_binary over the neurons that have P- or T-eligible branches
    neurons = table[table["split"].astype(str).isin(set(args.cv_splits))][["neuron_id", "cell_type_binary", "split"]].drop_duplicates("neuron_id")
    folds_by_rep = {r: neuron_folds(neurons, args.k, seed=args.seed + r, stratum="cell_type_binary") for r in range(args.repeats)}
    pd.concat([f.rename(f"fold_repeat_{r}") for r, f in folds_by_rep.items()], axis=1).rename_axis("neuron_id").reset_index().to_csv(out / "folds.csv", index=False)
    counts = neurons.assign(fold=neurons["neuron_id"].map(folds_by_rep[0])).groupby(["fold", "cell_type_binary"], dropna=False).size().unstack(fill_value=0)
    log(f"[cv] {len(neurons)} neurons ({', '.join(args.cv_splits)} of {args.splits.name}), {args.k} folds x {args.repeats} repetitions; neurons per fold (repetition 0):\n{counts}")
    (out / "run_info.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "n_neurons": len(neurons),
                                                   "pack_dir": str(d["pack_dir"]), "exclude_compartments": d.get("exclude_compartments")}, indent=2), encoding="utf-8")
    if "p0" in args.models:
        run_p0(args, cfg, table, pack, folds_by_rep, cell_type, out, log)
    if "t0" in args.models:
        run_t0(args, table, pack, folds_by_rep[0], out, log)
    log(f"[cv] -> {out}")


if __name__ == "__main__":
    main()
