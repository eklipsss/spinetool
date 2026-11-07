"""Compare VAE decoders on reconstructions of the SAME validation spines (models/spine_vae/decoder_study.py).

For --n-spines random spines of --split (fixed seed, the split of the first run; every run must share split and bbox):
  real      the real local_sealed mesh (base by the same finder as the generated meshes)
  real_mc   exact SDF of the real mesh -> Marching Cubes on the frozen bbox at --resolution (the chain of the VAE; reference)
  <run>     z = mu of point-cloud variant 1 -> decoder -> Marching Cubes on the same grid, postprocessed as the generation
Per run: paired standardised bias of the 11 morphometrics against real_mc (cluster bootstrap over neurons), mean |bias|
and its paired difference from the first run (the baseline), surface error to the real mesh (Chamfer-L1, HD95 in nm),
mesh validity and the real_mc-vs-reconstruction classifier AUC (logistic, as the generator report). Two trainings of the
same recipe passed as two runs give the training-noise floor.

    python scripts/neuron_model/compare_vae_decoders.py ^
        --run base=runs/training/neuron-model/2026-10-19_003408_vae-dev-reg0-cos ^
        --run base_rerun=runs/training/neuron-model/2026-11-04_015937_vae-dev-reg0-cos-curves ^
        --run beta1e-5=<run> --mesh-manifest C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet ^
        --split val --n-spines 300

Writes runs/analysis/neuron-model/<date>_vae-decoder-study/{spines.parquet, per_metric.csv, summary.csv, report.json, summary.md}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":  # torch + MKL OpenMP runtimes (see generate_vae.py)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch


def parse_runs(items):
    runs = []
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--run expects name=path, got {item!r}")
        runs.append((name, Path(path)))
    if len({n for n, _ in runs}) != len(runs):
        raise SystemExit("run names must be unique")
    return runs


def main() -> None:
    from src.neuron_model.spine_generation.data.bbox import FrozenBBox
    from src.neuron_model.spine_generation.data.index import load_spine_index
    from src.neuron_model.spine_generation.evaluation.attachment import finder_from_config
    from src.neuron_model.spine_generation.evaluation.classifier_ablation import repeated_cv
    from src.neuron_model.spine_generation.evaluation.distribution_metrics import JOINT_FEATURES
    from src.neuron_model.spine_generation.evaluation.morphometrics import generated_morphometrics
    from src.neuron_model.spine_generation.evaluation.real_controls import banded_sdf
    from src.neuron_model.spine_generation.experiment.artifacts import write_json
    from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
    from src.neuron_model.spine_generation.models.spine_vae.decoder_study import cluster_mean_ci, optimize_latent, paired_bias, surface_distances
    from src.neuron_model.spine_generation.models.spine_vae.generate import latent_to_mesh
    from src.neuron_model.spine_generation.models.spine_vae.latent_prior import encode_spines, load_split_index, load_vae_run
    from src.neuron_model.spine_generation.reconstruction.marching_cubes import GridSpec, sdf_to_mesh
    from src.neuron_model.spine_generation.reconstruction.mesh_validation import validate_mesh
    from src.neuron_model.spine_generation.reconstruction.postprocess import postprocess_mesh
    from src.neuron_model.spine_geometry import load_trimesh
    from src.neuron_model.spine_morphometrics import SCALAR_METRICS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=run_dir; the FIRST run is the baseline")
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-spines", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=128, help="MC grid on the frozen bbox, as the generation (128 in the report)")
    ap.add_argument("--mesh-manifest", type=Path, help="manifest with the real local_sealed meshes (a training cache has none)")
    ap.add_argument("--mesh-path-mode", default="absolute", choices=("relative", "absolute"))
    ap.add_argument("--bbox", type=Path, help="frozen bbox json (default: data.bbox of the first run)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--optimize-latent", type=int, default=0, metavar="STEPS",
                    help="also fit the code of every spine with the decoder FROZEN (Adam from mu, training SDF loss on the "
                         "spine's sdf_samples.npz): kind <run>_zopt. Bias gone -> the encoder is the bottleneck; bias kept -> the decoder")
    ap.add_argument("--opt-lr", type=float, default=1e-2)
    ap.add_argument("--opt-runs", nargs="*", help="runs to fit codes for (default: the baseline only)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    runs = parse_runs(args.run)
    started = time.time()

    loaded = [(name, *load_vae_run(path, args.checkpoint)) for name, path in runs]  # (name, model, run, cfg, ckpt, device)
    base_cfg = loaded[0][3]
    for name, _, _, cfg, _, _ in loaded[1:]:
        for key in ("splits", "bbox", "coord_scale"):
            if str(cfg["data"].get(key)) != str(base_cfg["data"].get(key)):
                raise SystemExit(f"run {name}: data.{key} differs from the baseline ({cfg['data'].get(key)} vs {base_cfg['data'].get(key)})")
    bbox = FrozenBBox.load(args.bbox or resolve_path(base_cfg["data"]["bbox"]))
    box = (tuple(bbox.low), tuple(bbox.high))
    grid = GridSpec(box[0], box[1], int(args.resolution))
    mc_cfg = load_config(resolve_path(base_cfg["reconstruction"]["marching_cubes_config"]))
    post_cfg, chunk = dict(mc_cfg["postprocess"]), int(mc_cfg.get("chunk_size", 262_144))
    finder = finder_from_config(load_config(REPO_ROOT / "configs/neuron-model/evaluation/default.yaml")["morphometrics"].get("attachment_finder"))

    index = load_split_index(base_cfg, args.split, args.n_spines, args.seed)
    frame = index.frame.set_index("spine_key", drop=False)
    meshes_frame = frame
    if args.mesh_manifest:
        meshes_frame = load_spine_index([args.mesh_manifest], path_mode=args.mesh_path_mode, check_files=()).frame.set_index("spine_key")
    keys = list(index.frame["spine_key"])
    neuron = {k: str(frame.loc[k, "group_id"]) for k in keys}
    print(f"[study] {len(keys)} {args.split} spines of {len(set(neuron.values()))} neurons, MC {args.resolution}^3 on the frozen bbox, "
          f"runs: {', '.join(n for n, _ in runs)}", flush=True)

    out = args.out or resolve_path(Path("runs/analysis/neuron-model") / f"{pd.Timestamp.now():%Y-%m-%d_%H%M%S}_vae-decoder-study")
    out.mkdir(parents=True, exist_ok=True)

    # references, once
    real_meshes, mc_meshes, rows = {}, {}, []
    for i, k in enumerate(keys, 1):
        real = load_trimesh(Path(meshes_frame.loc[k, "local_sealed_mesh_path"]), process=False)
        raw, _, _ = sdf_to_mesh(banded_sdf(real, grid), grid, chunk_size=chunk)
        post = postprocess_mesh(raw, **post_cfg)[0] if raw is not None else None
        real_meshes[k], mc_meshes[k] = real, post
        rows.append({"spine_key": k, "kind": "real_mc", "valid": bool(validate_mesh(post, bbox=box, check_self_intersections=False).get("is_valid")) if post is not None else False,
                     **surface_distances(post, real, seed=i)})
        if i % 50 == 0 or i == len(keys):
            print(f"  real_mc {i}/{len(keys)} ({time.time() - started:.0f} s)", flush=True)
    tables = [generated_morphometrics([real_meshes[k] for k in keys], attachment_finder=finder, compute_chords=False, sample_ids=keys).assign(kind="real"),
              generated_morphometrics([mc_meshes[k] for k in keys], attachment_finder=finder, compute_chords=False, sample_ids=keys).assign(kind="real_mc")]

    opt_runs = set(args.opt_runs) if args.opt_runs else {runs[0][0]}
    names = []
    for name, model, run, cfg, ckpt, device in loaded:
        codes = encode_spines(model, index, cfg["data"], variants=(1,), device=device, batch_size=16)
        mu = {k: codes["mu"][j, 0] for j, k in enumerate(codes["keys"])}
        coord_scale = float(cfg["data"]["coord_scale"])
        variants = [(name, mu, {})]
        if args.optimize_latent > 0 and name in opt_runs:
            clamp = cfg.get("loss", {}).get("sdf_clamp")
            clamp = None if clamp is None else float(clamp)
            far = int(cfg["data"].get("far_queries", 0)) > 0 and float(cfg.get("loss", {}).get("far", 0.0)) > 0
            zopt, info = {}, {}
            for i, k in enumerate(keys, 1):
                zopt[k], info[k] = optimize_latent(model, mu[k], frame.loc[k, "sdf_samples_path"], coord_scale=coord_scale, sdf_clamp=clamp,
                                                   far_bbox=box if far else None, steps=args.optimize_latent, lr=args.opt_lr,
                                                   far_clamp_nm=float(cfg["data"].get("far_clamp_nm", 300.0)), seed=i, device=device)
                if i % 50 == 0 or i == len(keys):
                    print(f"  {name}_zopt fit {i}/{len(keys)}: SDF loss {np.mean([v['loss_start'] for v in info.values()]):.2f} -> "
                          f"{np.mean([v['loss_end'] for v in info.values()]):.2f} nm ({time.time() - started:.0f} s)", flush=True)
            variants.append((f"{name}_zopt", zopt, info))
        for kind, z_by_key, info in variants:
            names.append(kind)
            meshes = []
            for i, k in enumerate(keys, 1):
                res = latent_to_mesh(model, z_by_key[k], bbox=box, resolution=args.resolution, coord_scale=coord_scale, device=device, chunk_size=chunk,
                                     iso_level=float(mc_cfg.get("iso_level", 0.0)), postprocess_cfg=post_cfg, check_self_intersections=False)
                mesh = res["mesh_postprocessed"]
                meshes.append(mesh)
                rows.append({"spine_key": k, "kind": kind, "valid": bool(res["validation_postprocessed"].get("is_valid")),
                             "valid_raw": bool(res["validation_raw"].get("is_valid")), **surface_distances(mesh, real_meshes[k], seed=i),
                             **({"opt_loss_start_nm": info[k]["loss_start"], "opt_loss_end_nm": info[k]["loss_end"]} if info else {})})
                if i % 50 == 0 or i == len(keys):
                    print(f"  {kind} ({ckpt}) {i}/{len(keys)} ({time.time() - started:.0f} s)", flush=True)
            tables.append(generated_morphometrics(meshes, attachment_finder=finder, compute_chords=False, sample_ids=keys).assign(kind=kind))

    morph = pd.concat(tables, ignore_index=True).rename(columns={"sample_id": "spine_key"})
    spines = morph.merge(pd.DataFrame(rows), on=["spine_key", "kind"], how="left")
    spines["neuron_id"] = spines["spine_key"].map(neuron)
    spines.to_parquet(out / "spines.parquet", index=False)

    metrics = [m for m in SCALAR_METRICS if m in spines.columns]
    res = paired_bias(spines, names + ["real"], reference="real_mc", metrics=metrics, baseline=names[0], n_boot=args.n_boot, seed=args.seed)
    res["per_metric"].to_csv(out / "per_metric.csv", index=False)
    summary = res["summary"]
    extra = []
    for kind in ["real_mc"] + names:
        g = spines[spines["kind"] == kind]
        ch = cluster_mean_ci(g["chamfer_l1"], g["neuron_id"], n_boot=args.n_boot, seed=args.seed)
        hd = cluster_mean_ci(g["hd95"], g["neuron_id"], n_boot=args.n_boot, seed=args.seed)
        entry = {"kind": kind, "valid_rate": float(g["valid"].mean()), "chamfer_l1_nm": ch["mean"], "chamfer_ci_low": ch["ci_low"],
                 "chamfer_ci_high": ch["ci_high"], "hd95_nm": hd["mean"]}
        if kind != "real_mc":
            ref = spines[spines["kind"] == "real_mc"]
            feats = [f for f in JOINT_FEATURES if f in spines.columns]
            x_ref, x_gen = ref[feats].to_numpy(float), g[feats].to_numpy(float)
            x_ref, x_gen = x_ref[np.isfinite(x_ref).all(1)], x_gen[np.isfinite(x_gen).all(1)]
            try:
                cv = repeated_cv(x_ref, x_gen, n_repeats=10, seed=args.seed)
                entry.update(auc_vs_real_mc=cv["roc_auc"], auc_sd=cv["roc_auc_sd"])
            except ValueError as exc:
                entry["auc_error"] = str(exc)
        extra.append(entry)
    summary = summary.merge(pd.DataFrame(extra), on="kind", how="outer")
    summary.to_csv(out / "summary.csv", index=False)
    write_json(out / "report.json", {"runs": {n: str(p) for n, p in runs}, "split": args.split, "n_spines": len(keys), "seed": args.seed,
                                     "resolution": args.resolution, "reference": "real_mc", "baseline": names[0],
                                     "summary": json.loads(summary.to_json(orient="records")), "seconds": time.time() - started})

    lines = ["# VAE decoder study", "", f"{len(keys)} {args.split} spines, {spines['neuron_id'].nunique()} neurons, MC {args.resolution}^3; "
             f"reference real_mc (exact SDF of the real mesh on the same grid); baseline {names[0]}.", "",
             "| kind | mean abs bias (SD) [95 % CI] | vs baseline [95 % CI] | metrics with bias | valid | Chamfer-L1 nm [CI] | HD95 nm | AUC vs real_mc |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary.to_dict("records"):
        f = lambda v, p=3: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"
        bias = f"{f(r.get('mean_abs_bias_sd'))} [{f(r.get('ci_low'))}; {f(r.get('ci_high'))}]" if np.isfinite(r.get("mean_abs_bias_sd", np.nan)) else ""
        diff = f"{f(r.get('diff_vs_baseline'), 3)} [{f(r.get('diff_ci_low'))}; {f(r.get('diff_ci_high'))}]" if np.isfinite(r.get("diff_vs_baseline", np.nan)) else ""
        nsig = f"{int(r['n_significant'])}/{int(r['n_metrics'])}" if np.isfinite(r.get("n_significant", np.nan)) else ""
        lines.append(f"| {r['kind']} | {bias} | {diff} | {nsig} | {f(r.get('valid_rate'), 2)} | {f(r.get('chamfer_l1_nm'), 1)} "
                     f"[{f(r.get('chamfer_ci_low'), 1)}; {f(r.get('chamfer_ci_high'), 1)}] | {f(r.get('hd95_nm'), 0)} | {f(r.get('auc_vs_real_mc'))} |")
    lines += ["", "Per metric (bias in SD of real_mc, 95 % CI, * = interval excludes 0):", "",
              "| metric | " + " | ".join(names + ["real"]) + " |", "| --- |" + " --- |" * (len(names) + 1)]
    pm = res["per_metric"]
    for m in metrics:
        cells = []
        for k in names + ["real"]:
            r = pm[(pm["kind"] == k) & (pm["metric"] == m)].iloc[0]
            cells.append(f"{r['bias_sd']:+.2f}{'*' if r['significant'] else ''}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
