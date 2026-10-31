"""Does the VAE code identify the spine, and how far is its latent from N(0, I)?  (diagnosis before a learned prior)

Two levels (src/neuron_model/spine_generation/models/spine_vae/latent_prior.py):

1. Latent (cheap, all --n-spines): every spine is encoded for each point-cloud resampling of the run's ``data.variants``.
   Printed: posterior sigma and resampling noise against the spread between spines, retrieval accuracy (is the
   posterior sample / the code of another resampling nearest to its own spine?), spectrum and effective rank of the
   aggregate posterior, KL(fitted Gaussian || N(0, I)), and the "hole ratio" of samples from N(0, I) and from a fitted
   Gaussian (NN distance sample -> held-out real code, relative to real code -> real code; ~1 = no holes).

2. Meshes (--decode-spines > 0; the colleague's test "one spine many times vs many spines"): for each of
   --decode-spines spines a mesh is decoded from z = mu, from --n-posterior samples z ~ q(z|x) and from mu of every
   other resampling. Morphometrics of every mesh (the same 11 scalar metrics as the evaluation, base by the
   down-facing-cap finder) give per metric:
     within_to_between   spread over the repeats of ONE spine / spread between spines (small = the code carries the shape)
     icc                 between-spine share of the variance (1 = all variation is between spines)
     recon_bias_sd       mean(reconstruction - real) / sd(real), the real value computed the same way (same finder) on
                         the real local_sealed mesh. This is the CEILING of any prior: if reconstructions of real codes
                         are already biased (e.g. shorter), no prior (Gaussian, flow) can remove that bias.
   Meshes are extracted in the box of the real spine (+ --crop-margin), as ``generate_vae.py --crop-to-spine``.

Windows (GPU), the run's own manifest / split; --manifest/--splits override them. The training cache has no meshes:
the real meshes come from the full manifest (--mesh-manifest, paths on the NAS, read only for --decode-spines spines):

    python scripts/neuron_model/analyze_vae_latent.py --run runs/training/neuron-model/2026-10-19_003408_vae-dev-reg0-cos \
        --split val --n-spines 1500 --decode-spines 40 --n-posterior 8 \
        --mesh-manifest C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet

Writes ``<run>/latent_analysis/<split>_seed<s>/{latent_report.json, codes.npz, mesh_repeats.parquet, mesh_report.json}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":  # torch + MKL OpenMP runtimes (see generate_vae.py)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.models.spine_vae.latent_prior import encode_spines, latent_report, load_split_index, load_vae_run, repeat_statistics


def load_index(cfg, args, split, n, seed):
    return load_split_index(cfg, split, n, seed, manifests=args.manifest, splits=args.splits, path_mode=args.path_mode)


def _fmt(report):
    lines = []
    for key, value in report.items():
        if isinstance(value, dict):
            value = ", ".join(f"{k} {v:.3g}" for k, v in value.items())
        elif isinstance(value, list):
            value = "[" + ", ".join(f"{v:.3g}" for v in value) + "]"
        elif isinstance(value, float):
            value = f"{value:.4g}"
        lines.append(f"  {key:42s} {value}")
    return "\n".join(lines)


def mesh_level(model, cfg, args, codes, index, device, out_dir):
    """Morphometrics of meshes decoded from repeats of the same spine (posterior samples, other resamplings) and from
    the codes of different spines; the real mesh of each spine through the same metric code."""
    from tqdm.auto import tqdm

    from src.neuron_model.spine_generation.evaluation.attachment import finder_from_config
    from src.neuron_model.spine_generation.evaluation.morphometrics import generated_morphometrics
    from src.neuron_model.spine_generation.experiment.config import load_config
    from src.neuron_model.spine_generation.models.spine_vae.generate import latent_to_mesh
    from src.neuron_model.spine_geometry import load_trimesh
    from src.neuron_model.spine_morphometrics import SCALAR_METRICS

    eval_cfg = load_config(REPO_ROOT / "configs/neuron-model/evaluation/default.yaml")
    finder = finder_from_config(eval_cfg["morphometrics"].get("attachment_finder"))
    coord_scale = float(cfg["data"]["coord_scale"])
    mc_cfg = load_config(resolve_path(cfg["reconstruction"]["marching_cubes_config"]))
    rng = np.random.default_rng(args.seed + 7)
    n = min(args.decode_spines, len(codes["keys"]))
    frame = index.frame.set_index("spine_key")
    if args.mesh_manifest:  # the training cache holds point clouds / SDF only; real meshes from another manifest
        from src.neuron_model.spine_generation.data.index import load_spine_index

        frame = load_spine_index([args.mesh_manifest], path_mode=args.mesh_path_mode, check_files=()).frame.set_index("spine_key")
    rows = []
    for i in tqdm(range(n), desc="decode"):
        key = codes["keys"][i]
        real = load_trimesh(Path(frame.loc[key, "local_sealed_mesh_path"]), process=False)
        v = np.asarray(real.vertices)
        box = (tuple((v.min(0) - args.crop_margin).tolist()), tuple((v.max(0) + args.crop_margin).tolist()))
        mu, sigma = codes["mu"][i], np.exp(0.5 * codes["logvar"][i])  # [V, D]
        jobs = [("mu", 0, mu[0])]
        jobs += [("resampling", k, mu[k]) for k in range(1, len(mu))]
        jobs += [("posterior", k, mu[0] + sigma[0] * rng.standard_normal(mu.shape[1])) for k in range(args.n_posterior)]
        meshes = [real]
        for _, _, z in jobs:
            res = latent_to_mesh(model, z, bbox=box, resolution=args.resolution, coord_scale=coord_scale, device=device,
                                 chunk_size=int(mc_cfg.get("chunk_size", 262_144)), iso_level=float(mc_cfg.get("iso_level", 0.0)),
                                 postprocess_cfg=mc_cfg["postprocess"], check_self_intersections=False)
            meshes.append(res["mesh_postprocessed"] if res["mesh_postprocessed"] is not None else res["mesh_raw"])
        table = generated_morphometrics(meshes, attachment_finder=finder, compute_chords=False)
        kinds = [("real", 0)] + [(k, j) for k, j, _ in jobs]
        for (kind, rep), (_, row) in zip(kinds, table.iterrows()):
            rows.append({"spine": i, "spine_key": key, "kind": kind, "repeat": rep, **{m: row[m] for m in SCALAR_METRICS}})
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "mesh_repeats.parquet", index=False)

    report = {"n_spines": int(n), "n_posterior": args.n_posterior, "resolution": args.resolution,
              "mesh_present": float(df.loc[df.kind != "real", "Volume"].notna().mean()),
              "metrics": repeat_statistics(df, SCALAR_METRICS)}
    write_json(out_dir / "mesh_report.json", report)
    print(f"\n[mesh] decoded meshes with metrics: {report['mesh_present']:.0%}")
    print("[mesh] per metric: within/between (posterior, resampling) | ICC posterior | reconstruction bias (sd of real) | median rel. error")
    for m, e in report["metrics"].items():
        print(f"  {m:20s} {e.get('within_to_between_posterior', float('nan')):.3f}  {e.get('within_to_between_resampling', float('nan')):.3f} | "
              f"{e.get('icc_posterior', float('nan')):.3f} | {e['recon_bias_sd']:+.3f} | {e['recon_median_rel_err']:.3f}  (n {e['n_pairs']})")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-spines", type=int, default=1500)
    parser.add_argument("--variants", type=int, nargs="+", help="point-cloud resamplings (default: the run's data.variants)")
    parser.add_argument("--decode-spines", type=int, default=0, help="mesh-level test on this many spines (0 = latent only)")
    parser.add_argument("--n-posterior", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--crop-margin", type=float, default=150.0)
    parser.add_argument("--mesh-manifest", type=Path, help="manifest with the real local_sealed meshes (default: the run's manifest; a training cache has none)")
    parser.add_argument("--mesh-path-mode", default="absolute", choices=("relative", "absolute"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manifest", type=Path, nargs="+")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--path-mode", choices=("relative", "absolute"))
    args = parser.parse_args()

    model, run, cfg, checkpoint_id, device = load_vae_run(args.run, args.checkpoint)
    variants = args.variants or [int(v) for v in cfg["data"].get("variants", (1, 2, 3, 4))]
    index = load_index(cfg, args, args.split, args.n_spines, args.seed)
    print(f"[latent] {checkpoint_id}: {len(index)} {args.split} spines, variants {variants}", flush=True)
    codes = encode_spines(model, index, cfg["data"], variants=variants, device=device, batch_size=args.batch_size, progress=True)
    out_dir = run.root / "latent_analysis" / f"{args.split}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "codes.npz", mu=codes["mu"], logvar=codes["logvar"], keys=np.array(codes["keys"]), variants=np.array(variants))
    report = {"checkpoint_id": checkpoint_id, "split": args.split, **latent_report(codes["mu"], codes["logvar"], seed=args.seed)}
    write_json(out_dir / "latent_report.json", report)
    print("\n[latent] report:\n" + _fmt(report))
    if args.decode_spines > 0:
        mesh_level(model, cfg, args, codes, index, device, out_dir)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
