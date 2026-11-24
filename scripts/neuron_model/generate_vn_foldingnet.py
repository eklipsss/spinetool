"""VN + FoldingNet as a generator: codes of the training spines -> normalizing flow -> samples -> meshes.

    python scripts/neuron_model/generate_vn_foldingnet.py --run runs/training/neuron-model/<vn-foldingnet run> \
        --n-samples 500 --seeds 1 2 [--checkpoint latest] [--mode prior|reconstruction]

1. Latent vectors of the TRAINING spines of the run's split (variant 1, eval mode): the invariant code z, plus the first
   two rows of the frame R in mode 'frame' (``VNFoldingAE.latent_vector``); validation spines for early stopping of the
   flow. Cached in ``<run>/latent_prior_flow/codes_{train,val}.npz``.
2. RealNVP prior on these vectors (``spine_vae.latent_prior.fit_flow``, the flow of our VAE: starts as the fitted Gaussian,
   early stopping on the validation NLL; ``--flow-noise`` dequantisation in whitened units) -> ``flow_prior.pt``.
3. ``--mode prior``: samples of the flow (``--temperature``) -> decoded on a grid of 8192 points (the decoder takes any
   grid) -> normals -> Screened Poisson with ``--poisson-config`` (default d8_pw8, the chain chosen on real spines) ->
   ``<run>/meshes_flow_d8_pw8/seed_<s>/sample_*`` in the MoGen layout (generated_mesh_postprocessed.off ...).
   ``--mode reconstruction``: the validation codes decoded instead (``recon_val``): the ceiling of the autoencoder.
Evaluation: evaluate_generators.py --model vn_foldingnet=<run>/meshes_flow_d8_pw8 (folders seed_<s>). GPU for steps
1 and 3, the Poisson chain on CPU (~2 s per spine).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.spine_generation.data.datasets import PointCloudDataset  # noqa: E402
from src.neuron_model.spine_generation.data.index import load_spine_index  # noqa: E402
from src.neuron_model.spine_generation.data.splits import load_splits  # noqa: E402
from src.neuron_model.spine_generation.experiment.artifacts import SampleWriter, write_json  # noqa: E402
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path  # noqa: E402
from src.neuron_model.spine_generation.experiment.run import open_run  # noqa: E402
from src.neuron_model.spine_generation.experiment.training import get_device  # noqa: E402
from src.neuron_model.spine_generation.models.spine_foldnet.train import decode_vectors, encode_dataset, load_run_model  # noqa: E402
from src.neuron_model.spine_generation.models.spine_mogen.infer import pointcloud_to_mesh, write_mogen_sample  # noqa: E402
from src.neuron_model.spine_generation.models.spine_vae.latent_prior import FlowPrior, fit_flow, hole_ratio, nll_per_dim  # noqa: E402
from src.neuron_model.spine_generation.reconstruction.mesh_validation import aggregate_validity  # noqa: E402
from src.neuron_model.spine_generation.reconstruction.screened_poisson import PoissonParams  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--mode", choices=["prior", "reconstruction"], default="prior")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--n-points", type=int, default=8192, help="decoded points per cloud (the Poisson chain input)")
    ap.add_argument("--max-train-codes", type=int, default=0, help="0 = every training spine of the split")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--flow-layers", type=int, default=8)
    ap.add_argument("--flow-hidden", type=int, default=512)
    ap.add_argument("--flow-epochs", type=int, default=400)
    ap.add_argument("--flow-noise", type=float, default=0.05)
    ap.add_argument("--refit-flow", action="store_true")
    ap.add_argument("--poisson-config", type=Path, default=REPO_ROOT / "configs/neuron-model/reconstruction/poisson_sweep/d8_pw8.yaml")
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--manifests", nargs="*", help="override data.manifests of the run (e.g. another copy of the cache)")
    args = ap.parse_args()

    started = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - started:7.0f} s] {msg}", flush=True)

    run = open_run(args.run)
    cfg = run.load_config()
    data_cfg = cfg["data"]
    device = get_device(cfg.get("device", "auto"))
    model = load_run_model(run, args.checkpoint, device)
    coord_scale = float(data_cfg["coord_scale"])
    checkpoint_id = f"{run.experiment_id}:{args.checkpoint}:step{getattr(model, 'checkpoint_step', -1)}"
    log(f"model {checkpoint_id}: mode {model.cfg.mode}, latent {model.cfg.latent_dim}, grid {model.cfg.grid}; device {device}")

    prior_dir = run.root / "latent_prior_flow"
    prior_dir.mkdir(exist_ok=True)
    codes_path = {s: prior_dir / f"codes_{s}.npz" for s in ("train", "val")}
    if not all(p.exists() for p in codes_path.values()) or args.refit_flow:
        manifests = args.manifests or data_cfg["manifests"]
        index = load_spine_index([resolve_path(p) for p in manifests], path_mode=data_cfg.get("path_mode", "relative"),
                                 check_files=(f"pointcloud_{data_cfg['n_points']}_path",))
        index = index.with_split(load_splits(resolve_path(data_cfg["splits"])), drop_unassigned=True)
        for split in ("train", "val"):
            idx = index.subset(index.frame.loc[index.frame["split"] == split, "spine_key"])
            if split == "train" and args.max_train_codes and len(idx) > args.max_train_codes:
                pick = np.sort(np.random.default_rng(0).choice(len(idx), args.max_train_codes, replace=False))
                idx = idx.subset(idx.frame["spine_key"].iloc[pick])
            ds = PointCloudDataset(idx, int(data_cfg["n_points"]), variants=(1,), random_variant=False, jitter_std=0.0, coord_scale=coord_scale, use_normals=False)
            vec, keys = encode_dataset(model, ds, device, log=log)
            np.savez(codes_path[split], codes=vec, spine_key=np.array(keys), checkpoint=checkpoint_id)
            log(f"{split}: {len(keys)} codes of dimension {vec.shape[1]} -> {codes_path[split]}")
    train = np.load(codes_path["train"], allow_pickle=True)["codes"]
    val = np.load(codes_path["val"], allow_pickle=True)
    val_codes, val_keys = val["codes"], list(val["spine_key"])

    if args.mode == "prior":
        flow_path = prior_dir / "flow_prior.pt"
        if flow_path.exists() and not args.refit_flow:
            prior = FlowPrior.load(flow_path)
            log(f"flow prior {flow_path}")
        else:
            prior, info = fit_flow(train, val_codes, n_layers=args.flow_layers, hidden=args.flow_hidden, epochs=args.flow_epochs,
                                   noise_std=args.flow_noise, log=log)
            prior.save(flow_path)
            write_json(prior_dir / "flow_fit.json", {k: v for k, v in info.items() if k != "history"})
            pd.DataFrame(info["history"]).to_csv(prior_dir / "flow_history.csv", index=False)
            log(f"flow fitted: best epoch {info['best_epoch']}, val NLL/dim {info['best_score_per_dim']:.4f} -> {flow_path}")
        report = {"val_nll_per_dim": nll_per_dim(prior, val_codes), "train_nll_per_dim": nll_per_dim(prior, train[: len(val_codes)]),
                  "holes": hole_ratio(prior.sample(2000, seed=123), val_codes)}
        write_json(prior_dir / "flow_report.json", report)
        log(f"flow: val NLL/dim {report['val_nll_per_dim']:.4f}, train {report['train_nll_per_dim']:.4f}; holes {report['holes']}")

    rec_cfg = load_config(resolve_path(args.poisson_config))
    poisson = PoissonParams.from_mapping(rec_cfg["poisson"])
    k_normal = int(rec_cfg["normals"]["k_normal"])
    out_root = run.root / f"meshes_flow_{Path(args.poisson_config).stem}"
    writer = SampleWriter(out_root)
    for seed in args.seeds:
        if args.mode == "prior":
            vectors = prior.sample(args.n_samples, temperature=args.temperature, seed=seed)
            tag = f"seed_{seed}" if args.temperature == 1.0 else f"T{args.temperature:g}_seed_{seed}"
        else:
            vectors = val_codes[: args.n_samples]
            tag = "recon_val"
        clouds = decode_vectors(model, vectors, device, n_points=args.n_points, coord_scale=coord_scale, grid_seed=seed)
        inference_config = {"model": "vn_foldingnet", "mode": args.mode, "checkpoint_id": checkpoint_id, "generation_seed": seed,
                            "temperature": args.temperature, "n_points": args.n_points, "coord_scale": coord_scale,
                            "ae_mode": model.cfg.mode, "grid": model.cfg.grid, "poisson_config": str(args.poisson_config)}
        raw, post = [], []
        for i, cloud in enumerate(clouds):
            rec = None if args.no_mesh else pointcloud_to_mesh(cloud, k_normal=k_normal, poisson=poisson, postprocess_cfg=rec_cfg["postprocess"])
            write_mogen_sample(writer, tag, i, cloud, rec, seed=seed, checkpoint_id=checkpoint_id, inference_config=inference_config, poisson=poisson, k_normal=k_normal)
            if args.mode == "reconstruction":
                (writer.sample_dir(tag, i) / "source_spine_key.txt").write_text(str(val_keys[i]), encoding="utf-8")
            if rec is not None:
                raw.append(rec["validation_raw"])
                post.append(rec["validation_postprocessed"])
            if (i + 1) % 25 == 0 or i + 1 == len(clouds):
                log(f"{tag}: meshes {i + 1}/{len(clouds)}")
        summary = {"n_samples": len(clouds), "inference_config": inference_config, "validity_raw": aggregate_validity(raw),
                   "validity_postprocessed": aggregate_validity(post), "extent_mean_physical": float((clouds.max(axis=1) - clouds.min(axis=1)).mean())}
        write_json(out_root / tag / "generation_summary.json", summary)
        log(f"{tag}: {summary['validity_postprocessed']}")
        if args.mode == "reconstruction":
            break
    log(f"-> {out_root}")


if __name__ == "__main__":
    main()
