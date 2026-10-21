"""Generate spines with a trained VAE: latent -> SDF on the frozen bbox grid -> Marching Cubes -> meshes.

Prior samples (the generative comparison with MoGen; ``z ~ N(0, I) * temperature``):

    python scripts/neuron_model/generate_vae.py --run runs/training/neuron-model/<experiment_id> \
        --n-samples 50 --seed 1

Reconstructions of real spines (``z = mu`` of the encoder; the VAE's own fidelity, there is no
MoGen counterpart):

    python scripts/neuron_model/generate_vae.py --run runs/training/neuron-model/<experiment_id> \
        --mode reconstruction --split test --n-samples 50 --seed 1 \
        --manifest O:/Datasets/Minnie65/preprocessed/manifest.parquet \
        --splits data/processed/neuron-model/splits/splits_dev_100n.parquet --drop-unassigned

Output layout is the one ``evaluate_generators.py`` reads (same as ``generate_mogen.py``):

    prior          <run>/meshes/seed_<s>/sample_<i>/generated_mesh_{raw,postprocessed}.off, ...
    reconstruction <run>/meshes_reconstruction/seed_<s>/sample_<i>/...  (+ the real spine_key in the metadata; with
                   --crop-to-spine: meshes_reconstruction_crop/, see the option)
    reconstruction also writes reconstruction_metrics.json per sample: Chamfer / Hausdorff-95 / F-score (nm) of the
    generated surface against the REAL point cloud of that spine

    python scripts/neuron_model/evaluate_generators.py --config configs/neuron-model/evaluation/default.yaml \
        --model vae=runs/training/neuron-model/<id>/meshes  "data.manifests=[...]"

Checkpoint: ``latest`` by default. NOT ``best_val_loss``: the validation loss contains the KL term,
whose weight grows during the warm-up, so "best" would pick an early checkpoint (small KL weight)
and not the best-trained one.
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

# Windows conda: torch (libomp) and MKL (libiomp5md) both load an OpenMP runtime; the first use of both (e.g. mesh
# extraction after training, SVD) aborts the process with "OMP: Error #15". Standard workaround, Windows only.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from src.neuron_model.spine_generation.data.bbox import FrozenBBox
from src.neuron_model.spine_generation.experiment.artifacts import SampleWriter, write_json
from src.neuron_model.spine_generation.experiment.checkpoint import CheckpointManager
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run
from src.neuron_model.spine_generation.experiment.training import get_device
from src.neuron_model.spine_generation.models.spine_vae.generate import latent_to_mesh, write_vae_sample
from src.neuron_model.spine_generation.models.spine_vae.model import SpineVAE, SpineVAEConfig
from src.neuron_model.spine_generation.reconstruction.mesh_validation import aggregate_validity


def _reconstruction_latents(model, args, cfg, device, *, split=None, n=None):
    """``(z [n, D], spine_keys, real_clouds [n, N, 3] in nm)``: encoder means of ``n`` random real spines of ``--split``
    and their real surface point clouds (the reference for ``--crop-to-spine`` and the reconstruction metrics)."""
    from src.neuron_model.spine_generation.data.datasets import PointCloudDataset
    from src.neuron_model.spine_generation.data.index import load_spine_index
    from src.neuron_model.spine_generation.data.splits import load_splits

    data = cfg["data"]
    manifests = args.manifest or [resolve_path(p) for p in data["manifests"]]
    splits_path = args.splits or resolve_path(data["splits"])
    index = load_spine_index(manifests, path_mode=data.get("path_mode", "relative"), check_files=())
    split = split or args.split
    index = index.with_split(load_splits(splits_path), split, drop_unassigned=args.drop_unassigned)
    if len(index) == 0:
        raise SystemExit(f"no spines in split {split!r}")
    rng = np.random.default_rng(args.seed)
    chosen = index.subset(index.frame["spine_key"].iloc[rng.permutation(len(index))[: n or args.n_samples]])
    dataset = PointCloudDataset(chosen, int(data["n_points"]), random_variant=False, coord_scale=float(data["coord_scale"]), use_normals=bool(data.get("use_normals", True)))
    latents, keys, clouds = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        normals = item.get("normals")
        with torch.no_grad():
            _, mu, _ = model.encode(item["points"][None].to(device), None if normals is None else normals[None].to(device), sample=False)
        latents.append(mu[0].cpu().numpy())
        keys.append(item["spine_key"])
        clouds.append(item["points"].numpy().astype(np.float64) / float(data["coord_scale"]))
    return np.stack(latents), keys, clouds


def _sample_data_prior(codes: np.ndarray, n: int, seed: int, components: int) -> np.ndarray:
    """``n`` latents from a distribution fitted to the encoder means of real spines: a full-covariance Gaussian
    (``components == 1``) or a Gaussian mixture. The standard-normal prior does not match the aggregate posterior when the
    KL weight is tiny (beta 1e-4: the VAE is nearly an autoencoder), so prior samples fall into poorly covered regions."""
    if components <= 1:
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(codes.mean(axis=0), np.cov(codes, rowvar=False), size=n)
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-4, random_state=seed).fit(codes)
    return gmm.sample(n)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="VAE training run directory")
    parser.add_argument("--checkpoint", default="latest", help="checkpoint tag inside --run (default: latest)")
    parser.add_argument("--mode", choices=("prior", "reconstruction"), default="prior")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="prior mode: z = temperature * N(0, I); <1 stays closer to the data manifold (less diverse)")
    parser.add_argument("--prior", choices=("normal", "data"), default="normal", help="prior mode: 'normal' = N(0, I); 'data' = a Gaussian (or mixture, --gmm-components) fitted to the encoder means of --fit-n real spines of --fit-split. Meshes go to meshes_data_prior/")
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--fit-n", type=int, default=2000)
    parser.add_argument("--gmm-components", type=int, default=1, help="--prior data: 1 = one full-covariance Gaussian")
    parser.add_argument("--resolution", type=int, help="marching-cubes grid (default: resolution_final of the reconstruction config, 256)")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--crop-to-spine", action="store_true", help="reconstruction mode, DIAGNOSTIC: extract the mesh only in the box of the real spine (+ --crop-margin) instead of the frozen train bbox. "
                        "The decoder is supervised only inside the box of its own spine; outside it is unconstrained and Marching Cubes returns thousands of fragments there")
    parser.add_argument("--crop-margin", type=float, default=150.0, help="nm, margin of the --crop-to-spine box")
    parser.add_argument("--check-self-intersections", action="store_true", help="slow; the spec's validity report includes it")
    # reconstruction mode
    parser.add_argument("--split", default="test", help="reconstruction mode: split to draw real spines from")
    parser.add_argument("--manifest", type=Path, nargs="+", help="reconstruction mode: manifest(s); default = the run's data.manifests")
    parser.add_argument("--splits", type=Path, help="reconstruction mode: splits parquet; default = the run's data.splits")
    parser.add_argument("--drop-unassigned", action="store_true", help="reconstruction mode: ignore spines the split does not cover")
    args = parser.parse_args()

    run = open_run(args.run)
    cfg = run.load_config()
    model_cfg = SpineVAEConfig.from_mapping(json.loads((run.root / "model_config.json").read_text(encoding="utf-8")))
    model = SpineVAE(model_cfg)
    payload = CheckpointManager(run.checkpoints).load(args.checkpoint, restore_rng=False)
    model.load_state_dict(payload["state"]["ema"]["model"] if "ema" in payload["state"] else payload["state"]["model"])
    device = get_device(cfg.get("device", "auto"))
    model.to(device).eval()
    checkpoint_id = f"{run.experiment_id}:{args.checkpoint}:step{payload['step']}"

    bbox = FrozenBBox.load(resolve_path(cfg["data"]["bbox"]))
    coord_scale = float(cfg["data"]["coord_scale"])
    mc_cfg = load_config(resolve_path(cfg["reconstruction"]["marching_cubes_config"]))
    resolution = int(args.resolution or mc_cfg["resolution_final"])
    postprocess_cfg = None if args.no_postprocess else mc_cfg["postprocess"]

    if args.crop_to_spine and args.mode != "reconstruction":
        parser.error("--crop-to-spine needs --mode reconstruction (a prior sample has no spine to take the box from)")
    real_clouds = None
    if args.mode == "prior" and args.prior == "data":
        codes, _, _ = _reconstruction_latents(model, args, cfg, device, split=args.fit_split, n=args.fit_n)
        print(f"fitted latent distribution on {len(codes)} {args.fit_split} spines ({args.gmm_components} component(s))", flush=True)
        z_all = _sample_data_prior(codes, args.n_samples, args.seed, args.gmm_components)
        keys = [None] * len(z_all)
        out_root = run.root / "meshes_data_prior"
    elif args.mode == "prior":
        z_all = (args.temperature * model.sample_prior(args.n_samples, generator=torch.Generator().manual_seed(args.seed), device="cpu")).numpy()
        keys = [None] * len(z_all)
        out_root = run.meshes
    else:
        z_all, keys, real_clouds = _reconstruction_latents(model, args, cfg, device)
        out_root = run.root / ("meshes_reconstruction_crop" if args.crop_to_spine else "meshes_reconstruction")
    tag = f"seed_{args.seed}"
    writer = SampleWriter(out_root)
    print(f"{args.mode}: {len(z_all)} samples, grid {resolution}^3, checkpoint {checkpoint_id} -> {out_root / tag}", flush=True)

    from tqdm.auto import tqdm

    from src.neuron_model.spine_generation.evaluation.geometry_metrics import chamfer_metrics, sample_surface

    raw_reports, post_reports, recon_rows = [], [], []
    for i, (z, key) in enumerate(tqdm(list(zip(z_all, keys)), desc=args.mode)):
        box = (bbox.low, bbox.high)
        if args.crop_to_spine:
            cloud = real_clouds[i]
            box = (tuple((cloud.min(axis=0) - args.crop_margin).tolist()), tuple((cloud.max(axis=0) + args.crop_margin).tolist()))
        result = latent_to_mesh(
            model, z, bbox=box, resolution=resolution, coord_scale=coord_scale, device=device,
            chunk_size=int(mc_cfg.get("chunk_size", 262_144)), iso_level=float(mc_cfg.get("iso_level", 0.0)),
            postprocess_cfg=postprocess_cfg, check_self_intersections=args.check_self_intersections,
        )
        config = {
            "mode": args.mode, "resolution": resolution, "coord_scale": coord_scale, "temperature": args.temperature if args.mode == "prior" else None,
            "spine_key": key, "split": args.split if args.mode == "reconstruction" else None, "generation_seed": args.seed,
        }
        config["bbox_nm"] = {"low": list(box[0]), "high": list(box[1]), "cropped_to_spine": bool(args.crop_to_spine)}
        write_vae_sample(writer, tag, i, z, result, seed=args.seed, checkpoint_id=checkpoint_id, generation_config=config)
        if real_clouds is not None:  # reconstruction vs the REAL surface (point cloud of the preprocessed spine), nm
            row = {"sample": i, "spine_key": key}
            for kind in ("raw", "postprocessed"):
                mesh = result["mesh_raw"] if kind == "raw" else result["mesh_postprocessed"]
                if mesh is None or len(mesh.faces) == 0:
                    continue
                m = chamfer_metrics(sample_surface(mesh, 8192, seed=i), real_clouds[i], f_score_thresholds=(10.0, 25.0, 50.0))
                row.update({f"{kind}_{k}": v for k, v in m.items()})
            recon_rows.append(row)
            write_json(writer.sample_dir(tag, i) / "reconstruction_metrics.json", row)
        raw_reports.append(result["validation_raw"])
        post_reports.append(result["validation_postprocessed"])

    summary = {
        "mode": args.mode,
        "n_samples": len(z_all),
        "checkpoint_id": checkpoint_id,
        "resolution": resolution,
        "temperature": args.temperature if args.mode == "prior" else None,
        "marching_cubes_frozen": bool(mc_cfg.get("frozen")),
        "validity_raw": aggregate_validity(raw_reports),
        "validity_postprocessed": aggregate_validity(post_reports),
    }
    if recon_rows:
        table = pd.DataFrame(recon_rows)
        table.to_parquet(out_root / tag / "reconstruction_metrics.parquet", index=False)
        summary["reconstruction_vs_real_nm"] = {
            f"{kind}_{k}_median": float(table[f"{kind}_{k}"].median())
            for kind in ("raw", "postprocessed") for k in ("chamfer_l1", "hausdorff_95", "f_score@25", "f_score@50") if f"{kind}_{k}" in table
        }
    write_json(out_root / tag / "generation_summary.json", summary)
    pd.DataFrame(raw_reports).to_parquet(out_root / tag / "validation_raw.parquet", index=False)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
