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
    reconstruction <run>/meshes_reconstruction/seed_<s>/sample_<i>/...  (+ the real spine_key in the metadata)

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


def _reconstruction_latents(model, args, cfg, device):
    """``(z [n, D], spine_keys)``: encoder means of ``n`` random real spines of ``--split``."""
    from src.neuron_model.spine_generation.data.datasets import PointCloudDataset
    from src.neuron_model.spine_generation.data.index import load_spine_index
    from src.neuron_model.spine_generation.data.splits import load_splits

    data = cfg["data"]
    manifests = args.manifest or [resolve_path(p) for p in data["manifests"]]
    splits_path = args.splits or resolve_path(data["splits"])
    index = load_spine_index(manifests, path_mode=data.get("path_mode", "relative"), check_files=())
    index = index.with_split(load_splits(splits_path), args.split, drop_unassigned=args.drop_unassigned)
    if len(index) == 0:
        raise SystemExit(f"no spines in split {args.split!r}")
    rng = np.random.default_rng(args.seed)
    chosen = index.subset(index.frame["spine_key"].iloc[rng.permutation(len(index))[: args.n_samples]])
    dataset = PointCloudDataset(chosen, int(data["n_points"]), random_variant=False, coord_scale=float(data["coord_scale"]), use_normals=bool(data.get("use_normals", True)))
    latents, keys = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        normals = item.get("normals")
        with torch.no_grad():
            _, mu, _ = model.encode(item["points"][None].to(device), None if normals is None else normals[None].to(device), sample=False)
        latents.append(mu[0].cpu().numpy())
        keys.append(item["spine_key"])
    return np.stack(latents), keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="VAE training run directory")
    parser.add_argument("--checkpoint", default="latest", help="checkpoint tag inside --run (default: latest)")
    parser.add_argument("--mode", choices=("prior", "reconstruction"), default="prior")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="prior mode: z = temperature * N(0, I); <1 stays closer to the data manifold (less diverse)")
    parser.add_argument("--resolution", type=int, help="marching-cubes grid (default: resolution_final of the reconstruction config, 256)")
    parser.add_argument("--no-postprocess", action="store_true")
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

    if args.mode == "prior":
        z_all = (args.temperature * model.sample_prior(args.n_samples, generator=torch.Generator().manual_seed(args.seed), device="cpu")).numpy()
        keys = [None] * len(z_all)
        out_root = run.meshes
    else:
        z_all, keys = _reconstruction_latents(model, args, cfg, device)
        out_root = run.root / "meshes_reconstruction"
    tag = f"seed_{args.seed}"
    writer = SampleWriter(out_root)
    print(f"{args.mode}: {len(z_all)} samples, grid {resolution}^3, checkpoint {checkpoint_id} -> {out_root / tag}", flush=True)

    from tqdm.auto import tqdm

    raw_reports, post_reports = [], []
    for i, (z, key) in enumerate(tqdm(list(zip(z_all, keys)), desc=args.mode)):
        result = latent_to_mesh(
            model, z, bbox=(bbox.low, bbox.high), resolution=resolution, coord_scale=coord_scale, device=device,
            chunk_size=int(mc_cfg.get("chunk_size", 262_144)), iso_level=float(mc_cfg.get("iso_level", 0.0)),
            postprocess_cfg=postprocess_cfg, check_self_intersections=args.check_self_intersections,
        )
        config = {
            "mode": args.mode, "resolution": resolution, "coord_scale": coord_scale, "temperature": args.temperature if args.mode == "prior" else None,
            "spine_key": key, "split": args.split if args.mode == "reconstruction" else None, "generation_seed": args.seed,
        }
        write_vae_sample(writer, tag, i, z, result, seed=args.seed, checkpoint_id=checkpoint_id, generation_config=config)
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
    write_json(out_root / tag / "generation_summary.json", summary)
    pd.DataFrame(raw_reports).to_parquet(out_root / tag / "validation_raw.parquet", index=False)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
