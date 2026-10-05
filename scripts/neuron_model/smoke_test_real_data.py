"""Smoke-test the data layer (``src/neuron_model/spine_generation/data/``) against REAL
preprocessed files, not the synthetic fixtures used by ``tests/neuron_model``.

Checks, in order: manifest loads and paths resolve, PointCloudDataset/SDFDataset read real
npz files with the expected shapes/dtypes, a frozen bbox computes from real point clouds, and
one forward pass each through SpineVAE and the MoGen PointInfinity port runs without error on
real (if tiny) data. Prints real coordinate extents (sanity check for the MoGen coord_scale
hypothesis, s-module-implementation-plan.md "Открытые вопросы" #1).

Not training - a few forward passes on <=10 spines, seconds on CPU.

    python scripts/neuron_model/smoke_test_real_data.py --manifest <output_root>/minnie65/manifest.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.neuron_model.spine_generation.data.bbox import compute_frozen_bbox
from src.neuron_model.spine_generation.data.datasets import PointCloudDataset, SDFDataset, make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.models.spine_mogen.pointinfinity import PointInfinity, PointInfinityConfig
from src.neuron_model.spine_generation.models.spine_vae.model import SpineVAE, SpineVAEConfig
from src.neuron_model.spine_generation.models.spine_vae.pointnext_encoder import PointNeXtConfig
from src.neuron_model.spine_generation.models.spine_vae.siren_sdf_decoder import SirenConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=2048, help="pointcloud size to load (2048/4096/8192)")
    parser.add_argument("--coord-scale-nm-to-um", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"=== loading {args.manifest} ===")
    index = load_spine_index([args.manifest], check_files=(f"pointcloud_{args.n_points}_path", "sdf_samples_path"))
    attrs = index.frame.attrs
    print(f"manifest rows: {attrs['n_manifest']}, train_eligible: {attrs['n_train_eligible']}, with files on disk: {attrs['n_with_files']}")
    if len(index) == 0:
        sys.exit("No usable spines found - check the manifest path and that files exist next to it.")
    print("spine keys:", index.keys)

    print("\n=== real coordinate extents (local frame, physical units = nm for Minnie/H01) ===")
    extents = []
    for path in index.frame[f"pointcloud_{args.n_points}_path"]:
        with np.load(path) as data:
            points = np.asarray(data["points"], dtype=float)
        extents.append(points.max(axis=0) - points.min(axis=0))
    extents = np.asarray(extents)
    print(f"bounding-box size per spine (nm): min={extents.min(axis=0)} max={extents.max(axis=0)} mean={extents.mean(axis=0)}")
    mogen_coord_scale = 1e-4  # spec's starting hypothesis: x_MoGen = x_um / 10 = x_nm * 1e-4
    print(f"-> after MoGen coord_scale={mogen_coord_scale}: mean extent (model units) = {extents.mean(axis=0) * mogen_coord_scale}")

    print("\n=== PointCloudDataset ===")
    pc_ds = PointCloudDataset(index, args.n_points, random_variant=False, coord_scale=args.coord_scale_nm_to_um)
    item = pc_ds[0]
    print(f"item['points']: shape={tuple(item['points'].shape)} dtype={item['points'].dtype} range=({item['points'].min():.3f}, {item['points'].max():.3f})")
    print(f"item['normals']: shape={tuple(item['normals'].shape)} unit-norm check: {item['normals'].norm(dim=-1).mean():.4f} (should be ~1.0)")
    loader = make_loader(pc_ds, batch_size=min(4, len(pc_ds)), shuffle=False, seed=0)
    batch = next(iter(loader))
    print(f"batch['points']: shape={tuple(batch['points'].shape)}")

    print("\n=== SDFDataset ===")
    sdf_ds = SDFDataset(index, args.n_points, {"surface": 256, "near": 512, "uniform": 256}, random_variant=False, coord_scale=args.coord_scale_nm_to_um)
    sdf_item = sdf_ds[0]
    print(f"query_points: shape={tuple(sdf_item['query_points'].shape)}, sdf range=({sdf_item['sdf'].min():.3f}, {sdf_item['sdf'].max():.3f})")
    print(f"sample_type counts: {torch.bincount(sdf_item['sample_type']).tolist()} (expect [256, 512, 256])")
    print(f"query_has_normal: {sdf_item['query_has_normal'].sum().item()} / {len(sdf_item['query_has_normal'])} (surface+near should carry normals, uniform should not)")

    print("\n=== FrozenBBox (not actually frozen - just a sanity check on this tiny subset) ===")
    bbox = compute_frozen_bbox(index, margin=50.0, n_points=args.n_points)
    print(f"bbox low={bbox.low} high={bbox.high} (physical units)")

    print("\n=== one forward pass: SpineVAE (random weights, tiny config) ===")
    vae = SpineVAE(SpineVAEConfig(
        latent_dim=8,
        encoder=PointNeXtConfig(input_points=args.n_points, base_width=8, stage_widths=(8, 16), strides=(4, 4), blocks=(0, 1), k_neighbors=8, global_feature_dim=16, sampler="random"),
        decoder=SirenConfig(hidden_dim=16, hidden_layers=2),
    )).eval()
    sdf_batch = next(iter(make_loader(sdf_ds, batch_size=min(2, len(sdf_ds)), shuffle=False, seed=0)))
    with torch.no_grad():
        from src.neuron_model.spine_generation.models.spine_vae.losses import LossWeights
        terms = vae.compute_losses(sdf_batch, LossWeights(), step=0, sample=False)
    print(f"sdf_loss on real data: {float(terms['sdf_loss']):.4f} (random weights - not expected to be small, just finite)")
    assert torch.isfinite(terms["total_loss"])

    print("\n=== one forward pass: MoGen PointInfinity port (random weights, tiny config) ===")
    mogen = PointInfinity(PointInfinityConfig(point_dim=16, latent_dim=32, n_latents=8, n_blocks=1, n_subblocks=1, n_heads=4, k_nn=8)).eval()
    with torch.no_grad():
        velocity = mogen(batch["points"], torch.rand(batch["points"].shape[0]))
    print(f"velocity: shape={tuple(velocity.shape)}, finite={torch.isfinite(velocity).all().item()}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
