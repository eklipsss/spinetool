"""Does the decoder USE the latent code?  (diagnosis of posterior / latent collapse of a VAE run)

    python scripts/neuron_model/diagnose_vae.py --run runs/training/neuron-model/<experiment_id> [--n-spines 512]

Reads the run's config (manifest / splits / variants), takes validation spines and computes the SDF loss
split by sample type (surface / near-surface / uniform) under four codes:

    real      z = mu(x) of the spine itself
    shuffled  z = mu of ANOTHER spine of the batch
    mean      z = the mean of mu over the evaluated spines (the "average shape" code)
    zero      z = 0

If ``shuffled`` is as good as ``real``, the decoder ignores the code: the model is an average-shape model.
Also prints how much the encoder output varies between spines (per-dimension std of mu, PCA), whether the
leading principal components follow the spine size, and which sample type dominates the loss.
Safe to run while training goes on (reads a checkpoint, small batch).
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

# Windows conda: torch (libomp) and MKL (libiomp5md) both load an OpenMP runtime and the first SVD aborts with
# "OMP: Error #15". Harmless for this read-only diagnostic.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from src.neuron_model.spine_generation.data.datasets import SAMPLE_NEAR, SAMPLE_SURFACE, SAMPLE_UNIFORM, make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.checkpoint import CheckpointManager
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run
from src.neuron_model.spine_generation.experiment.training import get_device
from src.neuron_model.spine_generation.models.spine_vae.model import SpineVAE, SpineVAEConfig
from src.neuron_model.spine_generation.models.spine_vae.train import sdf_datasets, to_device

TYPES = {"surface": SAMPLE_SURFACE, "near": SAMPLE_NEAR, "uniform": SAMPLE_UNIFORM}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--n-spines", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run = open_run(args.run)
    cfg = run.load_config()
    data_cfg = cfg["data"]
    device = get_device(cfg.get("device", "auto"))
    model = SpineVAE(SpineVAEConfig.from_mapping(json.loads((run.root / "model_config.json").read_text(encoding="utf-8"))))
    payload = CheckpointManager(run.checkpoints).load(args.checkpoint, restore_rng=False)
    model.load_state_dict(payload["state"]["model"])
    model.to(device).eval()
    print(f"run {run.experiment_id}, checkpoint {args.checkpoint} (step {payload['step']})")

    index = load_spine_index([resolve_path(p) for p in data_cfg["manifests"]], path_mode=data_cfg.get("path_mode", "relative"), check_files=())
    index = index.with_split(load_splits(resolve_path(data_cfg["splits"])), drop_unassigned=bool(data_cfg.get("drop_unassigned", False)))
    _, val = sdf_datasets(index, data_cfg)
    if val is None:
        raise SystemExit("no validation spines")
    n = min(args.n_spines, len(val))
    rng = np.random.default_rng(args.seed)
    subset_ids = rng.permutation(len(val))[:n]
    loader = make_loader(torch.utils.data.Subset(val, subset_ids.tolist()), batch_size=args.batch_size, shuffle=False, seed=args.seed, num_workers=0)

    sums = {mode: {t: [0.0, 0] for t in TYPES} for mode in ("real", "shuffled", "mean", "zero")}
    mus, sizes, feats = [], [], []
    with torch.no_grad():
        batches = []
        for batch in loader:
            batch = to_device(batch, device)
            h = model.encoder(batch["points"], batch.get("normals") if model.cfg.encoder.use_normals else None)
            _, mu, _ = model.latent(h, sample=False)
            feats.append(h.cpu())
            mus.append(mu.cpu())
            pts = batch["points"]
            sizes.append((pts.max(dim=1).values - pts.min(dim=1).values).cpu())  # bbox extent per spine
            batches.append((batch, mu))
        mu_all = torch.cat(mus)
        mu_mean = mu_all.mean(dim=0, keepdim=True).to(device)
        for batch, mu in batches:
            codes = {
                "real": mu,
                "shuffled": mu.roll(shifts=1, dims=0),
                "mean": mu_mean.expand_as(mu),
                "zero": torch.zeros_like(mu),
            }
            for mode, z in codes.items():
                err = (model.decode(batch["query_points"], z) - batch["sdf"]).abs()
                for name, code in TYPES.items():
                    mask = batch["sample_type"] == code
                    sums[mode][name][0] += float(err[mask].sum())
                    sums[mode][name][1] += int(mask.sum())

    def mean_err(mode, name):
        total, count = sums[mode][name]
        return total / max(count, 1)

    print(f"\nSDF L1 error (model units = um) on {n} validation spines, by sample type")
    print(f"{'code':10s} {'surface':>9s} {'near':>9s} {'uniform':>9s} {'weighted*':>10s}")
    weights = {"surface": 1024, "near": 3072, "uniform": 1024}
    for mode in ("real", "shuffled", "mean", "zero"):
        w = sum(weights[t] * mean_err(mode, t) for t in TYPES) / sum(weights.values())
        print(f"{mode:10s} {mean_err(mode, 'surface'):9.4f} {mean_err(mode, 'near'):9.4f} {mean_err(mode, 'uniform'):9.4f} {w:10.4f}")
    print("* weighted by the training query counts (1024 surface, 3072 near, 1024 uniform)")
    real_w = sum(weights[t] * mean_err("real", t) for t in TYPES) / sum(weights.values())
    shuf_w = sum(weights[t] * mean_err("shuffled", t) for t in TYPES) / sum(weights.values())
    gain = (shuf_w - real_w) / max(shuf_w, 1e-12)
    print(f"\nrelative gain of the spine's own code over another spine's code: {gain:.1%}"
          + ("   <-- the decoder IGNORES the code" if gain < 0.02 else "   (the code carries shape information)"))
    for name in TYPES:
        g = (mean_err("shuffled", name) - mean_err("real", name)) / max(mean_err("shuffled", name), 1e-12)
        print(f"   gain on {name:8s}: {g:.1%}")

    h_all = torch.cat(feats)
    h_std = h_all.std(dim=0)
    print(f"\nencoder pooled features h [{h_all.shape[1]}]: std between spines mean {float(h_std.mean()):.5f}, "
          f"mean |h| {float(h_all.abs().mean()):.4f}, ratio {float(h_std.mean() / h_all.abs().mean().clamp_min(1e-12)):.4f}"
          "  (ratio ~0: the encoder gives the same vector for every spine)")
    std = mu_all.std(dim=0)
    print(f"\nencoder output between spines: per-dimension std of mu: mean {float(std.mean()):.4f}, max {float(std.max()):.4f}; "
          f"dims with std > 0.1: {int((std > 0.1).sum())} / {len(std)}; mean |mu| {float(mu_all.abs().mean()):.4f}")
    centered = mu_all - mu_all.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    print("PCA of mu, explained variance of the top 5 components:", [f"{float(v):.1%}" for v in var[:5]])
    size = torch.cat(sizes)
    scores = centered @ vh[:5].T
    corr = [float(np.corrcoef(scores[:, k].numpy(), size[:, ax].numpy())[0, 1]) for k in range(min(3, scores.shape[1])) for ax in range(3)]
    print("correlation of the top-3 PC scores with the spine extent along x, y, z:")
    for k in range(min(3, scores.shape[1])):
        print(f"   PC{k + 1}: " + ", ".join(f"{c:+.2f}" for c in corr[3 * k:3 * k + 3]))
    print("\nReading: gain ~0 and tiny std of mu = the encoder output does not vary between spines or the decoder ignores it.")


if __name__ == "__main__":
    main()
