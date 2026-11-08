"""Directly optimised VAE codes (decoder frozen) for train / val spines, as input of fit_latent_prior.py.

Why (2026-11-07, decoder study 2026-11-07_195048): the reconstructions of encoder codes (z = mu) fill the spine necks
(ConvexHullRatio -0.59 SD, mean |bias| 0.226, AUC 0.95 against the real mesh through the same exact-SDF + MC chain), but the
SAME frozen decoder with the code of each spine fitted directly to its SDF samples has almost no bias (0.070, below the 0.161
of the real mesh itself through MC; ConvexHullRatio +0.06; Chamfer-L1 18 nm instead of 40). The bottleneck is the encoder
(amortisation gap), not the decoder. A generator does not need the encoder: prior -> code -> decoder. So the prior is fitted
on directly optimised codes instead of encoder codes.

    python scripts/neuron_model/fit_vae_codes.py --run runs/training/neuron-model/2026-10-19_003408_vae-dev-reg0-cos \
        --n-train 15000 --n-val 1500 --steps 500
    python scripts/neuron_model/fit_latent_prior.py --run <run> --variants 1 --out-name latent_prior_zopt --n-train 15000 --n-val 1500
    python scripts/neuron_model/generate_vae.py --run <run> --prior learned --latent-prior <run>/latent_prior_zopt/flow_prior.pt ...

Writes ``<run>/<out-name>/codes_{train,val}.npz`` (mu [n, 1, D] = optimised codes, logvar = the encoder's, keys, tag in the
format of fit_latent_prior.py, so it reuses them; same spine selection: load_split_index(cfg, split, n, seed)) and
``fit_info_{split}.parquet`` (SDF loss before / after per spine, nm; distance of the code from mu).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402,F401  (torch first: Windows DLL order)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    from src.neuron_model.spine_generation.data.bbox import FrozenBBox
    from src.neuron_model.spine_generation.experiment.config import resolve_path
    from src.neuron_model.spine_generation.models.spine_vae.decoder_study import optimize_latents
    from src.neuron_model.spine_generation.models.spine_vae.latent_prior import encode_spines, load_split_index, load_vae_run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--splits-to-fit", nargs="+", default=["train", "val"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32, help="spines optimised together (GPU memory ~ batch x SDF samples)")
    ap.add_argument("--seed", type=int, default=1, help="spine selection, as fit_latent_prior.py --seed")
    ap.add_argument("--out-name", default="latent_prior_zopt")
    ap.add_argument("--anchor-weight", type=float, default=0.0,
                    help="MAP penalty in nm * mean_d(((z - mu) / sigma)^2) (0 = none, as chain K; choose it with "
                         "compare_vae_decoders.py --opt-anchor-weights on dev val)")
    ap.add_argument("--anchor-scale", default="posterior", choices=("posterior", "unit"))
    ap.add_argument("--device")
    args = ap.parse_args()

    model, run, cfg, checkpoint_id, device = load_vae_run(args.run, args.checkpoint, device=args.device)
    data, loss_cfg = cfg["data"], cfg.get("loss", {})
    coord_scale = float(data["coord_scale"])
    clamp = None if loss_cfg.get("sdf_clamp") is None else float(loss_cfg["sdf_clamp"])
    far = int(data.get("far_queries", 0)) > 0 and float(loss_cfg.get("far", 0.0)) > 0
    box = None
    if far:  # the far-field hinge lives in the frozen bbox (only runs trained with it)
        bbox = FrozenBBox.load(resolve_path(data["bbox"]))
        box = (tuple(bbox.low), tuple(bbox.high))
    out_dir = run.root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for split in args.splits_to_fit:
        n = args.n_train if split == "train" else args.n_val
        index = load_split_index(cfg, split, n, args.seed)
        codes = encode_spines(model, index, data, variants=(1,), device=device, batch_size=16)
        keys, mu = list(codes["keys"]), codes["mu"][:, 0]
        paths = index.frame.set_index("spine_key").loc[keys, "sdf_samples_path"].tolist()
        sigma = np.exp(0.5 * np.asarray(codes["logvar"][:, 0], dtype=float))
        z_all, info_all = np.empty_like(mu), []
        for b in range(0, len(keys), args.batch_size):
            sl = slice(b, b + args.batch_size)
            z, info = optimize_latents(model, mu[sl], paths[sl], coord_scale=coord_scale, sdf_clamp=clamp, far_bbox=box,
                                       steps=args.steps, lr=args.lr, far_clamp_nm=float(data.get("far_clamp_nm", 300.0)),
                                       seed=args.seed + b, device=device, anchor_weight_nm=args.anchor_weight,
                                       anchor_scale=sigma[sl] if args.anchor_scale == "posterior" else None)
            z_all[sl] = z
            info_all += info
            done = min(b + args.batch_size, len(keys))
            if (b // args.batch_size) % 20 == 0 or done == len(keys):
                el = time.time() - started
                print(f"[codes] {split} {done}/{len(keys)}: SDF loss {np.mean([i['loss_start'] for i in info_all]):.2f} -> "
                      f"{np.mean([i['loss_end'] for i in info_all]):.2f} nm ({el:.0f} s)", flush=True)
        # the tag format of fit_latent_prior.get_codes (variants = [1]): it reuses these codes instead of encoding
        tag = f"{checkpoint_id}|{split}|{n}|{args.seed}|[1]"
        np.savez(out_dir / f"codes_{split}.npz", mu=z_all[:, None, :].astype(np.float64), logvar=codes["logvar"],
                 keys=np.array(keys), tag=np.array(tag), mu_encoder=mu.astype(np.float64))
        shift = np.linalg.norm(z_all - mu, axis=1)
        spread = float(np.median(np.linalg.norm(mu - mu.mean(axis=0), axis=1)))
        pd.DataFrame({"spine_key": keys, "loss_start_nm": [i["loss_start"] for i in info_all], "loss_end_nm": [i["loss_end"] for i in info_all],
                      "shift_from_mu": shift, "posterior_dist2": [i["anchor_dist2"] for i in info_all]}).to_parquet(out_dir / f"fit_info_{split}.parquet", index=False)
        print(f"[codes] {split}: {len(keys)} codes -> {out_dir / f'codes_{split}.npz'}; |z - mu| median {np.median(shift):.3f} "
              f"(median |mu - mean| {spread:.3f}); |z| median {np.median(np.linalg.norm(z_all, axis=1)):.3f} "
              f"vs |mu| {np.median(np.linalg.norm(mu, axis=1)):.3f}; ((z - mu)/sigma)^2 median "
              f"{np.median([i['anchor_dist2'] for i in info_all]):.2f} (anchor weight {args.anchor_weight:g} nm)", flush=True)
    print(f"-> {out_dir}  ({checkpoint_id}, {time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
