"""Fine-tune the VAE decoder jointly with per-spine codes (auto-decoder, DeepSDF style) -> a NEW run readable as a VAE run.

Why (2026-11-08, chain L): a flow prior on codes fitted with the decoder frozen does not give necks in generation (intact
meshes as convex as with encoder codes, 27-42 % broken at the neck), although the same codes reconstruct their own spines
with the necks. The decoder must learn necks on the codes the prior is fitted to. See models/spine_vae/autodecoder.py.

    python scripts/neuron_model/finetune_vae_autodecoder.py --base-run runs/training/neuron-model/2026-10-19_003408_vae-dev-reg0-cos \
        --codes-dir latent_prior_zopt_a1 --name vae-autodecoder-n0 --steps 20000 --code-noise 0 --anchor-weight 1 --num-workers 8

The new run (``runs/training/neuron-model/<date>_<name>``) holds the base model with the fine-tuned decoder (state
``model`` of ``checkpoints/latest.pt``; encoder unchanged), so load_vae_run / generate_vae / fit_vae_codes /
fit_latent_prior / compare_vae_decoders work on it unchanged; ``<run>/<out-name>/codes_train.npz`` holds the jointly
trained codes with the tag fit_latent_prior.py expects (same spines / count / seed as the base codes); the validation
codes are fitted afterwards with the decoder frozen:

    python scripts/neuron_model/fit_vae_codes.py --run <new run> --splits-to-fit val --n-val 1500 --anchor-weight 1 --out-name latent_prior_ad
    python scripts/neuron_model/fit_latent_prior.py --run <new run> --variants 1 --out-name latent_prior_ad --n-train 15000 --n-val 1500

Logs ``<run>/logs/autodecoder.jsonl`` (training terms; SDF loss at the clean code and at a code perturbed by its posterior
sigma, nm; ((c - mu)/sigma)^2; code norm) and ``<run>/autodecoder_curves.png``.
"""

from __future__ import annotations

import argparse
import copy
import json
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


def write_curves(path: Path, history) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(steps, [h["sdf_clean_nm"] for h in history], label="clean code")
    axes[0].plot(steps, [h["sdf_sigma_perturbed_nm"] for h in history], label="code + sigma * eps")
    axes[0].set_ylabel("SDF L1, nm (training batch)")
    axes[0].legend()
    axes[1].plot(steps, [h["anchor_dist2_median"] for h in history])
    axes[1].set_ylabel("median ((c - mu) / sigma)^2")
    axes[2].plot(steps, [h.get("eikonal_loss", np.nan) for h in history], label="eikonal")
    axes[2].plot(steps, [h.get("normal_loss", np.nan) for h in history], label="normal")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    from src.neuron_model.spine_generation.data.bbox import FrozenBBox
    from src.neuron_model.spine_generation.experiment.artifacts import write_json
    from src.neuron_model.spine_generation.experiment.checkpoint import CheckpointManager, state_dicts
    from src.neuron_model.spine_generation.experiment.config import resolve_path
    from src.neuron_model.spine_generation.experiment.run import create_run
    from src.neuron_model.spine_generation.models.spine_vae.autodecoder import finetune_autodecoder, query_dataset
    from src.neuron_model.spine_generation.models.spine_vae.latent_prior import encode_spines, load_split_index, load_vae_run
    from src.neuron_model.spine_generation.models.spine_vae.train import loss_weights

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--codes-dir", default="latent_prior_zopt_a1", help="folder of the base run with codes_train.npz (fit_vae_codes.py)")
    ap.add_argument("--name", default="vae-autodecoder")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5, help="decoder lr (the base run ended at 1e-5)")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--lr-codes", type=float, default=5e-3)
    ap.add_argument("--anchor-weight", type=float, default=1.0, help="nm * mean_d(((c - mu) / sigma)^2), as fit_vae_codes.py")
    ap.add_argument("--code-noise", type=float, default=0.0, help="z = c + noise * sigma * eps in training (0 = DeepSDF, 1 = the VAE's neighbourhood)")
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--out-name", default="latent_prior_ad", help="codes folder inside the NEW run")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device")
    args = ap.parse_args()

    model, base_run, base_cfg, base_ckpt, device = load_vae_run(args.base_run, args.checkpoint, device=args.device)
    data = base_cfg["data"]
    coord_scale = float(data["coord_scale"])
    base_codes = np.load(base_run.root / args.codes_dir / "codes_train.npz", allow_pickle=False)
    keys = [str(k) for k in base_codes["keys"]]
    codes0 = np.asarray(base_codes["mu"], dtype=np.float64)[:, 0]
    sigma = np.exp(0.5 * np.asarray(base_codes["logvar"], dtype=np.float64)[:, 0])
    base_tag = str(base_codes["tag"]).split("|")  # checkpoint | split | n | seed | variants
    if base_tag[0] != base_ckpt:
        print(f"[warn] codes were fitted with {base_tag[0]}, the base run is now {base_ckpt}", flush=True)
    index = load_split_index(base_cfg, "train", 10**9, int(base_tag[3]))
    index = index.subset(keys)
    if len(index) != len(keys):
        raise SystemExit(f"{len(keys) - len(index)} coded spines are not in the run's train split")
    if "mu_encoder" in base_codes.files:
        mu = np.asarray(base_codes["mu_encoder"], dtype=np.float64)
    else:  # codes written before 2026-11-08 do not store the encoder's mean: re-encode (FPS start point -> tiny noise)
        print(f"[ad] re-encoding {len(keys)} training spines for the anchor", flush=True)
        torch.manual_seed(args.seed)
        enc = encode_spines(model, index, data, variants=(1,), device=device, batch_size=16)
        pos = {k: i for i, k in enumerate(enc["keys"])}
        mu = enc["mu"][[pos[k] for k in keys], 0]

    far_bbox = None
    if int(data.get("far_queries", 0)) > 0:
        bbox = FrozenBBox.load(resolve_path(data["bbox"]))
        far_bbox = (bbox.low, bbox.high)
    dataset = query_dataset(index, data, far_bbox=far_bbox)

    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = args.name
    cfg["autodecoder"] = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                          "base_checkpoint": base_ckpt, "base_codes_tag": str(base_codes["tag"]), "n_codes": len(keys)}
    run = create_run(cfg, runs_root=base_run.root.parent, extra_info={"n_train": len(keys), "base_run": str(base_run.root)})
    (run.root / "model_config.json").write_text((base_run.root / "model_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    manager = CheckpointManager(run.checkpoints, keep_every=args.checkpoint_every)
    out_dir = run.root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = run.logs / "autodecoder.jsonl"
    print(f"run {run.root}\nbase {base_ckpt}; {len(keys)} training spines; device {device}; code noise {args.code_noise:g} sigma; "
          f"anchor {args.anchor_weight:g} nm; ((c0 - mu)/sigma)^2 median {np.median(((codes0 - mu) / sigma) ** 2):.2f}", flush=True)

    def log_fn(row):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if row["step"] % (10 * 100) == 0 or row["step"] == args.steps:
            print(f"[ad] step {row['step']}: SDF clean {row['sdf_clean_nm']:.2f} nm, +sigma {row['sdf_sigma_perturbed_nm']:.2f} nm; "
                  f"eik {row.get('eikonal_loss', float('nan')):.4f} normal {row.get('normal_loss', float('nan')):.4f}; "
                  f"((c-mu)/sigma)^2 {row['anchor_dist2_median']:.2f}; |c| {row['code_norm_mean']:.2f}; {row['elapsed_s']:.0f} s", flush=True)

    def save_codes(step, table):
        manager.save(step, state_dicts(model=model), {})
        ckpt = f"{run.experiment_id}:latest:step{step}"
        tag = "|".join([ckpt, *base_tag[1:]])  # fit_latent_prior.get_codes: the same spines / count / seed, the new checkpoint
        np.savez(out_dir / "codes_train.npz", mu=table[:, None, :].astype(np.float64), logvar=base_codes["logvar"],
                 keys=np.array(keys), tag=np.array(tag), mu_encoder=mu)

    started = time.time()
    codes, history = finetune_autodecoder(
        model, codes0, mu, sigma, keys, dataset, weights=loss_weights(base_cfg.get("loss", {})), coord_scale=coord_scale, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, lr_codes=args.lr_codes, min_lr_ratio=args.min_lr_ratio, warmup_steps=args.warmup_steps,
        anchor_weight_nm=args.anchor_weight, code_noise=args.code_noise, clip_norm=args.clip_norm, seed=args.seed, device=device,
        precision=base_cfg.get("precision"), num_workers=args.num_workers, log_fn=log_fn, checkpoint_every=args.checkpoint_every,
        checkpoint_fn=save_codes)
    write_curves(run.root / "autodecoder_curves.png", history)
    d2 = (((codes - mu) / sigma) ** 2).mean(axis=1)
    summary = {"steps": args.steps, "seconds": time.time() - started, "first": history[0] if history else None, "last": history[-1] if history else None,
               "code_shift_median": float(np.median(np.linalg.norm(codes - codes0, axis=1))),
               "posterior_dist2_median": float(np.median(d2)), "code_norm_median": float(np.median(np.linalg.norm(codes, axis=1)))}
    write_json(run.root / "autodecoder_summary.json", summary)
    print(f"[ad] codes moved {summary['code_shift_median']:.3f} (median |c - c0|); ((c - mu)/sigma)^2 median {summary['posterior_dist2_median']:.2f}; "
          f"|c| median {summary['code_norm_median']:.2f}")
    print(f"-> {run.root}  ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
