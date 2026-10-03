"""Export a MoGen (JAX/Flax/Orbax) checkpoint to plain .npz + reference outputs for the PyTorch parity test.

Runs in the SEPARATE JAX environment (plan stage 4.1-4.2), NOT in the
`neuron-model` env - see docs/neuron-model/s-module-implementation-plan.md
"Этап 4". UNTESTED here (no JAX on the dev laptop): verify on the first run.

    python scripts/neuron_model/export_mogen_weights.py \
        --checkpoint <dir>/mouse_mixed/best_checkpoints/750000/train_state \
        --train-config <dir>/mouse_mixed/config.json \
        --connectomics-repo <path to a clone of google-research/connectomics> \
        --out data/external/mogen/mouse_mixed_750000

Writes:
    <out>_weights.npz    keys "params/<flax path>" and "ema_params/<flax path>"
    <out>_reference.npz  fixed inputs (coord, t, cond) + official model outputs
                         (velocity, and a short midpoint trajectory from fixed noise)
    <out>_export.json    checkpoint path, train config, shapes, library versions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from flatten(value, f"{prefix}/{key}" if prefix else str(key))
    else:
        yield prefix, np.asarray(tree)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help=".../best_checkpoints/<step>/train_state")
    parser.add_argument("--train-config", type=Path, required=True, help="the checkpoint's config.json")
    parser.add_argument("--connectomics-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output path prefix")
    parser.add_argument("--n-points", type=int, default=8192)
    parser.add_argument("--n-ref", type=int, default=2)
    parser.add_argument("--ref-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.connectomics_repo))
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from connectomics.mogen.flow_matching import schedules
    from connectomics.mogen.models import pointinfinity

    train_cfg = json.loads(args.train_config.read_text())
    restored = ocp.PyTreeCheckpointer().restore(str(args.checkpoint.resolve()))
    arrays = {}
    for collection in ("params", "ema_params"):
        if collection not in restored:
            raise KeyError(f"'{collection}' not in checkpoint (has {list(restored)})")
        for key, value in flatten(restored[collection]):
            arrays[f"{collection}/{key}"] = value
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out) + "_weights.npz", **arrays)

    ema = restored["ema_params"]
    cond_dim = int(np.asarray(ema["Dense_0"]["kernel"]).shape[0]) if "Dense_0" in ema else 0
    config = pointinfinity.PointInfinityConfig(
        point_dim=train_cfg["pfty_point_dim"],
        latent_dim=train_cfg["pfty_latent_dim"],
        n_latents=train_cfg["pfty_n_latents"],
        n_blocks=train_cfg["pfty_n_blocks"],
        n_subblocks=train_cfg["pfty_n_subblocks"],
        n_heads=train_cfg["pfty_n_heads"],
        k_nn=train_cfg["pfty_k_nn"],
    )
    model = pointinfinity.PointInfinity(config)

    rng = np.random.default_rng(args.seed)
    coord = rng.normal(size=(args.n_ref, args.n_points, 3)).astype(np.float32)
    t = np.linspace(0.1, 0.9, args.n_ref).astype(np.float32)
    cond = np.zeros((args.n_ref, cond_dim), dtype=np.float32) if cond_dim else None

    def apply(x, tt):
        return model.apply({"params": ema}, jnp.asarray(x), t=jnp.asarray(tt), cond=None if cond is None else jnp.asarray(cond))

    velocity = np.asarray(apply(coord, t))

    # short midpoint trajectory from fixed noise, exactly as flow_matching.utils.generate_step_midpoint
    x = coord.copy()
    timesteps = np.asarray(schedules.t_schedule(jnp.linspace(0.0, 1.0, args.ref_steps + 1), train_cfg["sample_schedule"]))
    for i in range(args.ref_steps):
        t0, t1 = timesteps[i], timesteps[i + 1]
        dt = t1 - t0
        v0 = np.asarray(apply(x, np.full(args.n_ref, t0, np.float32)))
        x_mid = x + v0 * dt / 2
        v_mid = np.asarray(apply(x_mid, np.full(args.n_ref, t0 + dt / 2, np.float32)))
        x = x + dt * v_mid

    np.savez(
        str(args.out) + "_reference.npz",
        coord=coord,
        t=t,
        cond=np.zeros((args.n_ref, 0), np.float32) if cond is None else cond,
        velocity=velocity,
        trajectory_end=x.astype(np.float32),
        timesteps=timesteps.astype(np.float32),
        sample_schedule=np.asarray(train_cfg["sample_schedule"]),
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "train_config": train_cfg,
        "cond_dim": cond_dim,
        "n_arrays": len(arrays),
        "jax_version": jax.__version__,
        "orbax_version": getattr(ocp, "__version__", None),
    }
    Path(str(args.out) + "_export.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {len(arrays)} arrays (cond_dim={cond_dim}) -> {args.out}_weights.npz / _reference.npz")


if __name__ == "__main__":
    main()
