"""Generate spines with MoGen: point clouds -> normals -> Screened Poisson -> meshes (+ validity summary).

Zero-shot with the exported pretrained weights (plan stage 4.4 - checks the
coordinate adapter, ODE stability and domain shift; NOT a spine model):

    python scripts/neuron_model/generate_mogen.py --weights data/external/mogen/mouse_mixed_750000_weights.npz \
        --config configs/neuron-model/mogen/pretrained_finetune.yaml --n-samples 16 --seed 1

From a training run (EMA weights of best_val_loss by default):

    python scripts/neuron_model/generate_mogen.py --run runs/training/neuron-model/<experiment_id> --n-samples 1000 --seed 1

Outputs go to ``<run>/meshes/<tag>/`` (or ``runs/analysis/neuron-model/<id>/meshes`` for --weights).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.neuron_model.spine_generation.experiment.artifacts import SampleWriter, write_json
from src.neuron_model.spine_generation.experiment.checkpoint import CheckpointManager
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.run import create_run, open_run
from src.neuron_model.spine_generation.experiment.training import get_device
from src.neuron_model.spine_generation.models.spine_mogen.infer import generate_point_clouds, pointcloud_to_mesh, write_mogen_sample
from src.neuron_model.spine_generation.models.spine_mogen.pointinfinity import PointInfinity, PointInfinityConfig
from src.neuron_model.spine_generation.models.spine_mogen.train import build_model
from src.neuron_model.spine_generation.reconstruction.mesh_validation import aggregate_validity
from src.neuron_model.spine_generation.reconstruction.screened_poisson import PoissonParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="training run directory")
    source.add_argument("--weights", type=Path, help="exported pretrained .npz (zero-shot)")
    parser.add_argument("--config", type=Path, help="config for --weights (data/flow/reconstruction sections)")
    parser.add_argument("--checkpoint", default="best_val_loss", help="checkpoint tag inside --run")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, help="override flow.inference_steps")
    parser.add_argument("--no-mesh", action="store_true", help="only point clouds")
    args = parser.parse_args()

    if args.run:
        run = open_run(args.run)
        cfg = run.load_config()
        model = PointInfinity(PointInfinityConfig(**json.loads((run.root / "model_config.json").read_text())))
        payload = CheckpointManager(run.checkpoints).load(args.checkpoint, restore_rng=False)
        model.load_state_dict(payload["state"]["ema"]["model"] if "ema" in payload["state"] else payload["state"]["model"])
        checkpoint_id = f"{run.experiment_id}:{args.checkpoint}:step{payload['step']}"
        out_root = run.meshes
    else:
        if not args.config:
            parser.error("--weights needs --config")
        cfg = load_config(args.config)
        model = build_model({**cfg["model"], "init": "pretrained", "pretrained_weights": str(args.weights)})
        checkpoint_id = f"pretrained:{args.weights.name}"
        run = create_run(cfg, name="mogen-zero-shot", runs_root=REPO_ROOT / "runs" / "analysis" / "neuron-model", extra_info={"weights": str(args.weights)})
        out_root = run.meshes

    device = get_device(cfg.get("device", "auto"))
    model.to(device)
    flow = cfg["flow"]
    steps = int(args.steps or flow["inference_steps"])
    inference_config = {
        "solver": flow.get("inference_solver", "midpoint"),
        "n_steps": steps,
        "schedule": flow.get("inference_schedule", "mogen_cosine"),
        "coord_scale": float(cfg["data"]["coord_scale"]),
        "n_points": int(cfg["data"]["n_points"]),
        "generation_seed": args.seed,
        "checkpoint_id": checkpoint_id,
        "cond": "zeros (unconditional)",
    }
    points = generate_point_clouds(
        model,
        n_samples=args.n_samples,
        n_points=inference_config["n_points"],
        coord_scale=inference_config["coord_scale"],
        n_steps=steps,
        schedule=inference_config["schedule"],
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
    )
    rec_cfg = load_config(resolve_path(cfg["reconstruction"]["poisson_config"]))
    poisson = PoissonParams.from_mapping(rec_cfg["poisson"])
    k_normal = int(rec_cfg["normals"]["k_normal"])
    writer = SampleWriter(out_root)
    tag = f"seed_{args.seed}"
    raw_reports, post_reports = [], []
    for i, cloud in enumerate(points):
        reconstruction = None if args.no_mesh else pointcloud_to_mesh(cloud, k_normal=k_normal, poisson=poisson, postprocess_cfg=rec_cfg["postprocess"])
        write_mogen_sample(writer, tag, i, cloud, reconstruction, seed=args.seed, checkpoint_id=checkpoint_id, inference_config=inference_config, poisson=poisson, k_normal=k_normal)
        if reconstruction is not None:
            raw_reports.append(reconstruction["validation_raw"])
            post_reports.append(reconstruction["validation_postprocessed"])
    summary = {
        "n_samples": args.n_samples,
        "inference_config": inference_config,
        "poisson_frozen": bool(rec_cfg.get("frozen")),
        "validity_raw": aggregate_validity(raw_reports),
        "validity_postprocessed": aggregate_validity(post_reports),
        "extent_mean_physical": float((points.max(axis=1) - points.min(axis=1)).mean()),
    }
    write_json(out_root / tag / "generation_summary.json", summary)
    if raw_reports:
        pd.DataFrame(raw_reports).to_parquet(out_root / tag / "validation_raw.parquet", index=False)
    print(summary)


if __name__ == "__main__":
    main()
