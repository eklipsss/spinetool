"""Train MoGen on spine point clouds (fine-tune from mouse_mixed, or from scratch).

Windows workstation (RTX 4090):

    python scripts/neuron_model/train_mogen.py --config configs/neuron-model/mogen/pretrained_finetune.yaml \
        "data.manifests=['O:/Datasets/Minnie65/preprocessed/minnie65/manifest.parquet']" data.num_workers=8

Resume an interrupted run (same config is read back from the run directory):

    python scripts/neuron_model/train_mogen.py --resume runs/training/neuron-model/<experiment_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.data.datasets import make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
from src.neuron_model.spine_generation.experiment.run import create_run, open_run
from src.neuron_model.spine_generation.experiment.seed import seed_everything
from src.neuron_model.spine_generation.experiment.training import count_parameters, get_device
from src.neuron_model.spine_generation.models.spine_mogen.train import (
    build_model,
    make_loss_fn,
    make_optimizer,
    make_sample_fn,
    make_validate_fn,
    point_cloud_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path, help="existing run directory to continue")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides (new runs only)")
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("give exactly one of --config / --resume")

    if args.resume:
        run = open_run(args.resume)
        cfg = run.load_config()
    else:
        cfg = load_config(args.config, args.overrides)
    data_cfg = cfg["data"]
    if not data_cfg.get("manifests"):
        parser.error("set data.manifests=[...]")
    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    index = load_spine_index([resolve_path(p) for p in data_cfg["manifests"]], path_mode=data_cfg.get("path_mode", "relative"), check_files=(f"pointcloud_{data_cfg['n_points']}_path",))
    index = index.with_split(load_splits(resolve_path(data_cfg["splits"])))
    train_ds, val_ds = point_cloud_datasets(index, data_cfg)
    if not args.resume:
        cfg["data"]["dataset_version"] = index.dataset_version
        run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_train": len(train_ds), "n_val": len(val_ds or [])})

    model = build_model(cfg["model"])
    write_json(run.root / "model_config.json", model.cfg.to_dict())  # actual architecture (incl. cond_dim from the checkpoint)
    print(f"run {run.root}\ndevice {device}; params {count_parameters(model):,}; train {len(train_ds)} / val {len(val_ds or [])}")
    optimizer = make_optimizer(model, cfg["optimizer"])
    train_cfg = cfg["train"]
    loader = make_loader(train_ds, batch_size=int(data_cfg["batch_size"]), shuffle=True, seed=int(cfg["seed"]), num_workers=int(data_cfg.get("num_workers", 0)), drop_last=True)
    loop = LoopConfig.from_mapping({**train_cfg, "precision": cfg.get("precision")})
    result = run_training(
        run,
        loop,
        model,
        optimizer,
        loader,
        make_loss_fn(cfg["flow"], device),
        device=device,
        validate_fn=make_validate_fn(val_ds, cfg["flow"], device, batch_size=int(data_cfg["batch_size"]), seed=int(cfg["seed"]), max_batches=train_cfg.get("n_val_batches")),
        sample_fn=make_sample_fn(run.samples, cfg["flow"], data_cfg, device, n_samples=int(train_cfg.get("n_samples", 4)), seed=int(cfg["seed"])),
        resume=bool(args.resume),
    )
    print("finished", result)


if __name__ == "__main__":
    main()
