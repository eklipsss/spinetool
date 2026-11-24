"""Train the VN + FoldingNet autoencoder on spine point clouds (comparison model, src/.../spine_foldnet).

    python scripts/neuron_model/train_vn_foldingnet.py --config configs/neuron-model/vn_foldingnet/frame.yaml \
        "data.manifests=['cache/minnie65_dev/manifest.parquet']" data.splits=data/processed/neuron-model/splits/splits_dev_100n.parquet

The same training spines as MoGen A and the VAE (cache of splits_dev_100n), no random rotations. Resume:

    python scripts/neuron_model/train_vn_foldingnet.py --resume runs/training/neuron-model/<run>

Then codes -> flow -> meshes: scripts/neuron_model/generate_vn_foldingnet.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  (first on Windows: see generate_tp.py)

from src.neuron_model.spine_generation.data.datasets import make_loader  # noqa: E402
from src.neuron_model.spine_generation.data.index import load_spine_index  # noqa: E402
from src.neuron_model.spine_generation.data.splits import load_splits  # noqa: E402
from src.neuron_model.spine_generation.experiment.artifacts import write_json  # noqa: E402
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path  # noqa: E402
from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training  # noqa: E402
from src.neuron_model.spine_generation.experiment.run import create_run, open_run  # noqa: E402
from src.neuron_model.spine_generation.experiment.seed import seed_everything  # noqa: E402
from src.neuron_model.spine_generation.experiment.training import count_parameters, get_device, make_lr_scheduler  # noqa: E402
from src.neuron_model.spine_generation.models.spine_foldnet.train import build_model, make_loss_fn, make_sample_fn, make_validate_fn  # noqa: E402
from src.neuron_model.spine_generation.models.spine_mogen.train import make_optimizer, point_cloud_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("overrides", nargs="*")
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
    index = load_spine_index([resolve_path(p) for p in data_cfg["manifests"]], path_mode=data_cfg.get("path_mode", "relative"),
                             check_files=tuple(data_cfg.get("check_files", (f"pointcloud_{data_cfg['n_points']}_path",))))
    index = index.with_split(load_splits(resolve_path(data_cfg["splits"])), drop_unassigned=bool(data_cfg.get("drop_unassigned", False)))
    train_ds, val_ds = point_cloud_datasets(index, data_cfg)
    if len(train_ds) == 0:
        parser.error(f"no training spines: check data.splits and that the manifest has pointcloud_{data_cfg['n_points']}_path files "
                     f"(the training cache holds only the size it was built with; use data.subsample_points for fewer points)")
    model = build_model(cfg["model"])
    if not args.resume:
        cfg["data"]["dataset_version"] = index.dataset_version
        run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_train": len(train_ds), "n_val": len(val_ds or [])})
    write_json(run.root / "model_config.json", model.cfg.to_dict())
    coord_scale = float(data_cfg["coord_scale"])
    print(f"run {run.root}\ndevice {device}; mode {model.cfg.mode}, grid {model.cfg.grid}; params {count_parameters(model):,}; "
          f"train {len(train_ds)} / val {len(val_ds or [])}", flush=True)
    optimizer = make_optimizer(model, cfg["optimizer"])
    train_cfg = cfg["train"]
    scheduler = make_lr_scheduler(optimizer, cfg["optimizer"].get("schedule"), int(train_cfg["max_steps"]))
    loader = make_loader(train_ds, batch_size=int(data_cfg["batch_size"]), shuffle=True, seed=int(cfg["seed"]),
                         num_workers=int(data_cfg.get("num_workers", 0)), drop_last=True)
    loop = LoopConfig.from_mapping({**train_cfg, "precision": cfg.get("precision")})
    result = run_training(
        run, loop, model, optimizer, loader, make_loss_fn(device, coord_scale), device=device, scheduler=scheduler,
        validate_fn=make_validate_fn(val_ds, device, batch_size=int(data_cfg["batch_size"]), seed=int(cfg["seed"]), coord_scale=coord_scale,
                                     max_batches=train_cfg.get("n_val_batches")),
        sample_fn=make_sample_fn(run.samples, val_ds, device, n_samples=int(train_cfg.get("n_samples", 8)), coord_scale=coord_scale),
        resume=bool(args.resume),
    )
    print("finished", result)


if __name__ == "__main__":
    main()
