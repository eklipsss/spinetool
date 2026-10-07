"""Train the PointNeXt -> VAE -> SIREN-SDF model (model C).

Windows workstation (RTX 4090):

    python scripts/neuron_model/train_vae.py --config configs/neuron-model/vae/baseline.yaml \
        "data.manifests=['O:/Datasets/Minnie65/preprocessed/manifest.parquet']" data.num_workers=8

Resume: ``python scripts/neuron_model/train_vae.py --resume runs/training/neuron-model/<experiment_id>``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.neuron_model.spine_generation.data.bbox import FrozenBBox
from src.neuron_model.spine_generation.data.datasets import make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
from src.neuron_model.spine_generation.experiment.run import create_run, open_run
from src.neuron_model.spine_generation.experiment.seed import seed_everything
from src.neuron_model.spine_generation.experiment.training import count_parameters, get_device
from src.neuron_model.spine_generation.models.spine_vae.model import SpineVAE, SpineVAEConfig
from src.neuron_model.spine_generation.models.spine_vae.train import loss_weights, make_loss_fn, make_sample_fn, make_validate_fn, sdf_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("give exactly one of --config / --resume")
    run = open_run(args.resume) if args.resume else None
    cfg = run.load_config() if run else load_config(args.config, args.overrides)
    data_cfg = cfg["data"]
    if not data_cfg.get("manifests"):
        parser.error("set data.manifests=[...]")
    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    # see train_mogen.py: override with "data.check_files=[]" to skip the per-spine exists()
    # check when nothing is being staged to the NAS right now.
    default_check_files = (f"pointcloud_{data_cfg['n_points']}_path", "sdf_samples_path")
    index = load_spine_index(
        [resolve_path(p) for p in data_cfg["manifests"]],
        path_mode=data_cfg.get("path_mode", "relative"),
        check_files=tuple(data_cfg.get("check_files", default_check_files)),
    )
    index = index.with_split(load_splits(resolve_path(data_cfg["splits"])))
    train_ds, val_ds = sdf_datasets(index, data_cfg)
    bbox = FrozenBBox.load(resolve_path(data_cfg["bbox"]))
    if run is None:
        cfg["data"]["dataset_version"] = index.dataset_version
        run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_train": len(train_ds), "n_val": len(val_ds or []), "bbox": bbox.__dict__})

    model_cfg = SpineVAEConfig.from_mapping(cfg["model"])
    model = SpineVAE(model_cfg)
    write_json(run.root / "model_config.json", model_cfg.to_dict())
    print(f"run {run.root}\ndevice {device}; params {count_parameters(model):,}; train {len(train_ds)} / val {len(val_ds or [])}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)))
    weights = loss_weights(cfg["loss"])
    train_cfg = cfg["train"]
    sampling = cfg["sampling"]
    mc_cfg = load_config(resolve_path(cfg["reconstruction"]["marching_cubes_config"]))
    loader = make_loader(train_ds, batch_size=int(data_cfg["batch_size"]), shuffle=True, seed=int(cfg["seed"]), num_workers=int(data_cfg.get("num_workers", 0)), drop_last=True)
    result = run_training(
        run,
        LoopConfig.from_mapping({**train_cfg, "precision": cfg.get("precision")}),
        model,
        optimizer,
        loader,
        make_loss_fn(weights, device),
        device=device,
        validate_fn=make_validate_fn(val_ds, weights, device, batch_size=int(data_cfg["batch_size"]), seed=int(cfg["seed"]), max_batches=train_cfg.get("n_val_batches")),
        sample_fn=make_sample_fn(
            run.meshes,
            val_ds,
            bbox=(bbox.low, bbox.high),
            coord_scale=float(data_cfg["coord_scale"]),
            resolution=int(sampling["resolution"]),
            n_reconstructions=int(sampling["n_reconstructions"]),
            n_prior=int(sampling["n_prior"]),
            seed=int(cfg["seed"]),
            device=device,
            chunk_size=int(sampling["chunk_size"]),
            postprocess_cfg=mc_cfg["postprocess"],
            check_self_intersections=bool(sampling.get("check_self_intersections", False)),
        ),
        resume=bool(args.resume),
    )
    print("finished", result)


if __name__ == "__main__":
    main()
