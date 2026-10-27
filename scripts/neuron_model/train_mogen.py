"""Train MoGen on spine point clouds (fine-tune from mouse_mixed, or from scratch).

Windows workstation (RTX 4090):

    python scripts/neuron_model/train_mogen.py --config configs/neuron-model/mogen/pretrained_finetune.yaml \
        "data.manifests=['O:/Datasets/Minnie65/preprocessed/manifest.parquet']" data.num_workers=8

Resume an interrupted run (same config is read back from the run directory):

    python scripts/neuron_model/train_mogen.py --resume runs/training/neuron-model/<experiment_id>

Conditional fine-tuning of MoGen A (s-module-conditional-design.md; blocks chosen by condition_effects.py):

    python scripts/neuron_model/train_mogen.py --config configs/neuron-model/mogen/conditional_finetune.yaml \
        "data.manifests=['<cache>/manifest.parquet']" data.splits=<splits> data.drop_unassigned=true \
        model.init_run=runs/training/neuron-model/<mogen A run> condition.effects_report=<report.json>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

# Windows conda: torch (libomp) and MKL (libiomp5md) both load an OpenMP runtime; the first use of both (e.g. mesh
# extraction after training, SVD) aborts the process with "OMP: Error #15". Standard workaround, Windows only.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.neuron_model.spine_generation.data.datasets import make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
from src.neuron_model.spine_generation.experiment.run import create_run, open_run
from src.neuron_model.spine_generation.experiment.seed import seed_everything
from src.neuron_model.spine_generation.experiment.training import count_parameters, get_device, make_lr_scheduler
from src.neuron_model.spine_generation.models.spine_mogen.train import (
    build_model,
    make_loss_fn,
    make_optimizer,
    make_sample_fn,
    make_validate_fn,
    point_cloud_datasets,
)


def prepare_conditions(index, condition_cfg, cond_dim: int, *, resume_dir=None):
    """Encoder (enabled blocks from the config list, or from the condition_effects.py report) and one condition vector
    per spine. Spines of an excluded compartment (axon_like) are left out of training."""
    from src.neuron_model.spine_generation.conditioning import ConditionEncoder, condition_summary, enabled_from_report, spine_conditions

    if resume_dir is not None and (Path(resume_dir) / "condition_encoder.json").exists():
        encoder = ConditionEncoder.load(Path(resume_dir) / "condition_encoder.json")
    else:
        enabled = condition_cfg.get("blocks")
        if enabled is None and condition_cfg.get("effects_report"):
            enabled = enabled_from_report(resolve_path(condition_cfg["effects_report"]), mode=str(condition_cfg.get("mode", "auto")))
        if enabled is None and not condition_cfg.get("effects_report") and str(condition_cfg.get("mode", "auto")) != "all":
            print("WARNING: condition.blocks and condition.effects_report are not set: EVERY block is enabled; run "
                  "scripts/neuron_model/condition_effects.py first (design section 6, check 0) or set condition.mode=all explicitly", flush=True)
        encoder = ConditionEncoder(cond_dim=int(cond_dim), enabled=list(enabled) if enabled is not None else None)
    conditions = spine_conditions(index.frame, resolve_path(condition_cfg.get("metadata_root", "datasets")))
    excluded = set(condition_cfg.get("exclude_compartments", ["axon_like"]))
    keep = ~conditions["compartment"].isin(excluded)
    vectors = encoder.encode_frame(conditions[keep])
    cond_by_key = dict(zip(index.frame.loc[keep, "spine_key"], vectors))
    info = {"n_spines": int(keep.sum()), "n_excluded_compartment": int((~keep).sum()), "counts": condition_summary(conditions[keep], encoder),
            "encoder": encoder.to_dict()}
    return encoder, cond_by_key, info


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

    # check_files defaults to verifying every train-split spine's pointcloud file actually exists
    # (needed while a staging mover may still be moving files to the NAS); override with
    # "data.check_files=[]" to skip it when nothing is being staged - saves one NAS stat() per
    # spine (can be tens of minutes at full dataset scale).
    default_check_files = (f"pointcloud_{data_cfg['n_points']}_path",)
    index = load_spine_index(
        [resolve_path(p) for p in data_cfg["manifests"]],
        path_mode=data_cfg.get("path_mode", "relative"),
        check_files=tuple(data_cfg.get("check_files", default_check_files)),
    )
    index = index.with_split(load_splits(resolve_path(data_cfg["splits"])), drop_unassigned=bool(data_cfg.get("drop_unassigned", False)))
    model = build_model(cfg["model"])
    encoder, cond_by_key, cond_info = None, None, None
    condition_cfg = cfg.get("condition") or {}
    if condition_cfg.get("enabled"):
        encoder, cond_by_key, cond_info = prepare_conditions(index, condition_cfg, model.cfg.cond_dim, resume_dir=args.resume)
    train_ds, val_ds = point_cloud_datasets(index, data_cfg, cond_by_key)
    if not args.resume:
        cfg["data"]["dataset_version"] = index.dataset_version
        run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_train": len(train_ds), "n_val": len(val_ds or [])})
    if encoder is not None:
        encoder.save(run.root / "condition_encoder.json")
        write_json(run.root / "condition_summary.json", cond_info)
        print(f"conditional: blocks {encoder.enabled or encoder.names} ({encoder.width} of cond_dim {encoder.cond_dim}); "
              f"dropout p_block {condition_cfg.get('p_block', 0.15)}, p_all {condition_cfg.get('p_all', 0.10)}; {cond_info['counts']}", flush=True)

    write_json(run.root / "model_config.json", model.cfg.to_dict())  # actual architecture (incl. cond_dim from the checkpoint)
    print(f"run {run.root}\ndevice {device}; params {count_parameters(model):,}; train {len(train_ds)} / val {len(val_ds or [])}")
    optimizer = make_optimizer(model, cfg["optimizer"])
    train_cfg = cfg["train"]
    # optimizer.schedule: e.g. "optimizer.schedule={type: cosine, min_lr: 1.0e-5}" (default: constant lr); on --resume of a
    # run without one it starts at the current step (loop.run_training)
    scheduler = make_lr_scheduler(optimizer, cfg["optimizer"].get("schedule"), int(train_cfg["max_steps"]))
    loader = make_loader(train_ds, batch_size=int(data_cfg["batch_size"]), shuffle=True, seed=int(cfg["seed"]), num_workers=int(data_cfg.get("num_workers", 0)), drop_last=True)
    loop = LoopConfig.from_mapping({**train_cfg, "precision": cfg.get("precision")})
    result = run_training(
        run,
        loop,
        model,
        optimizer,
        loader,
        make_loss_fn(cfg["flow"], device, encoder=encoder, condition_cfg=condition_cfg),
        device=device,
        scheduler=scheduler,
        validate_fn=make_validate_fn(val_ds, cfg["flow"], device, batch_size=int(data_cfg["batch_size"]), seed=int(cfg["seed"]), max_batches=train_cfg.get("n_val_batches")),
        sample_fn=make_sample_fn(run.samples, cfg["flow"], data_cfg, device, n_samples=int(train_cfg.get("n_samples", 4)), seed=int(cfg["seed"])),
        resume=bool(args.resume),
    )
    print("finished", result)


if __name__ == "__main__":
    main()
