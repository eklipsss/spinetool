import json

import numpy as np
import torch

from src.neuron_model.spine_generation.data.datasets import make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import build_group_split
from src.neuron_model.spine_generation.experiment.artifacts import read_pointcloud_ply
from src.neuron_model.spine_generation.experiment.checkpoint import CheckpointManager
from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
from src.neuron_model.spine_generation.experiment.run import create_run
from src.neuron_model.spine_generation.experiment.seed import seed_everything
from src.neuron_model.spine_generation.models.spine_mogen.infer import generate_point_clouds
from src.neuron_model.spine_generation.models.spine_mogen.train import (
    build_model,
    make_loss_fn,
    make_optimizer,
    make_sample_fn,
    make_validate_fn,
    point_cloud_datasets,
)

TINY = {"point_dim": 16, "latent_dim": 32, "n_latents": 8, "n_blocks": 1, "n_subblocks": 1, "n_heads": 4, "k_nn": 4, "coord_dim": 3, "cond_dim": 6}
DATA = {"n_points": 2048, "coord_scale": 1e-2, "jitter_std": 1.0, "batch_size": 4}
FLOW = {"time_schedule": "mogen_cosine", "inference_schedule": "mogen_cosine", "inference_steps": 3}


def _setup(fake_dataset, tmp_path, experiment_id):
    seed_everything(0)
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    index = index.with_split(build_group_split(index.frame, seed=1))
    train_ds, val_ds = point_cloud_datasets(index, DATA)
    run = create_run({"seed": 0, "data": DATA}, experiment_id=experiment_id, runs_root=tmp_path / "runs")
    model = build_model({"init": "random", "architecture": TINY})
    optimizer = make_optimizer(model, {"name": "adamw", "lr": 1e-3})
    loader = make_loader(train_ds, batch_size=4, shuffle=True, seed=0, drop_last=True)
    device = torch.device("cpu")
    kwargs = dict(
        device=device,
        validate_fn=make_validate_fn(val_ds, FLOW, device, batch_size=4, seed=0),
        sample_fn=make_sample_fn(run.samples, FLOW, DATA, device, n_samples=2, seed=0),
    )
    return run, model, optimizer, loader, kwargs


def test_mogen_training_loop_checkpoints_samples_and_resume(fake_dataset, tmp_path):
    run, model, optimizer, loader, kwargs = _setup(fake_dataset, tmp_path, "mogen_tiny")
    loop = LoopConfig(max_steps=4, accumulation_steps=2, clip_norm=0.1, ema_decay=0.9, log_every=2, validate_every=2, sample_every=4, checkpoint_every=2)
    result = run_training(run, loop, model, optimizer, loader, make_loss_fn(FLOW, torch.device("cpu")), **kwargs)
    assert result["step"] == 4 and "val_loss" in result["metrics"]

    index = json.loads((run.checkpoints / "index.json").read_text())
    assert index["latest"]["step"] == 4 and "val_loss" in index["best"]
    points, normals = read_pointcloud_ply(run.samples / "step_0000004" / "sample_000.ply")
    assert points.shape == (2048, 3) and normals is None and np.isfinite(points).all()
    train_log = [json.loads(line) for line in (run.logs / "metrics_train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in train_log] == [2, 4] and all(np.isfinite(r["loss"]) for r in train_log)

    # resume to 6 steps from the saved state
    _, model2, optimizer2, loader2, kwargs2 = _setup(fake_dataset, tmp_path, "unused")
    kwargs2["sample_fn"] = make_sample_fn(run.samples, FLOW, DATA, torch.device("cpu"), n_samples=1, seed=0)
    loop2 = LoopConfig(**{**loop.__dict__, "max_steps": 6})
    result2 = run_training(run, loop2, model2, optimizer2, loader2, make_loss_fn(FLOW, torch.device("cpu")), resume=True, **kwargs2)
    assert result2["step"] == 6
    assert CheckpointManager(run.checkpoints).load("latest", restore_rng=False)["step"] == 6
    val_log = [json.loads(line)["step"] for line in (run.logs / "metrics_val.jsonl").read_text().splitlines()]
    assert val_log == [2, 4, 6]


def test_validation_loss_is_deterministic(fake_dataset, tmp_path):
    run, model, optimizer, loader, kwargs = _setup(fake_dataset, tmp_path, "mogen_val")
    validate = kwargs["validate_fn"]
    model.eval()
    with torch.no_grad():
        assert validate(model, 0) == validate(model, 1)


def test_generation_is_seeded_per_sample_independent_of_batch():
    model = build_model({"init": "random", "architecture": TINY}).eval()
    common = dict(n_samples=3, n_points=64, coord_scale=1e-2, n_steps=2, schedule="mogen_cosine", seed=7, device=torch.device("cpu"))
    a = generate_point_clouds(model, batch_size=1, **common)
    b = generate_point_clouds(model, batch_size=3, **common)
    assert a.shape == (3, 64, 3) and np.allclose(a, b, atol=1e-5)
    c = generate_point_clouds(model, batch_size=3, **{**common, "seed": 8})
    assert not np.allclose(a, c)
