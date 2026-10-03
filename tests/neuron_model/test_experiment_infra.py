import json

import pytest
import torch

from src.neuron_model.spine_generation.experiment.checkpoint import CheckpointManager, load_state_dicts, state_dicts
from src.neuron_model.spine_generation.experiment.config import apply_overrides, config_hash, load_config, save_config
from src.neuron_model.spine_generation.experiment.metrics_logger import MetricsLogger
from src.neuron_model.spine_generation.experiment.run import create_run, open_run
from src.neuron_model.spine_generation.experiment.seed import seed_everything
from src.neuron_model.spine_generation.experiment.training import EMA, OptimizerStep


def test_config_defaults_merge_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\nmodel: {width: 64, depth: 2}\nlr: 0.1\n")
    (tmp_path / "child.yaml").write_text("defaults: [base.yaml]\nmodel: {depth: 4}\n")
    config = load_config(tmp_path / "child.yaml", ["lr=1e-4", "model.use_normals=false", "new.key=[1, 2]"])
    assert config == {"seed": 1, "model": {"width": 64, "depth": 4, "use_normals": False}, "lr": 1e-4, "new": {"key": [1, 2]}}
    save_config(config, tmp_path / "saved.yaml")
    assert load_config(tmp_path / "saved.yaml") == config
    assert config_hash(config) == config_hash(dict(reversed(list(config.items()))))


def test_config_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\n")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_override_requires_key_value():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_run_dir_and_info(tmp_path):
    config = {"name": "unit", "seed": 7, "data": {"dataset_version": "v1:abc", "split_version": "splits_v1"}}
    run = create_run(config, experiment_id="exp1", runs_root=tmp_path)
    info = run.load_info()
    assert info["experiment_id"] == "exp1" and info["random_seed"] == 7
    assert info["dataset_version"] == "v1:abc" and info["split_version"] == "splits_v1"
    assert all((run.root / d).is_dir() for d in ("checkpoints", "logs", "samples", "meshes", "metrics"))
    assert open_run(run.root).load_config() == config
    with pytest.raises(FileExistsError):
        create_run(config, experiment_id="exp1", runs_root=tmp_path)


def _toy():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer


def _train_steps(model, optimizer, ema, n):
    for _ in range(n):
        x = torch.randn(16, 4)  # consumes the global RNG -> resume must restore it
        loss = (model(x) - x.sum(dim=1, keepdim=True)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)


def test_resume_reproduces_uninterrupted_training(tmp_path):
    seed_everything(123)
    model, optimizer = _toy()
    ema = EMA(model, decay=0.9)
    manager = CheckpointManager(tmp_path, tracked={"val_loss": "min"})
    _train_steps(model, optimizer, ema, 3)
    manager.save(3, state_dicts(model=model, optimizer=optimizer, ema=ema), {"val_loss": 1.0})
    _train_steps(model, optimizer, ema, 2)
    reference = [p.detach().clone() for p in model.parameters()]
    reference_ema = [p.detach().clone() for p in ema.model.parameters()]

    seed_everything(999)  # different state before resuming
    model2, optimizer2 = _toy()
    ema2 = EMA(model2, decay=0.5)
    payload = CheckpointManager(tmp_path, tracked={"val_loss": "min"}).load("latest")
    load_state_dicts(payload["state"], model=model2, optimizer=optimizer2, ema=ema2)
    assert payload["step"] == 3 and ema2.decay == 0.9
    _train_steps(model2, optimizer2, ema2, 2)
    for a, b in zip(reference, model2.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(reference_ema, ema2.model.parameters()):
        assert torch.equal(a, b)


def test_best_checkpoints_min_and_max(tmp_path):
    model, optimizer = _toy()
    manager = CheckpointManager(tmp_path, tracked={"val_loss": "min", "coverage": "max"})
    assert manager.save(1, state_dicts(model=model), {"val_loss": 2.0, "coverage": 0.1}) == {"val_loss": True, "coverage": True}
    assert manager.save(2, state_dicts(model=model), {"val_loss": 3.0, "coverage": 0.5}) == {"val_loss": False, "coverage": True}
    assert manager.save(3, state_dicts(model=model), {"val_loss": 1.0, "coverage": 0.2}) == {"val_loss": True, "coverage": False}
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["best"]["val_loss"]["step"] == 3 and index["best"]["coverage"]["step"] == 2
    assert manager.load("best_coverage", restore_rng=False)["step"] == 2
    # a manager re-opened on the same directory keeps the best values
    reopened = CheckpointManager(tmp_path, tracked={"val_loss": "min"})
    assert reopened.save(4, state_dicts(model=model), {"val_loss": 1.5}) == {"val_loss": False}


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(0)
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    model_a, _ = _toy()
    model_b, _ = _toy()
    model_b.load_state_dict(model_a.state_dict())
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)

    torch.nn.functional.mse_loss(model_a(x), y).backward()
    opt_a.step()

    stepper = OptimizerStep(opt_b, accumulation_steps=2)
    for part in (slice(0, 4), slice(4, 8)):
        stepper.backward(torch.nn.functional.mse_loss(model_b(x[part]), y[part]))
    assert stepper.ready()
    stepper.step()
    for a, b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_grad_clipping_bounds_norm():
    model, _ = _toy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    stepper = OptimizerStep(optimizer, clip_norm=0.1)
    stepper.backward(model(torch.randn(4, 4)).sum() * 1000)
    norm_before = stepper.step()
    assert norm_before > 0.1  # returned norm is the pre-clip total norm


def test_ema_update_formula():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    assert torch.allclose(ema.model.weight, torch.tensor([[0.1]]))


def test_metrics_logger_roundtrip_and_truncate(tmp_path):
    logger = MetricsLogger(tmp_path)
    for step in range(1, 6):
        logger.log("train", step, {"loss": 1.0 / step, "nan_metric": float("nan"), "t": torch.tensor(2.0)})
    frame = logger.read("train")
    assert frame["step"].tolist() == [1, 2, 3, 4, 5] and frame["t"].eq(2.0).all() and frame["nan_metric"].isna().all()
    logger.truncate_after("train", 3)
    assert logger.read("train")["step"].tolist() == [1, 2, 3]
