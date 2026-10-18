import math

import pytest
import torch

from src.neuron_model.spine_generation.experiment.config import apply_overrides
from src.neuron_model.spine_generation.experiment.training import make_lr_scheduler


def _opt(lr=1e-4):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_constant_by_default_and_cosine_shape():
    assert make_lr_scheduler(_opt(), None, 100) is None
    assert make_lr_scheduler(_opt(), {"type": "constant"}, 100) is None
    cfg = apply_overrides({"optimizer": {"lr": 1e-4}}, ["optimizer.schedule.type=cosine", "optimizer.schedule.min_lr=1.0e-5", "optimizer.schedule.warmup_steps=10"])
    opt = _opt()
    sched = make_lr_scheduler(opt, cfg["optimizer"]["schedule"], 110)
    lrs = []
    for _ in range(130):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-5) and lrs[9] == pytest.approx(1e-4)  # linear warm-up
    assert lrs[60] == pytest.approx(1e-5 + 0.5 * (1e-4 - 1e-5), rel=1e-6)  # half way of the cosine
    assert lrs[110] == pytest.approx(1e-5) and lrs[129] == pytest.approx(1e-5)  # floor after total_steps
    assert all(a >= b - 1e-15 for a, b in zip(lrs[9:111], lrs[10:111]))


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError):
        make_lr_scheduler(_opt(), {"type": "step"}, 10)


def test_schedule_added_on_resume_starts_at_the_current_step(tmp_path):
    from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
    from src.neuron_model.spine_generation.experiment.run import create_run, open_run

    run = create_run({"name": "t", "seed": 1}, experiment_id="lr", runs_root=tmp_path)
    data = [torch.randn(8, 4) for _ in range(4)]

    def loss_fn(model, batch, step):
        loss = model(batch).pow(2).mean()
        return loss, {"loss": float(loss)}

    def fresh():
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 1)
        return m, torch.optim.AdamW(m.parameters(), lr=1e-3)

    loop = lambda n: LoopConfig.from_mapping({"max_steps": n, "log_every": 1, "checkpoint_every": 5, "validate_every": 1000, "sample_every": 1000})
    m, o = fresh()
    run_training(run, loop(5), m, o, data, loss_fn, device=torch.device("cpu"))
    m, o = fresh()
    sched = make_lr_scheduler(o, {"type": "cosine", "min_lr": 1e-4}, 10)
    run_training(open_run(run.root), loop(10), m, o, data, loss_fn, device=torch.device("cpu"), scheduler=sched, resume=True)
    expected = 1e-4 + (1e-3 - 1e-4) * 0.5 * (1 + math.cos(math.pi * 10 / 10))
    assert o.param_groups[0]["lr"] == pytest.approx(expected)  # 10 scheduler steps in total, not 5
