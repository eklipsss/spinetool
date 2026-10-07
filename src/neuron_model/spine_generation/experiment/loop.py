"""Generic training loop shared by the generators (VAE, MoGen fine-tune / scratch).

The model-specific parts are callbacks:

- ``loss_fn(model, batch, step) -> (loss, logs)``  one micro-batch (``step`` = optimizer steps done so far);
- ``validate_fn(model_for_eval, step) -> metrics``  e.g. val loss, latent stats;
- ``sample_fn(model_for_eval, step) -> metrics``  writes intermediate samples.

Everything the spec requires of every training run lives here once: mixed
precision, gradient accumulation, global gradient clipping, EMA, periodic
validation and sampling, ``latest`` / ``best_<metric>`` checkpoints and exact
resume (model + EMA + optimizer + scheduler + scaler + RNG + step).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .checkpoint import CheckpointManager, load_state_dicts, state_dicts
from .metrics_logger import MetricsLogger
from .run import RunDir
from .training import EMA, OptimizerStep, autocast, infinite, make_grad_scaler


@dataclass
class LoopConfig:
    max_steps: int
    accumulation_steps: int = 1
    clip_norm: Optional[float] = None
    precision: Optional[str] = "fp32"
    ema_decay: Optional[float] = None  # None disables EMA
    log_every: int = 100
    validate_every: int = 1000
    sample_every: int = 5000
    checkpoint_every: int = 1000
    tracked_metrics: Mapping[str, str] = field(default_factory=lambda: {"val_loss": "min"})
    keep_every: Optional[int] = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LoopConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in values.items() if k in known})


def run_training(
    run: RunDir,
    loop: LoopConfig,
    model,
    optimizer,
    train_loader: Iterable,
    loss_fn: Callable[[Any, Any, int], Tuple[Any, Dict[str, float]]],
    *,
    device,
    scheduler=None,
    validate_fn: Optional[Callable[[Any, int], Dict[str, float]]] = None,
    sample_fn: Optional[Callable[[Any, int], Dict[str, float]]] = None,
    resume: bool = False,
    on_step: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Train until ``loop.max_steps`` optimizer steps; returns final step + last metrics."""
    import torch

    model.to(device)
    ema = EMA(model, loop.ema_decay) if loop.ema_decay else None
    scaler = make_grad_scaler(device, loop.precision)
    stepper = OptimizerStep(optimizer, accumulation_steps=loop.accumulation_steps, clip_norm=loop.clip_norm, scaler=scaler)
    checkpoints = CheckpointManager(run.checkpoints, tracked=loop.tracked_metrics, keep_every=loop.keep_every)
    logger = MetricsLogger(run.logs)

    step = 0
    if resume and checkpoints.exists("latest"):
        payload = checkpoints.load("latest", map_location=device)
        load_state_dicts(payload["state"], model=model, optimizer=optimizer, ema=ema, scheduler=scheduler, scaler=scaler)
        step = int(payload["step"])
        for stream in ("train", "val", "samples"):
            logger.truncate_after(stream, step)

    def eval_model():
        return ema.model if ema is not None else model

    def save(metrics: Dict[str, float]) -> None:
        checkpoints.save(step, state_dicts(model=model, optimizer=optimizer, ema=ema, scheduler=scheduler, scaler=scaler), metrics)

    batches = infinite(train_loader)
    window: Dict[str, float] = {}
    window_n = 0
    started = time.perf_counter()
    last_metrics: Dict[str, float] = {}
    model.train()
    print(f"training: step {step}/{loop.max_steps}, logging every {loop.log_every} steps", flush=True)
    while step < loop.max_steps:
        logs: Dict[str, float] = {}
        for _ in range(loop.accumulation_steps):
            batch = next(batches)
            with autocast(device, loop.precision):
                loss, logs = loss_fn(model, batch, step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {logs}")
            stepper.backward(loss)
        grad_norm = stepper.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        step += 1

        logs = {**logs, "grad_norm": grad_norm if grad_norm is not None else float("nan"), "lr": optimizer.param_groups[0]["lr"]}
        for key, value in logs.items():
            window[key] = window.get(key, 0.0) + float(value)
        window_n += 1
        if on_step is not None:
            on_step(step, logs)
        if step % loop.log_every == 0 or step == loop.max_steps:
            elapsed = time.perf_counter() - started
            avg = {k: v / window_n for k, v in window.items()}
            steps_per_second = window_n / max(elapsed, 1e-9)
            logger.log("train", step, avg, steps_per_second=steps_per_second)
            remaining_steps = loop.max_steps - step
            eta_min = remaining_steps / steps_per_second / 60 if steps_per_second > 0 else float("nan")
            metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in avg.items())
            print(f"step {step}/{loop.max_steps} ({steps_per_second:.2f} steps/s, ETA {eta_min:.1f} min) {metrics_str}", flush=True)
            window, window_n, started = {}, 0, time.perf_counter()

        metrics: Dict[str, float] = {}
        if validate_fn is not None and (step % loop.validate_every == 0 or step == loop.max_steps):
            model.eval()
            with torch.no_grad():
                metrics.update(validate_fn(eval_model(), step))
            model.train()
            logger.log("val", step, metrics)
            val_str = ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"  val @ step {step}: {val_str}", flush=True)
        if sample_fn is not None and (step % loop.sample_every == 0 or step == loop.max_steps):
            model.eval()
            sample_metrics = sample_fn(eval_model(), step) or {}
            model.train()
            logger.log("samples", step, sample_metrics)
            metrics.update(sample_metrics)
        if metrics:
            last_metrics = metrics
        if step % loop.checkpoint_every == 0 or step == loop.max_steps or metrics:
            save(metrics)
    return {"step": step, "metrics": last_metrics}
