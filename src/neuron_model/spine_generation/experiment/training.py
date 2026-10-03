"""Shared training helpers: device, mixed precision, EMA, gradient accumulation/clipping.

Model-specific training loops (VAE, MoGen) build on these; the helpers stay
model-agnostic so every generator gets the same, tested mechanics required by
the spec (mixed precision, gradient clipping, gradient accumulation, EMA).
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any, Dict, Iterable, Iterator, Optional


def get_device(preference: str = "auto"):
    """``auto`` -> cuda, then mps, then cpu; or an explicit torch device string."""
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(name: Optional[str]):
    import torch

    return {None: None, "none": None, "fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def autocast(device, precision: Optional[str]):
    """Autocast context for ``precision`` in {"bf16", "fp16", "fp32"/None}.

    Mixed precision is only enabled on CUDA; on MPS/CPU this is a no-op so the
    same config runs unchanged on the mac (fp32) and on the RTX 4090.
    """
    import torch

    dtype = autocast_dtype(precision)
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device, precision: Optional[str]):
    """GradScaler only for fp16 on CUDA (bf16 does not need loss scaling)."""
    import torch

    enabled = device.type == "cuda" and precision == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


class EMA:
    """Exponential moving average of model parameters (buffers are copied as-is).

    ``decay`` 0.999 is the MoGen reference (``polyak_decay`` in its config).
    """

    def __init__(self, model, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            ema_params = dict(self.model.named_parameters())
            for name, param in model.named_parameters():
                ema_params[name].lerp_(param.detach(), 1.0 - self.decay)
            ema_buffers = dict(self.model.named_buffers())
            for name, buffer in model.named_buffers():
                ema_buffers[name].copy_(buffer)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.model.load_state_dict(state["model"])


class OptimizerStep:
    """Gradient accumulation + global-norm clipping + (optional) fp16 loss scaling.

    Usage per micro-batch::

        stepper.backward(loss)
        if stepper.ready():      # after `accumulation_steps` micro-batches
            grad_norm = stepper.step()
    """

    def __init__(self, optimizer, *, accumulation_steps: int = 1, clip_norm: Optional[float] = None, scaler=None) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        self.optimizer = optimizer
        self.accumulation_steps = int(accumulation_steps)
        self.clip_norm = clip_norm
        self.scaler = scaler
        self._micro = 0

    def backward(self, loss) -> None:
        loss = loss / self.accumulation_steps
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._micro += 1

    def ready(self) -> bool:
        return self._micro >= self.accumulation_steps

    def step(self) -> Optional[float]:
        import torch

        params = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        scaled = self.scaler is not None and self.scaler.is_enabled()
        if scaled:
            self.scaler.unscale_(self.optimizer)
        grad_norm = None
        if self.clip_norm:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.clip_norm))
        elif params:
            grad_norm = float(torch.norm(torch.stack([p.grad.detach().norm() for p in params])))
        if scaled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._micro = 0
        return grad_norm


def infinite(loader: Iterable) -> Iterator:
    """Cycle a DataLoader forever (a new epoch - and new shuffle - each pass)."""
    while True:
        for batch in loader:
            yield batch


def count_parameters(model, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
