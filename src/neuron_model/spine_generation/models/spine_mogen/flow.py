"""Flow Matching for point clouds, as in MoGen (``connectomics/mogen/flow_matching``).

- path: ``x_t = (1 - t) x_0 + t x_1``, ``x_0 ~ N(0, I)`` noise, ``x_1`` data;
  target velocity ``x_1 - x_0``; loss = MSE (spec, "Обучение Flow Matching");
- training times: ``t ~ U(0, 1)`` passed through the schedule
  (``mogen_cosine`` = ``cosine_2.0`` from the ``mouse_mixed`` config);
- sampling: explicit midpoint ODE solver from t=0 to t=1 over the schedule
  applied to ``linspace(0, 1, n_steps + 1)`` (``generate_step_midpoint``).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

SCHEDULE_ALIASES = {"mogen_cosine": "cosine_2.0"}


def cosine_schedule(t: torch.Tensor, exponent: float = 1.0, skew: float = 1.0) -> torch.Tensor:
    """``schedules.cosine``: ``((|(|cos(pi t)| - 1)|^e) - 1) * ((t < 0.5) - 0.5) + 0.5`` with ``t := t^skew``."""
    t = t**skew
    return ((torch.abs(torch.abs(torch.cos(t * math.pi)) - 1) ** exponent) - 1) * ((t < 0.5).to(t.dtype) - 0.5) + 0.5


def t_schedule(t: torch.Tensor, name: str = "linear") -> torch.Tensor:
    """``schedules.t_schedule`` (linear / cosine / cosine_<exp> / cosine_<exp>_<skew> / linear_logsnr[_min_max])."""
    name = SCHEDULE_ALIASES.get(name, name)
    if name in ("linear", "uniform"):
        return t
    if name.startswith("cosine"):
        parts = name.split("_")
        if len(parts) == 1:
            return cosine_schedule(t)
        if len(parts) == 2:
            return cosine_schedule(t, float(parts[1]))
        return cosine_schedule(t, float(parts[1]), float(parts[2]))
    if name.startswith("linear_logsnr"):
        lo, hi = (-20.0, 20.0) if name == "linear_logsnr" else map(float, name.split("_")[-2:])
        return torch.sigmoid(t * (hi - lo) + lo)
    raise ValueError(f"Unknown time schedule: {name}")


def flow_matching_loss(
    model: Callable[..., torch.Tensor],
    x_1: torch.Tensor,
    *,
    schedule: str = "mogen_cosine",
    cond: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mean squared error between predicted and target velocity for a batch ``x_1`` ``[B, N, 3]``.

    With a ``generator`` (e.g. fixed-seed validation) noise and times are drawn
    on the generator's device (CPU) and moved to ``x_1``'s device, so the same
    draws are obtained on CUDA, MPS and CPU.
    """
    b = x_1.shape[0]
    draw_device = generator.device if generator is not None else x_1.device
    x_0 = torch.randn(x_1.shape, device=draw_device, dtype=x_1.dtype, generator=generator).to(x_1.device)
    t = torch.rand(b, device=draw_device, dtype=x_1.dtype, generator=generator).to(x_1.device)
    t = t_schedule(t, schedule)
    t_exp = t.view(b, *([1] * (x_1.dim() - 1)))
    x_t = (1 - t_exp) * x_0 + t_exp * x_1
    target = x_1 - x_0
    pred = model(x_t, t, cond) if cond is not None else model(x_t, t)
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def sample_midpoint(
    model: Callable[..., torch.Tensor],
    x_0: torch.Tensor,
    *,
    n_steps: int = 100,
    schedule: str = "mogen_cosine",
    cond: Optional[torch.Tensor] = None,
    return_history: bool = False,
):
    """Integrate ``dx/dt = v(x, t)`` from t=0 (noise ``x_0``) to t=1 with the explicit midpoint rule."""
    timesteps = t_schedule(torch.linspace(0.0, 1.0, n_steps + 1, device=x_0.device, dtype=x_0.dtype), schedule)
    b = x_0.shape[0]
    x = x_0
    history = [x] if return_history else None
    call = (lambda y, tt: model(y, tt, cond)) if cond is not None else (lambda y, tt: model(y, tt))
    for i in range(n_steps):
        t0, t1 = timesteps[i], timesteps[i + 1]
        dt = t1 - t0
        v0 = call(x, t0.expand(b))
        x_mid = x + v0 * dt / 2
        v_mid = call(x_mid, (t0 + dt / 2).expand(b))
        x = x + dt * v_mid
        if return_history:
            history.append(x)
    return torch.stack(history, dim=1) if return_history else x
