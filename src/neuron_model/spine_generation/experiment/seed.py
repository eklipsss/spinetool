"""Random seed initialisation and RNG state capture/restore (for exact resume)."""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (all devices).

    ``deterministic=True`` additionally asks torch for deterministic kernels
    (slower; some ops then raise if they have no deterministic implementation).
    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: derive numpy/python seeds from torch's per-worker seed."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state() -> Dict[str, Any]:
    import torch

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
