"""Checkpoints: ``latest`` + ``best_<metric>`` for any number of tracked metrics.

A checkpoint holds everything needed for an exact resume: model, EMA model,
optimizer, LR scheduler, grad scaler, RNG states, step/epoch and the metrics
the "best" decisions were based on. Writes are atomic (temp file + rename) so
an interrupted save never corrupts the previous ``latest``.

Files in ``<run>/checkpoints/``::

    latest.pt
    best_<metric>.pt      # e.g. best_val_loss.pt, best_morphology_metric.pt
    step_<N>.pt           # optional periodic snapshots (keep_every)
    index.json            # which step each file holds + best values so far
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .seed import capture_rng_state, restore_rng_state


@dataclass
class TrackedMetric:
    name: str
    mode: str = "min"  # "min" or "max"

    def better(self, value: float, best: Optional[float]) -> bool:
        if value is None or not math.isfinite(value):
            return False
        if best is None:
            return True
        return value < best if self.mode == "min" else value > best


@dataclass
class CheckpointManager:
    directory: Path
    tracked: Mapping[str, str] = field(default_factory=dict)  # metric name -> "min"/"max"
    keep_every: Optional[int] = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._metrics = {name: TrackedMetric(name, mode) for name, mode in self.tracked.items()}
        self.index: Dict[str, Any] = self._read_index()

    @property
    def index_path(self) -> Path:
        return self.directory / "index.json"

    def path(self, tag: str) -> Path:
        return self.directory / f"{tag}.pt"

    def best_value(self, metric: str) -> Optional[float]:
        return self.index.get("best", {}).get(metric, {}).get("value")

    def save(
        self,
        step: int,
        state: Mapping[str, Any],
        metrics: Optional[Mapping[str, float]] = None,
        *,
        include_rng: bool = True,
    ) -> Dict[str, bool]:
        """Save ``latest`` and every ``best_<metric>`` that improved. Returns which bests improved."""
        metrics = {k: float(v) for k, v in (metrics or {}).items() if v is not None}
        improved: Dict[str, bool] = {}
        for name, tracked in self._metrics.items():
            value = metrics.get(name)
            improved[name] = tracked.better(value, self.best_value(name))
            if improved[name]:
                self.index.setdefault("best", {})[name] = {"value": value, "step": int(step), "mode": tracked.mode}

        payload = {
            "step": int(step),
            "state": dict(state),
            "metrics": metrics,
            "rng": capture_rng_state() if include_rng else None,
            "best": self.index.get("best", {}),
        }
        self._atomic_save(payload, self.path("latest"))
        self.index["latest"] = {"step": int(step), "metrics": metrics}
        for name, did_improve in improved.items():
            if did_improve:
                self._atomic_save(payload, self.path(f"best_{name}"))

        if self.keep_every and step % self.keep_every == 0:
            self._atomic_save(payload, self.path(f"step_{step}"))
        self._write_index()
        return improved

    def exists(self, tag: str = "latest") -> bool:
        return self.path(tag).exists()

    def load(self, tag: str = "latest", *, map_location: Any = "cpu", restore_rng: bool = True) -> Dict[str, Any]:
        """Load a checkpoint payload; by default also restores python/numpy/torch RNG state."""
        import torch

        payload = torch.load(self.path(tag), map_location=map_location, weights_only=False)
        if restore_rng and payload.get("rng") is not None:
            restore_rng_state(payload["rng"])
        return payload

    def _atomic_save(self, payload: Mapping[str, Any], path: Path) -> None:
        import torch

        tmp = path.with_name(path.name + ".tmp")
        torch.save(dict(payload), tmp)
        tmp.replace(path)

    def _read_index(self) -> Dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {}

    def _write_index(self) -> None:
        tmp = self.index_path.with_name(self.index_path.name + ".tmp")
        tmp.write_text(json.dumps(self.index, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)


def state_dicts(**objects: Any) -> Dict[str, Any]:
    """``state_dicts(model=m, optimizer=o, ema=e)`` -> ``{"model": m.state_dict(), ...}`` (skips None)."""
    return {name: obj.state_dict() for name, obj in objects.items() if obj is not None}


def load_state_dicts(state: Mapping[str, Any], **objects: Any) -> None:
    """Inverse of :func:`state_dicts`; raises if the checkpoint lacks a requested entry."""
    for name, obj in objects.items():
        if obj is None:
            continue
        if name not in state:
            raise KeyError(f"Checkpoint has no '{name}' state (has: {sorted(state)})")
        obj.load_state_dict(state[name])
