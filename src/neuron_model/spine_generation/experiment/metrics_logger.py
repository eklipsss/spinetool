"""Scalar metric logging to JSON Lines (one file per stream, e.g. train / val).

Plain files on purpose: no TensorBoard/W&B dependency, easy to read back with
pandas for plots and reports, and safe to append to after a resume.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd


class MetricsLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def path(self, stream: str) -> Path:
        return self.log_dir / f"metrics_{stream}.jsonl"

    def log(self, stream: str, step: int, metrics: Mapping[str, Any], **extra: Any) -> None:
        record: Dict[str, Any] = {"step": int(step), "time": time.time(), **extra}
        for key, value in metrics.items():
            record[key] = _to_scalar(value)
        with self.path(stream).open("a", encoding="utf-8") as fd:
            fd.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self, stream: str) -> pd.DataFrame:
        return read_metrics(self.path(stream))

    def truncate_after(self, stream: str, step: int) -> None:
        """Drop records logged after ``step`` - used on resume from an older checkpoint,
        so the curve does not contain two different histories for the same steps."""
        path = self.path(stream)
        if not path.exists():
            return
        kept: List[str] = [
            line for line in path.read_text(encoding="utf-8").splitlines() if line and json.loads(line)["step"] <= step
        ]
        path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def read_metrics(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _to_scalar(value: Any) -> Optional[float]:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    value = float(value)
    return value if math.isfinite(value) else None
