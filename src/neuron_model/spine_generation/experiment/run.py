"""Run directories: ``runs/training/neuron-model/<experiment_id>/``.

Layout (s-module-technical-spec.md, "Структура runs")::

    <run_dir>/
      config.yaml       # fully resolved config actually used
      run_info.json     # experiment_id, git commit, dataset/split version, seed, ...
      checkpoints/
      logs/             # metrics JSONL, text logs
      samples/          # raw generator outputs + generation configs
      meshes/           # extracted meshes
      metrics/          # validation / generation metric tables
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import REPO_ROOT, config_hash, get_dotted, load_config, save_config

DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "training" / "neuron-model"
RUN_SUBDIRS = ("checkpoints", "logs", "samples", "meshes", "metrics")


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def experiment_id(self) -> str:
        return self.root.name

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def info_path(self) -> Path:
        return self.root / "run_info.json"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def meshes(self) -> Path:
        return self.root / "meshes"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    def load_config(self) -> Dict[str, Any]:
        return load_config(self.config_path)

    def load_info(self) -> Dict[str, Any]:
        return json.loads(self.info_path.read_text(encoding="utf-8"))


def git_state(repo: Path = REPO_ROOT) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def make_experiment_id(name: str, now: Optional[_dt.datetime] = None) -> str:
    now = now or _dt.datetime.now()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
    return f"{now:%Y-%m-%d_%H%M%S}_{slug}"


def create_run(
    config: Mapping[str, Any],
    *,
    name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> RunDir:
    """Create a fresh run directory, save the resolved config and ``run_info.json``.

    ``config`` should contain ``seed`` and, where applicable,
    ``data.dataset_version``/``data.split_version`` - they are copied into
    ``run_info.json`` so every run records exactly what it was trained on.
    """
    experiment_id = experiment_id or make_experiment_id(name or str(config.get("name", "run")))
    root = Path(runs_root) / experiment_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Run directory already exists and is not empty: {root} (use open_run to resume)")
    run = RunDir(root)
    for sub in RUN_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    save_config(config, run.config_path)
    info = {
        "experiment_id": experiment_id,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        **git_state(),
        "config_hash": config_hash(config),
        "random_seed": config.get("seed"),
        "dataset_version": get_dotted(config, "data.dataset_version"),
        "split_version": get_dotted(config, "data.split_version"),
        **_library_versions(),
        **dict(extra_info or {}),
    }
    run.info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return run


def open_run(root: Path) -> RunDir:
    run = RunDir(Path(root))
    if not run.config_path.exists():
        raise FileNotFoundError(f"Not a run directory (no config.yaml): {root}")
    return run


def _library_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for module in ("torch", "numpy"):
        try:
            versions[f"{module}_version"] = __import__(module).__version__
        except ImportError:
            versions[f"{module}_version"] = None
    return versions
