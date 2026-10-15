"""Branch preprocessing for modules T and P (docs/neuron-model/tp-branch-preprocessing.md)."""

from .config import BranchPreprocessingConfig, load_config
from .runner import consolidate, run

__all__ = ["BranchPreprocessingConfig", "consolidate", "load_config", "run"]
