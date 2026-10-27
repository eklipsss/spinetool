"""Condition vector of the spine generator (docs/neuron-model/s-module-conditional-design.md, section 3).

The condition is a concatenation of one-hot BLOCKS; a block of zeros means "unknown". This lets one model be
conditioned on partial information (a dataset without cell types, a spine without a compartment) and, with block
dropout during training (``drop_blocks``), also generate unconditionally (all zeros = the input MoGen was trained
with). The vector is padded with zeros to the model's ``cond_dim`` (22 for ``mouse_mixed``).

Blocks can be DISABLED (always zero): a block is used only when the real spines of its values differ more than the
neuron-level noise (``scripts/neuron_model/condition_effects.py``, design section 6, check 0), otherwise the model
could learn the style of the few neurons carrying a rare value. The schema (order, values, enabled blocks, width) is
written into the run directory (``condition_encoder.json``) and read back for generation, so a checkpoint is never
fed a vector of another layout.

Spine conditions are DERIVED here from the identity of the spine, not read from the manifest columns: the
compartment of the manifest was null for every neuron but the first of the table before the lookup fix
(``metadata.compartment_lookup``). Cell type and compartment come from the metadata tables of the repository
(``datasets/``), species / source / microscopy from the dataset, health / treatment of LabID from the group in the
cell id (``metadata.LABID_GROUPS``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_BLOCKS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("species", ("mouse", "human")),
    ("source", ("minnie65", "labid", "h01")),
    ("microscopy", ("electron", "confocal")),
    ("health", ("healthy", "disease")),
    ("treatment", ("9009",)),
    ("cell_type_binary", ("excitatory", "inhibitory")),
    ("compartment", ("apical", "basal", "uncertain")),
)
SOURCE_MICROSCOPY = {"minnie65": "electron", "h01": "electron", "labid": "confocal"}


@dataclass
class ConditionEncoder:
    blocks: List[Tuple[str, Tuple[str, ...]]] = field(default_factory=lambda: [(n, tuple(v)) for n, v in DEFAULT_BLOCKS])
    cond_dim: int = 22
    enabled: Optional[List[str]] = None  # None = every block

    def __post_init__(self) -> None:
        self.blocks = [(str(n), tuple(str(v) for v in values)) for n, values in self.blocks]
        names = [n for n, _ in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate block names: {names}")
        if self.width > self.cond_dim:
            raise ValueError(f"condition blocks need {self.width} inputs, the model has cond_dim={self.cond_dim}")
        if self.enabled is not None:
            unknown = set(self.enabled) - set(names)
            if unknown:
                raise ValueError(f"enabled blocks not in the schema: {sorted(unknown)}")
            self.enabled = [n for n in names if n in set(self.enabled)]

    # ------------------------------------------------------------------ layout
    @property
    def width(self) -> int:
        return sum(len(v) for _, v in self.blocks)

    @property
    def names(self) -> List[str]:
        return [n for n, _ in self.blocks]

    def slices(self) -> Dict[str, slice]:
        out, start = {}, 0
        for name, values in self.blocks:
            out[name] = slice(start, start + len(values))
            start += len(values)
        return out

    def is_enabled(self, name: str) -> bool:
        return self.enabled is None or name in self.enabled

    # ---------------------------------------------------------------- encoding
    def encode(self, condition: Mapping[str, Any]) -> np.ndarray:
        """``{block: value}`` -> float32 [cond_dim]. Missing / None / unknown value / disabled block -> zeros."""
        vec = np.zeros(self.cond_dim, dtype=np.float32)
        for (name, values), sl in zip(self.blocks, self.slices().values()):
            value = condition.get(name)
            if value is None or (isinstance(value, float) and np.isnan(value)) or not self.is_enabled(name):
                continue
            value = str(value)
            if value in values:
                vec[sl.start + values.index(value)] = 1.0
        return vec

    def encode_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """[N, cond_dim] for a frame with one column per block (missing columns = unknown)."""
        out = np.zeros((len(frame), self.cond_dim), dtype=np.float32)
        for (name, values), sl in zip(self.blocks, self.slices().values()):
            if name not in frame.columns or not self.is_enabled(name):
                continue
            col = frame[name].astype(object).where(frame[name].notna(), None).map(lambda v: None if v is None else str(v)).to_numpy()
            for k, value in enumerate(values):
                out[col == value, sl.start + k] = 1.0
        return out

    def decode(self, vec: Sequence[float]) -> Dict[str, Optional[str]]:
        vec = np.asarray(vec, dtype=float)
        out: Dict[str, Optional[str]] = {}
        for (name, values), sl in zip(self.blocks, self.slices().values()):
            part = vec[sl]
            out[name] = values[int(np.argmax(part))] if part.max() > 0.5 else None
        return out

    def parse(self, text: str) -> Dict[str, str]:
        """``"species=mouse,compartment=apical"`` -> dict; checks block names and values."""
        out: Dict[str, str] = {}
        for item in filter(None, (t.strip() for t in str(text).split(","))):
            if "=" not in item:
                raise ValueError(f"condition item {item!r} is not key=value")
            key, value = (s.strip() for s in item.split("=", 1))
            schema = dict(self.blocks)
            if key not in schema:
                raise ValueError(f"unknown condition block {key!r}; blocks: {self.names}")
            if value not in schema[key]:
                raise ValueError(f"unknown value {value!r} of {key!r}; values: {list(schema[key])}")
            if not self.is_enabled(key):
                raise ValueError(f"block {key!r} is disabled in this model (enabled: {self.enabled})")
            out[key] = value
        return out

    # --------------------------------------------------------------- dropout
    def drop_blocks(self, cond, *, p_block: float, p_all: float, generator=None):
        """Training-time condition dropout on a torch batch [B, cond_dim]: every block independently zeroed with
        probability ``p_block``, the whole vector with ``p_all`` (returns a new tensor)."""
        import torch

        if p_block <= 0 and p_all <= 0:
            return cond
        b = cond.shape[0]
        keep = torch.ones_like(cond)
        for sl in self.slices().values():
            drop = torch.rand(b, device="cpu", generator=generator) < p_block
            keep[drop.to(cond.device), sl] = 0.0
        if p_all > 0:
            drop_all = (torch.rand(b, device="cpu", generator=generator) < p_all).to(cond.device)
            keep[drop_all] = 0.0
        return cond * keep

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        return {"blocks": [[n, list(v)] for n, v in self.blocks], "cond_dim": self.cond_dim, "enabled": self.enabled, "width": self.width}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConditionEncoder":
        return cls([(n, tuple(v)) for n, v in d["blocks"]], int(d["cond_dim"]), d.get("enabled"))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ConditionEncoder":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def enabled_from_report(report_path: Path, *, mode: str = "auto") -> Optional[List[str]]:
    """Blocks to enable from a ``condition_effects.py`` report: ``auto`` = the blocks with a real difference
    (``enabled`` true in the report), ``all`` = None (every block)."""
    if mode == "all":
        return None
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return [name for name, block in report["blocks"].items() if block.get("enabled")]


def spine_conditions(frame: pd.DataFrame, metadata_root: Path) -> pd.DataFrame:
    """One row per spine of ``frame`` (needs ``dataset``, ``neuron_id``, ``limb_id``): the value of every block.
    Cell type / compartment from the metadata tables (fixed lookup), never from stale manifest columns."""
    from ..metadata import LABID_GROUPS, cell_type_binary, dataset_kind, labid_group, load_metadata_tables

    meta = load_metadata_tables(Path(metadata_root))
    kinds = frame["dataset"].astype(str).map(dataset_kind)
    neuron = frame["neuron_id"].astype(str)
    limb = pd.to_numeric(frame["limb_id"], errors="coerce")
    out = pd.DataFrame(index=frame.index)
    out["source"] = kinds.where(kinds.isin(list(SOURCE_MICROSCOPY)), None)
    out["species"] = kinds.map({"minnie65": "mouse", "labid": "mouse", "h01": "human"})
    out["microscopy"] = out["source"].map(SOURCE_MICROSCOPY)
    health, treatment, ctype, comp = [], [], [], []
    for k, n, l in zip(kinds, neuron, limb):
        h = t = None
        if k in ("minnie65", "h01"):
            h = "healthy"
        elif k == "labid":
            g = labid_group(n)
            if g is not None:
                h, t, _ = LABID_GROUPS[g]
        health.append(h)
        treatment.append(t)
        ctype.append(cell_type_binary(meta.get("cell_types", {}).get(k, {}).get(n)))
        c = None if pd.isna(l) else meta.get("compartments", {}).get(k, {}).get((n, int(l)))
        comp.append(c)
    out["health"] = health
    out["treatment"] = treatment
    out["cell_type_binary"] = ctype
    out["compartment"] = comp  # may contain axon_like: excluded from training by the caller, never encoded
    return out


def condition_summary(conditions: pd.DataFrame, encoder: ConditionEncoder) -> Dict[str, Dict[str, int]]:
    """Counts per block value (None = unknown), for the run log."""
    out = {}
    for name, _ in encoder.blocks:
        if name in conditions.columns:
            out[name] = {str(k): int(v) for k, v in conditions[name].fillna("unknown").value_counts().items()}
    return out
