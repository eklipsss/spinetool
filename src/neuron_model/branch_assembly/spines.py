"""Spine meshes for the assembly: a pool of closed meshes in the local spine frame (nm).

Local frame (module-interfaces-spec.md, section 3.4): origin = attachment centre, ``+y`` = into the spine body,
``x`` = branch tangent. Generated meshes (``generate_vae.py`` / ``generate_mogen.py``,
``generated_mesh_postprocessed.off``) and real ``local_sealed`` meshes are in this frame. The attachment region of a
mesh is found geometrically (``DownFacingCapFinder``, the evaluation's step E): its centre ``c`` is what is put on the
trunk surface; the base = vertices less than ``base_band_nm`` above ``c`` along ``y`` (what has to end up inside the trunk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class PoolSpine:
    key: str
    vertices: np.ndarray  # local frame, nm
    faces: np.ndarray
    center: np.ndarray    # attachment centre in the local frame
    height: float         # max y - center y
    base_mask: np.ndarray
    attachment_method: str


@dataclass
class SpinePool:
    spines: List[PoolSpine]
    rejected: List[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.spines)

    @classmethod
    def from_meshes(cls, meshes: Iterable[tuple], *, base_band_nm: float = 30.0, finder=None) -> "SpinePool":
        """``meshes``: (key, trimesh.Trimesh) pairs. Keeps closed, consistently wound meshes of positive volume with a found
        attachment region; the largest component of a multi-component mesh (the spine at the origin) is used."""
        from ..spine_generation.evaluation.attachment import DownFacingCapFinder

        finder = finder or DownFacingCapFinder()
        good, rejected = [], []
        for key, mesh in meshes:
            reason = None
            if mesh is None or len(mesh.faces) == 0:
                reason = "empty"
            else:
                parts = mesh.split(only_watertight=False)
                if len(parts) > 1:
                    mesh = max(parts, key=lambda m: abs(m.volume) if m.is_watertight else m.area)
                if not mesh.is_watertight:
                    reason = "not_watertight"
                elif not mesh.is_winding_consistent:
                    reason = "inconsistent_winding"
                elif mesh.volume <= 0:
                    reason = "non_positive_volume"
            region = None if reason else finder(mesh)
            if reason is None and region is None:
                reason = "no_attachment_region"
            if reason:
                rejected.append({"key": key, "reason": reason})
                continue
            v = np.asarray(mesh.vertices, dtype=float)
            c = np.asarray(region.center, dtype=float)
            good.append(PoolSpine(key=str(key), vertices=v, faces=np.asarray(mesh.faces, dtype=np.int64), center=c,
                                  height=float(v[:, 1].max() - c[1]), base_mask=v[:, 1] <= c[1] + base_band_nm, attachment_method=str(region.method)))
        return cls(good, rejected)

    @classmethod
    def from_paths(cls, paths: Sequence[Path], *, base_band_nm: float = 30.0, keys: Optional[Sequence[str]] = None) -> "SpinePool":
        import trimesh

        def gen():
            for i, p in enumerate(paths):
                try:
                    mesh = trimesh.load(p, process=True, force="mesh")
                except Exception:  # unreadable file -> rejected as empty
                    mesh = None
                yield (keys[i] if keys is not None else str(p)), mesh

        return cls.from_meshes(gen(), base_band_nm=base_band_nm)


def find_mesh_files(root: Path, name: str = "generated_mesh_postprocessed.off", limit: Optional[int] = None) -> List[Path]:
    """Mesh files under ``root``: ``<root>/**/<name>`` (the generators' sample layout), else every *.off / *.ply / *.obj."""
    root = Path(root)
    files = sorted(root.rglob(name))
    if not files:
        files = sorted(p for ext in ("*.off", "*.ply", "*.obj") for p in root.rglob(ext))
    return files[:limit] if limit else files
