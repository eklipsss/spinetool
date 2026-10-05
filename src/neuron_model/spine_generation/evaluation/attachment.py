"""Attachment region (spine base) of a GENERATED mesh - step E of the plan.

Real spines get their attachment region during sealing (preprocessing stage 1: the
boundary loop of the opening + the cap that closed it). A generated mesh is closed
everywhere, so the base has to be recognised geometrically. Generated meshes live in the
same local frame as the training data (``orient_spines_w_holes``): origin = attachment
centre, the spine grows along ``+y``, the attachment cap faces ``-y``.

Primary method - ``down_facing_cap``: the face where the ``y`` axis enters the mesh from
below is the seed; the cap is the connected set of faces around it whose normal is within
``max_tilt_deg`` of ``-y`` (region growing over face adjacency, stops where the surface
turns into the neck wall). Then

- ``cap_area``   = area of those faces (subtracted for "Area", like the real cap);
- ``loop_area``  = their area projected onto the ``xz`` plane = area enclosed by the
  cap's rim seen along the spine axis ("JunctionArea");
- ``center``     = area-weighted centroid of the cap.

Fallback - ``section``: if the seed face does not face down (e.g. a saddle-shaped base),
the mesh is cut by planes ``y = const`` from just above 0 up to ``section_band_frac`` of
the height; the centre is the centroid of the lowest single-contour cross-section, the
loop area the largest single-contour cross-section in that band, the cap area the surface
below that level.

Validated on 10 real spines (2026-10-05) against the stage-1 attachment region, both on the
raw ``local_sealed`` meshes and on smooth Marching-Cubes remeshes of them (closer to what
the generators output): the 6 centre-based metrics move by <1% (median) on raw meshes and
not measurably beyond the remeshing error itself on smooth ones; JunctionArea has ~12%
median error (+6% bias) on smooth meshes, Area <1%. The plane section at ``y = 0``
alone underestimates JunctionArea by ~37% (the real cap is not planar - it straddles
``y = 0``), hence the region-growing primary. See s-module-implementation-plan.md §7.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from ...spine_morphometrics import AttachmentRegion


def _axis_seed_face(mesh: Any) -> Tuple[int, str]:
    """Face where the +y ray through ``x = z = 0`` first enters the mesh; else the lowest face."""
    start = [0.0, float(mesh.vertices[:, 1].min()) - 1.0 - float(np.ptp(mesh.vertices[:, 1])), 0.0]
    try:
        locations, _, faces = mesh.ray.intersects_location([start], [[0.0, 1.0, 0.0]])
    except Exception:  # ray backend unavailable / degenerate mesh
        faces = []
    if len(faces):
        return int(faces[int(np.argmin(locations[:, 1]))]), "axis_ray"
    return int(np.argmin(mesh.triangles_center[:, 1])), "lowest_face"


def _grow(mesh: Any, seed: int, allowed: np.ndarray) -> np.ndarray:
    neighbours: List[List[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen = {seed}
    queue = deque([seed])
    while queue:
        face = queue.popleft()
        for other in neighbours[face]:
            if other not in seen and allowed[other]:
                seen.add(other)
                queue.append(other)
    return np.fromiter(sorted(seen), dtype=int)


def _section_contours(mesh: Any, y0: float) -> List[Tuple[float, np.ndarray]]:
    """``[(area, centroid_xyz)]`` of the closed contours of the cross-section ``y = y0``."""
    section = mesh.section(plane_origin=[0.0, y0, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if section is None:
        return []
    contours = []
    for points in section.discrete:
        points = np.asarray(points, dtype=float)
        if len(points) < 3:
            continue
        x, z = points[:, 0], points[:, 2]
        cross = x * np.roll(z, -1) - np.roll(x, -1) * z
        area = 0.5 * cross.sum()
        if abs(area) < 1e-12:
            continue
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * area)
        cz = ((z + np.roll(z, -1)) * cross).sum() / (6.0 * area)
        contours.append((abs(float(area)), np.array([cx, y0, cz])))
    return contours


@dataclass(frozen=True)
class DownFacingCapFinder:
    """Callable ``mesh -> AttachmentRegion | None`` (the evaluation's ``AttachmentFinder``)."""

    max_tilt_deg: float = 60.0         # cap faces: normal within this angle of -y
    section_fallback: bool = True
    section_band_frac: float = 0.05    # fallback scans y in (0, band * height]
    section_step_frac: float = 0.005   # ... in steps of this fraction of the height

    def __call__(self, mesh: Any) -> Optional[AttachmentRegion]:
        if mesh is None or len(mesh.faces) == 0:
            return None
        region = self.down_facing_cap(mesh)
        if region is None and self.section_fallback:
            region = self.section(mesh)
        return region

    def down_facing_cap(self, mesh: Any) -> Optional[AttachmentRegion]:
        seed, seed_method = _axis_seed_face(mesh)
        facing_down = -np.asarray(mesh.face_normals)[:, 1] > np.cos(np.radians(self.max_tilt_deg))
        if not facing_down[seed]:
            return None
        faces = _grow(mesh, seed, facing_down)
        areas = np.asarray(mesh.area_faces)[faces]
        if areas.sum() <= 0:
            return None
        projected = float((areas * -np.asarray(mesh.face_normals)[faces, 1]).sum())
        center = (np.asarray(mesh.triangles_center)[faces] * areas[:, None]).sum(axis=0) / areas.sum()
        return AttachmentRegion(center=center, cap_area=float(areas.sum()), loop_area=projected, method=f"down_facing_cap:{seed_method}")

    def section(self, mesh: Any) -> Optional[AttachmentRegion]:
        height = float(mesh.vertices[:, 1].max())
        if height <= 0:
            return None
        step = self.section_step_frac * height
        single = []
        for y0 in np.arange(step, self.section_band_frac * height + 0.5 * step, step):
            contours = _section_contours(mesh, float(y0))
            if len(contours) == 1:
                single.append((float(y0), contours[0]))
        if not single:
            return None
        center = single[0][1][1]
        y_max, (loop_area, _) = max(single, key=lambda item: item[1][0])
        cap_area = float(mesh.slice_plane([0.0, y_max, 0.0], [0.0, -1.0, 0.0]).area)
        return AttachmentRegion(center=center, cap_area=cap_area, loop_area=loop_area, method="section")


def finder_from_config(cfg: Optional[dict]) -> Optional[DownFacingCapFinder]:
    """``morphometrics.attachment_finder`` config -> finder (``null`` / ``none`` -> no finder)."""
    if not cfg:
        return None
    if isinstance(cfg, str):
        cfg = {"name": cfg}
    name = cfg.get("name", "down_facing_cap")
    if name in (None, "none", "null"):
        return None
    if name != "down_facing_cap":
        raise ValueError(f"unknown attachment finder {name!r}")
    return DownFacingCapFinder(**{k: v for k, v in cfg.items() if k != "name"})
