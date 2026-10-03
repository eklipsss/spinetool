"""Light mesh postprocessing - only what the spec allows.

Allowed: removing very small connected components, removing degenerate /
duplicate faces, merging duplicate vertices, fixing face orientation.
Not allowed: anything that changes the shape at scale (smoothing, hole
filling of real openings, remeshing to hide artifacts). Raw and postprocessed
meshes are evaluated separately, so a model that only looks good after heavy
fixing cannot win.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def postprocess_mesh(
    mesh: Any,
    *,
    min_component_area_fraction: float = 0.01,
    degenerate_area_tol: float = 1e-12,
) -> Tuple[Any, Dict[str, Any]]:
    """Return ``(postprocessed copy, report of what was changed)``.

    A component is removed if its surface area is below
    ``min_component_area_fraction`` of the largest component's area.
    """
    import trimesh

    work = mesh.copy()
    report: Dict[str, Any] = {
        "n_faces_before": int(len(work.faces)),
        "n_vertices_before": int(len(work.vertices)),
        "min_component_area_fraction": float(min_component_area_fraction),
    }

    areas = np.asarray(work.area_faces)
    keep = areas > degenerate_area_tol
    report["n_degenerate_faces_removed"] = int((~keep).sum())
    work.update_faces(keep)

    unique = work.unique_faces()
    report["n_duplicate_faces_removed"] = int(len(work.faces) - unique.sum())
    work.update_faces(unique)
    work.remove_unreferenced_vertices()
    work.merge_vertices()

    components = work.split(only_watertight=False)
    report["n_components_before"] = int(len(components))
    if len(components) > 1:
        component_areas = np.asarray([c.area for c in components])
        kept = [c for c, a in zip(components, component_areas) if a >= min_component_area_fraction * component_areas.max()]
        report["n_components_removed"] = int(len(components) - len(kept))
        work = trimesh.util.concatenate(kept)
    else:
        report["n_components_removed"] = 0

    trimesh.repair.fix_winding(work)
    if work.is_watertight and work.volume < 0:
        work.invert()
    report["n_faces_after"] = int(len(work.faces))
    report["n_vertices_after"] = int(len(work.vertices))
    return work, report
