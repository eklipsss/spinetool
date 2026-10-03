"""Technical validity of generated / reconstructed meshes and the aggregate rates.

Per-mesh checks (spec, "Тестирование"): watertight, manifold,
n_connected_components, self_intersections, degenerate_faces, genus,
positive_volume, surface_touches_bbox. Reuses the preprocessing QC
(``spine_geometry.mesh_geometry_report``, CGAL self-intersections) so real and
generated meshes are judged by exactly the same code.

Aggregates: valid_mesh_rate, watertight_rate, single_component_rate,
genus0_rate, self_intersection_rate - computed separately for raw and
postprocessed meshes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from ...spine_geometry import mesh_geometry_report


def surface_touches_bbox(mesh: Any, low: Sequence[float], high: Sequence[float], *, rel_tol: float = 1e-3) -> bool:
    """True if any vertex lies on (within ``rel_tol`` of the box size from) a face of the frozen box."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    tol = rel_tol * float(np.max(high - low))
    vertices = np.asarray(mesh.vertices, dtype=float)
    return bool(((vertices - low) <= tol).any() or ((high - vertices) <= tol).any())


def validate_mesh(
    mesh: Optional[Any],
    *,
    bbox: Optional[Sequence[Sequence[float]]] = None,
    check_self_intersections: bool = True,
    degenerate_area_tol: float = 1e-12,
) -> Dict[str, Any]:
    """Validity report for one mesh; ``mesh=None`` (no surface extracted) is reported as invalid."""
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return {"mesh_present": False, "is_valid": False}
    report = mesh_geometry_report(mesh, degenerate_area_tol=degenerate_area_tol, check_self_intersections=check_self_intersections)
    report["mesh_present"] = True
    report["surface_touches_bbox"] = surface_touches_bbox(mesh, *bbox) if bbox is not None else None
    report["is_valid"] = bool(
        report["is_watertight"]
        and report["is_manifold"]
        and report["n_connected_components"] == 1
        and report["n_degenerate_faces"] == 0
        and report["has_positive_finite_volume"]
        and report["has_finite_coordinates"]
        and report["has_self_intersections"] is not True
        and report["surface_touches_bbox"] is not True
    )
    return report


def aggregate_validity(reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    reports = list(reports)
    n = len(reports)
    if n == 0:
        return {"n": 0}

    def rate(predicate) -> float:
        return float(sum(1 for r in reports if predicate(r)) / n)

    present = [r for r in reports if r.get("mesh_present")]
    return {
        "n": n,
        "mesh_present_rate": len(present) / n,
        "valid_mesh_rate": rate(lambda r: r.get("is_valid", False)),
        "watertight_rate": rate(lambda r: r.get("is_watertight", False)),
        "single_component_rate": rate(lambda r: r.get("n_connected_components") == 1),
        "genus0_rate": rate(lambda r: r.get("genus") == 0),
        "self_intersection_rate": rate(lambda r: r.get("has_self_intersections") is True),
        "positive_volume_rate": rate(lambda r: r.get("has_positive_finite_volume", False)),
        "touches_bbox_rate": rate(lambda r: r.get("surface_touches_bbox") is True),
    }
