from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

SAMPLE_SURFACE = 0
SAMPLE_NEAR_SURFACE = 1
SAMPLE_UNIFORM = 2
SAMPLE_TYPE_CODES = {
    "surface": SAMPLE_SURFACE,
    "near_surface": SAMPLE_NEAR_SURFACE,
    "uniform": SAMPLE_UNIFORM,
}


def stable_seed(*parts: Any) -> int:
    """Deterministic 64-bit seed from arbitrary identifying parts."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def sample_surface_area_weighted(
    mesh: Any,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted uniform surface sampling (s-module-preprocessing.md, 3.6.1, 3.6.3).

    Faces are drawn with probability A_i / sum(A); points are placed with uniform
    barycentric coordinates. Normals interpolate vertex normals with the same
    barycentric weights and are re-normalised.

    Returns ``(points [N,3], normals [N,3], face_indices [N])``.
    """
    areas = np.asarray(mesh.area_faces, dtype=float)
    total = float(areas.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Mesh has zero or non-finite surface area.")
    face_indices = rng.choice(len(areas), size=int(n_points), p=areas / total)

    r1 = rng.random(int(n_points))
    r2 = rng.random(int(n_points))
    sqrt_r1 = np.sqrt(r1)
    bary = np.column_stack([1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2])

    faces = np.asarray(mesh.faces, dtype=int)[face_indices]
    vertices = np.asarray(mesh.vertices, dtype=float)
    points = np.einsum("nk,nkd->nd", bary, vertices[faces])

    vertex_normals = np.asarray(mesh.vertex_normals, dtype=float)
    normals = np.einsum("nk,nkd->nd", bary, vertex_normals[faces])
    norms = np.linalg.norm(normals, axis=1)
    face_normals = np.asarray(mesh.face_normals, dtype=float)[face_indices]
    bad = ~np.isfinite(norms) | (norms <= 1e-12)
    normals[bad] = face_normals[bad]
    norms[bad] = np.linalg.norm(normals[bad], axis=1)
    normals = normals / np.maximum(norms, 1e-12)[:, None]
    return points, normals, face_indices.astype(np.int64)


def winding_numbers(points: np.ndarray, triangles: np.ndarray, max_chunk_elements: int = 2_000_000) -> np.ndarray:
    """Generalized winding number of ``points`` w.r.t. a triangle soup.

    For a closed outward-oriented mesh the value is ~1 inside and ~0 outside.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=float)
    chunk = max(1, int(max_chunk_elements // max(1, len(triangles))))
    result = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk][:, None, :]
        a = triangles[None, :, 0, :] - p
        b = triangles[None, :, 1, :] - p
        c = triangles[None, :, 2, :] - p
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        det = np.einsum("pfd,pfd->pf", a, np.cross(b, c))
        denom = (
            la * lb * lc
            + np.einsum("pfd,pfd->pf", a, b) * lc
            + np.einsum("pfd,pfd->pf", b, c) * la
            + np.einsum("pfd,pfd->pf", c, a) * lb
        )
        result[start:start + chunk] = np.arctan2(det, denom).sum(axis=1) / (2.0 * np.pi)
    return result


def unsigned_distance(mesh: Any, points: np.ndarray, max_chunk_elements: int = 2_000_000) -> np.ndarray:
    """Distance to the closest surface point.

    Uses ``trimesh.proximity`` (needs a working ``rtree``); otherwise falls back to
    an exact chunked brute-force search over all triangles.
    """
    import trimesh

    points = np.asarray(points, dtype=float)
    try:
        _, distances, _ = trimesh.proximity.closest_point(mesh, points)
        return np.asarray(distances, dtype=float)
    except (ImportError, OSError):
        pass
    triangles = np.asarray(mesh.triangles, dtype=float)
    n_faces = len(triangles)
    chunk = max(1, int(max_chunk_elements // max(1, n_faces)))
    result = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        repeated_points = np.repeat(block, n_faces, axis=0)
        repeated_triangles = np.tile(triangles, (len(block), 1, 1))
        closest = trimesh.triangles.closest_point(repeated_triangles, repeated_points)
        distances = np.linalg.norm(closest - repeated_points, axis=1).reshape(len(block), n_faces)
        result[start:start + chunk] = distances.min(axis=1)
    return result


def signed_distance(mesh: Any, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Signed distance with the ``negative_inside`` convention (3.7.1).

    Magnitude: exact distance to the closest surface point. Sign: generalized
    winding number (> 0.5 means inside). Returns ``(sdf, winding)``.
    """
    points = np.asarray(points, dtype=float)
    distances = unsigned_distance(mesh, points)
    winding = winding_numbers(points, np.asarray(mesh.triangles, dtype=float))
    sign = np.where(winding > 0.5, -1.0, 1.0)
    return sign * np.asarray(distances, dtype=float), winding


def split_pool(pool_size: int, fractions: Sequence[float]) -> Tuple[int, int, int]:
    total = float(sum(fractions))
    if total <= 0:
        raise ValueError("SDF sample fractions must sum to a positive value.")
    n_surface = int(round(pool_size * fractions[0] / total))
    n_near = int(round(pool_size * fractions[1] / total))
    return n_surface, n_near, int(pool_size) - n_surface - n_near


def build_sdf_samples(
    mesh: Any,
    cap_face_mask: np.ndarray,
    *,
    pool_size: int,
    fractions: Sequence[float],
    near_sigmas: Sequence[float],
    bbox_margin: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """SDF query pool Q_surface + Q_near + Q_uniform (3.7.3-3.7.5).

    ``surface_normals`` holds the normal of the originating surface point for
    surface and near-surface samples and NaN for uniform samples.
    """
    n_surface, n_near, n_uniform = split_pool(pool_size, fractions)
    cap_face_mask = np.asarray(cap_face_mask, dtype=bool)

    surface_points, surface_normals, surface_faces = sample_surface_area_weighted(mesh, n_surface, rng)
    near_base, near_normals, near_faces = sample_surface_area_weighted(mesh, n_near, rng)
    sigmas = rng.choice(np.asarray(near_sigmas, dtype=float), size=n_near)
    near_points = near_base + rng.normal(size=(n_near, 3)) * sigmas[:, None]

    bounds = np.asarray(mesh.bounds, dtype=float)
    low = bounds[0] - float(bbox_margin)
    high = bounds[1] + float(bbox_margin)
    uniform_points = rng.uniform(low, high, size=(n_uniform, 3))

    query_points = np.vstack([surface_points, near_points, uniform_points])
    sdf, winding = signed_distance(mesh, query_points)
    source_faces = np.concatenate([surface_faces, near_faces, np.full(n_uniform, -1, dtype=np.int64)])

    return {
        "query_points": query_points.astype(np.float32),
        "sdf": sdf.astype(np.float32),
        "sample_type": np.concatenate(
            [
                np.full(n_surface, SAMPLE_SURFACE, dtype=np.uint8),
                np.full(n_near, SAMPLE_NEAR_SURFACE, dtype=np.uint8),
                np.full(n_uniform, SAMPLE_UNIFORM, dtype=np.uint8),
            ]
        ),
        "surface_normals": np.vstack(
            [surface_normals, near_normals, np.full((n_uniform, 3), np.nan)]
        ).astype(np.float32),
        "is_attachment_cap": np.concatenate(
            [cap_face_mask[surface_faces], cap_face_mask[near_faces], np.zeros(n_uniform, dtype=bool)]
        ),
        "source_face_indices": source_faces.astype(np.int32),
        "noise_sigma": np.concatenate(
            [np.zeros(n_surface), sigmas, np.zeros(n_uniform)]
        ).astype(np.float32),
        "winding_number": winding.astype(np.float32),
        "bbox_low": low.astype(np.float32),
        "bbox_high": high.astype(np.float32),
    }


def sdf_quality(
    mesh: Any,
    samples: Dict[str, np.ndarray],
    *,
    expected_counts: Tuple[int, int, int],
    surface_tolerance: float,
    interior_probe: float,
    bbox_margin: float,
    n_probe_points: int = 256,
    min_interior_fraction: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """QC of an SDF sample pool, including sign-convention sanity checks (3.7.2, 3.7.7)."""
    rng = np.random.default_rng(0) if rng is None else rng
    sdf = np.asarray(samples["sdf"], dtype=float)
    query = np.asarray(samples["query_points"], dtype=float)
    sample_type = np.asarray(samples["sample_type"])
    counts = tuple(int(np.sum(sample_type == code)) for code in (SAMPLE_SURFACE, SAMPLE_NEAR_SURFACE, SAMPLE_UNIFORM))

    all_finite = bool(np.isfinite(sdf).all() and np.isfinite(query).all())
    has_positive = bool(np.any(sdf > 0))
    has_negative = bool(np.any(sdf < 0))
    surface_abs = np.abs(sdf[sample_type == SAMPLE_SURFACE])
    surface_max_abs = float(surface_abs.max()) if len(surface_abs) else None
    surface_ok = surface_max_abs is not None and surface_max_abs <= surface_tolerance

    bounds = np.asarray(mesh.bounds, dtype=float)
    extent = float(np.max(bounds[1] - bounds[0]))
    exterior_point = bounds[1] + extent + float(bbox_margin)
    exterior_sdf = float(signed_distance(mesh, exterior_point[None, :])[0][0])

    probe_points, probe_normals, _ = sample_surface_area_weighted(mesh, n_probe_points, rng)
    inner = probe_points - probe_normals * float(interior_probe)
    interior_fraction = float(np.mean(winding_numbers(inner, np.asarray(mesh.triangles, dtype=float)) > 0.5))

    # The local frame puts the attachment centroid at the origin, so the query
    # bounding box of a local-coordinate mesh must contain (0, 0, 0).
    in_local_coordinates = bool(np.all(query.min(axis=0) <= 0) and np.all(query.max(axis=0) >= 0))

    sdf_valid = bool(
        all_finite
        and has_positive
        and has_negative
        and surface_ok
        and exterior_sdf > 0
        and interior_fraction >= min_interior_fraction
        and in_local_coordinates
        and counts == tuple(expected_counts)
    )
    return {
        "sdf_valid": sdf_valid,
        "sdf_min": float(np.min(sdf)) if len(sdf) else None,
        "sdf_max": float(np.max(sdf)) if len(sdf) else None,
        "sdf_abs_mean": float(np.mean(np.abs(sdf))) if len(sdf) else None,
        "n_surface_samples": counts[0],
        "n_near_samples": counts[1],
        "n_uniform_samples": counts[2],
        "sdf_all_finite": all_finite,
        "sdf_has_positive": has_positive,
        "sdf_has_negative": has_negative,
        "sdf_surface_max_abs": surface_max_abs,
        "sdf_exterior_check": exterior_sdf,
        "sdf_interior_fraction": interior_fraction,
        "sdf_in_local_coordinates": in_local_coordinates,
        "sdf_counts_match_config": counts == tuple(expected_counts),
    }
