"""Geometry of dendritic branch shafts for modules T and P.

Pure functions (no file I/O) behind the branch preprocessing
(``docs/neuron-model/tp-branch-preprocessing.md``):

- spine-vertex matching by coordinates (shaft-only surface);
- centreline recovery from cross-sections: the raw ``branch_skeleton.npy`` lies on the
  membrane (offset from the section centre ~ the shaft radius), so it only orders the
  sections; each section is a thin slab perpendicular to a smoothed tangent, the connected
  fragment nearest to the skeleton point is kept and a circle is fitted robustly (Kasa);
- section QC flags, smoothing spline of the centres, uniform arc-length nodes;
- rotation-minimising frame (double reflection) with the bending reference normal,
  bending components (kappa1, kappa2) from consecutive frames and their inverse
  (integration), which T uses to rebuild a branch from features;
- projection of attachment points onto the centreline -> (s, theta);
- tube self-contact check and tube mesh.

All lengths in nanometres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# section flag bits (a section with flag 0 is reliable)
FLAG_FEW_POINTS = 1
FLAG_LARGE_RADIUS = 2
FLAG_KINK = 4
FLAG_OUTSIDE_MESH = 8
FLAG_END_SECTION = 16
FLAG_NAMES = {
    FLAG_FEW_POINTS: "few_points",
    FLAG_LARGE_RADIUS: "large_radius",
    FLAG_KINK: "kink",
    FLAG_OUTSIDE_MESH: "outside_mesh",
    FLAG_END_SECTION: "end_section",
}


# ---------------------------------------------------------------------------
# Vertex matching
# ---------------------------------------------------------------------------


def vertex_keys(vertices: np.ndarray, decimals: int = 3) -> np.ndarray:
    """One hashable key per vertex: coordinates rounded to ``10**-decimals`` (nm)."""
    q = np.round(np.asarray(vertices, dtype=np.float64) * 10.0**decimals).astype(np.int64)
    q = np.ascontiguousarray(q.reshape(-1, 3))
    return q.view(np.dtype((np.void, q.dtype.itemsize * 3))).ravel()


def match_vertices(vertices: np.ndarray, other: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Boolean mask: which ``vertices`` coincide (after rounding) with a vertex of ``other``."""
    if len(other) == 0 or len(vertices) == 0:
        return np.zeros(len(vertices), dtype=bool)
    return np.isin(vertex_keys(vertices, decimals), vertex_keys(other, decimals))


# ---------------------------------------------------------------------------
# Polylines
# ---------------------------------------------------------------------------


def arclength(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    """Points at (almost) uniform arc-length spacing ``step``, both ends kept."""
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]
    points = points[keep]
    s = arclength(points)
    total = float(s[-1]) if len(s) else 0.0
    if len(points) < 2 or total <= 0:
        raise ValueError("polyline is degenerate (fewer than two distinct points)")
    n = max(int(math.ceil(total / step)) + 1, 2)
    targets = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(targets, s, points[:, k]) for k in range(3)])


def smoothed_tangents(points: np.ndarray, half_window: int) -> np.ndarray:
    """Unit tangents from the points ``half_window`` samples ahead and behind (clamped at the ends)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    idx = np.arange(n)
    ahead = points[np.minimum(idx + half_window, n - 1)]
    behind = points[np.maximum(idx - half_window, 0)]
    t = ahead - behind
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def section_basis(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """An orthonormal basis (ea, eb) of the plane perpendicular to the unit vector ``t``."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ea = np.cross(t, helper)
    ea /= np.linalg.norm(ea)
    return ea, np.cross(t, ea)


# ---------------------------------------------------------------------------
# Cross-sections
# ---------------------------------------------------------------------------


def fit_circle_2d(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic least-squares circle (Kasa): x^2 + y^2 = 2ax + 2by + c."""
    a_mat = np.c_[2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))]
    rhs = (xy**2).sum(axis=1)
    (a, b, c), *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    return np.array([a, b]), float(np.sqrt(max(c + a * a + b * b, 0.0)))


def robust_fit_circle_2d(xy: np.ndarray, k: float = 0.35, iterations: int = 5, min_points: int = 10) -> Tuple[np.ndarray, float]:
    """Kasa fit repeated without points whose distance to the circle exceeds ``k * r``.

    Suppresses unsegmented spine necks / bases touching the shaft ring."""
    keep = np.ones(len(xy), dtype=bool)
    center, radius = fit_circle_2d(xy)
    for _ in range(iterations):
        new = np.abs(np.linalg.norm(xy - center, axis=1) - radius) < k * radius
        if new.sum() < min_points or (new == keep).all():
            break
        keep = new
        center, radius = fit_circle_2d(xy[keep])
    return center, radius


def fit_ellipse_2d(xy: np.ndarray) -> Optional[Tuple[np.ndarray, float, float, np.ndarray]]:
    """Direct least-squares ellipse (Fitzgibbon et al. 1999, numerically stable form of Halir & Flusser 1998).

    Returns (centre [2], a, b, major-axis unit vector [2]) with a >= b, or None if the conic is not an ellipse."""
    if len(xy) < 6:
        return None
    m = xy.mean(axis=0)
    sc = float(np.sqrt(((xy - m) ** 2).sum(axis=1).mean())) or 1.0
    x, y = ((xy - m) / sc).T
    d1 = np.c_[x * x, x * y, y * y]
    d2 = np.c_[x, y, np.ones_like(x)]
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        t = -np.linalg.solve(s3, s2.T)
        mm = s1 + s2 @ t
        mm = np.array([mm[2] / 2.0, -mm[1], mm[0] / 2.0])
        w, v = np.linalg.eig(mm)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    cond = 4.0 * v[0] * v[2] - v[1] ** 2
    ok = np.flatnonzero(cond > 0)
    if len(ok) == 0:
        return None
    a1 = v[:, ok[0]]
    A, B, C = a1
    D, E, F = t @ a1
    q = np.array([[A, B / 2.0], [B / 2.0, C]])
    try:
        c = np.linalg.solve(2.0 * q, -np.array([D, E]))
    except np.linalg.LinAlgError:
        return None
    f0 = float(c @ q @ c + D * c[0] + E * c[1] + F)
    lam, vec = np.linalg.eigh(q)
    if not (np.all(lam * -f0 > 0)):
        return None
    axes = np.sqrt(-f0 / lam)  # semi-axes along the eigenvectors
    i_major = int(np.argmax(axes))
    a, b = float(axes[i_major]) * sc, float(axes[1 - i_major]) * sc
    return m + sc * c, a, b, vec[:, i_major] / np.linalg.norm(vec[:, i_major])


def ellipse_radial_residual(xy: np.ndarray, center: np.ndarray, a: float, b: float, major: np.ndarray) -> np.ndarray:
    """|p - c| minus the ellipse radius in the direction of p (an approximation of the geometric distance)."""
    d = xy - center
    u = d @ major
    w = d @ np.array([-major[1], major[0]])
    psi = np.arctan2(w, u)
    r_e = a * b / np.sqrt((b * np.cos(psi)) ** 2 + (a * np.sin(psi)) ** 2)
    return np.linalg.norm(d, axis=1) - r_e


def robust_fit_ellipse_2d(xy: np.ndarray, circle_center: np.ndarray, circle_r: float, k: float = 0.35, iterations: int = 4, min_points: int = 10,
                          min_ratio: float = 0.2, max_a_factor: float = 2.5, k_init: float = 0.6) -> Optional[Tuple[np.ndarray, float, float, np.ndarray]]:
    """Ellipse fit started from the robust circle: points farther than ``k_init * r`` from the circle are dropped first (spine
    necks / bases; looser than the circle's own ``k``, otherwise the ends of the minor axis of a flattened section are lost
    too), then the ellipse is refitted without points farther than ``k * sqrt(a b)`` from it. Rejected (None) when b / a <
    ``min_ratio``, a > ``max_a_factor * r`` or the centre is more than r from the circle centre (partial arcs)."""
    # start from ALL points of the fragment (a circle-based trim would drop the ends of the minor axis of a flattened
    # section: the robust circle of such a section converges to the major-axis arcs), protrusions are removed by the
    # ellipse residual below; ``k_init`` only guards against far outliers (> (1 + k_init) r from the circle centre)
    keep = np.linalg.norm(xy - circle_center, axis=1) < (1.0 + k_init) * circle_r
    if keep.sum() < min_points:
        return None
    fit = fit_ellipse_2d(xy[keep])
    for _ in range(iterations):
        if fit is None:
            return None
        c, a, b, major = fit
        new = np.abs(ellipse_radial_residual(xy, c, a, b, major)) < k * math.sqrt(a * b)
        if new.sum() < min_points or (new == keep).all():
            break
        keep = new
        fit = fit_ellipse_2d(xy[keep])
    if fit is None:
        return None
    c, a, b, major = fit
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0 or b / a < min_ratio or a > max_a_factor * circle_r or np.linalg.norm(c - circle_center) > circle_r:
        return None
    return fit


def nearest_cluster(xy: np.ndarray, eps: float) -> np.ndarray:
    """Mask of the eps-connected fragment of the slice points containing the point nearest to the origin."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    pairs = cKDTree(xy).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    _, labels = connected_components(graph, directed=False)
    return labels == labels[int(np.argmin(np.linalg.norm(xy, axis=1)))]


@dataclass
class Sections:
    """Per-section estimates along the initial (skeleton) curve."""

    points: np.ndarray  # [m,3] skeleton points g_j
    tangents: np.ndarray  # [m,3]
    centers: np.ndarray  # [m,3] (NaN where no fit)
    radii: np.ndarray  # [m] (NaN where no fit)
    n_points: np.ndarray  # [m] points of the kept fragment
    ellipse_center: Optional[np.ndarray] = None  # [m,3] (NaN where no ellipse); only with fit_sections(fit_ellipse=True)
    ellipse_a: Optional[np.ndarray] = None  # [m] semi-major axis, nm
    ellipse_b: Optional[np.ndarray] = None  # [m] semi-minor axis, nm
    ellipse_major: Optional[np.ndarray] = None  # [m,3] unit vector of the major axis (in the section plane)


def fit_sections(
    points: np.ndarray,
    tangents: np.ndarray,
    shaft_vertices: np.ndarray,
    *,
    slab: float,
    max_r: float,
    cluster_eps: float,
    robust_k: float,
    min_points: int,
    tree=None,
    fit_ellipse: bool = False,
) -> Sections:
    """Circle fit of every cross-section (see module docstring); with ``fit_ellipse`` also an ellipse (flattened sections)."""
    from scipy.spatial import cKDTree

    shaft_vertices = np.asarray(shaft_vertices, dtype=float)
    m = len(points)
    centers = np.full((m, 3), np.nan)
    radii = np.full(m, np.nan)
    npts = np.zeros(m, dtype=np.int64)
    ell = (np.full((m, 3), np.nan), np.full(m, np.nan), np.full(m, np.nan), np.full((m, 3), np.nan)) if fit_ellipse else (None, None, None, None)
    if len(shaft_vertices) == 0:
        return Sections(points, tangents, centers, radii, npts, *ell)
    tree = tree if tree is not None else cKDTree(shaft_vertices)
    reach = math.sqrt(slab * slab + max_r * max_r)
    candidates = tree.query_ball_point(points, r=reach)
    for j in range(m):
        idx = candidates[j]
        if len(idx) < min_points:
            continue
        g, t = points[j], tangents[j]
        ea, eb = section_basis(t)
        d = shaft_vertices[idx] - g
        along = d @ t
        xy = np.c_[d @ ea, d @ eb]
        sel = (np.abs(along) < slab) & (np.linalg.norm(xy, axis=1) < max_r)
        xy = xy[sel]
        if len(xy) < min_points:
            continue
        xy = xy[nearest_cluster(xy, cluster_eps)]
        npts[j] = len(xy)
        if len(xy) < min_points:
            continue
        c2, r = robust_fit_circle_2d(xy, robust_k, min_points=min_points)
        centers[j] = g + c2[0] * ea + c2[1] * eb
        radii[j] = r
        if fit_ellipse:
            e = robust_fit_ellipse_2d(xy, c2, r, robust_k, min_points=min_points)
            if e is not None:
                ce, a, b, major = e
                ell[0][j] = g + ce[0] * ea + ce[1] * eb
                ell[1][j], ell[2][j] = a, b
                ell[3][j] = major[0] * ea + major[1] * eb
    return Sections(points, tangents, centers, radii, npts, *ell)


def flag_sections(
    sections: Sections,
    *,
    min_points: int,
    large_radius_factor: float,
    kink_factor: float,
    end_distance_factor: float,
    inside: Optional[np.ndarray] = None,
    kink_floor_nm: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Bit flags per section (0 = reliable) and the branch median radius ``r_med``.

    A centre is a kink when it is farther than ``max(kink_factor * r_med, kink_floor_nm)`` from the midpoint of its
    neighbours: the noise of the centres (mesh roughness, ~60-90 nm on Minnie65) does not scale with the radius, so a
    limit proportional to r alone flags most sections of thin branches."""
    m = len(sections.radii)
    flags = np.zeros(m, dtype=np.int64)
    fitted = np.isfinite(sections.radii)
    flags[~fitted | (sections.n_points < min_points)] |= FLAG_FEW_POINTS
    if not fitted.any():
        return flags, float("nan")
    r_med = float(np.median(sections.radii[fitted]))
    flags[fitted & (sections.radii > large_radius_factor * r_med)] |= FLAG_LARGE_RADIUS
    # kink: centre far from the midpoint of the neighbouring fitted centres. One displaced centre also
    # shifts the midpoints of its two neighbours, so only the local maximum of the deviation is flagged.
    fitted_idx = np.flatnonzero(fitted)
    dev = np.zeros(len(fitted_idx))
    for k in range(1, len(fitted_idx) - 1):
        a, j, b = fitted_idx[k - 1], fitted_idx[k], fitted_idx[k + 1]
        dev[k] = np.linalg.norm(sections.centers[j] - 0.5 * (sections.centers[a] + sections.centers[b]))
    for k in range(1, len(fitted_idx) - 1):
        if dev[k] > max(kink_factor * r_med, kink_floor_nm) and dev[k] >= dev[k - 1] and dev[k] >= dev[k + 1]:
            flags[fitted_idx[k]] |= FLAG_KINK
    if inside is not None:
        flags[fitted & ~np.asarray(inside, dtype=bool)] |= FLAG_OUTSIDE_MESH
    s = arclength(sections.points)
    d_end = end_distance_factor * r_med
    flags[(s < d_end) | (s > s[-1] - d_end)] |= FLAG_END_SECTION
    return flags, r_med


def flag_reasons(flags: np.ndarray) -> dict:
    return {name: int(((flags & bit) != 0).sum()) for bit, name in FLAG_NAMES.items()}


# ---------------------------------------------------------------------------
# Smoothed centreline and nodes
# ---------------------------------------------------------------------------


@dataclass
class Centerline:
    """Smoothed centreline sampled densely; ``s`` is the arc length (nm) of the dense samples."""

    dense: np.ndarray  # [n,3]
    s: np.ndarray  # [n]
    param: np.ndarray  # [n] chord parameter of the fitted spline
    spline: object  # callable param -> [.,3]
    rms_to_centers: float

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s_query: np.ndarray) -> np.ndarray:
        p = np.interp(s_query, self.s, self.param)
        return self.spline(p)

    def tangent_at(self, s_query: np.ndarray) -> np.ndarray:
        p = np.interp(s_query, self.s, self.param)
        d = self.spline.derivative()(p)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def s_of_param(self, p: np.ndarray) -> np.ndarray:
        return np.interp(p, self.param, self.s)


class _VectorSpline:
    def __init__(self, splines):
        self._s = splines

    def __call__(self, p):
        return np.column_stack([sp(p) for sp in self._s])

    def derivative(self):
        return _VectorSpline([sp.derivative() for sp in self._s])


def smoothing_lambda(x: np.ndarray, smoothing_nm: float) -> float:
    """Penalty of a cubic smoothing spline with an equivalent kernel width ``smoothing_nm``.

    For sum (y_i - f(x_i))^2 + lam * int f''^2 with samples every ``delta`` the equivalent kernel has
    width h ~ (lam * delta)^(1/4) (Silverman), so lam = h^4 / delta. An explicit physical scale is used
    instead of generalised cross-validation: GCV picks lam ~ 0 (pure interpolation) for densely and
    correlatedly sampled section centres, and the bending then follows the fitting noise."""
    delta = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    return float(smoothing_nm) ** 4 / max(delta, 1e-9)


def smooth_series(x: np.ndarray, y: np.ndarray, smoothing_nm: float):
    """Smoothing spline of samples ``y`` [n] or [n,k] over increasing ``x`` with a physical smoothing width."""
    from scipy.interpolate import make_smoothing_spline

    lam = smoothing_lambda(x, smoothing_nm)
    if y.ndim == 1:
        return make_smoothing_spline(x, y, lam=lam)
    return _VectorSpline([make_smoothing_spline(x, y[:, k], lam=lam) for k in range(y.shape[1])])


def smooth_centerline(centers: np.ndarray, *, smoothing_nm: float, dense_step: float = 50.0) -> Centerline:
    """Smoothing spline through ordered reliable section centres (width ``smoothing_nm``), sampled densely;
    ``rms_to_centers`` is reported for QC."""
    centers = np.asarray(centers, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(centers, axis=0), axis=1) > 1e-6]
    centers = centers[keep]
    if len(centers) < 5:
        raise ValueError(f"need at least 5 reliable section centres, got {len(centers)}")
    x = arclength(centers)
    spline = smooth_series(x, centers, smoothing_nm)
    rms = float(np.sqrt(np.mean(np.sum((spline(x) - centers) ** 2, axis=1))))
    n = max(int(math.ceil(x[-1] / dense_step)) + 1, 5)
    param = np.linspace(0.0, float(x[-1]), n)
    dense = spline(param)
    return Centerline(dense=dense, s=arclength(dense), param=param, spline=spline, rms_to_centers=rms)


def node_grid(length: float, step: float) -> Tuple[np.ndarray, float]:
    """Nodes s_j = j * ds_b with M = ceil(L / step) + 1 and ds_b = L / (M - 1) (L reproduced exactly)."""
    m = max(int(math.ceil(length / step)) + 1, 2)
    ds = length / (m - 1)
    return np.arange(m) * ds, ds


# ---------------------------------------------------------------------------
# Rotation-minimising frame and bending
# ---------------------------------------------------------------------------


def initial_normal(points: np.ndarray, t0: np.ndarray, straight_fraction: float) -> Tuple[np.ndarray, str]:
    """Zero-angle direction e1(0): towards the bend (farthest point from the chord), projected
    on the normal plane; a global axis for practically straight branches."""
    points = np.asarray(points, dtype=float)
    chord = points[-1] - points[0]
    length = float(arclength(points)[-1])
    chord_len = float(np.linalg.norm(chord))
    if chord_len > 0:
        u = chord / chord_len
        rel = points - points[0]
        dist = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    else:
        dist = np.linalg.norm(points - points[0], axis=1)
    j = int(np.argmax(dist))
    if dist[j] >= straight_fraction * length:
        a = points[j] - points[0]
        e1 = a - (a @ t0) * t0
        norm = float(np.linalg.norm(e1))
        if norm > 1e-9 * max(length, 1.0):
            return e1 / norm, "bending"
    axes = np.eye(3)
    a = axes[int(np.argmin(np.abs(axes @ t0)))]
    e1 = a - (a @ t0) * t0
    return e1 / np.linalg.norm(e1), "global_axis"


def rotation_minimizing_frames(points: np.ndarray, tangents: np.ndarray, e1_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Double-reflection RMF (Wang et al., 2008): returns (T, E1, E2), each [m,3]."""
    points = np.asarray(points, dtype=float)
    t = np.asarray(tangents, dtype=float)
    m = len(points)
    r = np.zeros((m, 3))
    r[0] = e1_0 - (e1_0 @ t[0]) * t[0]
    r[0] /= np.linalg.norm(r[0])
    for i in range(m - 1):
        v1 = points[i + 1] - points[i]
        c1 = v1 @ v1
        if c1 <= 0:
            r[i + 1] = r[i]
            continue
        r_l = r[i] - (2.0 / c1) * (v1 @ r[i]) * v1
        t_l = t[i] - (2.0 / c1) * (v1 @ t[i]) * v1
        v2 = t[i + 1] - t_l
        c2 = v2 @ v2
        r_next = r_l if c2 <= 1e-30 else r_l - (2.0 / c2) * (v2 @ r_l) * v2
        r_next = r_next - (r_next @ t[i + 1]) * t[i + 1]
        r[i + 1] = r_next / np.linalg.norm(r_next)
    e2 = np.cross(t, r)
    return t, r, e2


def frames_matrix(t: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """[m,3,3] with columns (t, e1, e2)."""
    return np.stack([t, e1, e2], axis=2)


def bending_from_frames(t: np.ndarray, e1: np.ndarray, e2: np.ndarray, ds: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kappa1, kappa2, twist) per step [m-1] from relative rotations R_j^T R_{j+1} = exp([w]x):
    kappa1 = w_e2 / ds, kappa2 = -w_e1 / ds, twist = |w_t| / ds (~0 for an RMF)."""
    from scipy.spatial.transform import Rotation

    frames = frames_matrix(t, e1, e2)
    rel = np.einsum("mji,mjk->mik", frames[:-1], frames[1:])  # R_j^T R_{j+1}
    w = Rotation.from_matrix(rel).as_rotvec()
    return w[:, 2] / ds, -w[:, 1] / ds, np.abs(w[:, 0]) / ds


def integrate_bending(kappa1: np.ndarray, kappa2: np.ndarray, ds: float, gamma0: Optional[np.ndarray] = None, frame0: Optional[np.ndarray] = None):
    """Inverse of :func:`bending_from_frames`: R_{j+1} = R_j exp([w_j]x), w_j = ds (0, -kappa2, kappa1),
    gamma_{j+1} = gamma_j + ds * normalize(t_j + t_{j+1}). Returns (gamma [m,3], frames [m,3,3])."""
    from scipy.spatial.transform import Rotation

    kappa1 = np.asarray(kappa1, dtype=float)
    kappa2 = np.asarray(kappa2, dtype=float)
    m = len(kappa1) + 1
    frames = np.zeros((m, 3, 3))
    frames[0] = np.eye(3) if frame0 is None else frame0
    gamma = np.zeros((m, 3))
    gamma[0] = 0.0 if gamma0 is None else gamma0
    steps = Rotation.from_rotvec(np.column_stack([np.zeros(m - 1), -kappa2 * ds, kappa1 * ds])).as_matrix()
    for j in range(m - 1):
        nxt = frames[j] @ steps[j]
        u, _, vt = np.linalg.svd(nxt)
        frames[j + 1] = u @ vt
        direction = frames[j][:, 0] + frames[j + 1][:, 0]
        gamma[j + 1] = gamma[j] + ds * direction / np.linalg.norm(direction)
    return gamma, frames


# ---------------------------------------------------------------------------
# Projection of attachment points
# ---------------------------------------------------------------------------


@dataclass
class Projection:
    s: np.ndarray  # arc length of the foot point [n]
    foot: np.ndarray  # [n,3]
    v_perp: np.ndarray  # [n,3] component of (A - foot) perpendicular to the tangent
    theta: np.ndarray  # [n] in (-pi, pi]
    beyond: np.ndarray  # [n] distance past an end of the centreline along its tangent (0 inside)
    at_end: np.ndarray  # [n] -1 start, +1 end, 0 interior


def project_points(points: np.ndarray, gamma: np.ndarray, s_nodes: np.ndarray, t: np.ndarray, e1: np.ndarray) -> Projection:
    """Closest point of each attachment centre on the node polyline, arc length, angle in the RMF."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    a = gamma[:-1]
    seg = gamma[1:] - gamma[:-1]
    seg_len2 = np.maximum((seg**2).sum(axis=1), 1e-30)
    n = len(points)
    s_out = np.zeros(n)
    foot = np.zeros((n, 3))
    v_perp = np.zeros((n, 3))
    theta = np.zeros(n)
    beyond = np.zeros(n)
    at_end = np.zeros(n, dtype=np.int64)
    for i, p in enumerate(points):
        raw = ((p - a) * seg).sum(axis=1) / seg_len2
        u = np.clip(raw, 0.0, 1.0)
        closest = a + u[:, None] * seg
        j = int(np.argmin(((p - closest) ** 2).sum(axis=1)))
        uj = float(u[j])
        s_out[i] = s_nodes[j] + uj * (s_nodes[j + 1] - s_nodes[j])
        foot[i] = closest[j]
        if j == 0 and raw[0] < 0:
            beyond[i] = float(-(p - gamma[0]) @ t[0])
            at_end[i] = -1
        elif j == len(seg) - 1 and raw[-1] > 1:
            beyond[i] = float((p - gamma[-1]) @ t[-1])
            at_end[i] = 1
        tt = (1 - uj) * t[j] + uj * t[j + 1]
        tt /= np.linalg.norm(tt)
        ee1 = (1 - uj) * e1[j] + uj * e1[j + 1]
        ee1 = ee1 - (ee1 @ tt) * tt
        ee1 /= np.linalg.norm(ee1)
        ee2 = np.cross(tt, ee1)
        v = p - foot[i]
        v_perp[i] = v - (v @ tt) * tt
        theta[i] = math.atan2(float(v_perp[i] @ ee2), float(v_perp[i] @ ee1))
        if theta[i] <= -math.pi:
            theta[i] += 2 * math.pi
    return Projection(s_out, foot, v_perp, theta, np.maximum(beyond, 0.0), at_end)


# ---------------------------------------------------------------------------
# Tube checks / mesh
# ---------------------------------------------------------------------------


def tube_self_contact(gamma: np.ndarray, radius: np.ndarray, s_nodes: np.ndarray, ds: float) -> Tuple[bool, float]:
    """Non-local self-contact of the round tube: two nodes farther apart along the axis than
    ``pi * max(r) + 4 ds`` whose centres are closer than r_i + r_j. Returns (contact, worst margin nm)."""
    from scipy.spatial import cKDTree

    radius = np.asarray(radius, dtype=float)
    excl = math.pi * float(radius.max()) + 4.0 * ds
    tree = cKDTree(gamma)
    pairs = tree.query_pairs(2.0 * float(radius.max()), output_type="ndarray")
    if len(pairs) == 0:
        return False, float("inf")
    i, j = pairs[:, 0], pairs[:, 1]
    far = np.abs(s_nodes[i] - s_nodes[j]) > excl
    if not far.any():
        return False, float("inf")
    margin = np.linalg.norm(gamma[i[far]] - gamma[j[far]], axis=1) - radius[i[far]] - radius[j[far]]
    worst = float(margin.min())
    return worst < 0, worst


def tube_mesh(gamma: np.ndarray, radius: np.ndarray, e1: np.ndarray, e2: np.ndarray, n_theta: int = 16, caps: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Round tube X(s, theta) = gamma + r (cos theta e1 + sin theta e2) with fan caps; outward faces."""
    m = len(gamma)
    ang = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    ring = np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :]
    verts = (gamma[:, None, :] + radius[:, None, None] * ring).reshape(-1, 3)
    faces = []
    for j in range(m - 1):
        for k in range(n_theta):
            a = j * n_theta + k
            b = j * n_theta + (k + 1) % n_theta
            c = (j + 1) * n_theta + k
            d = (j + 1) * n_theta + (k + 1) % n_theta
            faces.append((a, b, c))  # normal ~ e_theta x t = outward radial direction
            faces.append((b, d, c))
    faces = np.asarray(faces, dtype=np.int64)
    if caps:
        start = len(verts)
        verts = np.vstack([verts, gamma[0], gamma[-1]])
        cap0 = [(start, (k + 1) % n_theta, k) for k in range(n_theta)]
        last = (m - 1) * n_theta
        cap1 = [(start + 1, last + k, last + (k + 1) % n_theta) for k in range(n_theta)]
        faces = np.vstack([faces, np.asarray(cap0 + cap1, dtype=np.int64)])
    # make the winding outward (positive signed volume)
    tri = verts[faces]
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0
    if vol < 0:
        faces = faces[:, ::-1]
    return verts, faces

def ellipse_tube_mesh(gamma: np.ndarray, a: np.ndarray, b: np.ndarray, angle: np.ndarray, e1: np.ndarray, e2: np.ndarray, n_theta: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Elliptic tube X(s, th) = gamma + a cos(th) u + b sin(th) v, u = cos(phi) e1 + sin(phi) e2, v = -sin(phi) e1 + cos(phi) e2
    (phi = ``angle`` of the major axis in the RMF), no caps."""
    u = np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * e2
    v = -np.sin(angle)[:, None] * e1 + np.cos(angle)[:, None] * e2
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    ring = (a[:, None, None] * np.cos(th)[None, :, None] * u[:, None, :] + b[:, None, None] * np.sin(th)[None, :, None] * v[:, None, :])
    verts = (gamma[:, None, :] + ring).reshape(-1, 3)
    m = len(gamma)
    faces = []
    for j in range(m - 1):
        for k in range(n_theta):
            a0, a1 = j * n_theta + k, j * n_theta + (k + 1) % n_theta
            b0, b1 = a0 + n_theta, a1 + n_theta
            faces.append((a0, a1, b1))
            faces.append((a0, b1, b0))
    return verts, np.asarray(faces, dtype=np.int64)
