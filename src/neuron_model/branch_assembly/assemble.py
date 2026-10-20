"""Assemble(T, P, S): trunk tube + spine meshes -> one closed branch surface (module-interfaces-spec.md, section 8).

For every placement (s_i, theta_i) of P:

1. anchor a_i = gamma(s_i) + r(s_i) n_i, n_i = cos(theta_i) e1 + sin(theta_i) e2; tangent t'_i = t(s_i) orthogonalised to
   n_i; rotation R_i with rows [t'_i; n_i; t'_i x n_i] (local x -> tangent, y -> outward normal, z -> t' x n);
2. spine vertices p = R_i^T (p_loc - c_i) + a_i - delta_i n_i, c_i = attachment centre of the spine mesh;
   delta_i >= 0 = the smallest sink along -n_i that puts the base (vertices < ``base_band_nm`` above c_i) inside the
   trunk by ``base_margin_nm`` (bisection on the tube SDF); more than ``max_immersion_fraction`` of the spine height ->
   the spine is rejected (``base_not_immersed``: e.g. a base wider than a thin trunk);
3. checks before the union: the spine body (vertices higher than delta_i + base band above c_i) must not enter the
   trunk (``penetrates_trunk``), and a spine must not overlap an already placed one by more than
   ``collision_volume_nm3`` (``collides``; Manifold intersection, only for overlapping bounding boxes). A rejected
   placement is retried with another pool spine (``on_failure: resample``, up to ``max_tries``) or dropped (``drop``);
4. the trunk and all accepted spines are united by one Manifold boolean (manifold3d): the immersed bases vanish inside
   the union, no explicit stitching of boundary loops;
5. the result is checked: closed, consistently wound, one component, no degenerate faces, positive volume (Manifold
   output is free of self-intersections by construction).

The seam is not remeshed (spec 8.1 step 6 is not implemented in this version): the junction keeps the triangles the
boolean cut out of both surfaces. All coordinates are shifted by the trunk centroid inside the computation and back on
output (float64 throughout). Units: nm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .spines import PoolSpine, SpinePool
from .trunk import DenseTrunk, densify, trunk_mesh


@dataclass
class AssemblyConfig:
    trunk_step_nm: float = 100.0          # dense centreline step of the tube
    trunk_n_theta: int = 32               # vertices per tube ring
    base_band_nm: float = 30.0            # spine base = vertices up to this height above the attachment centre
    base_margin_nm: float = 10.0          # the base ends up at least this deep inside the trunk
    base_inside_quantile: float = 1.0     # share of base vertices that must be inside (1 = all)
    max_immersion_fraction: float = 0.5   # delta <= this * spine height
    penetration_tol_nm: float = 20.0      # body vertex deeper than this inside the trunk -> penetrates_trunk
    max_penetration_fraction: float = 0.0
    collision_volume_nm3: float = 1000.0  # overlap of two spines above this -> collides (10 nm cube)
    on_failure: str = "resample"          # resample | drop
    max_tries: int = 5
    seed: int = 0

    @classmethod
    def from_mapping(cls, values: Optional[Dict[str, Any]]) -> "AssemblyConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (values or {}).items() if k in known})


@dataclass
class AssemblyResult:
    vertices: Optional[np.ndarray]
    faces: Optional[np.ndarray]
    branch: Dict[str, Any]
    spines: List[Dict[str, Any]] = field(default_factory=list)


def _manifold(vertices: np.ndarray, faces: np.ndarray):
    import manifold3d as m3d

    mf = m3d.Manifold(m3d.Mesh64(vert_properties=np.ascontiguousarray(vertices, dtype=np.float64), tri_verts=np.ascontiguousarray(faces, dtype=np.uint64)))
    return mf, str(mf.status()).split(".")[-1]


def rotation(tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Rows [t'; n; t' x n], t' = tangent orthogonalised to the normal (same convention as ``local_to_global``)."""
    n = normal / np.linalg.norm(normal)
    t = tangent - (tangent @ n) * n
    t /= np.linalg.norm(t)
    return np.stack([t, n, np.cross(t, n)])


def place(spine: PoolSpine, trunk: DenseTrunk, s: float, theta: float, cfg: AssemblyConfig) -> Dict[str, Any]:
    """Pose of one spine on the trunk: vertices in the trunk frame + immersion depth, or a failure reason."""
    f = trunk.surface_frame(np.array([s]), np.array([theta]))
    a, n, t, r = f["anchor"][0], f["normal"][0], f["tangent"][0], float(f["radius"][0])
    R = rotation(t, n)
    v0 = (spine.vertices - spine.center) @ R + a  # R^T (p - c) + a for row vectors
    base0 = v0[spine.base_mask]
    q = cfg.base_inside_quantile

    def excess(delta: float) -> float:  # > 0: the base is not deep enough
        return float(np.quantile(trunk.signed_distance(base0 - delta * n), q)) + cfg.base_margin_nm

    # the base must not pass the axis (deeper than r it comes out on the other side: excess is not monotonic)
    delta_max = min(cfg.max_immersion_fraction * spine.height, r)
    out: Dict[str, Any] = {"anchor": a, "normal": n, "radius_at_s": r, "delta_nm": None, "status": "ok", "reason": None}
    grid = np.linspace(0.0, delta_max, 33)
    hit = next((k for k, d in enumerate(grid) if excess(float(d)) <= 0), None)
    if hit is None:
        out.update(status="rejected", reason="base_not_immersed")
        return out
    delta = float(grid[hit])
    if hit > 0:  # refine between the last failing and the first passing grid depth
        lo, hi = float(grid[hit - 1]), delta
        while hi - lo > 0.5:
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if excess(mid) <= 0 else (mid, hi)
        delta = hi
    v = v0 - delta * n
    out.update(delta_nm=float(delta), vertices=v)
    body = (spine.vertices[:, 1] - spine.center[1]) > delta + cfg.base_band_nm + cfg.penetration_tol_nm
    if body.any():
        deep = trunk.signed_distance(v[body]) < -cfg.penetration_tol_nm
        out["penetration_fraction"] = float(deep.mean())
        if deep.mean() > cfg.max_penetration_fraction:
            out.update(status="rejected", reason="penetrates_trunk")
    return out


def _boxes_overlap(a, b) -> bool:
    return all(a[k] <= b[k + 3] and b[k] <= a[k + 3] for k in range(3))


def assemble_branch(trunk_record: Dict[str, Any], s: Sequence[float], theta: Sequence[float], pool: SpinePool, cfg: AssemblyConfig,
                    rng: Optional[np.random.Generator] = None) -> AssemblyResult:
    import manifold3d as m3d
    import trimesh

    rng = rng or np.random.default_rng(cfg.seed)
    trunk = densify(trunk_record, cfg.trunk_step_nm)
    origin = trunk.gamma.mean(axis=0)
    trunk.gamma = trunk.gamma - origin  # work near 0, shift back on output
    tv, tf = trunk_mesh(trunk, cfg.trunk_n_theta)
    trunk_mf, status = _manifold(tv, tf)
    branch: Dict[str, Any] = {"length_nm": trunk.length, "n_placements": int(len(s)), "trunk_status": status, "seam_remesh": "none"}
    if status != "NoError":
        branch.update(status="failed", reason=f"trunk_mesh:{status}")
        return AssemblyResult(None, None, branch)

    accepted: List[Any] = []  # (manifold, bbox)
    spine_rows: List[Dict[str, Any]] = []
    for i, (s_i, th_i) in enumerate(zip(s, theta)):
        tries = cfg.max_tries if cfg.on_failure == "resample" else 1
        row: Dict[str, Any] = {"spine_index": i, "s": float(s_i), "theta": float(th_i)}
        reasons = []
        for attempt in range(tries):
            sp = pool.spines[int(rng.integers(len(pool)))]
            pose = place(sp, trunk, float(s_i), float(th_i), cfg)
            reason = pose["reason"]
            mf = None
            if reason is None:
                mf, st = _manifold(pose["vertices"], sp.faces)
                if st != "NoError":
                    reason = f"spine_mesh:{st}"
            if reason is None:
                box = mf.bounding_box()
                for other, obox in accepted:
                    if _boxes_overlap(box, obox) and (mf ^ other).volume() > cfg.collision_volume_nm3:
                        reason = "collides"
                        break
            if reason is None:
                accepted.append((mf, box))
                row.update(status="placed", spine_key=sp.key, attachment_method=sp.attachment_method, delta_nm=pose["delta_nm"], attempts=attempt + 1,
                           radius_at_s=pose["radius_at_s"], penetration_fraction=pose.get("penetration_fraction", 0.0),
                           anchor_x=pose["anchor"][0] + origin[0], anchor_y=pose["anchor"][1] + origin[1], anchor_z=pose["anchor"][2] + origin[2])
                break
            reasons.append(reason)
        else:
            row.update(status="dropped", attempts=tries)
        row["failure_reasons"] = ";".join(reasons) if reasons else None
        spine_rows.append(row)

    union = m3d.Manifold.batch_boolean([trunk_mf] + [m for m, _ in accepted], m3d.OpType.Add) if accepted else trunk_mf
    out = union.to_mesh64()
    verts = np.asarray(out.vert_properties, dtype=np.float64)[:, :3] + origin
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    areas = mesh.area_faces
    n_placed = sum(r["status"] == "placed" for r in spine_rows)
    branch.update(
        status="ok", reason=None, n_spines_placed=int(n_placed), n_spines_dropped=int(len(spine_rows) - n_placed),
        union_status=str(union.status()).split(".")[-1], n_vertices=int(len(verts)), n_faces=int(len(faces)),
        watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
        n_components=int(len(mesh.split(only_watertight=False))) if len(faces) else 0,
        n_degenerate_faces=int((areas < 1e-6).sum()), volume_nm3=float(mesh.volume), area_nm2=float(areas.sum()), genus=int(union.genus()),
        volume_trunk_nm3=float(trunk_mf.volume()),
    )
    problems = [name for name, bad in (("not_watertight", not branch["watertight"]), ("inconsistent_winding", not branch["winding_consistent"]),
                                       ("several_components", branch["n_components"] != 1), ("degenerate_faces", branch["n_degenerate_faces"] > 0),
                                       ("non_positive_volume", branch["volume_nm3"] <= 0)) if bad]
    if problems:
        branch.update(status="invalid", reason=";".join(problems))
    return AssemblyResult(verts, faces, branch, spine_rows)


def config_dict(cfg: AssemblyConfig) -> Dict[str, Any]:
    return asdict(cfg)
