from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3 as _Polyhedron_3
    from CGAL.CGAL_Polygon_mesh_processing import volume as _cgal_volume
    _CGAL = True
except ImportError:
    _CGAL = False
    _Polyhedron_3 = None  # type: ignore

try:
    import trimesh
    import trimesh.repair
    _TRIMESH = True
except ImportError:
    _TRIMESH = False

try:
    from CGAL.CGAL_Polygon_mesh_processing import does_self_intersect as _does_self_intersect
    _CGAL_SELF_INTERSECT = True
except ImportError:
    _CGAL_SELF_INTERSECT = False

try:
    from CGAL.CGAL_Polygon_mesh_processing import stitch_borders as _stitch_borders
    _CGAL_STITCH = True
except ImportError:
    _CGAL_STITCH = False

__all__ = [
    "MeshReport",
    "check_mesh",
    "repair_mesh",
]


# ===========================================================================
# Report dataclass
# ===========================================================================

@dataclass
class MeshReport:

    n_vertices: int = 0
    n_faces: int = 0
    is_valid: bool = False
    is_closed: bool = False
    n_boundary_edges: int = 0
    volume: Optional[float] = None
    volume_negative: bool = False
    has_self_intersections: Optional[bool] = None
    n_degenerate_faces: int = 0
    issues: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def __str__(self) -> str:
        lines = [
            "Mesh report:",
            f"  Vertices              : {self.n_vertices}",
            f"  Faces                 : {self.n_faces}",
            f"  Valid (lib check)     : {self.is_valid}",
            f"  Closed / watertight   : {self.is_closed}",
            f"  Boundary halfedges    : {self.n_boundary_edges}",
        ]
        if self.volume is not None:
            lines.append(f"  Volume                : {self.volume:.6f}")
        lines.append(f"  Volume negative       : {self.volume_negative}")
        if self.has_self_intersections is not None:
            lines.append(f"  Self-intersections    : {self.has_self_intersections}")
        lines.append(f"  Degenerate faces      : {self.n_degenerate_faces}")
        if self.issues:
            lines.append("  Issues detected       :")
            for issue in self.issues:
                lines.append(f"    ✗ {issue}")
        else:
            lines.append("  Issues detected       : none ✓")
        return "\n".join(lines)


# ===========================================================================
# Detection
# ===========================================================================

def check_mesh(mesh: Any) -> MeshReport:
    if _CGAL and _Polyhedron_3 is not None and isinstance(mesh, _Polyhedron_3):
        return _check_cgal(mesh)
    if _TRIMESH and isinstance(mesh, trimesh.Trimesh):
        return _check_trimesh(mesh)
    warnings.warn(
        "check_mesh: unsupported mesh type — pass a Polyhedron_3 or trimesh.Trimesh.",
        stacklevel=2,
    )
    return MeshReport()


def _check_cgal(mesh: Any) -> MeshReport:
    r = MeshReport()
    r.n_vertices = mesh.size_of_vertices()
    r.n_faces = mesh.size_of_facets()

    try:
        r.is_valid = bool(mesh.is_valid())
    except Exception:
        r.is_valid = False
    if not r.is_valid:
        r.issues.append("CGAL Polyhedron_3.is_valid() returned False")

    try:
        r.is_closed = bool(mesh.is_closed())
    except Exception:
        r.is_closed = False

    # Count boundary halfedges
    try:
        boundary = sum(1 for h in mesh.halfedges() if h.is_border())
        r.n_boundary_edges = boundary
        if boundary > 0:
            r.issues.append(
                f"Mesh has {boundary} boundary halfedges "
                f"({boundary // 2} boundary edges) — open holes exist"
            )
    except Exception:
        pass

    # Volume (only meaningful for closed meshes)
    if r.is_closed:
        try:
            vol = float(_cgal_volume(mesh))
            r.volume = vol
            r.volume_negative = vol < 0
            if r.volume_negative:
                r.issues.append(
                    f"Volume is negative ({vol:.6f}): face normals are likely inverted "
                    "(faces oriented clockwise when viewed from outside)"
                )
        except Exception as exc:
            r.issues.append(f"Volume computation failed: {exc}")
    else:
        r.issues.append("Mesh is not closed — volume is undefined")

    # Self-intersections (CGAL PMP)
    if _CGAL_SELF_INTERSECT:
        try:
            r.has_self_intersections = bool(_does_self_intersect(mesh))
            if r.has_self_intersections:
                r.issues.append("Mesh has self-intersecting face pairs")
        except Exception:
            pass

    # Degenerate faces via vertex coordinates
    try:
        from spine_analysis.mesh.utils import _mesh_to_v_f
        v, f = _mesh_to_v_f(mesh)
        if len(f) > 0:
            e1 = v[f[:, 1]] - v[f[:, 0]]
            e2 = v[f[:, 2]] - v[f[:, 0]]
            areas = np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5
            r.n_degenerate_faces = int(np.sum(areas < 1e-12))
            if r.n_degenerate_faces > 0:
                r.issues.append(f"{r.n_degenerate_faces} degenerate (zero-area) faces")
    except Exception:
        pass

    return r


def _check_trimesh(mesh: trimesh.Trimesh) -> MeshReport:
    r = MeshReport()
    r.n_vertices = len(mesh.vertices)
    r.n_faces = len(mesh.faces)

    try:
        r.is_valid = bool(mesh.is_valid)
    except Exception:
        r.is_valid = False
    if not r.is_valid:
        r.issues.append("trimesh.is_valid returned False")

    try:
        r.is_closed = bool(mesh.is_watertight)
    except Exception:
        r.is_closed = False
    if not r.is_closed:
        r.issues.append("Mesh is not watertight (has open boundary edges / holes)")

    # Boundary edges
    try:
        _, counts = np.unique(np.sort(mesh.edges_sorted, axis=1), axis=0, return_counts=True)
        r.n_boundary_edges = int(np.sum(counts == 1))
    except Exception:
        pass

    # Volume
    if r.is_closed:
        try:
            r.volume = float(mesh.volume)
            r.volume_negative = r.volume < 0
            if r.volume_negative:
                r.issues.append(
                    f"Volume is negative ({r.volume:.6f}): face normals are likely inverted"
                )
        except Exception as exc:
            r.issues.append(f"Volume computation failed: {exc}")

    # Winding / self-intersections
    try:
        if not mesh.is_winding_consistent:
            r.has_self_intersections = True
            r.issues.append(
                "Winding is inconsistent — possible self-intersections or "
                "incorrectly oriented face groups"
            )
        else:
            r.has_self_intersections = False
    except Exception:
        pass

    # Degenerate faces
    try:
        if len(mesh.faces) > 0:
            r.n_degenerate_faces = int(np.sum(mesh.area_faces < 1e-12))
            if r.n_degenerate_faces > 0:
                r.issues.append(f"{r.n_degenerate_faces} degenerate (zero-area) faces")
    except Exception:
        pass

    return r


# ===========================================================================
# Repair
# ===========================================================================


def _fill_holes_fan(tm: "trimesh.Trimesh") -> None:
    edges = tm.edges_sorted 
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]

    if len(boundary_edges) == 0:
        return

    adj: Dict[int, set] = defaultdict(set)
    for e in boundary_edges:
        u, v = int(e[0]), int(e[1])
        adj[u].add(v)
        adj[v].add(u)

    changed = True
    while changed:
        changed = False
        for v in list(adj.keys()):
            if len(adj[v]) != 2:
                for neighbor in list(adj[v]):
                    adj[neighbor].discard(v)
                del adj[v]
                changed = True

    loops: List[List[int]] = []
    remaining = set(adj.keys())
    while remaining:
        start = next(iter(remaining))
        loop: List[int] = [start]
        visited = {start}
        remaining.discard(start)
        current = start
        prev = -1
        closed = False
        for _ in range(len(adj)):  
            candidates = adj[current] - {prev}
            if not candidates:
                break
            nxt = next(iter(candidates))
            if nxt == start:
                closed = True
                break
            if nxt in visited:
                break  
            loop.append(nxt)
            visited.add(nxt)
            remaining.discard(nxt)
            prev, current = current, nxt
        if closed and len(loop) >= 3:
            loops.append(loop)

    if not loops:
        return

    new_verts: List = tm.vertices.tolist()
    new_faces: List = tm.faces.tolist()

    for loop in loops:
        pts = tm.vertices[loop]
        centroid = pts.mean(axis=0).tolist()
        cid = len(new_verts)
        new_verts.append(centroid)
        n = len(loop)
        for k in range(n):
            new_faces.append([loop[k], loop[(k + 1) % n], cid])

    tm.vertices = np.array(new_verts, dtype=float)
    tm.faces = np.array(new_faces, dtype=int)
    tm._cache.clear()
    trimesh.repair.fix_normals(tm, multibody=True)


def _split_nonmanifold_vertices(tm: "trimesh.Trimesh") -> int:
    faces = np.asarray(tm.faces, dtype=int)

    vertex_faces: Dict[int, List[int]] = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for vi in face:
            vertex_faces[int(vi)].append(face_idx)

    new_vertices: List = tm.vertices.tolist()
    new_faces = faces.tolist()
    n_split = 0

    for vertex_id, face_list in vertex_faces.items():
        if len(face_list) <= 1:
            continue

        edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for face_idx in face_list:
            face = new_faces[face_idx]
            k = face.index(vertex_id)
            for other in (face[(k + 1) % 3], face[(k - 1) % 3]):
                edge_to_faces[tuple(sorted((vertex_id, other)))].append(face_idx)

        face_adj: Dict[int, set] = defaultdict(set)
        for adjacent_faces in edge_to_faces.values():
            for i in range(len(adjacent_faces)):
                for j in range(i + 1, len(adjacent_faces)):
                    face_adj[adjacent_faces[i]].add(adjacent_faces[j])
                    face_adj[adjacent_faces[j]].add(adjacent_faces[i])

        remaining = set(face_list)
        components: List[List[int]] = []
        while remaining:
            start = next(iter(remaining))
            stack = [start]
            remaining.discard(start)
            comp = []
            while stack:
                current = stack.pop()
                comp.append(current)
                for neighbor in face_adj.get(current, ()):
                    if neighbor in remaining:
                        remaining.discard(neighbor)
                        stack.append(neighbor)
            components.append(comp)

        if len(components) <= 1:
            continue  

        n_split += 1
        for comp in components[1:]:
            new_id = len(new_vertices)
            new_vertices.append(list(tm.vertices[vertex_id]))
            for face_idx in comp:
                new_faces[face_idx] = [new_id if vi == vertex_id else vi for vi in new_faces[face_idx]]

    if n_split:
        tm.vertices = np.array(new_vertices, dtype=float)
        tm.faces = np.array(new_faces, dtype=int)
        tm._cache.clear()

    return n_split


# ===========================================================================

def repair_mesh(
    mesh: Any,
    fix_normals: bool = True,
    fill_holes: bool = True,
    remove_degenerate: bool = True,
    stitch_borders: bool = True,
    verbose: bool = True,
) -> Tuple[Any, MeshReport, MeshReport]:
    if not _TRIMESH:
        raise ImportError("trimesh is required for mesh repair — install it with: pip install trimesh")

    is_cgal = _CGAL and _Polyhedron_3 is not None and isinstance(mesh, _Polyhedron_3)

    report_before = check_mesh(mesh)
    if verbose:
        print("=== Before repair ===")
        print(report_before)

    # ------------------------------------------------------------------
    # Step 1: CGAL stitch_borders (merges duplicated boundary vertices)
    # ------------------------------------------------------------------
    working_cgal = mesh  # we may modify this copy
    if is_cgal and stitch_borders and _CGAL_STITCH:
        try:
            working_cgal = mesh.deepcopy()
            _stitch_borders(working_cgal)
            if verbose:
                print("  [1/5] CGAL stitch_borders applied.")
        except Exception as exc:
            if verbose:
                print(f"  [1/5] CGAL stitch_borders skipped ({exc}).")
            working_cgal = mesh
    elif verbose:
        print("  [1/5] CGAL stitch_borders skipped (not available or not requested).")

    # ------------------------------------------------------------------
    # Convert to trimesh for the remaining steps
    # ------------------------------------------------------------------
    if is_cgal:
        from dendrite_analysis.surface_distances import polyhedron_to_trimesh
        tm = polyhedron_to_trimesh(working_cgal)
    else:
        tm = mesh.copy()

    # ------------------------------------------------------------------
    # Step 2: Remove degenerate faces
    # ------------------------------------------------------------------
    if remove_degenerate and len(tm.faces) > 0:
        mask = tm.area_faces > 1e-12
        n_removed = int(np.sum(~mask))
        if n_removed > 0:
            tm.update_faces(mask)
            tm.remove_unreferenced_vertices()
            if verbose:
                print(f"  [2/5] Removed {n_removed} degenerate faces.")
        elif verbose:
            print("  [2/5] No degenerate faces found.")
    elif verbose:
        print("  [2/5] Degenerate-face removal skipped.")

    # ------------------------------------------------------------------
    # Step 3: Split non-manifold ("pinch point") vertices
    # ------------------------------------------------------------------
    n_split = _split_nonmanifold_vertices(tm)
    if verbose:
        if n_split:
            print(f"  [3/5] Split {n_split} non-manifold vertex(es) into separate fans.")
        else:
            print("  [3/5] No non-manifold vertices found.")

    # ------------------------------------------------------------------
    # Step 4: Fill holes
    # ------------------------------------------------------------------
    if fill_holes:
        was_watertight = tm.is_watertight
        trimesh.repair.fill_holes(tm)
        if not was_watertight and not tm.is_watertight:
            _fill_holes_fan(tm)
        if verbose:
            if not was_watertight and tm.is_watertight:
                print("  [4/5] Holes filled — mesh is now watertight.")
            elif not was_watertight:
                print("  [4/5] Hole filling attempted; mesh is still not watertight.")
            else:
                print("  [4/5] Mesh was already watertight — no holes to fill.")
    elif verbose:
        print("  [4/5] Hole filling skipped.")

    # ------------------------------------------------------------------
    # Step 5: Fix normals / winding
    # ------------------------------------------------------------------
    if fix_normals:
        vol_before = tm.volume if tm.is_watertight else None
        # Pass 4a: fix locally inconsistent winding (mixed CW/CCW faces)
        trimesh.repair.fix_normals(tm, multibody=True)
        if tm.is_watertight:
            vol_after = tm.volume
            if vol_after < 0:
                tm.faces = tm.faces[:, ::-1].copy()
                tm._cache.clear()
                vol_after = tm.volume
                if verbose:
                    print(
                        f"  [5/5] Mesh was consistently inverted (all normals inward, "
                        f"volume {vol_before:.4f}) → flipped all face orientations. "
                        f"New volume: {vol_after:.4f}."
                    )
            elif verbose:
                if vol_before is not None and vol_before < 0:
                    print(
                        f"  [5/5] Inverted normals fixed: "
                        f"volume {vol_before:.4f} → {vol_after:.4f}."
                    )
                else:
                    print(f"  [5/5] Normals OK (volume: {vol_after:.4f}).")
        else:
            normals = tm.face_normals
            centers = tm.triangles_center
            centroid = tm.vertices.mean(axis=0)
            outward_dot = np.sum(normals * (centers - centroid), axis=1)
            frac_inward = float(np.mean(outward_dot < 0))
            if frac_inward > 0.75:
                tm.faces = tm.faces[:, ::-1].copy()
                tm._cache.clear()
                if verbose:
                    print(
                        f"  [5/5] Non-watertight mesh: {frac_inward:.0%} of face normals "
                        f"point inward — flipped all face orientations."
                    )
            elif verbose:
                print(
                    f"  [5/5] Non-watertight mesh: {frac_inward:.0%} inward normals "
                    f"(threshold 75%) — no global flip applied."
                )
    elif verbose:
        print("  [5/5] Normal fix skipped.")

    # ------------------------------------------------------------------
    # Convert back to Polyhedron_3 if input was CGAL
    # ------------------------------------------------------------------
    if verbose:
        watertight_str = "watertight" if tm.is_watertight else "not watertight"
        print(f"  [pre-convert] trimesh: {watertight_str}, "
              f"volume = {tm.volume:.6f}" if tm.is_watertight else
              f"  [pre-convert] trimesh: {watertight_str} — volume undefined.")

    repaired: Any = tm
    if is_cgal:
        try:
            from spine_analysis.mesh.utils import v_f_to_mesh_isolated
            v = np.asarray(tm.vertices, dtype=float)
            f = np.asarray(tm.faces, dtype=int)
            if verbose:
                print(f"  Converting {len(v)} vertices / {len(f)} faces to Polyhedron_3 "
                      f"(may take a moment for large meshes)...")
            repaired = v_f_to_mesh_isolated(v, f)
            if verbose:
                print("  Converted repaired mesh back to Polyhedron_3.")

            if _CGAL:
                try:
                    cgal_closed = bool(repaired.is_closed())
                    if cgal_closed:
                        cgal_vol = float(_cgal_volume(repaired))
                        if verbose:
                            print(f"  [post-convert] CGAL mesh closed=True, "
                                  f"volume = {cgal_vol:.6f}")
                        if cgal_vol < 0:
                            if verbose:
                                print(
                                    "  [post-convert] Negative CGAL volume detected — "
                                    "rebuilding with reversed face order."
                                )
                            repaired = v_f_to_mesh_isolated(v, f[:, ::-1].copy())
                    elif verbose:
                        print("  [post-convert] CGAL mesh is not closed — "
                              "volume check skipped.")
                except Exception as exc:
                    if verbose:
                        print(f"  [post-convert] CGAL volume check failed: {exc}")
        except Exception as exc:
            warnings.warn(
                f"Could not convert repaired mesh back to Polyhedron_3: {exc}. "
                "Returning trimesh.Trimesh instead.",
                stacklevel=2,
            )
            repaired = tm

    report_after = check_mesh(repaired)
    if verbose:
        print("\n=== After repair ===")
        print(report_after)

    return repaired, report_before, report_after
