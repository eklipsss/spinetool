from __future__ import annotations

import math
from typing import List, Dict, Any

import numpy as np
import trimesh

from CGAL.CGAL_Kernel import Vector_3, Point_3
from CGAL.CGAL_Polygon_mesh_processing import area, face_area
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3_Facet_handle, Polyhedron_3, Polyhedron_3_Halfedge_handle
from spine_analysis.mesh.utils import v_f_to_mesh, V_F, _mesh_to_v_f
from spine_analysis.shape_metric.metric_core import SpineMetric
from spine_segmentation import point_2_list


_MESH_ATTACHMENT_CENTERS: Dict[int, np.ndarray] = {}
_MESH_DENDRITE_SKELETONS: Dict[int, Any] = {}


def _vec_2_point(vector: Vector_3) -> Point_3:
    return Point_3(vector.x(), vector.y(), vector.z())


def _vec_2_list(vector: Vector_3) -> list:
    return [vector.x(), vector.y(), vector.z()]


def _point_2_vec(point: Point_3) -> Vector_3:
    return Vector_3(point.x(), point.y(), point.z())


def _point_2_list(point: Point_3) -> list:
    return [point.x(), point.y(), point.z()]


def _calculate_facet_center(facet: Polyhedron_3_Facet_handle) -> Vector_3:
    circulator = facet.facet_begin()
    begin = facet.facet_begin()
    center = Vector_3(0, 0, 0)
    while circulator.hasNext():
        halfedge = circulator.next()
        pnt = halfedge.vertex().point()
        center += Vector_3(pnt.x(), pnt.y(), pnt.z())
        # check for end of loop
        if circulator == begin:
            break
    center /= 3
    return center


def get_facet_norm(facet):
    circulator = facet.facet_begin()
    begin = facet.facet_begin()
    points = []
    for i in range(3):
        if not circulator.hasNext():
            break
        halfedge = circulator.next()
        pnt = halfedge.vertex().point()
        points.append([pnt.x(), pnt.y(), pnt.z()])
        if circulator == begin:
            break
    points = np.array(points)
    ba = points[0] - points[1]
    bc = points[2] - points[1]

    norm = np.cross(bc, ba)
    return norm / np.linalg.norm(norm)


def register_attachment_center(spine_mesh: Polyhedron_3, attachment_center: np.ndarray) -> None:
    _MESH_ATTACHMENT_CENTERS[id(spine_mesh)] = np.asarray(attachment_center, dtype=float)


def get_attachment_center(spine_mesh: Polyhedron_3):
    return _MESH_ATTACHMENT_CENTERS.get(id(spine_mesh))


def register_dendrite_skeleton(dendrite_mesh: Polyhedron_3, skeleton: Any) -> None:
    _MESH_DENDRITE_SKELETONS[id(dendrite_mesh)] = skeleton


def get_dendrite_skeleton(dendrite_mesh: Polyhedron_3):
    return _MESH_DENDRITE_SKELETONS.get(id(dendrite_mesh))


def _calculate_junction_center_old(spine_mesh: Polyhedron_3) -> Vector_3:
    junction_triangles = _get_junction_triangles_old(spine_mesh)
    if len(junction_triangles) > 0:
        junction_center = Vector_3(0, 0, 0)
        for facet in junction_triangles:
            junction_center += _calculate_facet_center(facet)
        junction_center /= len(junction_triangles)
    else:
        junction_center = _point_2_vec(spine_mesh.points().next())
    return junction_center


def _get_junction_triangles_old(spine_mesh: Polyhedron_3) -> set:
    junction_triangles = set()
    for v in spine_mesh.vertices():
        if v.vertex_degree() > 10:
            # mark adjacent triangles
            for h in spine_mesh.halfedges():
                if h.vertex() == v:
                    junction_triangles.add(h.facet())
    return junction_triangles


def _get_junction_triangles(spine_mesh: Polyhedron_3, use_old: bool = False) -> set:
    if use_old:
        return _get_junction_triangles_old(spine_mesh)

    attachment_center = get_attachment_center(spine_mesh)
    if attachment_center is None:
        return _get_junction_triangles_old(spine_mesh)

    vertices, faces = _mesh_to_v_f(spine_mesh)
    triangles = vertices[faces]
    finite_mask = np.isfinite(triangles).all(axis=(1, 2))
    triangle_norms = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid_mask = finite_mask & (triangle_norms > 1e-12)
    if not np.any(valid_mask):
        return _get_junction_triangles_old(spine_mesh)

    valid_indices = np.flatnonzero(valid_mask)
    valid_triangles = triangles[valid_mask]
    repeated_points = np.repeat(np.asarray(attachment_center, dtype=float).reshape(1, 3), len(valid_triangles), axis=0)
    projected_points = trimesh.triangles.closest_point(valid_triangles, repeated_points)
    finite_projected = np.isfinite(projected_points).all(axis=1)
    if not np.any(finite_projected):
        return _get_junction_triangles_old(spine_mesh)

    candidate_indices = valid_indices[finite_projected]
    candidate_points = repeated_points[finite_projected]
    candidate_projections = projected_points[finite_projected]
    distances = np.linalg.norm(candidate_projections - candidate_points, axis=1)
    seed_face_index = int(candidate_indices[int(np.argmin(distances))])

    facet_handles = [facet for facet in spine_mesh.facets()]
    for i, facet in enumerate(facet_handles):
        facet.set_id(i)
    if seed_face_index >= len(facet_handles):
        return _get_junction_triangles_old(spine_mesh)

    seed_face = faces[seed_face_index]
    seed_vertex_ids = set(int(v) for v in seed_face.tolist())
    junction_triangles = set()
    for facet in facet_handles:
        circulator = facet.facet_begin()
        begin = facet.facet_begin()
        facet_vertex_ids = set()
        while circulator.hasNext():
            halfedge = circulator.next()
            facet_vertex_ids.add(int(halfedge.vertex().id()))
            if circulator == begin or len(facet_vertex_ids) == 3:
                break
        if seed_vertex_ids.intersection(facet_vertex_ids):
            junction_triangles.add(facet)

    ##### DEBUG VISUALIZATION START #####
#     try:
#         from IPython.display import display
#         import plotly.io as pio
#         import plotly.graph_objects as go

#         try:
#             if pio.renderers.default in ("", None):
#                 pio.renderers.default = "notebook_connected"
#         except Exception:
#             pass

#         fig = go.Figure()
#         fig.add_trace(
#             go.Mesh3d(
#                 x=vertices[:, 0],
#                 y=vertices[:, 1],
#                 z=vertices[:, 2],
#                 i=faces[:, 0],
#                 j=faces[:, 1],
#                 k=faces[:, 2],
#                 color="#8cb6d9",
#                 opacity=0.35,
#                 name="mesh",
#                 showscale=False,
#             )
#         )

#         junction_face_indices = []
#         for facet in junction_triangles:
#             junction_face_indices.append(facet.id())
#         if junction_face_indices:
#             junction_faces = faces[np.asarray(junction_face_indices, dtype=int)]
#             fig.add_trace(
#                 go.Mesh3d(
#                     x=vertices[:, 0],
#                     y=vertices[:, 1],
#                     z=vertices[:, 2],
#                     i=junction_faces[:, 0],
#                     j=junction_faces[:, 1],
#                     k=junction_faces[:, 2],
#                     color="#39ff14",
#                     opacity=0.85,
#                     name="junction_triangles",
#                     showscale=False,
#                 )
#             )

#         fig.add_trace(
#             go.Scatter3d(
#                 x=[attachment_center[0]],
#                 y=[attachment_center[1]],
#                 z=[attachment_center[2]],
#                 mode="markers+text",
#                 marker={"size": 6, "color": "purple"},
#                 text=["attachment"],
#                 textposition="top center",
#                 name="attachment_center",
#             )
#         )

#         all_points = np.vstack([vertices, attachment_center.reshape(1, 3)])
#         mins = all_points.min(axis=0)
#         maxs = all_points.max(axis=0)
#         center = (mins + maxs) / 2.0
#         radius = np.max(maxs - mins) / 2.0
#         if radius == 0:
#             radius = 1.0

#         fig.update_layout(
#             title="Debug junction triangles",
#             scene={
#                 "xaxis": {"range": [center[0] - radius, center[0] + radius], "title": "X"},
#                 "yaxis": {"range": [center[1] - radius, center[1] + radius], "title": "Y"},
#                 "zaxis": {"range": [center[2] - radius, center[2] + radius], "title": "Z"},
#                 "aspectmode": "cube",
#             },
#             margin={"l": 0, "r": 0, "t": 40, "b": 0},
#         )
#         display(fig)
#         fig.show()
#     except Exception:
#         pass
    ##### DEBUG VISUALIZATION END #####

    return junction_triangles if len(junction_triangles) > 0 else {facet_handles[seed_face_index]}


def _calculate_junction_center(spine_mesh: Polyhedron_3, use_old: bool = False) -> Vector_3:
    if use_old:
        return _calculate_junction_center_old(spine_mesh)

    attachment_center = get_attachment_center(spine_mesh)
    if attachment_center is None:
        return _calculate_junction_center_old(spine_mesh)
    return Vector_3(float(attachment_center[0]), float(attachment_center[1]), float(attachment_center[2]))


def calculate_surface_center(mesh: Polyhedron_3) -> np.ndarray:
    vertices = np.ndarray((mesh.size_of_vertices(), 3))
    for i, vertex in enumerate(mesh.vertices()):
        vertex.set_id(i)
        vertices[i, :] = point_2_list(vertex.point())
    return np.mean(vertices, axis=0)


def calculate_metrics(spine_mesh: Polyhedron_3,
                      metric_names: List[str], params: List[Dict[str, Any]] = None) -> List[SpineMetric]:
    if params is None:
        params = [{}] * len(metric_names)
    return [create_metric_by_name(name, spine_mesh, **params[i]) for i, name in enumerate(metric_names)]

import sys
import tblib.pickling_support
tblib.pickling_support.install()
class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise self.ee.with_traceback(self.tb)


def calculate_metrics_parallel(mesh_v_f: V_F, metric_names, params):
    if params is None:
        params = [{}] * len(metric_names)
    spine_mesh = v_f_to_mesh(*mesh_v_f)
    res = []
    for i, name in enumerate(metric_names):
        try:
            metric = create_metric_by_name(name, spine_mesh, **params[i])
            res.append(metric)
        except Exception as e:
            return ExceptionWrapper(e)
    return {metric_name: metric.value if metric is not None else None for metric_name, metric in zip(metric_names, res)}


def create_metric_by_name(metric_name: str, *args, **kwargs):
    path = str(__package__).split('.')
    metric = __import__(__package__)
    for i in range(1, len(path)):
        metric = getattr(metric, path[i])
    klass = getattr(metric, metric_name + "SpineMetric")
    return klass(*args, **kwargs)


def polar2cart(az: np.ndarray, elev: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.stack([radius * np.sin(elev) * np.cos(az),
                     radius * np.sin(elev) * np.sin(az),
                     radius * np.cos(elev)], axis=-1)


def cart2polar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.array(x), np.array(y), np.array(z)
    XsqPlusYsq = np.power(x, 2) + np.power(y, 2)
    r = np.sqrt(XsqPlusYsq + np.power(z, 2))    # r
    elev = np.arctan2(np.sqrt(XsqPlusYsq), z)   # theta
    az = np.arctan2(y, x)                       # phi
    return [r, elev, az]


def get_rotation_matrix(angle, axis):
    sin_t = np.sin(angle)
    cos_t = np.cos(angle)
    x, y, z = axis[0], axis[1], axis[2]
    return np.array([
        [cos_t + x ** 2 * (1 - cos_t), -z * sin_t + y * x * (1 - cos_t), y * sin_t + z * x * (1 - cos_t)],
        [z * sin_t + x * y * (1 - cos_t), cos_t + y ** 2 * (1 - cos_t), -x * sin_t + z * y * (1 - cos_t)],
        [-y * sin_t + x * z * (1 - cos_t), x * sin_t + y * z * (1 - cos_t), cos_t + z ** 2 * (1 - cos_t)]
    ])

def point_in_circle(circle, point):
    return circle is not None and math.hypot(*np.subtract(point, circle[:-1])) - circle[-1] <= 1e-14

def get_enclosing_circle(points):
    if len(points) < 1:
        return None
    c = (*points[0], 0.0)
    for i, point_1 in enumerate(points[:]):
        if not point_in_circle(c, point_1):
            c = (*point_1, 0.0)
            for j, point_2 in enumerate(points[:i+1]):
                if not point_in_circle(c, point_2):
                    c = _make_circle(points[:j+1], point_1, point_2)
    return c


def _make_circle(points, p, q):

    def get_2points_circle(a, b):
        points = np.array([a, b])
        center = np.sum(points, axis=0) / 2
        return center[0], center[1], np.sqrt(max(np.sum((center - points) ** 2, axis=-1)))
    
    def _cross(x0, y0, x1, y1, x2, y2):
        return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    
    def get_circumscribed_circle(points: List):
        points = np.array(points)
        r = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
        sides = points - r
        subtractions = np.array([np.roll(sides[:,1],-1) - np.roll(sides[:,1],-2), np.roll(sides[:,0],-2) - np.roll(sides[:,0],-1)])
        d = np.dot(sides[:, 0], subtractions[0])
        if d == 0:
            return None
        xy = np.add(np.matmul(np.sum(sides ** 2, axis=-1) / (d * 2), subtractions.T), r)
        return xy[0], xy[1], np.sqrt(max(np.sum((xy - points) ** 2, axis=-1)))

    circle = get_2points_circle(p, q)
    left, right = None, None

    for h in points:
        if point_in_circle(circle, h):
            continue

        side = _cross(*p, *q, *h)
        circle = get_circumscribed_circle([p, q, h])
        if circle is None or side == 0:
            continue

        center_orient = _cross(*p, *q, *circle[:-1])
        
        if side > 0 :
            if left is None or center_orient > _cross(*p, *q, *left[:-1]):
                left = circle
        else:
            if right is None or center_orient < _cross(*p, *q, *right[:-1]):
                right = circle

    res = None
    if left is not None and right is not None:
        res = left if left[-1] <= right[-1] else right
    elif left is None:
        res = right
    return circle if res is None else res

def get_incident_halfedges(facet_halfedge: Polyhedron_3_Halfedge_handle) -> List[Polyhedron_3_Halfedge_handle]:
    return [facet_halfedge, facet_halfedge.next(), facet_halfedge.next().next()]

def subdivide_mesh(mesh: Polyhedron_3,
                    relative_max_facet_area: float = 0.001) -> Polyhedron_3:
    out: Polyhedron_3 = mesh.deepcopy()
    for i, facet in enumerate(out.facets()):
        facet.set_id(i)

    facets = [facet for facet in out.facets()]
    total_area = area(out)

    for facet in facets:
        facet_area = face_area(facet, out)
        relative_area = facet_area / total_area

        # facet already small enough
        if relative_area <= relative_max_facet_area:
            continue

        subdivision_number = int(np.ceil(math.log(relative_area / relative_max_facet_area, 3)))
        triangles: List[Polyhedron_3_Halfedge_handle] = [facet.halfedge()]
        for i in range(subdivision_number):
            new_triangles = []
            for halfedge in triangles:
                new_triangles.extend(get_incident_halfedges(halfedge))
                center = _vec_2_point(_calculate_facet_center(halfedge.facet()))
                new_v = out.create_center_vertex(halfedge).vertex()
                new_v.set_point(center)
            triangles = new_triangles

    return out
