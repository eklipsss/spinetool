from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_modifier
from spine_segmentation import point_2_list

MeshDataset = Dict[str, Polyhedron_3]
V_F = Tuple[np.ndarray, np.ndarray]
LineSet = List[Tuple[Point_3, Point_3]]


def _mesh_to_v_f(mesh: Polyhedron_3) -> V_F:
    vertices = np.ndarray((mesh.size_of_vertices(), 3))
    for i, vertex in enumerate(mesh.vertices()):
        vertex.set_id(i)
        vertices[i, :] = point_2_list(vertex.point())

    facets = np.ndarray((mesh.size_of_facets(), 3)).astype("uint")
    for i, facet in enumerate(mesh.facets()):
        circulator = facet.facet_begin()
        j = 0
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge = circulator.next()
            v = halfedge.vertex()
            facets[i, j] = (v.id())
            j += 1
            # check for end of loop
            if circulator == begin or j == 3:
                break
    return vertices, facets


def load_spine_meshes(folder_path: str = "output",
                      spine_file_pattern: str = "**/spine_*.off") -> Dict[str, Polyhedron_3]:
    output = {}
    path = Path(folder_path)
    spine_names = list(path.glob(spine_file_pattern))
    for spine_name in spine_names:
        output[str(spine_name)] = Polyhedron_3(str(spine_name))
    return output


def rotate(mesh: Polyhedron_3, matrix) -> Polyhedron_3:
    from spine_analysis.shape_metric.utils import calculate_surface_center
    output = mesh.deepcopy()
    center = calculate_surface_center(mesh)
    for v in output.vertices():
        p = v.point()
        rotated_p = np.matmul(matrix, [p.x() - center[0], p.y() - center[1], p.z() - center[2]])
        v.set_point(Point_3(rotated_p[0] + center[0], rotated_p[1] + center[1], rotated_p[2] + center[2]))

    return output


def preprocess_meshes(spine_meshes: MeshDataset) -> Dict[str, V_F]:
    output = {}
    for (spine_name, spine_mesh) in spine_meshes.items():
        output[spine_name] = _mesh_to_v_f(spine_mesh)
    return output


def polylines_to_line_set(polylines: Polylines) -> LineSet:
    output = []
    for line in polylines:
        for i in range(len(line) - 1):
            output.append((line[i], line[i + 1]))
    return output


def v_f_to_mesh(v: np.ndarray, f: np.ndarray) -> Polyhedron_3:
    p = Polyhedron_3()
    modifier = Polyhedron_modifier()
    modifier.begin_surface(len(v), len(f))
    for vertex in v:
        modifier.add_vertex(Point_3(*vertex.tolist()))
    for facet in f:
        modifier.begin_facet()
        for vi in facet.tolist():
            modifier.add_vertex_to_facet(int(vi))
        modifier.end_facet()
    modifier.end_surface()
    p.delegate(modifier)
    return p

def write_off(fd, v, f):
    fd.write('OFF\n')
    fd.write(f'{len(v)} {len(f)} 0\n')

    for v_row in v:
        fd.write(f'{v_row[0]} {v_row[1]} {v_row[2]}\n')

    for facet in f:
        fd.write(f'3 {facet[0]} {facet[1]} {facet[2]}\n')


def v_f_to_mesh_isolated(v: np.ndarray, f: np.ndarray, timeout: float = 600.0) -> Polyhedron_3:
    import subprocess
    import sys
    import os
    import tempfile

    worker_path = os.path.join(os.path.dirname(__file__), "_v_f_to_off_worker.py")

    with tempfile.TemporaryDirectory() as tmp_dir:
        npz_path = os.path.join(tmp_dir, "v_f.npz")
        off_path = os.path.join(tmp_dir, "mesh.off")
        np.savez(npz_path, v=np.asarray(v, dtype=float), f=np.asarray(f, dtype=int))

        try:
            result = subprocess.run(
                [sys.executable, worker_path, npz_path, off_path],
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"v_f_to_mesh_isolated: building Polyhedron_3 did not finish within "
                f"{timeout}s and was terminated."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"v_f_to_mesh_isolated: building Polyhedron_3 crashed the worker "
                f"process (exit code {result.returncode}) — the input is likely "
                "non-manifold or otherwise invalid for CGAL's incremental builder. "
                f"stderr:\n{result.stderr}"
            )
        if not os.path.exists(off_path):
            raise RuntimeError(
                "v_f_to_mesh_isolated: worker exited cleanly but produced no output. "
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return Polyhedron_3(off_path)
