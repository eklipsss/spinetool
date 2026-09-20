from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import numpy as np

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_modifier


def _build_polyhedron(vertices: np.ndarray, faces: np.ndarray) -> Polyhedron_3:
    polyhedron = Polyhedron_3()
    modifier = Polyhedron_modifier()
    modifier.begin_surface(int(len(vertices)), int(len(faces)))
    for vertex in vertices:
        modifier.add_vertex(Point_3(float(vertex[0]), float(vertex[1]), float(vertex[2])))
    for face in faces:
        modifier.begin_facet()
        for vertex_index in face[:3]:
            modifier.add_vertex_to_facet(int(vertex_index))
        modifier.end_facet()
    modifier.end_surface()
    polyhedron.delegate(modifier)
    return polyhedron


def _write_off(polyhedron: Polyhedron_3, output_path: Path) -> None:
    vertices = []
    for index, vertex in enumerate(polyhedron.vertices()):
        vertex.set_id(index)
        point = vertex.point()
        vertices.append((float(point.x()), float(point.y()), float(point.z())))

    faces = []
    for facet in polyhedron.facets():
        face = []
        circulator = facet.facet_begin()
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge = circulator.next()
            face.append(int(halfedge.vertex().id()))
            if circulator == begin:
                break
        if len(face) >= 3:
            faces.append(face[:3])

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("OFF\n")
        handle.write(f"{len(vertices)} {len(faces)} 0\n")
        for x, y, z in vertices:
            handle.write(f"{x} {y} {z}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _v_f_to_off_worker.py input.npz output.off", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    data = np.load(input_path)
    vertices = np.asarray(data["v"], dtype=float)
    faces = np.asarray(data["f"], dtype=int)

    polyhedron = _build_polyhedron(vertices, faces)
    _write_off(polyhedron, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
