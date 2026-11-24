"""Subprocess worker: CGAL Poisson surface reconstruction (the mesher of SpineTool 1) of an oriented point set.

    python -m src.neuron_model._cgal_poisson_worker <in.npz with points, normals> <out.off>

Isolated in its own process because the CGAL binding can abort the whole interpreter (seen on the macOS build: "invalid
solver eigen" followed by an abort), like the skeletonisation in ``_cgal_skeletonize_worker.py``."""

import sys

import numpy as np


def main() -> None:
    from CGAL.CGAL_Kernel import Point_3, Vector_3
    from CGAL.CGAL_Point_set_3 import Point_set_3
    from CGAL.CGAL_Poisson_surface_reconstruction import poisson_surface_reconstruction
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

    data = np.load(sys.argv[1])
    ps = Point_set_3()
    ps.add_normal_map()
    for x, n in zip(data["points"], data["normals"]):
        ps.insert(Point_3(*map(float, x)), Vector_3(*map(float, n)))
    poly = Polyhedron_3()
    poisson_surface_reconstruction(ps, poly)
    if poly.size_of_facets() == 0:
        sys.exit(2)
    poly.write_to_file(sys.argv[2])


if __name__ == "__main__":
    main()
