"""Standalone CGAL skeletonization worker, run as a disposable subprocess.

``CGAL_Surface_mesh_skeletonization.surface_mesh_skeletonization`` (mean-curvature-flow
skeleton) is a complex native algorithm that on pathological input can abort the whole
process instead of raising a catchable exception — observed on the Windows workstation
during the full Minnie65 run (false-spine detection stage, stage 2): ``python.exe``
crashed with ``STATUS_HEAP_CORRUPTION`` (Windows Event Viewer, faulting module
``ntdll.dll``), which took down the whole ``ProcessPoolExecutor`` worker and raised
``BrokenProcessPool`` in the parent, aborting the entire multi-hour run.

Running the call here, in a throwaway process, turns such a crash into an ordinary
non-zero exit code that the caller
(``spine_geometry.cgal_skeleton_segments_from_trimesh_isolated``) can catch and report
as an ordinary per-spine failure, the same way every other exception in the pipeline is
already handled (``spine_preprocessing._execute_stage_worker``), instead of killing a
whole worker process and the batch it was processing.

    python _cgal_skeletonize_worker.py <mesh.off> <output.npz>

Exit codes: 0 success (``output.npz`` has key ``segments``, shape ``(N, 2, 3)``, ``N``
may be 0 only if the mesh genuinely degenerates to no skeleton, which does not happen
here since that case is reported as an error below); 1 an ordinary CGAL-level error
(mesh not closed, empty skeleton) - message on stderr, matching the in-process
function's error text; any other exit code or no output file at all = native crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _cgal_skeletonize_worker.py mesh.off output.npz", file=sys.stderr)
        return 2

    mesh_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    from CGAL.CGAL_Polygon_mesh_processing import Polylines
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization

    polyhedron = Polyhedron_3(str(mesh_path))
    if not bool(polyhedron.is_closed()):
        print("CGAL skeletonization requires a closed/watertight mesh.", file=sys.stderr)
        return 1

    skeleton_polylines = Polylines()
    correspondence_polylines = Polylines()
    surface_mesh_skeletonization(polyhedron, skeleton_polylines, correspondence_polylines)

    segments = []
    for polyline in skeleton_polylines:
        points = [np.array([p.x(), p.y(), p.z()], dtype=float) for p in polyline]
        for start, end in zip(points[:-1], points[1:]):
            if np.linalg.norm(end - start) > 0:
                segments.append((start, end))

    if not segments:
        print("CGAL returned an empty spine skeleton.", file=sys.stderr)
        return 1

    np.savez(output_path, segments=np.asarray(segments, dtype=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
