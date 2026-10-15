"""Distance to the soma along the dendritic path (limb skeleton graph).

Root = limb-skeleton node nearest to the soma-mesh centroid; distance of a point = shortest-path
length to the node nearest to that point. The limb skeleton (like the branch skeleton) may run on
the membrane; over a path of tens of micrometres that biases the length by about a percent.
Unreachable nodes (a limb skeleton split into pieces) give NaN, never a Euclidean substitute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class SomaDistance:
    def __init__(self, neuron_dir: Path) -> None:
        self.neuron_dir = Path(neuron_dir)
        self._soma_center: Optional[np.ndarray] = None
        self._limbs: Dict[int, Optional[Tuple[np.ndarray, np.ndarray, object]]] = {}

    @property
    def soma_center(self) -> Optional[np.ndarray]:
        if self._soma_center is None:
            path = self.neuron_dir / "soma" / "soma_mesh.off"
            if not path.exists():
                return None
            from ..spine_geometry import load_trimesh

            self._soma_center = np.asarray(load_trimesh(path, process=False).vertices, dtype=float).mean(axis=0)
        return self._soma_center

    def _limb(self, limb: int):
        if limb in self._limbs:
            return self._limbs[limb]
        result = None
        path = self.neuron_dir / f"limb_{limb:03d}" / "limb_skeleton.npy"
        center = self.soma_center
        if path.exists() and center is not None:
            import networkx as nx
            from scipy.spatial import cKDTree

            from ..spine_geometry import graph_from_segments, load_skeleton_segments

            graph = graph_from_segments(load_skeleton_segments(path))
            nodes = np.array(list(graph.nodes))
            positions = np.vstack([graph.nodes[n]["pos"] for n in nodes])
            root = int(nodes[np.argmin(np.linalg.norm(positions - center, axis=1))])
            lengths = nx.single_source_dijkstra_path_length(graph, root, weight="length")
            dist = np.array([lengths.get(int(n), np.nan) for n in nodes], dtype=float)
            result = (positions, dist, cKDTree(positions))
        self._limbs[limb] = result
        return result

    def distance(self, limb: int, point: np.ndarray) -> float:
        data = self._limb(int(limb))
        if data is None:
            return float("nan")
        _, dist, tree = data
        _, idx = tree.query(np.asarray(point, dtype=float))
        return float(dist[int(idx)])
