"""Oriented normals for an unoriented point cloud (input to Screened Poisson).

MoGen generates coordinates only; Poisson needs oriented normals. Per the spec:

1. PCA normal: eigenvector of the smallest eigenvalue of the covariance of the
   ``k`` nearest neighbours (``k_normal`` 20-50);
2. consistent orientation by propagation over the neighbourhood graph
   (Hoppe et al. 1992: minimum spanning tree with edge cost ``1 - |n_i·n_j|``,
   flip a child whose normal disagrees with its parent's);
3. global orientation: each connected component is seeded at its point
   farthest from the cloud centroid, where the outward normal must point away
   from the centroid (the surface is tangent to the enclosing sphere there).

``outward_fraction`` (share of normals with ``n·(p - c) > 0``) is returned as a
diagnostic only - for strongly non-convex shapes it is legitimately < 1.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree


def pca_normals(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unoriented normals + neighbour indices + distances (``k`` neighbours incl. the point itself)."""
    points = np.asarray(points, dtype=np.float64)
    k = min(int(k), len(points))
    tree = cKDTree(points)
    distances, neighbours = tree.query(points, k=k)
    local = points[neighbours] - points[neighbours].mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local) / k
    _, eigenvectors = np.linalg.eigh(covariance)  # ascending eigenvalues
    normals = eigenvectors[:, :, 0]
    return normals / np.linalg.norm(normals, axis=1, keepdims=True), neighbours, distances


def orient_normals(points: np.ndarray, normals: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Consistent + outward orientation via MST propagation (see module docstring)."""
    points = np.asarray(points, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64, copy=True)
    n, k = neighbours.shape
    rows = np.repeat(np.arange(n), k)
    cols = neighbours.reshape(-1)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    # epsilon keeps parallel-normal edges (cost 0) present in the sparse graph
    cost = 1.0 - np.abs(np.einsum("ij,ij->i", normals[rows], normals[cols])) + 1e-9
    graph = sparse.coo_matrix((cost, (rows, cols)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    tree = csgraph.minimum_spanning_tree(graph)
    tree = tree.maximum(tree.T)

    n_components, labels = csgraph.connected_components(tree, directed=False)
    centroid = points.mean(axis=0)
    radial = points - centroid
    for component in range(n_components):
        members = np.flatnonzero(labels == component)
        seed = members[np.argmax(np.einsum("ij,ij->i", radial[members], radial[members]))]
        if np.dot(normals[seed], radial[seed]) < 0:
            normals[seed] = -normals[seed]
        order, predecessors = csgraph.breadth_first_order(tree, seed, directed=False, return_predecessors=True)
        for node in order[1:]:
            parent = predecessors[node]
            if np.dot(normals[node], normals[parent]) < 0:
                normals[node] = -normals[node]
    return normals


def estimate_normals(points: np.ndarray, k: int = 30) -> Tuple[np.ndarray, Dict[str, float]]:
    """PCA normals oriented consistently and outward; returns ``(normals [N,3] float32, diagnostics)``."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError(f"points must be [N>=4, 3], got {points.shape}")
    normals, neighbours, distances = pca_normals(points, k)
    normals = orient_normals(points, normals, neighbours)
    radial = points - points.mean(axis=0)
    outward = np.einsum("ij,ij->i", normals, radial) > 0
    diagnostics = {
        "k_normal": int(neighbours.shape[1]),
        "outward_fraction": float(outward.mean()),
        "mean_neighbour_distance": float(distances[:, 1:].mean()) if distances.shape[1] > 1 else 0.0,
    }
    return normals.astype(np.float32), diagnostics


def normal_angle_error(estimated: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Angle between estimated and reference normals, oriented (sign matters) and unoriented."""
    e = estimated / np.linalg.norm(estimated, axis=1, keepdims=True)
    r = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    cos = np.clip(np.einsum("ij,ij->i", e, r), -1.0, 1.0)
    oriented = np.degrees(np.arccos(cos))
    unoriented = np.degrees(np.arccos(np.abs(cos)))
    return {
        "oriented_mean_deg": float(oriented.mean()),
        "unoriented_mean_deg": float(unoriented.mean()),
        "flipped_fraction": float((cos < 0).mean()),
    }
