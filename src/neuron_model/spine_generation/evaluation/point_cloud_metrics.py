"""Surface-level metrics on sampled point clouds (spec: MMD-CD, Coverage-CD, 1-NNA-CD).

Every real / generated mesh is sampled with the same number of points (e.g. 2048) and
the clouds are compared by Chamfer distance (CD), following the 3D generative-model
literature (Achlioptas et al. 2018; Yang et al. 2019 "PointFlow")::

    CD(X, Y) = mean_x min_y ||x - y||^2 + mean_y min_x ||x - y||^2

NO per-shape normalisation (unlike many papers): the local frame and the physical scale
are part of what a spine generator must reproduce, so clouds are compared as they are
(optionally multiplied by one global ``scale``, e.g. nm -> um, purely for readable numbers).

Given ``D_rg[i, j] = CD(real_i, gen_j)``:

- MMD-CD (minimum matching distance) = mean_i min_j D_rg  -> fidelity (lower is better);
- COV-CD (coverage) = share of real clouds that are the nearest real neighbour of at
  least one generated cloud -> diversity / mode coverage (higher is better);
- 1-NNA-CD = leave-one-out 1-NN accuracy of telling real from generated in the pooled
  set -> 0.5 means indistinguishable, 1.0 means trivially separable (target 0.5).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def chamfer_matrix(
    a: np.ndarray,
    b: Optional[np.ndarray] = None,
    *,
    device: str = "cpu",
    pair_chunk: int = 16,
    scale: float = 1.0,
) -> np.ndarray:
    """``[len(a), len(b)]`` Chamfer distances between two sets of clouds ``[n, P, 3]``.

    ``b=None`` -> symmetric ``a`` vs ``a`` (only the upper triangle is computed; diagonal 0).
    ``pair_chunk`` clouds of ``b`` are compared to one cloud of ``a`` at a time: memory
    ~ ``pair_chunk * P_a * P_b * 4`` bytes (16 x 2048 x 2048 -> 256 MB).
    """
    import torch

    symmetric = b is None
    a_t = torch.as_tensor(np.asarray(a, dtype=np.float32) * scale, device=device)
    b_t = a_t if symmetric else torch.as_tensor(np.asarray(b, dtype=np.float32) * scale, device=device)
    n, m = a_t.shape[0], b_t.shape[0]
    out = torch.zeros(n, m, dtype=torch.float64)
    with torch.no_grad():
        for i in range(n):
            start = i + 1 if symmetric else 0
            for j0 in range(start, m, pair_chunk):
                block = b_t[j0:j0 + pair_chunk]  # [c, P_b, 3]
                d2 = torch.cdist(a_t[i].unsqueeze(0).expand(block.shape[0], -1, -1), block).pow(2)  # [c, P_a, P_b]
                values = d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)
                out[i, j0:j0 + block.shape[0]] = values.double().cpu()
    result = out.numpy()
    if symmetric:
        result = result + result.T
    return result


def mmd_cd(d_real_gen: np.ndarray) -> float:
    """Minimum matching distance: for every real cloud, the CD to its closest generated cloud, averaged."""
    return float(np.asarray(d_real_gen).min(axis=1).mean())


def coverage_cd(d_real_gen: np.ndarray) -> float:
    """Share of real clouds matched (as nearest real neighbour) by at least one generated cloud."""
    d = np.asarray(d_real_gen)
    return float(np.unique(d.argmin(axis=0)).size / d.shape[0])


def one_nna(d_real_real: np.ndarray, d_gen_gen: np.ndarray, d_real_gen: np.ndarray) -> Dict[str, float]:
    """Leave-one-out 1-NN accuracy on the pooled set; needs equally many real and generated clouds.

    Returns overall accuracy plus the per-class accuracies (a generator that collapses onto
    a few real shapes shows up as accuracy_generated >> accuracy_real).
    """
    n, m = d_real_real.shape[0], d_gen_gen.shape[0]
    if n != m:
        raise ValueError(f"1-NNA needs equal set sizes (got {n} real vs {m} generated) - subsample first")
    pooled = np.block([[d_real_real, d_real_gen], [d_real_gen.T, d_gen_gen]]).astype(float)
    np.fill_diagonal(pooled, np.inf)
    labels = np.r_[np.zeros(n, dtype=int), np.ones(m, dtype=int)]
    correct = labels[pooled.argmin(axis=1)] == labels
    return {
        "accuracy": float(correct.mean()),
        "accuracy_real": float(correct[:n].mean()),
        "accuracy_generated": float(correct[n:].mean()),
    }


def surface_metrics(d_real_real: np.ndarray, d_gen_gen: np.ndarray, d_real_gen: np.ndarray) -> Dict[str, float]:
    nna = one_nna(d_real_real, d_gen_gen, d_real_gen)
    return {
        "mmd_cd": mmd_cd(d_real_gen),
        "coverage_cd": coverage_cd(d_real_gen),
        "one_nna_cd": nna["accuracy"],
        "one_nna_accuracy_real": nna["accuracy_real"],
        "one_nna_accuracy_generated": nna["accuracy_generated"],
    }


def cloud_descriptors(clouds: np.ndarray, *, chunk: int = 4096) -> np.ndarray:
    """Cheap 7-D shape descriptor per cloud: centroid (3), sorted PCA std (3), mean radius (1).

    Used to pre-select Chamfer candidates among a huge reference set; computed chunk by chunk
    so ``clouds`` can be a memmap over the whole train split.
    """
    out = np.empty((len(clouds), 7))
    for start in range(0, len(clouds), chunk):
        block = np.asarray(clouds[start:start + chunk], dtype=np.float64)
        centroid = block.mean(axis=1)
        centered = block - centroid[:, None, :]
        cov = np.einsum("npi,npj->nij", centered, centered) / max(block.shape[1] - 1, 1)
        eig = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0, None))  # ascending
        radius = np.linalg.norm(centered, axis=2).mean(axis=1, keepdims=True)
        out[start:start + len(block)] = np.hstack([centroid, eig, radius])
    return out


def nearest_chamfer_with_candidates(
    queries: np.ndarray,
    references: np.ndarray,
    candidates: np.ndarray,
    *,
    device: str = "cpu",
    scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Nearest reference cloud (by CD) for each query, searched only among ``candidates[i]``.

    Exact CD against hundreds of thousands of train spines is infeasible, so candidates come
    from a cheap pre-filter (k nearest neighbours by :func:`cloud_descriptors`) and CD
    re-ranks them. ``references`` may be a memmap (only candidate rows are read).
    Returns ``distance`` and ``index`` (into ``references``) per query.
    """
    candidates = np.asarray(candidates, dtype=int)
    best_d = np.empty(len(queries))
    best_i = np.empty(len(queries), dtype=int)
    for i in range(len(queries)):
        order = np.sort(candidates[i])  # sorted -> sequential reads when references is a memmap
        d = chamfer_matrix(queries[i:i + 1], np.asarray(references[order]), device=device, scale=scale)[0]
        k = int(d.argmin())
        best_d[i], best_i[i] = d[k], order[k]
    return {"distance": best_d, "index": best_i}
