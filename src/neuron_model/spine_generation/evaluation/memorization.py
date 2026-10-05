"""Memorization: does the generator copy training spines? (spec, "memorization")

Primary test - DCR (distance to closest record, standard in synthetic-data evaluation):
for every generated spine, the distance to its nearest spine in the FULL train set is
compared with the same distance for the real TEST spines (held out, so not memorised by
construction, and measured against the very same reference set - no set-size bias):

- ``dcr_median_ratio`` = median(gen -> train) / median(test -> train): ~1 for a generator
  that has learned the distribution; << 1 means generated spines sit closer to training
  spines than unseen real spines do;
- ``near_copy_rate`` = share of generated spines closer to train than the ``quantile`` (5%)
  of test -> train distances; expected ~``quantile`` without memorization;
- ``mannwhitney_pvalue``: one-sided, H1 = gen -> train distances are smaller.

Secondary - the spec's literal check "closer to train than to test", per generated spine.
Train must be subsampled to the test size (an 8x larger train set is closer by chance),
which also DILUTES the signal: a copied train spine is in the subsample only with
probability |test| / |train|, so a pure copier scores ~0.5 + 0.5 * |test| / |train|, not 1.
Kept for completeness; rely on the DCR numbers.

Scale: morphometric search uses KD-trees (10-D, fine for the full ~0.6M train set). Chamfer
on the full train set uses candidate re-ranking: the k nearest train clouds by cheap shape
descriptors (``point_cloud_metrics.cloud_descriptors``), then exact Chamfer among them.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree


def dcr_report(d_gen_train: np.ndarray, d_holdout_train: np.ndarray, *, quantile: float = 0.05) -> Dict[str, float]:
    """Compare nearest-train distances of generated vs held-out real samples (see module docstring)."""
    g = np.asarray(d_gen_train, dtype=float)
    h = np.asarray(d_holdout_train, dtype=float)
    threshold = float(np.quantile(h, quantile))
    return {
        "dcr_median_generated": float(np.median(g)),
        "dcr_median_holdout": float(np.median(h)),
        "dcr_median_ratio": float(np.median(g) / np.median(h)) if np.median(h) > 0 else np.nan,
        "near_copy_threshold": threshold,
        "near_copy_rate": float((g < threshold).mean()),
        "near_copy_rate_expected": float(quantile),
        "mannwhitney_pvalue": float(stats.mannwhitneyu(g, h, alternative="less").pvalue),
        "n_generated": int(g.size),
        "n_holdout": int(h.size),
    }


def morphometric_dcr(gen_x: np.ndarray, holdout_x: np.ndarray, train_x: np.ndarray, *, quantile: float = 0.05) -> Dict[str, float]:
    """DCR in standardised morphometric space against the full train set (KD-tree)."""
    tree = cKDTree(np.asarray(train_x, dtype=float))
    return dcr_report(tree.query(np.asarray(gen_x, float), k=1)[0], tree.query(np.asarray(holdout_x, float), k=1)[0], quantile=quantile)


def chamfer_dcr(
    gen_clouds: np.ndarray,
    holdout_clouds: np.ndarray,
    train_clouds: np.ndarray,
    *,
    k_candidates: int = 50,
    quantile: float = 0.05,
    device: str = "cpu",
    scale: float = 1.0,
    train_descriptors: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """DCR by Chamfer distance against the full train set via descriptor-candidate re-ranking.

    ``train_clouds`` may be a ``np.load(..., mmap_mode="r")`` memmap - only candidate rows are read.
    ``train_descriptors`` (precomputed ``cloud_descriptors(train_clouds)``) avoids re-reading
    the whole train set for every evaluated model.
    """
    from .point_cloud_metrics import cloud_descriptors, nearest_chamfer_with_candidates

    train_desc = cloud_descriptors(train_clouds) if train_descriptors is None else np.asarray(train_descriptors, float)
    mean, std = train_desc.mean(axis=0), train_desc.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    tree = cKDTree((train_desc - mean) / std)
    k = min(k_candidates, len(train_desc))

    def nearest(clouds: np.ndarray) -> np.ndarray:
        cand = tree.query((cloud_descriptors(clouds) - mean) / std, k=k)[1].reshape(len(clouds), k)
        return nearest_chamfer_with_candidates(clouds, train_clouds, cand, device=device, scale=scale)["distance"]

    result = dcr_report(nearest(gen_clouds), nearest(holdout_clouds), quantile=quantile)
    result["k_candidates"] = int(k)
    return result


def _closer_to_train(
    nearest_train: Callable[[np.ndarray], np.ndarray],
    nearest_test: np.ndarray,
    n_train: int,
    n_test: int,
    *,
    n_repeats: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    repeats = n_repeats if n_train > n_test else 1
    fractions = []
    for _ in range(repeats):
        cols = rng.choice(n_train, size=n_test, replace=False) if n_train > n_test else np.arange(n_train)
        fractions.append(float((nearest_train(cols) < nearest_test).mean()))
    return {
        "fraction_closer_to_train": float(np.mean(fractions)),
        "fraction_closer_to_train_std": float(np.std(fractions)),
        # what a pure copier would score after the dilution (see module docstring)
        "fraction_closer_to_train_if_copying": float(0.5 + 0.5 * min(1.0, n_test / n_train)),
    }


def closer_to_train_points(gen_x: np.ndarray, train_x: np.ndarray, test_x: np.ndarray, *, n_repeats: int = 20, seed: int = 0) -> Dict[str, float]:
    """Spec's literal check in feature space (train subsampled to the test size; diluted - see docstring)."""
    gen_x, train_x = np.asarray(gen_x, float), np.asarray(train_x, float)
    nearest_test = cKDTree(np.asarray(test_x, float)).query(gen_x, k=1)[0]
    return _closer_to_train(lambda cols: cKDTree(train_x[cols]).query(gen_x, k=1)[0], nearest_test, len(train_x), len(test_x), n_repeats=n_repeats, seed=seed)


def closer_to_train_matrices(d_gen_train: np.ndarray, d_gen_test: np.ndarray, *, n_repeats: int = 20, seed: int = 0) -> Dict[str, float]:
    """Spec's literal check from precomputed distance matrices (e.g. Chamfer on subsets)."""
    d_gen_train, d_gen_test = np.asarray(d_gen_train, float), np.asarray(d_gen_test, float)
    return _closer_to_train(lambda cols: d_gen_train[:, cols].min(axis=1), d_gen_test.min(axis=1), d_gen_train.shape[1], d_gen_test.shape[1], n_repeats=n_repeats, seed=seed)


def memorization_report(
    *,
    gen_x: np.ndarray,
    train_x: np.ndarray,
    test_x: np.ndarray,
    n_repeats: int = 20,
    quantile: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Morphometric-space memorization (DCR + the spec's literal check). Chamfer DCR: :func:`chamfer_dcr`."""
    result = {f"morphometric_{k}": v for k, v in morphometric_dcr(gen_x, test_x, train_x, quantile=quantile).items()}
    result.update({f"morphometric_literal_{k}": v for k, v in closer_to_train_points(gen_x, train_x, test_x, n_repeats=n_repeats, seed=seed).items()})
    return result
