"""Torch datasets over preprocessed spines (point clouds, SDF samples).

Coordinates: preprocessing stores local coordinates in the dataset's physical
unit (nm for Minnie/H01). Every model uses **one global** ``coord_scale``
(physical -> model units, e.g. 1e-3 for nm -> µm; MoGen's reference is
``x_µm / 10`` = ``x_nm * 1e-4``), never a per-object normalisation - that would
hide scale errors. SDF values are scaled by the same factor.

Augmentation: no random 3D rotations (the local frame is biologically
meaningful); only an optional small Gaussian jitter in physical units and a
random choice among the 4 precomputed surface-resampling variants
(``pointcloud_<N>_<1..4>.npz``). Randomness uses torch's RNG, so it is
reproducible through the DataLoader ``generator`` / worker seeding.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .index import SpineIndex, pointcloud_variant_path

SAMPLE_SURFACE, SAMPLE_NEAR, SAMPLE_UNIFORM = 0, 1, 2  # spine_sampling.SAMPLE_TYPE_CODES


def _load_npz(path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class PointCloudDataset:
    """One item = one spine's surface point cloud (+ normals).

    Args:
        index: spines to serve (usually one split).
        n_points: 2048 / 4096 / 8192 - which precomputed size to read.
        variants: which resampling variants may be used.
        random_variant: pick a random variant per item (training) or always the first (evaluation).
        jitter_std: Gaussian jitter std **in physical units** (0 disables).
        coord_scale: physical -> model units.
        use_normals: also return normals.
    """

    def __init__(
        self,
        index: SpineIndex,
        n_points: int,
        *,
        variants: Sequence[int] = (1, 2, 3, 4),
        random_variant: bool = True,
        jitter_std: float = 0.0,
        coord_scale: float = 1.0,
        use_normals: bool = True,
    ) -> None:
        self.index = index
        self.n_points = int(n_points)
        self.variants = tuple(int(v) for v in variants)
        self.random_variant = random_variant
        self.jitter_std = float(jitter_std)
        self.coord_scale = float(coord_scale)
        self.use_normals = use_normals
        self._paths = index.frame[f"pointcloud_{self.n_points}_path"].tolist()
        self._keys = index.keys

    def __len__(self) -> int:
        return len(self._keys)

    def _variant(self) -> int:
        import torch

        if not self.random_variant:
            return self.variants[0]
        return self.variants[int(torch.randint(len(self.variants), (1,)))]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        import torch

        variant = self._variant()
        data = _load_npz(pointcloud_variant_path(self._paths[i], variant))
        points = torch.from_numpy(np.asarray(data["points"], dtype=np.float32))
        if points.shape[0] != self.n_points:
            raise ValueError(f"{self._keys[i]}: expected {self.n_points} points, file has {points.shape[0]}")
        if self.jitter_std > 0:
            points = points + torch.randn_like(points) * self.jitter_std
        item: Dict[str, Any] = {
            "points": points * self.coord_scale,
            "spine_key": self._keys[i],
            "variant": variant,
        }
        if self.use_normals:
            item["normals"] = torch.from_numpy(np.asarray(data["normals"], dtype=np.float32))
        return item


class SDFDataset:
    """One item = encoder point cloud + a minibatch of SDF query points for the same spine.

    ``query_counts`` = how many queries to draw per item from each sample type
    of ``sdf_samples.npz`` (surface / near-surface / uniform), without
    replacement. Uniform samples carry no surface normal; ``query_normals`` is
    0 there and ``query_has_normal`` is False. No jitter here: moving the
    encoder points would not move the SDF labels.
    """

    def __init__(
        self,
        index: SpineIndex,
        encoder_points: int,
        query_counts: Mapping[str, int],
        *,
        variants: Sequence[int] = (1, 2, 3, 4),
        random_variant: bool = True,
        coord_scale: float = 1.0,
        use_normals: bool = True,
    ) -> None:
        self.pointclouds = PointCloudDataset(
            index,
            encoder_points,
            variants=variants,
            random_variant=random_variant,
            jitter_std=0.0,
            coord_scale=coord_scale,
            use_normals=use_normals,
        )
        self.counts = {
            SAMPLE_SURFACE: int(query_counts.get("surface", 0)),
            SAMPLE_NEAR: int(query_counts.get("near", 0)),
            SAMPLE_UNIFORM: int(query_counts.get("uniform", 0)),
        }
        if sum(self.counts.values()) == 0:
            raise ValueError("query_counts must request at least one query point")
        self.random_queries = random_variant
        self.coord_scale = float(coord_scale)
        self._sdf_paths = index.frame["sdf_samples_path"].tolist()

    def __len__(self) -> int:
        return len(self.pointclouds)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        import torch

        item = self.pointclouds[i]
        sdf = _load_npz(self._sdf_paths[i])
        sample_type = np.asarray(sdf["sample_type"])
        chosen = []
        for code, count in self.counts.items():
            if count == 0:
                continue
            pool = np.flatnonzero(sample_type == code)
            if len(pool) < count:
                raise ValueError(f"{item['spine_key']}: asked {count} queries of type {code}, file has {len(pool)}")
            if self.random_queries:
                pick = pool[torch.randperm(len(pool))[:count].numpy()]
            else:
                pick = pool[:count]
            chosen.append(pick)
        idx = np.concatenate(chosen)
        normals = np.asarray(sdf["surface_normals"], dtype=np.float32)[idx]
        has_normal = np.isfinite(normals).all(axis=1)
        normals[~has_normal] = 0.0
        item.update(
            {
                "query_points": torch.from_numpy(np.asarray(sdf["query_points"], dtype=np.float32)[idx]) * self.coord_scale,
                "sdf": torch.from_numpy(np.asarray(sdf["sdf"], dtype=np.float32)[idx]) * self.coord_scale,
                "sample_type": torch.from_numpy(sample_type[idx].astype(np.int64)),
                "query_normals": torch.from_numpy(normals),
                "query_has_normal": torch.from_numpy(has_normal),
            }
        )
        return item


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0, drop_last: bool = False, pin_memory: Optional[bool] = None):
    """DataLoader with reproducible shuffling and worker seeding."""
    import torch

    from ..experiment.seed import torch_generator, worker_init_fn

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=torch_generator(seed),
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=num_workers > 0,
    )
