import numpy as np
import pandas as pd
import pytest
import torch

from src.neuron_model.spine_generation.data.bbox import FrozenBBox, compute_frozen_bbox
from src.neuron_model.spine_generation.data.datasets import SDFDataset, PointCloudDataset, make_loader
from src.neuron_model.pointcloud_io import load_points
from src.neuron_model.spine_generation.data.index import load_spine_index, pointcloud_variant_path, spine_relative_dir
from conftest import N_SURFACE
from src.neuron_model.spine_generation.data.splits import (
    assert_no_leakage,
    build_group_split,
    load_splits,
    save_splits,
    split_summary,
)

def test_index_reroots_windows_paths_and_filters(fake_dataset):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    assert len(index) == 23  # one spine not train_eligible
    assert index.frame.attrs["n_manifest"] == 24
    first = index.frame.iloc[0]
    assert first["spine_key"] == "minnie65/86469000/000/001/000"
    assert first["pointcloud_8192_path"].startswith(str(fake_dataset))
    assert first["pointclouds_path"] == first["pointcloud_8192_path"]  # packed file rerooted too
    assert pointcloud_variant_path(first["pointcloud_8192_path"], 3).name == "pointclouds.npz"
    assert index.dataset_version == "v1:abc123"


def test_index_drops_spines_with_missing_files(fake_dataset):
    (fake_dataset / spine_relative_dir("86469005", "000", "001", "000") / "sdf_samples.npz").unlink()
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    assert len(index) == 22 and "minnie65/86469005/000/001/000" not in index.keys


def test_group_split_no_leakage_and_group_fractions(fake_dataset):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    splits = build_group_split(index.frame, seed=1)
    assert_no_leakage(splits)
    summary = split_summary(splits, index.frame)
    assert summary["n_groups"] == {"train": 10, "val": 1, "test": 1}  # round(0.8*12)=10, round(0.1*12)=1
    assert sum(summary["n_spines"].values()) == len(index)
    assert set(splits["group_id"].str.split("/").str[0]) == {"minnie65"}
    again = build_group_split(index.frame, seed=1)
    pd.testing.assert_frame_equal(splits, again)
    assert not build_group_split(index.frame, seed=2)["split"].equals(splits["split"])


def test_split_stratification_spreads_strata():
    frame = pd.DataFrame(
        {
            "spine_key": [f"d/n{i}/0/0/0" for i in range(100)],
            "dataset": "d", "neuron_id": [f"n{i}" for i in range(100)],
            "cell_type_binary": ["excitatory"] * 50 + ["inhibitory"] * 50,
        }
    )
    splits = build_group_split(frame, seed=3, stratify=("cell_type_binary",)).merge(frame, on="spine_key")
    counts = splits.groupby(["split", "cell_type_binary"]).size().unstack()
    assert (counts.loc["train"] == 40).all() and (counts.loc["val"] == 5).all() and (counts.loc["test"] == 5).all()


def test_save_splits_is_frozen(fake_dataset, tmp_path):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    splits = build_group_split(index.frame, seed=1)
    path = tmp_path / "splits_v1.parquet"
    save_splits(splits, path, version="splits_v1", info={"seed": 1})
    assert load_splits(path)["split_version"].eq("splits_v1").all()
    with pytest.raises(FileExistsError):
        save_splits(splits, path, version="splits_v1", info={})
    with_split = index.with_split(load_splits(path), "val")
    assert set(with_split.frame["split"]) == {"val"}


def test_pointcloud_dataset_scale_variants_and_jitter(fake_dataset):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    raw = load_points(index.frame.iloc[0]["pointcloud_2048_path"], 2048, 1)
    evaluation = PointCloudDataset(index, 2048, random_variant=False, coord_scale=1e-3)
    item = evaluation[0]
    assert item["points"].shape == (2048, 3) and item["normals"].shape == (2048, 3) and item["variant"] == 1
    assert torch.allclose(item["points"], torch.from_numpy(raw) * 1e-3)

    torch.manual_seed(0)
    train = PointCloudDataset(index, 2048, random_variant=True, jitter_std=5.0, use_normals=False)
    variants = {train[0]["variant"] for _ in range(40)}
    assert variants == {1, 2, 3, 4} and "normals" not in train[0]
    assert not torch.allclose(train[0]["points"], evaluation[0]["points"] / 1e-3)


def test_sdf_dataset_query_counts_and_normal_mask(fake_dataset):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    dataset = SDFDataset(index, 2048, {"surface": 5, "near": 10, "uniform": 4}, coord_scale=0.5)
    item = dataset[0]
    assert item["query_points"].shape == (19, 3) and item["sdf"].shape == (19,)
    assert item["sample_type"].tolist() == [0] * 5 + [1] * 10 + [2] * 4
    assert item["query_has_normal"].tolist() == [True] * 15 + [False] * 4
    assert torch.all(item["query_normals"][~item["query_has_normal"]] == 0)
    deterministic = SDFDataset(index, 2048, {"surface": 5}, random_variant=False, coord_scale=0.5)
    assert deterministic[0]["sdf"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]  # first 5 surface samples, scaled
    with pytest.raises(ValueError):
        SDFDataset(index, 2048, {"surface": N_SURFACE + 1})[0]


def test_loader_batches_and_is_reproducible(fake_dataset):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    dataset = PointCloudDataset(index, 2048, random_variant=True)
    keys_a = [b["spine_key"] for b in make_loader(dataset, batch_size=4, shuffle=True, seed=5)]
    keys_b = [b["spine_key"] for b in make_loader(dataset, batch_size=4, shuffle=True, seed=5)]
    assert keys_a == keys_b and len(keys_a) == 6
    batch = next(iter(make_loader(dataset, batch_size=4, shuffle=False, seed=5)))
    assert batch["points"].shape == (4, 2048, 3)


def test_frozen_bbox(fake_dataset, tmp_path):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    splits = build_group_split(index.frame, seed=1)
    train = index.with_split(splits, "train")
    bbox = compute_frozen_bbox(train, margin=50.0, n_points=2048, split_version="splits_v1")
    all_points = np.concatenate([load_points(p, 2048, 1) for p in train.frame["pointcloud_2048_path"]])
    assert np.allclose(bbox.low, all_points.min(axis=0) - 50.0) and np.allclose(bbox.high, all_points.max(axis=0) + 50.0)
    bbox.save(tmp_path / "bbox.json")
    assert FrozenBBox.load(tmp_path / "bbox.json") == bbox
    with pytest.raises(FileExistsError):
        bbox.save(tmp_path / "bbox.json")
    with pytest.raises(ValueError):
        compute_frozen_bbox(index.with_split(splits), margin=1.0, n_points=2048)


def test_legacy_pointcloud_format_still_loads(fake_dataset_legacy):
    index = load_spine_index([fake_dataset_legacy / "manifest.parquet"])
    first = index.frame.iloc[0]
    assert pointcloud_variant_path(first["pointcloud_8192_path"], 3).name == "pointcloud_8192_3.npz"
    dataset = PointCloudDataset(index, 2048, random_variant=False, variants=(3,))
    item = dataset[0]
    raw = np.load(pointcloud_variant_path(first["pointcloud_2048_path"], 3))["points"]
    assert item["variant"] == 3 and torch.allclose(item["points"], torch.from_numpy(raw))
