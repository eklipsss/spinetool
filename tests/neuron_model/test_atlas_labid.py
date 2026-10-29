"""LabID spines on the maps of the spine atlas: one record list per experimental group, tagged, encoded by the (here faked) VAE."""

from types import SimpleNamespace

import numpy as np

from src.neuron_model.spine_generation import atlas as A
from src.neuron_model.spine_generation.data.index import SpineIndex, load_spine_index


def test_labid_records_group_the_spines_and_tag_them(fake_dataset, monkeypatch):
    index = load_spine_index([fake_dataset / "manifest.parquet"], check_files=())
    frame = index.frame.copy()
    names = ["Wt_1_1", "Wt_1_2", "Ab_1_1", "Ab+9009_3_1"] * 3  # 12 neurons -> four LabID-style ids (a prefix decides the group)
    mapping = {n: names[i] + f"_{i}" for i, n in enumerate(sorted(frame["neuron_id"].unique()))}
    frame["neuron_id"] = frame["neuron_id"].map(mapping)
    labid_index = SpineIndex(frame, index.dataset_versions)
    monkeypatch.setattr(A, "encode", lambda vae, points, normals: np.arange(4, dtype=float))
    vae = SimpleNamespace(n_points=2048)
    out = A.labid_records(vae, labid_index, n_per_group=4, n_meshes=0, seed=0, progress=False)
    assert set(out) == {"Wt", "Ab", "Ab+9009"}
    assert all(len(rs) <= 4 for rs in out.values()) and len(out["Wt"]) == 4
    for g, rs in out.items():
        assert all(r.source == f"labid_{g}" and r.meta["labid_group"] == g and r.latent is not None for r in rs)
    only = A.labid_records(vae, labid_index, groups=["Ab"], n_per_group=2, n_meshes=0, progress=False)
    assert list(only) == ["Ab"] and len(only["Ab"]) == 2
