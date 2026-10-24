"""LabID -> Minnie65 raw layout for the S preprocessing (src/neuron_model/labid_layout.py) and LabID metadata."""

from pathlib import Path

import numpy as np
import pytest

from src.neuron_model import labid_layout
from src.neuron_model.metadata import LABID_GROUPS, labid_group, labid_unit_id

REPO = Path(__file__).resolve().parents[2]


def _sealed_patch():
    """A spine cut from a sphere surface and sealed by a fan: (surface, sealed spine, centre of the cut)."""
    import trimesh

    surface = trimesh.creation.icosphere(subdivisions=4, radius=1000.0)
    patch = trimesh.Trimesh(surface.vertices, surface.faces[surface.triangles_center[:, 2] > 800.0], process=False)
    patch.remove_unreferenced_vertices()
    edges = patch.edges[trimesh.grouping.group_rows(patch.edges_sorted, require_count=1)]
    centre = patch.vertices[np.unique(edges)].mean(0)
    v = np.vstack([patch.vertices, centre])
    sealed = trimesh.Trimesh(v, np.vstack([patch.faces, np.column_stack([edges, np.full(len(edges), len(v) - 1)])]), process=False)
    return surface, sealed, centre


def test_unseal_removes_exactly_the_seal_and_leaves_one_loop():
    from scipy.spatial import cKDTree

    surface, sealed, _ = _sealed_patch()
    faces = set(map(tuple, np.sort(surface.faces, axis=1)))
    res = labid_layout.unseal(sealed, faces, cKDTree(surface.vertices))
    assert res["unseal_status"] == "ok" and res["n_boundary_loops"] == 1 and res["n_seal_patches"] == 1
    opened = res["mesh"]
    assert len(opened.faces) == len(sealed.faces) - res["n_seal_faces"]
    # the opened spine keeps the surface vertices exactly (stage 1 / the branch preprocessing match by coordinates)
    d, _ = cKDTree(surface.vertices).query(opened.vertices)
    assert d.max() < 1e-9


def test_unseal_leaves_a_mesh_without_seal_unchanged():
    from scipy.spatial import cKDTree

    surface, _, _ = _sealed_patch()
    res = labid_layout.unseal(surface, set(map(tuple, np.sort(surface.faces, axis=1))), cKDTree(surface.vertices))
    assert res["unseal_status"] == "seal_not_found" and res["mesh"] is surface


def test_labid_group_metadata():
    assert labid_unit_id("Ab+9009", "24-2") == "Ab+9009_24_2"
    assert labid_group("Ab+9009_24_2") == "Ab+9009" and labid_group("Ab_ab23_1") == "Ab"
    assert labid_group("Wt_old_1003_22") == "Wt_old" and labid_group("Wt+9009_13_1") == "Wt+9009" and labid_group("Wt_1_1") == "Wt"
    assert labid_group("864691134886335738") is None
    assert LABID_GROUPS["Ab+9009"] == ("disease", "9009", "ab+9009")


def test_dataset_metadata_for_labid_and_minnie():
    from types import SimpleNamespace

    from src.neuron_model.metadata import dataset_metadata

    lab = dataset_metadata(SimpleNamespace(dataset="labid", neuron_id="Wt+9009_13_1", limb_id="000", branch_id="000"), {})
    assert (lab["species"], lab["health"], lab["treatment"], lab["imaging_series"]) == ("mouse", "healthy", "9009", "wt+9009")
    mn = dataset_metadata(SimpleNamespace(dataset="minnie65", neuron_id="1", limb_id="000", branch_id="000"), {})
    assert mn["health"] == "healthy" and mn["treatment"] is None and mn["imaging_series"] is None


@pytest.mark.skipif(not (REPO / "datasets/labid/Wt/1-1_processed/surface_mesh.off").exists(), reason="local LabID data not present")
def test_convert_one_labid_cell(tmp_path):
    import pandas as pd

    table = labid_layout.convert(REPO / "datasets/labid", tmp_path, units=["Wt_1_1"], log=None)
    branch = tmp_path / "labid" / "Wt_1_1" / "limb_000" / "branch_000"
    assert (branch / "branch_mesh.off").exists() and (branch / "shaft_mesh.off").exists()
    seg = np.load(branch / "branch_skeleton.npy")
    assert seg.ndim == 3 and seg.shape[1:] == (2, 3) and np.abs(seg).max() > 1000.0  # nm, not um
    assert len(table) == 9 and (table["unseal_status"] == "ok").all()
    assert len(list((branch / "spines").glob("spine_*.off"))) == 9
    assert (pd.read_parquet(tmp_path / "labid" / "labid_conversion.parquet")["neuron_id"] == "Wt_1_1").all()
    # second call: already converted, nothing redone
    again = labid_layout.convert(REPO / "datasets/labid", tmp_path, units=["Wt_1_1"], log=None)
    assert len(again) == 9
    # another scale into the same tree is refused (it would mix geometries; downstream hashes do not see the scale)
    with pytest.raises(ValueError, match="NEW output folder"):
        labid_layout.convert(REPO / "datasets/labid", tmp_path, units=["Wt_1_1"], z_scale={"all": 0.5}, log=None)
    # the z scale is applied to every mesh of the group
    scaled = labid_layout.convert(REPO / "datasets/labid", tmp_path / "z05", units=["Wt_1_1"], z_scale={"all": 0.5}, log=None)
    assert (scaled["z_scale"] == 0.5).all()
    import trimesh
    a = trimesh.load(branch / "branch_mesh.off", process=False)
    b = trimesh.load(tmp_path / "z05" / "labid" / "Wt_1_1" / "limb_000" / "branch_000" / "branch_mesh.off", process=False)
    assert np.allclose(b.vertices[:, 2], 0.5 * a.vertices[:, 2]) and np.allclose(b.vertices[:, :2], a.vertices[:, :2])
