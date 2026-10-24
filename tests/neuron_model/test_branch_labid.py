"""LabID inputs of the branch preprocessing (branch_preprocessing/labid.py)."""

from pathlib import Path

import numpy as np
import pytest

from src.neuron_model.branch_preprocessing import labid

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("group,name,cell", [
    ("Ab", "ab23-1_processed", "Ab_23"), ("Ab+9009", "24-2", "Ab+9009_24"), ("Wt", "8-1_processed-1", "Wt_8"),
    ("Wt_old", "1003-22", "Wt_old_1003"), ("Wt_old", "6", "Wt_old_6"), ("Wt_old", "3_full_res (10-2)", "Wt_old_3_full_res_10_2"),
])
def test_cell_of_parses_folder_names(group, name, cell):
    assert labid.cell_of(group, name) == cell


def test_attachment_centre_is_the_seal_cap():
    """A spine cut from a surface and sealed: the cap faces are the ones absent from the surface mesh."""
    import trimesh
    from scipy.spatial import cKDTree

    full = trimesh.creation.icosphere(subdivisions=4, radius=1000.0)
    keep = full.triangles_center[:, 2] > 800.0
    patch = trimesh.Trimesh(full.vertices, full.faces[keep], process=False)
    patch.remove_unreferenced_vertices()
    boundary = np.unique(patch.edges[trimesh.grouping.group_rows(patch.edges_sorted, require_count=1)])
    centre = patch.vertices[boundary].mean(0)
    # fan cap over the boundary loop (orientation irrelevant here)
    edges = patch.edges[trimesh.grouping.group_rows(patch.edges_sorted, require_count=1)]
    v = np.vstack([patch.vertices, centre])
    cap = np.column_stack([edges, np.full(len(edges), len(v) - 1)])
    spine = trimesh.Trimesh(v, np.vstack([patch.faces, cap]), process=False)
    shaft = trimesh.creation.icosphere(subdivisions=2, radius=500.0)
    got, area, method = labid.attachment_of_spine(spine, cKDTree(full.vertices), set(map(tuple, np.sort(full.faces, axis=1))), shaft)
    assert method == "seal_cap"
    assert np.linalg.norm(got - centre) < 30.0  # centroid of the fan ~ centre of the loop
    assert area == pytest.approx(np.pi * (1000.0**2 - 800.0**2), rel=0.2)  # jagged boundary, fan is not flat


@pytest.mark.skipif(not (REPO / "datasets/labid/Ab+9009/24-1/pure_surface_mesh.off").exists(), reason="local LabID data not present")
def test_labid_cell_end_to_end(tmp_path):
    from src.neuron_model.branch_preprocessing.config import load_config
    from src.neuron_model.branch_preprocessing.runner import run

    cfg = load_config(None, {"dataset": "labid", "raw_data_root": REPO / "datasets", "metadata_root": REPO / "datasets",
                             "output_root": tmp_path, "neuron_ids": ["Ab+9009_24"], "workers": 1})
    out = run(cfg)
    assert out["processed"] == 1 and out["branches_rows"] == 2
    import pandas as pd

    b = pd.read_parquet(tmp_path / "branches.parquet")
    s = pd.read_parquet(tmp_path / "branch_spines.parquet")
    assert set(b["status"]) == {"ok"} and set(b["health"]) == {"disease"} and set(b["treatment"]) == {"9009"}
    assert (b["radius_median_nm"].between(300, 1000)).all()  # micrometres converted to nm
    assert (s["attachment_method"] == "seal_cap").all() and s["has_position"].sum() >= 5
    assert (tmp_path / "Ab+9009_24" / "_done.json").exists()
