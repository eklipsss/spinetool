"""LabID inputs of the branch preprocessing (branch_preprocessing/labid.py) on the converted tree."""

from pathlib import Path

import numpy as np
import pytest

from src.neuron_model.metadata import labid_name_numbers, labid_unit_id

REPO = Path(__file__).resolve().parents[2]
HAVE_LABID = (REPO / "datasets/labid/Ab+9009/24-1/pure_surface_mesh.off").exists()


@pytest.mark.parametrize("group,name,unit,numbers", [
    ("Ab", "ab23-1_processed", "Ab_ab23_1", (23, 1)), ("Ab+9009", "24-2", "Ab+9009_24_2", (24, 2)), ("Wt", "8-1_processed-1", "Wt_8_1_1", (8, 1)),
    ("Wt_old", "1003-22", "Wt_old_1003_22", (1003, 22)), ("Wt_old", "6", "Wt_old_6", (6, None)),
    ("Wt_old", "3_full_res (10-2)", "Wt_old_3_full_res_10_2", (3, None)),
])
def test_every_folder_is_its_own_branch(group, name, unit, numbers):
    assert labid_unit_id(group, name) == unit
    assert labid_name_numbers(name) == numbers


def test_folder_ids_are_unique_on_the_local_data():
    if not HAVE_LABID:
        pytest.skip("local LabID data not present")
    from src.neuron_model.labid_layout import discover_raw

    frame = discover_raw(REPO / "datasets/labid")
    assert len(frame) == frame["neuron_id"].nunique() == 179


def test_raw_layout_is_refused(tmp_path):
    from src.neuron_model.branch_preprocessing.config import load_config
    from src.neuron_model.branch_preprocessing.labid import discover

    raw = tmp_path / "labid" / "Wt" / "1-1"
    raw.mkdir(parents=True)
    (raw / "pure_surface_mesh.off").write_text("OFF\n0 0 0\n")
    with pytest.raises(ValueError, match="convert it first"):
        discover(load_config(None, {"dataset": "labid", "raw_data_root": tmp_path, "metadata_root": tmp_path}))


@pytest.mark.skipif(not HAVE_LABID, reason="local LabID data not present")
def test_labid_branch_end_to_end(tmp_path):
    import pandas as pd

    from src.neuron_model.branch_preprocessing.config import load_config
    from src.neuron_model.branch_preprocessing.runner import run
    from src.neuron_model.labid_layout import convert

    units = ["Ab+9009_24_1", "Ab+9009_24_2"]
    convert(REPO / "datasets/labid", tmp_path / "raw", units=units, log=None)
    cfg = load_config(None, {"dataset": "labid", "raw_data_root": tmp_path / "raw", "metadata_root": REPO / "datasets",
                             "output_root": tmp_path / "out", "workers": 1})
    out = run(cfg)
    assert out["processed"] == 2 and out["branches_rows"] == 2
    b = pd.read_parquet(tmp_path / "out" / "branches.parquet")
    s = pd.read_parquet(tmp_path / "out" / "branch_spines.parquet")
    assert set(b["neuron_id"]) == set(units) and set(b["merged_branch_id"]) == {0}
    assert set(b["status"]) == {"ok"} and set(b["health"]) == {"disease"} and set(b["treatment"]) == {"9009"}
    assert set(b["name_number_1"]) == {24} and set(b["unit_nm"]) == {1000.0}
    assert (b["radius_median_nm"].between(300, 1000)).all()  # micrometres converted to nm
    assert (s["attachment_method"] == "boundary_loop").all() and s["has_position"].sum() >= 5
    assert (tmp_path / "out" / "Ab+9009_24_1" / "_done.json").exists()
