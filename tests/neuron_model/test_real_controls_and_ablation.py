"""Untouched test split, real controls (generator layout, readable by the evaluation) and classifier ablation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.neuron_model.spine_generation.evaluation.classifier_ablation import classifier_ablation, default_groups
from src.neuron_model.spine_generation.evaluation.distribution_metrics import JOINT_FEATURES


def test_untouched_split_drops_dev_test_and_dev_neurons():
    from scripts.neuron_model.make_untouched_test_split import untouched_split

    dev = pd.DataFrame({"spine_key": ["a1", "a2", "b1", "c1"], "group_id": ["A", "A", "B", "C"], "split": ["train", "train", "val", "test"]})
    final = pd.DataFrame({"spine_key": ["a1", "a2", "b1", "c1", "d1", "e1", "f1"], "group_id": ["A", "A", "B", "C", "D", "E", "F"],
                          "split": ["test", "train", "train", "test", "test", "train", "test"]})
    t = untouched_split(dev, final).set_index("spine_key")["split"].to_dict()
    # dev train/val kept; dev test neuron C gone; final test neurons that are in dev (A) are not test; D and F are
    assert t == {"a1": "train", "a2": "train", "b1": "val", "d1": "test", "f1": "test"}


def _sphere(path, radius=200.0):
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    mesh.apply_translation([0.0, radius, 0.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return mesh


def test_controls_write_generator_layout_readable_by_evaluation(tmp_path):
    from src.neuron_model.spine_generation.evaluation.inputs import discover_seed_dirs, load_generated_set
    from src.neuron_model.spine_generation.evaluation.real_controls import assign_samples, run_control

    keys = [f"k{i}" for i in range(6)]
    picks = assign_samples(keys, n_per_seed=2, seeds=[1, 2], base_seed=0)
    assert len(set(picks[1]) | set(picks[2])) == 4  # disjoint subsets
    for i, key in enumerate(keys[:2]):
        mesh_path = tmp_path / "meshes" / f"{key}.off"
        _sphere(mesh_path, radius=150.0 + 20 * i)
        spine = {"spine_key": key, "local_sealed_mesh_path": str(mesh_path), "pointcloud_8192_path": None}
        for control, kwargs in (("asis", {"postprocess_cfg": {"min_component_area_fraction": 0.01}}),
                                ("mc", {"bbox_low": [-400.0, -50.0, -400.0], "bbox_high": [400.0, 600.0, 400.0], "resolution": 32,
                                        "postprocess_cfg": {"min_component_area_fraction": 0.01}})):
            row = run_control({"control": control, "seed": 1, "spine": spine, "kwargs": kwargs,
                               "sample_dir": str(tmp_path / "out" / control / "seed_1" / f"sample_{i:05d}")})
            assert row["error"] is None, row
            assert row["post_valid"]
    for control in ("asis", "mc"):
        seeds = discover_seed_dirs(tmp_path / "out" / control)
        meshes, reports, ids = load_generated_set(seeds[1], "postprocessed")
        assert len(meshes) == 2 and all(m is not None for m in meshes)
        vol = [m.volume for m in meshes]
        # the mc control approximates the real volume at this coarse grid (voxel 25 nm on a 150-170 nm sphere)
        if control == "mc":
            true = [4 / 3 * np.pi * r ** 3 for r in (150.0, 170.0)]
            assert np.allclose(vol, true, rtol=0.12)
        meta = json.loads((seeds[1] / "sample_00000" / "generation_metadata.json").read_text())
        assert meta["model"] == f"real_{control}" and meta["spine_key"] == "k0"


def test_banded_sdf_gives_the_same_marching_cubes_mesh(tmp_path):
    from src.neuron_model.spine_generation.evaluation.real_controls import banded_sdf
    from src.neuron_model.spine_generation.reconstruction.marching_cubes import GridSpec, sdf_to_mesh
    from src.neuron_model.spine_sampling import signed_distance

    mesh = _sphere(tmp_path / "s.off", radius=150.0)
    grid = GridSpec((-1200.0, -300.0, -900.0), (1300.0, 1800.0, 1000.0), 40)
    full, _, v_full = sdf_to_mesh(lambda q: signed_distance(mesh, q)[0], grid)
    band, _, v_band = sdf_to_mesh(banded_sdf(mesh, grid), grid)
    assert np.array_equal(np.sign(v_full), np.sign(v_band))
    assert np.allclose(np.sort(full.vertices, axis=0), np.sort(band.vertices, axis=0))


def test_failed_control_is_a_row_not_a_crash(tmp_path):
    from src.neuron_model.spine_generation.evaluation.real_controls import run_control

    row = run_control({"control": "asis", "seed": 1, "spine": {"spine_key": "x", "local_sealed_mesh_path": str(tmp_path / "missing.off")},
                       "kwargs": {}, "sample_dir": str(tmp_path / "out" / "asis" / "seed_1" / "sample_00000")})
    assert row["error"] is not None


def _sets(shift_feature=None, n=400, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.full((len(JOINT_FEATURES), len(JOINT_FEATURES)), 0.3) + 0.7 * np.eye(len(JOINT_FEATURES))
    real = pd.DataFrame(rng.multivariate_normal(np.zeros(len(JOINT_FEATURES)), cov, size=n), columns=JOINT_FEATURES)
    gen = pd.DataFrame(rng.multivariate_normal(np.zeros(len(JOINT_FEATURES)), cov, size=n), columns=JOINT_FEATURES)
    if shift_feature:
        gen[shift_feature] += 1.5
    return real, gen


def test_ablation_finds_the_only_shifted_feature():
    real, gen = _sets("Area")
    t = classifier_ablation(real, gen, n_repeats=3)
    sub = t["subsets"]
    groups = sub[sub.kind == "group"].set_index("name")
    assert groups.loc["all", "roc_auc"] > 0.8
    assert groups.loc["no_area", "roc_auc"] < 0.6          # removing the area group removes the signal
    assert groups.loc["no_attachment", "roc_auc"] < 0.6    # Area is an attachment metric too
    single = sub[sub.kind == "single"].sort_values("roc_auc", ascending=False)
    assert single.iloc[0]["name"] == "Area"
    drop = sub[sub.kind == "drop_one"].sort_values("auc_minus_all")
    assert drop.iloc[0]["name"] == "without Area"
    coef = t["coefficients"].set_index("feature")["coef_mean"]
    assert coef.abs().idxmax() == "Area" and coef["Area"] > 0  # + = larger in the generated set
    assert t["shifts"].set_index("feature").loc["Area", "median_shift_sd"] > 1.0


def test_ablation_null_is_near_chance_and_groups_are_defined():
    real, gen = _sets(None)
    t = classifier_ablation(real, gen, n_repeats=3)
    assert abs(t["subsets"].set_index("name").loc["all", "roc_auc"] - 0.5) < 0.07
    g = default_groups()
    assert set(g["no_area_no_attachment"]) == {"Volume", "ConvexHullVolume", "ConvexHullRatio"}
    assert "Area" not in g["no_area"] and "LengthAreaRatio" not in g["no_area"]


def test_ablation_cli_on_an_evaluation_folder(tmp_path, monkeypatch, capsys):
    import sys

    from scripts.neuron_model import classifier_ablation as cli

    real, gen = _sets("Volume")
    run = tmp_path / "eval"
    (run / "postprocessed").mkdir(parents=True)
    real.to_parquet(run / "real_test_morphometrics.parquet")
    gen.iloc[:200].to_parquet(run / "postprocessed" / "morphometrics_vae_flow_seed1.parquet")
    gen.iloc[200:].to_parquet(run / "postprocessed" / "morphometrics_vae_flow_seed2.parquet")
    real.iloc[:300].to_parquet(run / "postprocessed" / "morphometrics_real_asis_seed1.parquet")
    monkeypatch.setattr(sys, "argv", ["x", "--eval-run", str(run), "--n-repeats", "2"])
    cli.main()
    summary = pd.read_csv(run / "classifier_ablation" / "postprocessed" / "summary_real.csv").set_index("model")
    assert set(summary.index) == {"vae_flow", "real_asis"}
    assert summary.loc["vae_flow", "n_per_class"] == 400
    assert summary.loc["vae_flow", "auc_all"] > 0.75 and summary.loc["real_asis", "auc_all"] < 0.6
    assert summary.loc["vae_flow", "auc_no_attachment"] > 0.75   # Volume is attachment-free


def test_make_real_controls_cli_end_to_end(fake_dataset, tmp_path, monkeypatch):
    import sys

    from scripts.neuron_model import make_real_controls as cli
    from src.neuron_model.spine_generation.data.index import load_spine_index

    manifest = fake_dataset / "manifest.parquet"
    index = load_spine_index([manifest], check_files=())
    for path in index.frame["local_sealed_mesh_path"]:
        _sphere(__import__("pathlib").Path(path), radius=150.0)
    groups = index.frame["group_id"] if "group_id" in index.frame else index.frame["neuron_id"]
    splits = pd.DataFrame({"spine_key": index.frame["spine_key"], "group_id": groups,
                           "split": np.where(index.frame["neuron_id"].str.endswith(("8", "9", "10", "11")), "test", "train")})
    splits.to_parquet(tmp_path / "splits.parquet")
    bbox = tmp_path / "bbox.json"
    bbox.write_text(json.dumps({"low": [-400, -50, -400], "high": [400, 600, 400], "margin": 0, "unit": "nm", "source": "test", "split_version": "t", "n_spines": 1}))
    out = tmp_path / "controls"
    monkeypatch.setattr(sys, "argv", ["x", "--manifest", str(manifest), "--splits", str(tmp_path / "splits.parquet"), "--controls", "asis", "mc",
                                      "--bbox", str(bbox), "--resolution", "24", "--n-samples", "2", "--seeds", "1", "2", "--out", str(out)])
    cli.main()
    table = pd.read_csv(out / "controls.csv")
    assert len(table) == 2 * 2 * 2 and table["error"].isna().all()
    assert set(table.groupby("seed")["spine_key"].nunique()) == {2}
    assert (out / "mc" / "seed_2" / "sample_00001" / "generated_mesh_postprocessed.off").exists()
