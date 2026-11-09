"""reconstruct_mogen_poisson.py --out: re-meshing into a copy leaves the source samples untouched (Poisson stubbed: pymeshlab
does not load on the mac)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_remesh_into_a_copy(tmp_path, monkeypatch):
    import trimesh

    from src.neuron_model.spine_generation.experiment.artifacts import write_pointcloud_ply
    from src.neuron_model.spine_generation.experiment.run import create_run

    run = create_run({"name": "mogen"}, experiment_id="mogen_test", runs_root=tmp_path / "runs")
    src = run.meshes / "seed_1"
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=200.0)
    for i in range(3):
        d = src / f"sample_{i:05d}"
        d.mkdir(parents=True)
        pts, _ = trimesh.sample.sample_surface(sphere, 2048, seed=i)
        write_pointcloud_ply(d / "generated_pointcloud.ply", np.asarray(pts))
        (d / "generated_mesh_postprocessed.off").write_text("OLD")
        (d / "sample_info.json").write_text("{}")

    spec = importlib.util.spec_from_file_location("remesh", REPO_ROOT / "scripts/neuron_model/reconstruct_mogen_poisson.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "screened_poisson", lambda points, normals, params: trimesh.creation.icosphere(subdivisions=2, radius=200.0))
    out = tmp_path / "meshes_pw4"
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run.root), "--seed", "1", "--out", str(out),
                                      "--poisson-config", str(REPO_ROOT / "configs/neuron-model/reconstruction/poisson_sweep/d8_pw4.yaml")])
    mod.main()
    for i in range(3):
        assert (src / f"sample_{i:05d}" / "generated_mesh_postprocessed.off").read_text() == "OLD"
        new = out / "seed_1" / f"sample_{i:05d}"
        assert (new / "sample_info.json").exists() and (new / "generated_pointcloud.ply").exists()
        assert (new / "generated_mesh_postprocessed.off").read_text() != "OLD"
        assert '"point_weight": 4.0' in (new / "poisson_config.json").read_text()
    assert (out / "reconstruction_source.json").exists()
