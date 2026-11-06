"""Decoder comparison on paired reconstructions (models/spine_vae/decoder_study.py, compare_vae_decoders.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.neuron_model.spine_generation.models.spine_vae.decoder_study import cluster_mean_ci, paired_bias, surface_distances

from test_vae_latent_prior import _tiny_vae_run  # noqa: E402 (tests dir is on sys.path, rootdir conftest)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _table(rng, n_neurons=12, per=15, shifts=None):
    shifts = shifts or {}
    rows = []
    for g in range(n_neurons):
        style = rng.normal(0, 1.0, 2)  # neuron style shared by its spines (and by their reconstructions)
        for s in range(per):
            ref = style + rng.normal(0, 1.0, 2)
            key = f"n{g}_s{s}"
            rows.append({"spine_key": key, "neuron_id": f"n{g}", "kind": "ref", "a": ref[0], "b": ref[1]})
            for kind, (da, db) in shifts.items():
                rows.append({"spine_key": key, "neuron_id": f"n{g}", "kind": kind,
                             "a": ref[0] + da + rng.normal(0, 0.3), "b": ref[1] + db + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


def test_paired_bias_finds_the_planted_bias_and_the_better_decoder():
    rng = np.random.default_rng(0)
    sd = np.sqrt(2.0)  # sd of ref = style (1) + spine (1)
    table = _table(rng, shifts={"base": (0.5 * sd, 0.0), "better": (0.1 * sd, 0.0), "same": (0.5 * sd, 0.0)})
    res = paired_bias(table, ["base", "better", "same"], reference="ref", metrics=["a", "b"], baseline="base", n_boot=500, seed=1)
    pm = res["per_metric"].set_index(["kind", "metric"])
    assert abs(pm.loc[("base", "a"), "bias_sd"] - 0.5) < 0.12 and pm.loc[("base", "a"), "significant"]
    assert not pm.loc[("base", "b"), "significant"]
    s = res["summary"].set_index("kind")
    assert s.loc["better", "diff_ci_high"] < 0                       # clearly better than the baseline
    assert s.loc["same", "diff_ci_low"] < 0 < s.loc["same", "diff_ci_high"]   # same recipe: no difference


def test_surface_distances_and_cluster_ci():
    import trimesh

    a = trimesh.creation.icosphere(subdivisions=3, radius=100.0)
    b = trimesh.creation.icosphere(subdivisions=3, radius=110.0)
    d = surface_distances(a, b, n_points=5000)
    assert 8 < d["chamfer_l1"] < 12 and d["hd95"] < 14
    assert np.isnan(surface_distances(None, b)["chamfer_l1"])
    ci = cluster_mean_ci(pd.Series([1.0, 2.0, np.nan, 3.0]), pd.Series(["x", "x", "y", "y"]), n_boot=200)
    assert ci["n"] == 3 and ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_compare_vae_decoders_script_on_a_tiny_run(fake_dataset, tmp_path):
    import trimesh

    from src.neuron_model.spine_generation.data.index import load_spine_index

    run = _tiny_vae_run(fake_dataset, tmp_path)
    for path in load_spine_index([fake_dataset / "manifest.parquet"]).frame["local_sealed_mesh_path"]:
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=150.0)
        sphere.apply_translation([0.0, 150.0, 0.0])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sphere.export(path)
    bbox = tmp_path / "bbox.json"
    bbox.write_text(json.dumps({"low": [-400.0, -100.0, -400.0], "high": [400.0, 500.0, 400.0], "margin": 0.0, "unit": "nm", "source": "test", "split_version": "test", "n_spines": 0}))
    out = tmp_path / "study"
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/neuron_model/compare_vae_decoders.py"), "--run", f"a={run.root}",
                           "--run", f"b={run.root}", "--split", "train", "--n-spines", "6", "--resolution", "24", "--bbox", str(bbox),
                           "--n-boot", "50", "--out", str(out)], capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    summary = pd.read_csv(out / "summary.csv").set_index("kind")
    assert {"a", "b", "real", "real_mc"} <= set(summary.index)
    assert summary.loc["real_mc", "chamfer_l1_nm"] < 30   # exact SDF + MC 24^3 of a sphere is close to the sphere
    # the untrained tiny decoder extracts no surface: its rows are NaN (counted, not dropped); the real mesh against its own
    # exact-SDF + MC copy is paired and almost unbiased
    assert summary.loc["a", "valid_rate"] == 0 and np.isnan(summary.loc["a", "mean_abs_bias_sd"])
    assert np.isfinite(summary.loc["real", "mean_abs_bias_sd"])
    assert (out / "summary.md").exists()
