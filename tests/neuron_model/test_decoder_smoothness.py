"""Decoder smoothness diagnostic (scripts/neuron_model/diagnose_decoder_smoothness.py) on a tiny run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from test_vae_autodecoder import REPO_ROOT, _run, _trainable_run  # noqa: E402 (tests dir is on sys.path)


def test_summarise_pairs_each_condition_with_its_reference():
    import importlib.util

    spec = importlib.util.spec_from_file_location("diag", REPO_ROOT / "scripts/neuron_model/diagnose_decoder_smoothness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []
    for i in range(4):
        rows.append({"experiment": "perturb", "condition": "s=0", "idx": i, "is_reference": True, "single_component": True, "valid": True,
                     "ConvexHullRatio": 0.5 + 0.1 * i, "chamfer_to_reference_nm": 0.0})
        rows.append({"experiment": "perturb", "condition": "s=1", "idx": i, "is_reference": False, "single_component": i < 2, "valid": True,
                     "ConvexHullRatio": 0.3 + 0.1 * i, "chamfer_to_reference_nm": 30.0})
    s = mod.summarise(pd.DataFrame(rows)).set_index("condition")
    assert abs(s.loc["s=1", "paired_change_ConvexHullRatio"] + 0.2) < 1e-12
    assert s.loc["s=1", "single_component"] == 0.5 and s.loc["s=1", "median_chamfer_nm"] == 30.0
    assert "| perturb | s=1 |" in mod.to_markdown(s.reset_index())


def test_diagnostic_script_runs_on_a_tiny_run(fake_dataset, tmp_path):
    base = _trainable_run(fake_dataset, tmp_path)
    scripts = REPO_ROOT / "scripts/neuron_model"
    _run([scripts / "fit_vae_codes.py", "--run", base.root, "--n-train", "6", "--splits-to-fit", "train", "--steps", "3", "--batch-size", "4",
          "--anchor-weight", "1", "--out-name", "latent_prior_zopt_a1"])
    _run([scripts / "fit_vae_codes.py", "--run", base.root, "--n-val", "2", "--splits-to-fit", "val", "--steps", "3", "--anchor-weight", "1",
          "--out-name", "latent_prior_zopt_a1"])
    _run([scripts / "fit_latent_prior.py", "--run", base.root, "--variants", "1", "--out-name", "latent_prior_zopt_a1", "--n-train", "6", "--n-val", "2",
          "--gmm", "--flow-epochs", "2", "--flow-layers", "2", "--flow-hidden", "8"])
    out = _run([scripts / "diagnose_decoder_smoothness.py", "--run", base.root, "--codes-dir", "latent_prior_zopt_a1", "--n", "3", "--resolution", "16",
                "--scales", "0.5", "--lerp-t", "0.5", "--pull", "0.5"])
    folder = base.root / "smoothness_latent_prior_zopt_a1"
    summary = pd.read_csv(folder / "summary.csv")
    assert {"perturb", "lerp", "flow"} <= set(summary.experiment) and (folder / "summary.md").exists()
    assert json.loads((folder / "report.json").read_text())["n"] == 3
    assert "nearest-neighbour" in out
