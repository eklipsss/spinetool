"""Joint decoder + code fine-tuning (models/spine_vae/autodecoder.py, finetune_vae_autodecoder.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.experiment.config import save_config
from src.neuron_model.spine_generation.models.spine_vae.latent_prior import load_vae_run

from test_vae_latent_prior import _tiny_vae_run  # noqa: E402 (tests dir is on sys.path)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _trainable_run(fake_dataset, tmp_path):
    """The tiny VAE run with the training settings the auto-decoder reads (queries, far field, regularisers)."""
    run = _tiny_vae_run(fake_dataset, tmp_path)
    cfg = run.load_config()
    bbox = tmp_path / "bbox.json"
    bbox.write_text(json.dumps({"low": [-400.0, -400.0, -400.0], "high": [400.0, 400.0, 400.0], "margin": 0.0, "unit": "nm",
                                "source": "test", "split_version": "test", "n_spines": 0}))
    cfg["data"].update({"query_counts": {"surface": 8, "near": 8, "uniform": 8}, "far_queries": 8, "bbox": str(bbox)})
    cfg["loss"] = {"sdf": 1.0, "far": 1.0, "eikonal": 0.1, "normal": 0.3, "sdf_clamp": 0.3}
    save_config(cfg, run.config_path)
    return run


def _run(args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return proc.stdout


def test_finetune_script_writes_a_vae_run_whose_codes_the_prior_tools_reuse(fake_dataset, tmp_path):
    base = _trainable_run(fake_dataset, tmp_path)
    scripts = REPO_ROOT / "scripts/neuron_model"
    _run([scripts / "fit_vae_codes.py", "--run", base.root, "--n-train", "4", "--splits-to-fit", "train", "--steps", "3",
          "--batch-size", "4", "--anchor-weight", "1", "--out-name", "latent_prior_zopt_a1"])
    base_codes = np.load(base.root / "latent_prior_zopt_a1" / "codes_train.npz")
    assert "mu_encoder" in base_codes.files
    out = _run([scripts / "finetune_vae_autodecoder.py", "--base-run", base.root, "--steps", "6", "--batch-size", "2",
                "--checkpoint-every", "3", "--warmup-steps", "2", "--code-noise", "1", "--lr", "1e-3", "--lr-codes", "1e-1",
                "--name", "ad-test"])
    new_root = Path(out.strip().splitlines()[-1].split()[1])
    assert new_root.parent == base.root.parent and new_root.name.endswith("ad-test")

    old, _, _, _, _ = load_vae_run(base.root)
    new, _, _, ckpt, _ = load_vae_run(new_root)
    so, sn = old.state_dict(), new.state_dict()
    assert all(torch.equal(so[k], sn[k]) for k in so if not k.startswith("decoder."))       # encoder untouched
    assert any(not torch.equal(so[k], sn[k]) for k in so if k.startswith("decoder."))       # decoder fine-tuned
    codes = np.load(new_root / "latent_prior_ad" / "codes_train.npz")
    assert str(codes["tag"]) == ckpt + "|train|4|1|[1]" and list(codes["keys"]) == list(base_codes["keys"])
    assert not np.allclose(codes["mu"], base_codes["mu"])
    rows = [json.loads(line) for line in (new_root / "logs" / "autodecoder.jsonl").read_text().splitlines()]
    assert rows and {"sdf_clean_nm", "sdf_sigma_perturbed_nm", "anchor_dist2_median", "eikonal_loss", "far_loss"} <= set(rows[-1])
    assert (new_root / "checkpoints" / "step_3.pt").exists()

    # validation codes with the fine-tuned decoder frozen, then the prior script reuses BOTH files (no re-encoding)
    _run([scripts / "fit_vae_codes.py", "--run", new_root, "--n-val", "2", "--splits-to-fit", "val", "--steps", "3",
          "--anchor-weight", "1", "--out-name", "latent_prior_ad"])
    out = _run([scripts / "fit_latent_prior.py", "--run", new_root, "--variants", "1", "--out-name", "latent_prior_ad",
                "--n-train", "4", "--n-val", "2", "--gmm", "--flow-epochs", "2", "--flow-layers", "2", "--flow-hidden", "8"])
    assert out.count("[prior] reuse") == 2


def test_strong_anchor_pulls_the_codes_to_the_encoder_mean(fake_dataset, tmp_path):
    from src.neuron_model.spine_generation.models.spine_vae.autodecoder import finetune_autodecoder, query_dataset
    from src.neuron_model.spine_generation.models.spine_vae.latent_prior import load_split_index
    from src.neuron_model.spine_generation.models.spine_vae.train import loss_weights

    run = _trainable_run(fake_dataset, tmp_path)
    model, _, cfg, _, device = load_vae_run(run.root)
    index = load_split_index(cfg, "train", 4, 1)
    cfg["data"]["far_queries"] = 0
    ds = query_dataset(index, cfg["data"])
    keys = index.keys
    mu = np.zeros((len(keys), model.cfg.latent_dim))
    c0 = np.ones_like(mu)
    common = dict(weights=loss_weights({"sdf": 1.0}), coord_scale=1e-3, steps=40, batch_size=2, lr=1e-4, lr_codes=5e-2,
                  warmup_steps=1, log_every=10, device=device)
    free, hist = finetune_autodecoder(model, c0, mu, np.ones_like(mu), keys, ds, anchor_weight_nm=0.0, **common)
    pulled, _ = finetune_autodecoder(model, c0, mu, np.ones_like(mu), keys, ds, anchor_weight_nm=1e6, **common)
    assert np.abs(pulled).mean() < np.abs(free).mean() - 0.3   # Adam moves a code by <= ~lr per update
    assert len(hist) == 4 and all(np.isfinite(h["sdf_clean_nm"]) for h in hist)
    assert not any(p.requires_grad for p in model.parameters())                        # frozen again afterwards
    with pytest.raises(ValueError, match="no code"):
        finetune_autodecoder(model, c0[:1], mu[:1], np.ones_like(mu[:1]), keys[:1], ds, anchor_weight_nm=0.0, **common)
