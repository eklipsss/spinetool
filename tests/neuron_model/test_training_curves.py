"""Post-hoc training curves (evaluation/training_curves.py) on the synthetic fake dataset: tiny VAE and tiny MoGen,
a few steps with keep_every, then the evaluation of the periodic checkpoints."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.neuron_model.spine_generation.data.bbox import compute_frozen_bbox
from src.neuron_model.spine_generation.data.datasets import make_loader
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import build_group_split
from src.neuron_model.spine_generation.evaluation.training_curves import gap_trend, list_checkpoints, run_curves, summarize
from src.neuron_model.spine_generation.evaluation.training_curves_report import write_html, write_png
from src.neuron_model.spine_generation.experiment.config import load_config, REPO_ROOT
from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
from src.neuron_model.spine_generation.experiment.run import create_run
from src.neuron_model.spine_generation.experiment.seed import seed_everything
from src.neuron_model.spine_generation.models.spine_mogen import train as mogen_train
from src.neuron_model.spine_generation.models.spine_vae import train as vae_train
from src.neuron_model.spine_generation.models.spine_vae.model import SpineVAE, SpineVAEConfig
from src.neuron_model.spine_generation.models.spine_vae.pointnext_encoder import PointNeXtConfig
from src.neuron_model.spine_generation.models.spine_vae.siren_sdf_decoder import SirenConfig

MOGEN_TINY = {"point_dim": 16, "latent_dim": 32, "n_latents": 8, "n_blocks": 1, "n_subblocks": 1, "n_heads": 4, "k_nn": 4, "coord_dim": 3, "cond_dim": 6}
FLOW = {"time_schedule": "mogen_cosine", "inference_schedule": "mogen_cosine", "inference_steps": 3}


def _splits(fake_dataset, tmp_path):
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    splits = build_group_split(index.frame, seed=1, fractions={"train": 0.5, "val": 0.34, "test": 0.16})
    path = tmp_path / "splits.parquet"
    splits.to_parquet(path, index=False)
    return index, path


def _train_vae(fake_dataset, tmp_path, *, steps=3, keep_every=1, checkpoint_every=100):
    seed_everything(0)
    index, splits_path = _splits(fake_dataset, tmp_path)
    split_index = index.with_split(pd.read_parquet(splits_path))
    bbox = compute_frozen_bbox(split_index.subset(split_index.frame.loc[split_index.frame["split"] == "train", "spine_key"]), margin=50.0, n_points=2048)
    bbox_path = tmp_path / "bbox.json"
    bbox.save(bbox_path)
    data = {"manifests": [str(fake_dataset / "manifest.parquet")], "path_mode": "relative", "splits": str(splits_path), "bbox": str(bbox_path), "n_points": 2048,
            "coord_scale": 1e-3, "use_normals": True, "query_counts": {"surface": 8, "near": 8, "uniform": 8}, "far_queries": 4, "batch_size": 4}
    loss = {"sdf": 1.0, "far": 1.0, "eikonal": 0.1, "normal": 0.3, "beta": 1e-3, "kl_start_steps": 1, "kl_warmup_steps": 4}
    train = {"max_steps": steps, "log_every": 1, "validate_every": 100000, "sample_every": 100000, "checkpoint_every": checkpoint_every, "keep_every": keep_every, "ema_decay": 0.9}
    cfg = {"name": "vae-curves-tiny", "seed": 0, "data": data, "loss": loss, "train": train, "sampling": {}, "device": "cpu"}
    run = create_run(cfg, experiment_id="vae_curves", runs_root=tmp_path / "runs")
    model_cfg = SpineVAEConfig(latent_dim=4, encoder=PointNeXtConfig(input_points=2048, base_width=8, stage_widths=(8, 16), strides=(4, 4), blocks=(0, 1), k_neighbors=4, global_feature_dim=8, sampler="random"),
                               decoder=SirenConfig(hidden_dim=16, hidden_layers=2))
    (run.root / "model_config.json").write_text(json.dumps(model_cfg.to_dict()), encoding="utf-8")
    model = SpineVAE(model_cfg)
    weights = vae_train.loss_weights(loss)
    train_ds, val_ds = vae_train.sdf_datasets(split_index, data, far_bbox=(bbox.low, bbox.high))
    device = torch.device("cpu")
    run_training(run, LoopConfig.from_mapping(train), model, torch.optim.Adam(model.parameters(), lr=1e-3),
                 make_loader(train_ds, batch_size=4, shuffle=True, seed=0, drop_last=True), vae_train.make_loss_fn(weights, device), device=device)
    return run


def _train_mogen(fake_dataset, tmp_path, *, steps=3, keep_every=1):
    seed_everything(0)
    index, splits_path = _splits(fake_dataset, tmp_path)
    split_index = index.with_split(pd.read_parquet(splits_path))
    data = {"manifests": [str(fake_dataset / "manifest.parquet")], "path_mode": "relative", "splits": str(splits_path), "n_points": 2048, "coord_scale": 1e-2, "jitter_std": 1.0, "batch_size": 4}
    train = {"max_steps": steps, "log_every": 1, "validate_every": 2, "sample_every": 100000, "checkpoint_every": 100, "keep_every": keep_every, "ema_decay": 0.9}
    cfg = {"name": "mogen-curves-tiny", "seed": 0, "data": data, "flow": FLOW, "train": train, "device": "cpu", "model": {"init": "random", "architecture": MOGEN_TINY}}
    run = create_run(cfg, experiment_id="mogen_curves", runs_root=tmp_path / "runs")
    model = mogen_train.build_model(cfg["model"])
    (run.root / "model_config.json").write_text(json.dumps(model.cfg.to_dict()), encoding="utf-8")
    train_ds, val_ds = mogen_train.point_cloud_datasets(split_index, data)
    device = torch.device("cpu")
    run_training(run, LoopConfig.from_mapping(train), model, mogen_train.make_optimizer(model, {"name": "adamw", "lr": 1e-3}),
                 make_loader(train_ds, batch_size=4, shuffle=True, seed=0, drop_last=True), mogen_train.make_loss_fn(FLOW, device), device=device,
                 validate_fn=mogen_train.make_validate_fn(val_ds, FLOW, device, batch_size=4, seed=0))
    return run


def test_keep_every_writes_snapshots_independent_of_checkpoint_every(fake_dataset, tmp_path):
    run = _train_vae(fake_dataset, tmp_path, steps=3, keep_every=1, checkpoint_every=100)
    assert [s for s, _ in list_checkpoints(run.checkpoints, include_latest=False)] == [1, 2, 3]
    assert (run.checkpoints / "latest.pt").exists()


def test_keep_every_is_read_from_config_and_cli_override():
    for name in ("vae/reg.yaml", "mogen/pretrained_finetune.yaml"):
        cfg = load_config(REPO_ROOT / "configs" / "neuron-model" / name, ["train.keep_every=1000", "train.checkpoint_every=1000"])
        assert LoopConfig.from_mapping({**cfg["train"], "precision": cfg.get("precision")}).keep_every == 1000
        assert LoopConfig.from_mapping(load_config(REPO_ROOT / "configs" / "neuron-model" / name)["train"]).keep_every is None


def test_summarize_ratio_estimator_and_draw_spread():
    values = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    s = summarize(values, np.array([4, 4, 4]))
    assert s["mean"] == pytest.approx(2.0) and s["se"] == pytest.approx(1.0 / np.sqrt(3)) and s["draw_sd"] == pytest.approx(0.0)
    assert np.isnan(summarize(values[:1], np.array([4, 4, 4]))["draw_sd"])


def test_gap_trend_detects_growth_and_flat():
    steps = np.arange(1, 21) * 1000.0
    se = np.full(20, 0.01)
    assert gap_trend(steps, 0.01 * np.arange(20), se)["growing"]
    assert not gap_trend(steps, np.full(20, 0.1) + np.random.default_rng(0).normal(0, 0.005, 20), se)["growing"]


def test_vae_curves_one_row_per_checkpoint_same_estimator_and_bit_identical(fake_dataset, tmp_path):
    run = _train_vae(fake_dataset, tmp_path, steps=3, keep_every=1)
    kwargs = dict(weights_kinds=("ema", "raw"), n_spines=8, batch_size=2, seed=3, device="cpu", reuse=False, log=lambda *_: None)
    out = run_curves(run.root, **kwargs)
    curves = out["curves"]
    assert sorted(curves["step"].unique()) == [1, 2, 3] and len(curves) == 6  # 3 checkpoints x (ema, raw)
    ema = curves[curves["weights"] == "ema"].sort_values("step")
    assert len(ema) == 3
    for col in ("val_sdf", "val_kl", "val_far", "val_eikonal", "val_normal", "val_total", "val_total_final", "train_total", "train_total_final", "kl_weight", "kl_weight_final",
                "val_total_se", "gap_total_final", "gap_total_final_se"):
        assert col in curves, col
    # KL warm-up: weight at the step differs from the final weight, the final weight is the same on every row
    assert ema["kl_weight"].tolist()[0] == 0.0 and ema["kl_weight"].iloc[-1] > 0
    assert ema["kl_weight_final"].nunique() == 1 and ema["kl_weight_final"].iloc[0] == pytest.approx(5e-4)
    # last step: weights at the step == final weights -> both totals coincide
    assert ema["val_total"].iloc[-1] == pytest.approx(ema["val_total_final"].iloc[-1])
    # same function for train and validation: evaluating the train subset AS validation reproduces the train columns
    assert ema["n_val"].iloc[0] == ema["n_train"].iloc[0] >= 4  # n_spines capped by the split size, same size on both sides
    again = run_curves(run.root, **kwargs)["curves"]
    pd.testing.assert_frame_equal(curves.sort_values(["step", "weights"]).reset_index(drop=True), again.sort_values(["step", "weights"]).reset_index(drop=True), check_exact=True)
    assert (run.root / "training_curves" / "curves.csv").exists() and len(pd.read_csv(run.root / "training_curves" / "curves.csv")) == 6
    # reuse of the stored rows, report files
    reused = run_curves(run.root, **{**kwargs, "reuse": True})
    pd.testing.assert_frame_equal(curves.sort_values(["step", "weights"]).reset_index(drop=True), reused["curves"].sort_values(["step", "weights"]).reset_index(drop=True), check_exact=False, rtol=1e-12)
    write_png(out["out_dir"] / "curves.png", "vae", curves, out["logs"], out["result"]["summary"])
    write_html(out["out_dir"] / "curves.html", "vae", curves, out["logs"], out["result"]["summary"], out["cfg"], "vae_curves", out["result"]["settings"])
    assert (out["out_dir"] / "curves.png").stat().st_size > 1000 and "plot_total_final" in (out["out_dir"] / "curves.html").read_text(encoding="utf-8")


def test_vae_estimator_is_the_same_function_for_train_and_val(fake_dataset, tmp_path):
    """Both splits go through evaluate_checkpoint: with the validation spines as the 'train' list the two columns agree."""
    from src.neuron_model.spine_generation.evaluation import training_curves as tc

    run = _train_vae(fake_dataset, tmp_path, steps=1, keep_every=1)
    cfg = run.load_config()
    inputs = tc.build_inputs("vae", cfg, run.root, n_spines=8, batch_size=4, seed=3)
    inputs.train_batches = inputs.val_batches  # same inputs on both sides
    model = tc.build_model("vae", run.root)
    row = tc.evaluate_checkpoint("vae", model, inputs, cfg, step=1, final_step=3, device=torch.device("cpu"), n_draws=1, seed=3)
    for name in ("sdf", "kl", "far", "eikonal", "normal", "total", "total_final"):
        assert row[f"val_{name}"] == row[f"train_{name}"] and row[f"gap_{name}"] == 0.0


def test_mogen_curves_draws_null_condition_and_determinism(fake_dataset, tmp_path):
    run = _train_mogen(fake_dataset, tmp_path, steps=3, keep_every=1)
    kwargs = dict(weights_kinds=("ema",), n_spines=8, batch_size=2, seed=5, n_draws=2, device="cpu", reuse=False, log=lambda *_: None)
    out = run_curves(run.root, **kwargs)
    curves = out["curves"]
    assert sorted(curves["step"]) == [1, 2, 3] and len(curves) == 3
    for col in ("val_loss", "val_loss_se", "val_loss_draw_sd", "train_loss", "gap_loss", "gap_loss_se"):
        assert col in curves, col
    assert np.isfinite(curves["val_loss_draw_sd"]).all() and (curves["val_loss_se"] > 0).all()
    again = run_curves(run.root, **kwargs)["curves"]
    pd.testing.assert_frame_equal(curves.reset_index(drop=True), again.reset_index(drop=True), check_exact=True)
    # in-loop validation points are merged (validate_every=2 -> steps 2 and 3)
    assert out["logs"]["val_inloop"]["step"].tolist() == [2, 3]
    write_html(out["out_dir"] / "curves.html", "mogen", curves, out["logs"], out["result"]["summary"], out["cfg"], "mogen_curves", out["result"]["settings"])
    write_png(out["out_dir"] / "curves.png", "mogen", curves, out["logs"], out["result"]["summary"])
    assert out["result"]["summary"]["ema"]["best_step"] in (1, 2, 3)


def test_include_init_row_equals_direct_evaluation_and_resume_computes_only_step0(fake_dataset, tmp_path, monkeypatch):
    from src.neuron_model.spine_generation.evaluation import training_curves as tc

    for kind, make in (("vae", _train_vae), ("mogen", _train_mogen)):
        sub = tmp_path / kind
        sub.mkdir()
        run = make(fake_dataset, sub, steps=3, keep_every=1)
        kwargs = dict(weights_kinds=("ema",), n_spines=8, batch_size=2, seed=3, n_draws=2, device="cpu", log=lambda *_: None)
        base = run_curves(run.root, reuse=False, **kwargs)["curves"]
        calls = []
        original = tc.evaluate_checkpoint
        monkeypatch.setattr(tc, "evaluate_checkpoint", lambda *a, **k: (calls.append(k["step"]), original(*a, **k))[1])
        out = run_curves(run.root, reuse=True, include_init=True, **kwargs)
        assert calls == [0]  # the evaluated checkpoints are reused, only the init row is computed
        curves = out["curves"]
        assert curves["step"].tolist() == [0, 1, 2, 3] and bool(curves.loc[0, "init"]) and curves.loc[0, "checkpoint"] == "init"
        pd.testing.assert_frame_equal(curves.drop(columns="init").iloc[1:].reset_index(drop=True), base.reset_index(drop=True), check_dtype=False)
        # direct evaluation of a freshly built model with the same estimator
        cfg = run.load_config()
        inputs = tc.build_inputs(kind, cfg, run.root, n_spines=8, batch_size=2, seed=3)
        direct = tc.evaluate_checkpoint(kind, tc.build_initial_model(kind, cfg, run.root), inputs, cfg, step=0, final_step=3, device=torch.device("cpu"), n_draws=2, seed=3)
        main = "total_final" if kind == "vae" else "loss"
        assert curves.loc[0, f"val_{main}"] == direct[f"val_{main}"] and curves.loc[0, f"train_{main}"] == direct[f"train_{main}"]
        # repeated call identical; trend and best ignore / include step 0 correctly
        again = run_curves(run.root, reuse=False, include_init=True, **kwargs)["curves"]
        pd.testing.assert_frame_equal(curves, again, check_exact=True)
        summary = out["result"]["summary"]["ema"]
        assert summary["gap_trend"]["n_points"] <= 2  # steps 1..3, second half
        monkeypatch.setattr(tc, "evaluate_checkpoint", original)
