"""Conditional spine generation: condition encoder, derived spine conditions, guidance, conditional training,
initialisation from a run, the neuron-level test of condition effects (s-module-conditional-design.md)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.neuron_model.spine_generation.conditioning import DEFAULT_BLOCKS, ConditionEncoder, enabled_from_report, spine_conditions

REPO = Path(__file__).resolve().parents[2]
TINY_C = {"point_dim": 16, "latent_dim": 32, "n_latents": 8, "n_blocks": 1, "n_subblocks": 1, "n_heads": 4, "k_nn": 4, "coord_dim": 3, "cond_dim": 22}
DATA = {"n_points": 2048, "coord_scale": 1e-2, "jitter_std": 1.0, "batch_size": 4}
FLOW = {"time_schedule": "mogen_cosine", "inference_schedule": "mogen_cosine", "inference_steps": 3}


def test_encoder_layout_encode_decode_parse_and_persistence(tmp_path):
    enc = ConditionEncoder()
    assert enc.width == 15 and enc.cond_dim == 22  # + radius context (2) at stage 2 = 17
    v = enc.encode({"species": "mouse", "compartment": "basal", "cell_type_binary": None, "health": "nonsense"})
    assert v.shape == (22,) and v.sum() == 2 and np.all(v[15:] == 0)
    assert enc.decode(v) == {"species": "mouse", "source": None, "microscopy": None, "health": None, "treatment": None,
                             "cell_type_binary": None, "compartment": "basal"}
    assert enc.parse("compartment=apical, cell_type_binary=inhibitory") == {"compartment": "apical", "cell_type_binary": "inhibitory"}
    for bad in ("compartment=dendrite", "colour=red", "compartment"):
        with pytest.raises(ValueError):
            enc.parse(bad)
    part = ConditionEncoder(enabled=["compartment"])
    assert part.encode({"species": "mouse", "compartment": "basal"}).sum() == 1  # disabled blocks stay zero
    with pytest.raises(ValueError, match="disabled"):
        part.parse("species=mouse")
    frame = pd.DataFrame({"species": ["mouse", None], "compartment": ["apical", "axon_like"]})
    assert np.array_equal(enc.encode_frame(frame), np.stack([enc.encode(r) for r in frame.to_dict("records")]))
    enc.save(tmp_path / "enc.json")
    assert ConditionEncoder.load(tmp_path / "enc.json").to_dict() == enc.to_dict()
    with pytest.raises(ValueError, match="cond_dim"):
        ConditionEncoder(cond_dim=6)


def test_block_dropout_rates():
    enc = ConditionEncoder()
    full = torch.from_numpy(np.tile(enc.encode({n: v[0] for n, v in DEFAULT_BLOCKS}), (20000, 1)))
    out = enc.drop_blocks(full, p_block=0.15, p_all=0.10, generator=torch.Generator().manual_seed(0))
    zero_all = (out.sum(1) == 0).float().mean().item()
    sl = enc.slices()["compartment"]
    zero_block = (out[:, sl].sum(1) == 0).float().mean().item()
    assert zero_all == pytest.approx(0.10 + 0.9 * 0.15**7, abs=0.01)
    assert zero_block == pytest.approx(1 - 0.9 * 0.85, abs=0.015)
    assert torch.equal(enc.drop_blocks(full, p_block=0.0, p_all=0.0), full)


def test_spine_conditions_use_the_fixed_metadata_lookup():
    if not (REPO / "datasets/minnie65/minnie65_limb_compartments_length_rule.parquet").exists():
        pytest.skip("local metadata tables not present")
    frame = pd.DataFrame({
        "dataset": ["minnie65", "minnie65", "minnie65", "labid", "minnie65"],
        "neuron_id": ["864691134886335738"] * 3 + ["Ab+9009_24_2", "1"],
        "limb_id": ["000", "001", "002", "000", "000"],
        "compartment": [None] * 5,  # stale manifest column: ignored
    })
    c = spine_conditions(frame, REPO / "datasets")
    assert list(c["compartment"]) == ["apical", "axon_like", "basal", None, None]
    assert list(c["cell_type_binary"][:3]) == ["excitatory"] * 3 and c["cell_type_binary"][3] is None
    assert list(c["source"]) == ["minnie65", "minnie65", "minnie65", "labid", "minnie65"]
    assert list(c["microscopy"]) == ["electron"] * 3 + ["confocal", "electron"]
    assert (c.loc[3, "health"], c.loc[3, "treatment"]) == ("disease", "9009") and c.loc[0, "health"] == "healthy"


def test_guidance_interpolates_between_null_and_conditional_velocity():
    from src.neuron_model.spine_generation.models.spine_mogen.flow import sample_midpoint

    def model(x, t, cond=None):  # velocity = sum(cond) everywhere: zero for the null condition
        s = torch.zeros(x.shape[0]) if cond is None else cond.sum(-1)
        return torch.ones_like(x) * s[:, None, None]

    x0 = torch.zeros(2, 5, 3)
    cond = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    plain = sample_midpoint(model, x0, n_steps=4, schedule="linear", cond=cond)
    assert torch.allclose(sample_midpoint(model, x0, n_steps=4, schedule="linear", cond=cond, guidance=1.0), plain)
    assert torch.allclose(sample_midpoint(model, x0, n_steps=4, schedule="linear", cond=cond, guidance=2.0), 2 * plain)
    assert torch.allclose(sample_midpoint(model, x0, n_steps=4, schedule="linear", cond=cond, guidance=0.0), torch.zeros_like(plain))


def test_conditional_training_step_validation_generation_and_init_from_run(fake_dataset, tmp_path):
    from src.neuron_model.spine_generation.data.datasets import make_loader
    from src.neuron_model.spine_generation.data.index import load_spine_index
    from src.neuron_model.spine_generation.data.splits import build_group_split
    from src.neuron_model.spine_generation.experiment.loop import LoopConfig, run_training
    from src.neuron_model.spine_generation.experiment.run import create_run
    from src.neuron_model.spine_generation.experiment.seed import seed_everything
    from src.neuron_model.spine_generation.models.spine_mogen.infer import generate_point_clouds
    from src.neuron_model.spine_generation.models.spine_mogen.train import (build_model, make_loss_fn, make_optimizer, make_validate_fn,
                                                                            point_cloud_datasets)

    seed_everything(0)
    index = load_spine_index([fake_dataset / "manifest.parquet"])
    index = index.with_split(build_group_split(index.frame, seed=1))
    enc = ConditionEncoder(enabled=["cell_type_binary", "compartment"])
    vectors = enc.encode_frame(index.frame[["cell_type_binary", "compartment"]])
    keys = list(index.frame["spine_key"])
    cond_by_key = {k: v for k, v in zip(keys[1:], vectors[1:])}  # the first spine has no condition: dropped
    train_ds, val_ds = point_cloud_datasets(index, DATA, cond_by_key)
    assert len(train_ds) + len(val_ds or []) <= len(keys) - 1
    item = train_ds[0]
    assert item["cond"].shape == (22,) and float(item["cond"].sum()) == 2.0

    model = build_model({"init": "random", "architecture": TINY_C})
    device = torch.device("cpu")
    loss_fn = make_loss_fn(FLOW, device, encoder=enc, condition_cfg={"p_block": 0.15, "p_all": 0.1})
    batch = next(iter(make_loader(train_ds, batch_size=4, shuffle=False, seed=0)))
    loss, logs = loss_fn(model, batch, 0)
    assert np.isfinite(logs["loss"])
    dense0_before = model.Dense_0.weight.detach().clone()

    run = create_run({"seed": 0, "data": DATA}, experiment_id="mogen_cond", runs_root=tmp_path / "runs")
    validate = make_validate_fn(val_ds, FLOW, device, batch_size=4, seed=0)
    loop = LoopConfig(max_steps=4, accumulation_steps=1, clip_norm=0.1, ema_decay=0.9, log_every=1, validate_every=4, sample_every=100, checkpoint_every=4)
    result = run_training(run, loop, model, make_optimizer(model, {"name": "adamw", "lr": 1e-3}),
                          make_loader(train_ds, batch_size=4, shuffle=True, seed=0, drop_last=True), loss_fn, device=device, validate_fn=validate)
    assert {"val_loss", "val_loss_null_cond"} <= set(result["metrics"])
    # the condition input is trained (at random init the zero-initialised output projections pass no gradient to it on
    # the very first step, hence several steps)
    assert not torch.allclose(model.Dense_0.weight.detach(), dense0_before)

    # a new model initialised from the run's EMA weights generates the same clouds as the EMA model
    (run.root / "model_config.json").write_text(__import__("json").dumps(model.cfg.to_dict()), encoding="utf-8")
    again = build_model({"init": "run", "init_run": str(run.root), "init_checkpoint": "latest"}).eval()
    common = dict(n_samples=2, n_points=64, coord_scale=1e-2, n_steps=2, schedule="mogen_cosine", seed=3, device=device,
                  cond=enc.encode({"compartment": "apical"}))
    a = generate_point_clouds(again, guidance=1.0, **common)
    b = generate_point_clouds(again, guidance=3.0, **common)
    assert a.shape == (2, 64, 3) and np.isfinite(b).all() and not np.allclose(a, b)


def test_neuron_level_permutation_separates_a_real_effect_from_neuron_style(tmp_path):
    """Spines of 40 neurons; every neuron has its own style (offset). 'real' shifts the morphology of its neurons,
    'style' is assigned to 3 neurons only and adds nothing: a spine-level test would call both significant."""
    from src.neuron_model.spine_generation.evaluation.condition_effects import condition_effects

    rng = np.random.default_rng(0)
    rows, feats = [], []
    for n in range(40):
        style = rng.normal(0, 1.0, 4)
        real = "a" if n % 2 else "b"
        rare = "x" if n < 3 else "y"
        for _ in range(60):
            feats.append(style + rng.normal(0, 1.0, 4) + (1.5 if real == "a" else 0.0))
            rows.append({"neuron": f"n{n}", "real": real, "rare": rare})
    meta = pd.DataFrame(rows)
    features = pd.DataFrame(np.array(feats), columns=["f1", "f2", "f3", "f4"])
    report = condition_effects(features, meta[["real", "rare"]], meta["neuron"], [("real", ("a", "b")), ("rare", ("x", "y")), ("none", ("u",))],
                               min_neurons=3, n_permutations=100, log=None)
    assert report["blocks"]["real"]["enabled"] and report["blocks"]["real"]["pairs"][0]["p_value"] < 0.02
    assert not report["blocks"]["rare"]["enabled"]
    assert report["blocks"]["none"]["reason"] == "no column"
    path = tmp_path / "effects.json"
    path.write_text(__import__("json").dumps(report, default=float), encoding="utf-8")
    assert enabled_from_report(path) == ["real"] and enabled_from_report(path, mode="all") is None
