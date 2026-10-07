"""Evaluate generated spine sets against the real test split (plan stage 7) and compare models.

Every model is judged by the same protocol (configs/neuron-model/evaluation/default.yaml):
real test morphometrics + test surface clouds, train-fitted standardiser / MMD bandwidth,
Chamfer-DCR memorization against the FULL train split (local cloud cache).

    python scripts/neuron_model/evaluate_generators.py --config configs/neuron-model/evaluation/default.yaml \
        --model mogen_finetuned=runs/training/neuron-model/<id>/meshes \
        --model vae=runs/training/neuron-model/<id>/meshes \
        "data.manifests=['O:/Datasets/Minnie65/preprocessed/manifest.parquet']"

``<model_root>`` contains ``seed_<s>/sample_<i>/generated_mesh_{raw,postprocessed}.off``
(written by generate_mogen.py and the VAE generator). Output: one run directory under
``runs/analysis/neuron-model`` with report.json / summary.csv / summary.md per mesh kind.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.evaluation.attachment import finder_from_config
from src.neuron_model.spine_generation.evaluation.inputs import (
    build_cloud_cache,
    discover_seed_dirs,
    load_generated_set,
    load_real_clouds,
    open_cloud_cache,
)
from src.neuron_model.spine_generation.evaluation.morphometrics import finder_bias, generated_morphometrics, real_morphometrics, sample_clouds
from src.neuron_model.spine_generation.evaluation.report import EvaluationConfig, build_real_reference, evaluate_model, write_report
from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.run import create_run
from src.neuron_model.spine_generation.experiment.training import get_device


def _parse_model(value: str):
    name, sep, root = value.partition("=")
    if not sep or not name or not root:
        raise argparse.ArgumentTypeError("--model expects NAME=PATH")
    return name, Path(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/neuron-model/evaluation/default.yaml")
    parser.add_argument("--model", type=_parse_model, action="append", required=True, help="NAME=<model_root> (repeatable)")
    parser.add_argument("--rebuild-cloud-cache", action="store_true", help="re-read all train clouds into the local cache")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    data_cfg, real_cfg, proto = cfg["data"], cfg["real"], cfg["protocol"]
    if not data_cfg.get("manifests"):
        parser.error("set data.manifests=[...]")
    n_points = int(real_cfg["surface_points"])
    eval_cfg = EvaluationConfig.from_mapping({**cfg["evaluation"], "device": str(get_device(cfg["evaluation"].get("device", "auto")))})

    # --- real reference (shared by all models)
    index = load_spine_index(
        [resolve_path(p) for p in data_cfg["manifests"]],
        path_mode=data_cfg.get("path_mode", "relative"),
        check_files=(f"pointcloud_{n_points}_path", "morphometrics_path"),
    ).with_split(load_splits(resolve_path(data_cfg["splits"])))
    train_index = index.subset(index.frame.loc[index.frame["split"] == "train", "spine_key"])
    test_index = index.subset(index.frame.loc[index.frame["split"] == real_cfg["test_split"], "spine_key"])
    print(f"real: {len(train_index)} train / {len(test_index)} {real_cfg['test_split']} spines")

    train_clouds = None
    if real_cfg.get("chamfer_memorization"):
        cache = resolve_path(real_cfg["train_cloud_cache"])
        if args.rebuild_cloud_cache or not cache.exists():
            print(f"building train cloud cache {cache} (one pass over {len(train_index)} files)")
            build_cloud_cache(train_index, n_points, cache)
        train_clouds = open_cloud_cache(cache, train_index)
    real = build_real_reference(
        real_morphometrics(train_index),
        real_morphometrics(test_index),
        test_clouds=load_real_clouds(test_index, n_points),
        train_clouds=train_clouds,
        seed=eval_cfg.seed,
    )

    run = create_run(cfg, name=cfg.get("name", "generator-evaluation"), runs_root=resolve_path(cfg["runs_root"]),
                     extra_info={"models": {name: str(root) for name, root in args.model}})
    morph_cfg = cfg.get("morphometrics", {})
    chords = bool(morph_cfg.get("compute_chord_distribution", True))
    finder = finder_from_config(morph_cfg.get("attachment_finder"))
    if finder is not None and morph_cfg.get("finder_bias_spines", 0):
        # the same finder on real test meshes vs their stage-1 attachment: how much of a
        # junction-metric gap between real and generated could be the finder's own error
        bias = finder_bias(test_index, finder, max_spines=int(morph_cfg["finder_bias_spines"]), seed=eval_cfg.seed)
        write_json(run.root / "finder_bias.json", bias)
        print(f"finder bias on {bias['n_spines']} real test spines: found {bias['found_rate']:.1%}, methods {bias['methods']}")
    expected_seeds = set(proto.get("generation_seeds", []))
    for mesh_kind in proto.get("mesh_kinds", ["raw"]):
        models = {}
        for name, root in args.model:
            seed_dirs = discover_seed_dirs(root)
            missing = expected_seeds - set(seed_dirs)
            if missing:
                print(f"WARNING {name}: missing generation seeds {sorted(missing)} (protocol: {sorted(expected_seeds)})")
            per_seed = {}
            for seed, seed_dir in seed_dirs.items():
                meshes, reports, ids = load_generated_set(seed_dir, mesh_kind)
                per_seed[seed] = {
                    "gen_morph": generated_morphometrics(meshes, attachment_finder=finder, compute_chords=chords, sample_ids=ids),
                    "gen_clouds": sample_clouds(meshes, n_points, seed=seed),
                    "validity_raw" if mesh_kind == "raw" else "validity_postprocessed": reports,
                }
                print(f"{mesh_kind} {name} seed {seed}: {sum(m is not None for m in meshes)}/{len(meshes)} meshes")
            models[name] = evaluate_model(real, per_seed, eval_cfg)
        out = write_report(models, run.root / mesh_kind)
        print(f"{mesh_kind}: report -> {out}")


if __name__ == "__main__":
    main()
