"""Calibrate the reconstruction backends (Screened Poisson, Marching Cubes) on real validation spines.

Stage 3 of docs/neuron-model/s-module-implementation-plan.md - run on the
Windows workstation BEFORE training any generator:

    python scripts/neuron_model/calibrate_reconstruction.py \
        --config configs/neuron-model/reconstruction/calibration.yaml \
        "data.manifests=['O:/Datasets/Minnie65/preprocessed/manifest.parquet']" workers=16

Writes ``runs/analysis/neuron-model/<id>/``: per-spine results (parquet),
per-candidate summaries (csv) and ``selection.json`` with the proposed frozen
parameters. Copy the selected values into ``configs/neuron-model/reconstruction/
{poisson,marching_cubes}.yaml`` and set ``frozen: true`` there.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.data.bbox import FrozenBBox
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.run import create_run
from src.neuron_model.spine_generation.reconstruction.calibration import (
    calibrate_marching_cubes_spine,
    calibrate_poisson_spine,
    poisson_param_grid,
    run_calibration,
    select_best,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    data = cfg["data"]
    if not data.get("manifests"):
        parser.error("set data.manifests=[...]")

    # only n_spines end up read - skip load_spine_index's default per-file exists() check over
    # the whole manifest (same NAS cost as build_splits.py); a missing file among the sampled
    # ones just surfaces as an "error" row from run_calibration, not a crash.
    index = load_spine_index([resolve_path(p) for p in data["manifests"]], path_mode=data.get("path_mode", "relative"), check_files=())
    index = index.with_split(load_splits(resolve_path(data["splits"])), cfg["calibration_split"])
    frame = index.frame.sample(n=min(int(cfg["n_spines"]), len(index)), random_state=int(cfg["seed"])).sort_values("spine_key")
    spines = frame.to_dict(orient="records")

    cfg.setdefault("data", {})["dataset_version"] = index.dataset_version
    run = create_run(cfg, name=cfg.get("name", "reconstruction-calibration"), runs_root=resolve_path(cfg["runs_root"]))
    print(f"calibrating on {len(spines)} {cfg['calibration_split']} spines -> {run.root}")
    common = {
        "compare_kwargs": dict(cfg["compare"]),
        "postprocess_kwargs": dict(cfg["postprocess"]),
        "check_self_intersections": bool(cfg["check_self_intersections"]),
    }
    selection = {}

    if cfg["poisson"]["enabled"]:
        pcfg = cfg["poisson"]
        params = poisson_param_grid(pcfg["grid"], fixed=pcfg.get("fixed"))
        worker = functools.partial(
            calibrate_poisson_spine,
            params_list=params,
            k_normals=pcfg["k_normals"],
            n_points=int(pcfg["n_points"]),
            true_normals_control=bool(pcfg["true_normals_control"]),
            **common,
        )
        results = run_calibration(spines, worker, workers=int(cfg["workers"]), label="poisson")
        results.to_parquet(run.metrics / "poisson_results.parquet", index=False)
        ok = results[results.get("error").isna()] if "error" in results else results
        summary = summarize(ok, ["normals_source", "k_normal", "depth", "point_weight", "samples_per_node", "scale"])
        summary.to_csv(run.metrics / "poisson_summary.csv", index=False)
        estimated = summary[summary["normals_source"] == "estimated"]
        selection["poisson"] = select_best(
            estimated,
            error_column=cfg["selection"]["error_column"],
            validity_column="raw_is_valid_mean",
            min_validity=float(cfg["selection"]["min_validity"]),
        )
        print("poisson selection:", json.dumps(selection["poisson"], indent=2, default=str))

    if cfg["marching_cubes"]["enabled"]:
        mcfg = cfg["marching_cubes"]
        bbox = FrozenBBox.load(resolve_path(data["bbox"]))
        worker = functools.partial(
            calibrate_marching_cubes_spine,
            bbox_low=bbox.low,
            bbox_high=bbox.high,
            resolutions=mcfg["resolutions"],
            chunk_size=int(mcfg["chunk_size"]),
            **common,
        )
        results = run_calibration(spines, worker, workers=int(cfg["workers"]), label="marching_cubes")
        results.to_parquet(run.metrics / "marching_cubes_results.parquet", index=False)
        ok = results[results.get("error").isna()] if "error" in results else results
        summary = summarize(ok, ["resolution"])
        summary.to_csv(run.metrics / "marching_cubes_summary.csv", index=False)
        selection["marching_cubes"] = {"per_resolution": summary.to_dict(orient="records")}

    write_json(run.metrics / "selection.json", selection)
    print(f"done -> {run.metrics / 'selection.json'}")


if __name__ == "__main__":
    main()
