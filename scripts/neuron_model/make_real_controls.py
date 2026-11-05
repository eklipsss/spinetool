"""Real TEST spines through the generators' surface pipelines (evaluation/real_controls.py): the detectability floor.

    python scripts/neuron_model/make_real_controls.py \
        --manifest cache\\manifest_minnie65_meta.parquet --path-mode absolute \
        --splits data\\processed\\neuron-model\\splits\\splits_dev_100n_untouched_test.parquet \
        --vae-run runs\\training\\neuron-model\\2026-10-19_003408_vae-dev-reg0-cos \
        --controls asis poisson mc --n-samples 1000 --seeds 1 2 3 --workers 16

Writes ``runs/analysis/neuron-model/<date>_real-controls/<control>/seed_<s>/sample_<i>/...`` (generator layout) and
``controls.csv`` (which real spine is behind which sample). Then evaluate them exactly like the generators, on the same split:

    python scripts/neuron_model/evaluate_generators.py --model real_asis=<out>/asis --model real_poisson=<out>/poisson
        --model real_mc=<out>/mc --model vae_flow=<...> --model mogen_a=<...> ... data.splits=<same split> data.drop_unassigned=true

Every seed takes a DIFFERENT random subset of the test spines (independent sets, as the generation seeds). The resolution of
``mc`` is that of the VAE generation being compared (``--resolution``, default 128 as in the report). pymeshlab (Poisson) is
not imported together with torch here, so the Windows OpenMP clash does not apply.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.neuron_model.spine_generation.data.bbox import FrozenBBox
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.evaluation.real_controls import CONTROLS, assign_samples, run_control
from src.neuron_model.spine_generation.experiment.artifacts import write_json
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--path-mode", default="relative", choices=["relative", "absolute"])
    ap.add_argument("--splits", type=Path, required=True, help="the split the evaluation will use (its test part)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--controls", nargs="+", default=list(CONTROLS), choices=CONTROLS)
    ap.add_argument("--n-samples", type=int, default=1000, help="spines per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--vae-run", type=Path, help="VAE run: frozen bbox (data.bbox) and marching-cubes config of the mc control")
    ap.add_argument("--bbox", type=Path, help="frozen bbox json (instead of --vae-run)")
    ap.add_argument("--resolution", type=int, default=128, help="mc grid resolution = that of the compared VAE generation")
    ap.add_argument("--poisson-config", type=Path, default=REPO_ROOT / "configs/neuron-model/reconstruction/poisson.yaml")
    ap.add_argument("--mc-config", type=Path, default=REPO_ROOT / "configs/neuron-model/reconstruction/marching_cubes.yaml")
    ap.add_argument("--check-self-intersections", action="store_true", help="off by default (CGAL subprocess per mesh), as in the generation")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, help="default runs/analysis/neuron-model/<date>_real-controls")
    args = ap.parse_args()

    index = load_spine_index(args.manifest, path_mode=args.path_mode, check_files=())
    index = index.with_split(load_splits(args.splits), args.split, drop_unassigned=True)
    frame = index.frame
    print(f"{len(frame)} {args.split} spines of {frame['group_id'].nunique()} neurons in {args.splits.name}", flush=True)
    picks = assign_samples(frame["spine_key"].tolist(), n_per_seed=args.n_samples, seeds=args.seeds, base_seed=args.base_seed)
    by_key = frame.set_index("spine_key", drop=False)

    kwargs = {}
    check = bool(args.check_self_intersections)
    if "asis" in args.controls:
        kwargs["asis"] = {"postprocess_cfg": None, "check_self_intersections": check}
    if "poisson" in args.controls:
        pc = load_config(args.poisson_config)
        kwargs["poisson"] = {"k_normal": int(pc["normals"]["k_normal"]), "poisson": dict(pc["poisson"]), "postprocess_cfg": dict(pc["postprocess"]),
                             "check_self_intersections": check}
    if "mc" in args.controls:
        mc_path = args.mc_config
        if args.bbox:
            bbox = FrozenBBox.load(args.bbox)
        elif args.vae_run:
            vcfg = open_run(args.vae_run).load_config()
            bbox = FrozenBBox.load(resolve_path(vcfg["data"]["bbox"]))
            mc_path = resolve_path(vcfg["reconstruction"]["marching_cubes_config"])
        else:
            ap.error("the mc control needs --vae-run or --bbox")
        mc = load_config(mc_path)
        kwargs["mc"] = {"bbox_low": list(bbox.low), "bbox_high": list(bbox.high), "resolution": int(args.resolution),
                        "postprocess_cfg": dict(mc["postprocess"]), "chunk_size": int(mc.get("chunk_size", 262_144)), "check_self_intersections": check}
    # the as-is control keeps the mesh as it is; postprocessing it like the others (small components) is the fair choice
    if "asis" in kwargs and "poisson" in kwargs:
        kwargs["asis"]["postprocess_cfg"] = kwargs["poisson"]["postprocess_cfg"]

    out = args.out or resolve_path(Path("runs/analysis/neuron-model") / f"{pd.Timestamp.now():%Y-%m-%d_%H%M%S}_real-controls")
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for control in args.controls:
        for seed, keys in picks.items():
            for i, key in enumerate(keys):
                row = by_key.loc[key]
                spine = {c: (str(row[c]) if c in row and pd.notna(row[c]) else None) for c in ("spine_key", "local_sealed_mesh_path", "pointcloud_8192_path")}
                jobs.append({"control": control, "seed": seed, "spine": spine, "kwargs": kwargs[control],
                             "sample_dir": str(out / control / f"seed_{seed}" / f"sample_{i:05d}")})
    write_json(out / "controls_config.json", {"splits": str(args.splits), "split": args.split, "n_samples": args.n_samples, "seeds": args.seeds,
                                             "base_seed": args.base_seed, "controls": {c: kwargs[c] for c in args.controls},
                                             "n_test_spines": int(len(frame)), "n_test_neurons": int(frame["group_id"].nunique())})
    print(f"{len(jobs)} jobs ({', '.join(args.controls)} x seeds {args.seeds} x {args.n_samples}) -> {out}", flush=True)

    rows = []
    step = max(1, len(jobs) // 20)
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            for k, row in enumerate(pool.map(run_control, jobs, chunksize=4), 1):
                rows.append(row)
                if k % step == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)}", flush=True)
    else:
        for k, job in enumerate(jobs, 1):
            rows.append(run_control(job))
            if k % step == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "controls.csv", index=False)
    for control, g in table.groupby("control"):
        print(f"{control:8s}: errors {g['error'].notna().sum()}/{len(g)}, valid raw {g.get('raw_valid', pd.Series(dtype=float)).mean():.2f}, "
              f"valid postprocessed {g.get('post_valid', pd.Series(dtype=float)).mean():.2f}, {g['seconds'].median():.1f} s/spine")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
