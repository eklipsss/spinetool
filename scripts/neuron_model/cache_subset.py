"""Copy a random subset of the training data to a LOCAL disk, in a compact form.

Why: with the dataset on the NAS the training is bound by the NAS read throughput (~4 MB/s
measured: 0.5-0.7 steps/s, GPU idle ~95 % of the time). Training reads every spine every
epoch, so a subset that fits on the local NVMe turns it into a local-disk job.

The result is a mini dataset with the SAME layout the loaders expect
(``<out>/<neuron>/limb_<L>/branch_<B>/spines/spine_<S>/``) + its own ``manifest.parquet``, so
nothing changes in the training code - only ``data.manifests`` points at it:

    python scripts/neuron_model/cache_subset.py \
        --manifest O:/Datasets/Minnie65/preprocessed/manifest.parquet \
        --splits data/processed/neuron-model/splits/splits_dev_100n.parquet \
        --out D:/cache/minnie65_dev --n-train 15000 --n-val 1500 --threads 32

    python scripts/neuron_model/train_vae.py --config ... \
        "data.manifests=['D:/cache/minnie65_dev/manifest.parquet']" "data.check_files=[]" \
        data.drop_unassigned=true data.splits=<same splits> data.bbox=<same bbox> ...

Only what training needs is copied: the ``--n-points`` point clouds (points + normals) of the
first ``--variants`` resamplings, and the SDF arrays ``query_points / sdf / sample_type /
surface_normals``. Meshes, quality files etc. are not copied, so the cache is for TRAINING only
(evaluation against real meshes still reads the NAS manifest).

Safe to interrupt (Ctrl+C) and to re-run (finished spines are skipped); the manifest is rewritten
every ``--manifest-every`` spines and at the end, and lists only spines whose files are complete.
Validation spines are copied first, then training spines in random order.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.neuron_model.pointcloud_io import PACKED_NAME, load_pointcloud, write_pointclouds
from src.neuron_model.spine_generation.data.index import load_spine_index, spine_relative_dir
from src.neuron_model.spine_generation.data.splits import load_splits

SDF_KEEP = ("query_points", "sdf", "sample_type", "surface_normals")
DROP_COLUMNS = ("spine_key", "manifest_path", "split", "group_id")


def _cache_one(row: dict, out: Path, n_points: int, variants, pc_column: str) -> int:
    """Copy one spine; returns the number of bytes written (0 if it was already complete)."""
    spine_dir = out / spine_relative_dir(row["neuron_id"], row["limb_id"], row["branch_id"], row["spine_id"])
    pc_out, sdf_out = spine_dir / PACKED_NAME, spine_dir / "sdf_samples.npz"
    if pc_out.exists() and sdf_out.exists():
        return 0
    spine_dir.mkdir(parents=True, exist_ok=True)
    clouds = {}
    for variant in variants:
        data = load_pointcloud(row[pc_column], n_points, variant)
        clouds[(n_points, variant)] = {"points": data["points"], "normals": data["normals"]}
    with np.load(row["sdf_samples_path"], allow_pickle=False) as sdf:
        arrays = {key: sdf[key] for key in SDF_KEEP}
    tmp_sdf = sdf_out.with_name("sdf_samples.tmp.npz")
    np.savez(tmp_sdf, **arrays)
    os.replace(tmp_sdf, sdf_out)
    tmp_pc = pc_out.with_name("pointclouds.tmp.npz")
    write_pointclouds(tmp_pc, clouds)  # the pack is the completeness marker: written last
    os.replace(tmp_pc, pc_out)
    return int(sdf_out.stat().st_size + pc_out.stat().st_size)


def _write_manifest(frame: pd.DataFrame, done_keys, out: Path, n_points: int) -> None:
    kept = frame[frame["spine_key"].isin(done_keys)].drop(columns=list(DROP_COLUMNS), errors="ignore").copy()
    # Re-rooting only uses the file NAME of each stored path: point them at the cached files.
    kept[f"pointcloud_{n_points}_path"] = PACKED_NAME
    kept["pointclouds_path"] = PACKED_NAME
    kept["sdf_samples_path"] = "sdf_samples.npz"
    for size in (2048, 4096, 8192):
        if size != n_points and f"pointcloud_{size}_path" in kept.columns:
            kept[f"pointcloud_{size}_path"] = None
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "manifest.tmp.parquet"
    kept.to_parquet(tmp)
    os.replace(tmp, out / "manifest.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=15000)
    parser.add_argument("--n-val", type=int, default=1500)
    parser.add_argument("--n-points", type=int, default=8192)
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2], help="point-cloud resamplings to keep (each ~0.2 MB/spine)")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--manifest-every", type=int, default=2000)
    args = parser.parse_args()

    index = load_spine_index([args.manifest], check_files=(), path_mode="relative")
    index = index.with_split(load_splits(args.splits), drop_unassigned=True)
    frame = index.frame
    pc_column = f"pointcloud_{args.n_points}_path"
    rng = np.random.default_rng(args.seed)

    def pick(split: str, n: int) -> pd.DataFrame:
        part = frame[frame["split"] == split]
        if len(part) == 0:
            print(f"WARNING: no spines in split {split!r}")
            return part
        return part.iloc[rng.permutation(len(part))[: min(n, len(part))]]

    val_rows, train_rows = pick("val", args.n_val), pick("train", args.n_train)
    todo = pd.concat([val_rows, train_rows])
    print(f"caching {len(val_rows)} val + {len(train_rows)} train spines -> {args.out} "
          f"(variants {args.variants}, {args.n_points} points, {args.threads} threads)")

    from tqdm.auto import tqdm

    done, failed = set(), []
    written = 0
    started = time.perf_counter()
    bar = tqdm(total=len(todo), desc="cache", mininterval=5.0, unit="spine")
    columns = ["spine_key", "neuron_id", "limb_id", "branch_id", "spine_id", pc_column, "sdf_samples_path"]
    rows = todo[columns].to_dict("records")
    pool = ThreadPoolExecutor(max_workers=args.threads)
    futures = {pool.submit(_cache_one, row, args.out, args.n_points, args.variants, pc_column): row["spine_key"] for row in rows}
    try:
        for n_finished, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                written += future.result()
                done.add(key)
            except Exception as exc:  # one bad spine must not stop the copy
                failed.append((key, repr(exc)))
            bar.update()
            if n_finished % 500 == 0:
                rate = written / max(time.perf_counter() - started, 1e-9) / 1e6
                bar.set_postfix_str(f"{rate:.1f} MB/s written, {len(failed)} failed")
            if n_finished % args.manifest_every == 0:
                _write_manifest(frame, done, args.out, args.n_points)
        pool.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\ninterrupted - finishing the spines in flight, then writing the manifest of what is complete")
        pool.shutdown(wait=True, cancel_futures=True)
        for future, key in futures.items():  # spines that finished while we were waiting
            if future.done() and not future.cancelled() and future.exception() is None:
                done.add(key)
    finally:
        bar.close()
        _write_manifest(frame, done, args.out, args.n_points)
    split_of = dict(zip(frame["spine_key"], frame["split"]))
    counts = pd.Series([split_of[k] for k in done]).value_counts().to_dict()
    print(f"manifest -> {args.out / 'manifest.parquet'}: {len(done)} spines {counts}, {len(failed)} failed, "
          f"{written / 1e9:.2f} GB written in {(time.perf_counter() - started) / 60:.1f} min")
    for key, error in failed[:10]:
        print(f"  failed {key}: {error}")


if __name__ == "__main__":
    main()
