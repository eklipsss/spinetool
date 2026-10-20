"""Assemble branch meshes: trunks + placements of a ``generate_tp.py`` set + a pool of spine meshes (module-interfaces-spec.md §8).

Windows (local disk only, CPU, ~seconds per branch):

    python scripts/neuron_model/assemble_branches.py \
        --tp-dir runs/analysis/neuron-model/tp_generation/<date>_generated_seed1 \
        --spines runs/training/neuron-model/<vae_run>/meshes_reconstruction/seed_1 \
        --n-branches 20 --seed 1

``--spines``: a directory with spine meshes in the local spine frame: ``**/generated_mesh_postprocessed.off`` (the
layout of generate_vae.py / generate_mogen.py) or any *.off / *.ply. Spines are drawn from the pool at random (no
conditioning yet). Options of the algorithm: ``configs/neuron-model/assembly/default.yaml`` (+ ``key=value`` overrides).

Writes ``<tp-dir>/assembly_<date>/``: ``meshes/branch_<i>.ply`` (one closed surface per branch), ``assembly_branches.parquet``
(checks per branch), ``assembly_spines.parquet`` (per placement: spine used, immersion depth, attempts, reasons),
``spine_pool_rejected.parquet`` and ``assembly_info.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from src.neuron_model.branch_assembly import AssemblyConfig, SpinePool, assemble_branch, find_mesh_files, load_tp_set
from src.neuron_model.spine_generation.experiment.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp-dir", type=Path, required=True, help="output directory of generate_tp.py")
    parser.add_argument("--spines", type=Path, required=True, help="directory with spine meshes (local spine frame)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/neuron-model/assembly/default.yaml")
    parser.add_argument("--n-branches", type=int, default=20, help="first N branches of the set (0 = all)")
    parser.add_argument("--max-pool", type=int, default=0, help="use at most this many spine meshes (0 = all)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    raw = load_config(args.config, args.overrides) if args.config.exists() else {}
    cfg = AssemblyConfig.from_mapping(raw.get("assembly", raw))
    if args.seed is not None:
        cfg.seed = args.seed
    from tqdm.auto import tqdm

    files = find_mesh_files(args.spines, limit=args.max_pool or None)
    if not files:
        raise SystemExit(f"no spine meshes under {args.spines}")
    t0 = time.perf_counter()
    pool = SpinePool.from_paths(files, base_band_nm=cfg.base_band_nm, keys=[str(p.relative_to(args.spines)) for p in files])
    print(f"[assembly] spine pool: {len(pool)} of {len(files)} meshes usable ({time.perf_counter() - t0:.1f}s); rejected: "
          f"{pd.Series([r['reason'] for r in pool.rejected]).value_counts().to_dict() if pool.rejected else {}}", flush=True)
    if not len(pool):
        raise SystemExit("no usable spine mesh")

    branches, trunks, placements = load_tp_set(args.tp_dir)
    n = len(trunks) if args.n_branches <= 0 else min(args.n_branches, len(trunks))
    out = args.out or args.tp_dir / f"assembly_{_dt.datetime.now():%Y-%m-%d_%H%M%S}"
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    import trimesh

    rng = np.random.default_rng(cfg.seed)
    brows, srows = [], []
    for i in tqdm(range(n), desc="branches"):
        p = placements.get(i)
        s = p["s"].to_numpy(float) if p is not None else np.zeros(0)
        th = p["theta"].to_numpy(float) if p is not None else np.zeros(0)
        started = time.perf_counter()
        try:
            res = assemble_branch(trunks[i], s, th, pool, cfg, rng)
        except Exception as exc:  # one broken branch must not stop the set
            brows.append({"branch_index": i, "status": "failed", "reason": repr(exc)[:300]})
            continue
        row = {"branch_index": i, **res.branch, "seconds": round(time.perf_counter() - started, 2)}
        if res.vertices is not None:
            path = out / "meshes" / f"branch_{i:05d}.ply"
            trimesh.Trimesh(res.vertices, res.faces, process=False).export(path)
            row["mesh_path"] = str(path.relative_to(out))
        brows.append(row)
        srows.extend({"branch_index": i, **r} for r in res.spines)

    bt = pd.DataFrame(brows)
    if len(branches):
        bt = bt.merge(branches.drop(columns=[c for c in ("length_nm", "n_spines") if c in branches.columns]), on="branch_index", how="left")
    bt.to_parquet(out / "assembly_branches.parquet", index=False)
    st = pd.DataFrame(srows)
    st.to_parquet(out / "assembly_spines.parquet", index=False)
    pd.DataFrame(pool.rejected or [], columns=["key", "reason"]).to_parquet(out / "spine_pool_rejected.parquet", index=False)
    info = {"tp_dir": str(args.tp_dir), "spines": str(args.spines), "n_branches": n, "pool_size": len(pool), "pool_rejected": len(pool.rejected),
            "config": cfg.__dict__, "status_counts": bt["status"].value_counts().to_dict()}
    if len(st):
        info["spines_placed"] = int((st["status"] == "placed").sum())
        info["spines_dropped"] = int((st["status"] == "dropped").sum())
        reasons = st["failure_reasons"].dropna().str.split(";").explode()
        info["failure_reasons"] = reasons.value_counts().to_dict()
    (out / "assembly_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    if "reason" in bt.columns and bt["reason"].notna().any():
        info["invalid_reasons"] = bt["reason"].dropna().str.split(";").explode().value_counts().to_dict()
        (out / "assembly_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        print(f"[assembly] not ok: {info['invalid_reasons']} (branches {bt.loc[bt['reason'].notna(), 'branch_index'].tolist()})", flush=True)
    print(f"[assembly] branches: {info['status_counts']}; spines placed {info.get('spines_placed', 0)}, dropped {info.get('spines_dropped', 0)}; "
          f"attempt failures {info.get('failure_reasons', {})}", flush=True)
    if len(bt) and "watertight" in bt.columns:
        ok = bt[bt["status"].isin(["ok", "invalid"])]
        print(f"[assembly] watertight {ok['watertight'].mean():.0%}, one component {(ok['n_components'] == 1).mean():.0%}, "
              f"median {ok['seconds'].median():.1f} s/branch -> {out}", flush=True)


if __name__ == "__main__":
    main()
