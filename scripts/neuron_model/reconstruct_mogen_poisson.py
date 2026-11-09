"""Screened Poisson reconstruction for point clouds written by ``generate_mogen.py --no-mesh``.

Workaround for a real Windows crash: ``pymeshlab`` bundles its own OpenMP runtime
(``libiomp5md.dll``), which conflicts with the one already loaded by torch/MKL when both
are used in the *same* process (``OMP: Error #15``) - and ``generate_mogen.py`` needs torch
for the flow-matching model. This script never imports torch, so running the two steps as
separate processes avoids the conflict entirely:

    python scripts/neuron_model/generate_mogen.py --run <run> --no-mesh --n-samples 50 --seed 1
    python scripts/neuron_model/reconstruct_mogen_poisson.py --run <run> --seed 1

Fills in the same files ``generate_mogen.py`` would have written itself (``generated_mesh_raw.off``,
``generated_mesh_postprocessed.off``, ``poisson_config.json``, ``mesh_validation_*.json``) using the
Poisson parameters from ``configs/neuron-model/reconstruction/poisson.yaml`` (or ``--poisson-config``).

Re-meshing already generated samples with OTHER Poisson parameters without touching them (chain R, 2026-11-09):

    python scripts/neuron_model/reconstruct_mogen_poisson.py --run <run> --seed 1 --meshes-dir <run>/meshes \
        --out <run>/meshes_poisson_d8_pw4 --poisson-config configs/neuron-model/reconstruction/poisson_sweep/d8_pw4.yaml --workers 8

copies every sample folder (point cloud, metadata) into ``--out/seed_<s>/`` and writes the new meshes there, so
``evaluate_generators.py --model name=<out>`` reads it like any generated set.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.experiment.artifacts import read_pointcloud_ply, write_json, write_mesh_off
from src.neuron_model.spine_generation.experiment.config import load_config
from src.neuron_model.spine_generation.experiment.run import open_run
from src.neuron_model.spine_generation.reconstruction.estimate_normals import estimate_normals
from src.neuron_model.spine_generation.reconstruction.mesh_validation import validate_mesh
from src.neuron_model.spine_generation.reconstruction.postprocess import postprocess_mesh
from src.neuron_model.spine_generation.reconstruction.screened_poisson import PoissonParams, screened_poisson


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="training run directory (same as generate_mogen.py --run)")
    parser.add_argument("--seed", type=int, required=True, help="seed tag used by generate_mogen.py --seed")
    parser.add_argument("--poisson-config", type=Path, default=REPO_ROOT / "configs/neuron-model/reconstruction/poisson.yaml")
    parser.add_argument("--check-self-intersections", action="store_true", help="off by default - CGAL subprocess per mesh is slow")
    parser.add_argument("--meshes-dir", type=Path, help="folder with seed_<s>/sample_* (default: the run's meshes folder)")
    parser.add_argument("--out", type=Path, help="write into a COPY of the sample folders here (the source stays untouched)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run = open_run(args.run)
    src_seed_dir = (args.meshes_dir or run.meshes) / f"seed_{args.seed}"
    rec_cfg = load_config(args.poisson_config)
    sample_dirs = sorted(src_seed_dir.glob("sample_*"), key=lambda p: int(p.name.split("_")[1]))
    if not sample_dirs:
        parser.error(f"no sample_* directories under {src_seed_dir} - run generate_mogen.py --no-mesh first")
    seed_dir = src_seed_dir
    if args.out is not None:
        seed_dir = args.out / f"seed_{args.seed}"
        if seed_dir.resolve() == src_seed_dir.resolve():
            parser.error("--out must differ from the source meshes folder")
        for d in sample_dirs:
            target = seed_dir / d.name
            target.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file() and not f.name.startswith(("generated_mesh_", "mesh_validation_", "poisson_config")):
                    shutil.copy2(f, target / f.name)
        sample_dirs = [seed_dir / d.name for d in sample_dirs]
        write_json(args.out / "reconstruction_source.json", {"source": str(src_seed_dir), "poisson_config": str(args.poisson_config)})

    jobs = [(str(d), rec_cfg, bool(args.check_self_intersections)) for d in sample_dirs]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_reconstruct, jobs, chunksize=4))
    else:
        results = [_reconstruct(j) for j in jobs]
    n_ok = sum(r is not None for r in results)
    n_valid = sum(bool(r and r.get("is_valid")) for r in results)
    print(f"reconstructed {n_ok}/{len(sample_dirs)} samples (valid postprocessed {n_valid}) -> {seed_dir}")


def _reconstruct(job):
    """One sample folder: estimated normals -> Screened Poisson -> postprocess; returns the postprocessed validation or None."""
    d, rec_cfg, check = job
    d = Path(d)
    ply = d / "generated_pointcloud.ply"
    if not ply.exists():
        print(f"  {d.name}: no {ply.name}, skipping", flush=True)
        return None
    poisson = PoissonParams.from_mapping(rec_cfg["poisson"])
    k_normal = int(rec_cfg["normals"]["k_normal"])
    points, _ = read_pointcloud_ply(ply)
    normals, diag = estimate_normals(points, k=k_normal)
    mesh = screened_poisson(points, normals, poisson)
    post, changes = postprocess_mesh(mesh, **dict(rec_cfg["postprocess"]))
    write_json(d / "poisson_config.json", {**poisson.to_dict(), "k_normal": k_normal, "normal_diagnostics": diag})
    write_mesh_off(d / "generated_mesh_raw.off", mesh)
    write_mesh_off(d / "generated_mesh_postprocessed.off", post)
    write_json(d / "mesh_validation_raw.json", validate_mesh(mesh, check_self_intersections=check))
    valid_post = {**validate_mesh(post, check_self_intersections=check), "postprocess_changes": changes}
    write_json(d / "mesh_validation_postprocessed.json", valid_post)
    return valid_post


if __name__ == "__main__":
    main()
