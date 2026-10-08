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
"""

from __future__ import annotations

import argparse
import sys
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
    args = parser.parse_args()

    run = open_run(args.run)
    seed_dir = run.meshes / f"seed_{args.seed}"
    rec_cfg = load_config(args.poisson_config)
    poisson = PoissonParams.from_mapping(rec_cfg["poisson"])
    k_normal = int(rec_cfg["normals"]["k_normal"])
    postprocess_cfg = dict(rec_cfg["postprocess"])

    sample_dirs = sorted(seed_dir.glob("sample_*"), key=lambda p: int(p.name.split("_")[1]))
    if not sample_dirs:
        parser.error(f"no sample_* directories under {seed_dir} - run generate_mogen.py --no-mesh first")

    n_ok = 0
    for d in sample_dirs:
        ply = d / "generated_pointcloud.ply"
        if not ply.exists():
            print(f"  {d.name}: no {ply.name}, skipping")
            continue
        points, _ = read_pointcloud_ply(ply)
        normals, diag = estimate_normals(points, k=k_normal)
        mesh = screened_poisson(points, normals, poisson)
        post, changes = postprocess_mesh(mesh, **postprocess_cfg)
        write_json(d / "poisson_config.json", {**poisson.to_dict(), "k_normal": k_normal, "normal_diagnostics": diag})
        write_mesh_off(d / "generated_mesh_raw.off", mesh)
        write_mesh_off(d / "generated_mesh_postprocessed.off", post)
        write_json(d / "mesh_validation_raw.json", validate_mesh(mesh, check_self_intersections=args.check_self_intersections))
        write_json(
            d / "mesh_validation_postprocessed.json",
            {**validate_mesh(post, check_self_intersections=args.check_self_intersections), "postprocess_changes": changes},
        )
        n_ok += 1
        print(f"  {d.name}: reconstructed ({len(mesh.faces) if mesh is not None else 0} faces)")

    print(f"reconstructed {n_ok}/{len(sample_dirs)} samples -> {seed_dir}")


if __name__ == "__main__":
    main()
