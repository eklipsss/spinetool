"""Which part of a fragmented generated mesh is the spine?

A VAE trained with a truncated SDF loss (``loss.sdf_clamp``) is not supervised far from the surface; the
decoder can cross zero there and Marching Cubes returns thousands of floating fragments next to the spine
(validation: ``n_connected_components`` ~ 10^4, surface area ~ 10^8 nm^2 instead of ~ 10^6..10^7).
This script splits every raw mesh of a generated set into connected components, finds the SPINE CANDIDATE
(the largest component that comes near the origin - in the local frame the attachment is at the origin
and the spine grows along +y) and validates it. Nothing is changed in the run: results go to new files
(``generated_mesh_component.off``, ``mesh_validation_component.json``) and ``component_analysis.parquet``.

    python scripts/neuron_model/analyze_vae_meshes.py --run runs/training/neuron-model/<id> \
        --set meshes/step_0020000/reconstruction       (or meshes/seed_1, meshes_reconstruction/seed_1)

``--overwrite-postprocessed`` also writes the candidate as ``generated_mesh_postprocessed.off`` (+ its
validation json), so ``evaluate_generators.py`` can score it as "postprocessed" (raw stays untouched and is
evaluated separately). Report it as "largest-component-near-origin postprocessing", not as a plain
small-fragment removal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from src.neuron_model.spine_generation.data.bbox import FrozenBBox
from src.neuron_model.spine_generation.experiment.artifacts import write_json, write_mesh_off
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run
from src.neuron_model.spine_generation.reconstruction.mesh_validation import validate_mesh
from src.neuron_model.spine_geometry import load_trimesh

# a real spine has a surface of ~0.5..20 um^2 (nm^2 here) - used only for the plausibility column
PLAUSIBLE_AREA_NM2 = (5.0e5, 3.0e7)


def split_components(mesh):
    parts = mesh.split(only_watertight=False)
    return sorted(parts, key=lambda p: -p.area)


def candidate_component(parts, origin_radius: float):
    """Largest component whose nearest vertex is within ``origin_radius`` of the origin (the attachment)."""
    for part in parts:
        if np.linalg.norm(np.asarray(part.vertices), axis=1).min() <= origin_radius:
            return part
    return None


def analyze_sample(sample_dir: Path, bbox, origin_radius: float, write_postprocessed: bool):
    raw_path = sample_dir / "generated_mesh_raw.off"
    if not raw_path.exists():
        return {"sample": sample_dir.name, "status": "no_mesh"}
    mesh = load_trimesh(raw_path, process=False)
    parts = split_components(mesh)
    areas = np.array([p.area for p in parts])
    row = {
        "sample": sample_dir.name,
        "status": "ok",
        "n_components": len(parts),
        "total_area_nm2": float(areas.sum()),
        "largest_area_nm2": float(areas[0]),
        "largest_area_fraction": float(areas[0] / areas.sum()),
        "n_components_over_1pct_of_largest": int((areas > 0.01 * areas[0]).sum()),
    }
    cand = candidate_component(parts, origin_radius)
    if cand is None:
        return {**row, "status": "no_component_near_origin"}
    report = validate_mesh(cand, bbox=bbox, check_self_intersections=False)
    write_mesh_off(sample_dir / "generated_mesh_component.off", cand)
    write_json(sample_dir / "mesh_validation_component.json", report)
    if write_postprocessed:
        write_mesh_off(sample_dir / "generated_mesh_postprocessed.off", cand)
        write_json(sample_dir / "mesh_validation_postprocessed.json", {**report, "postprocess_changes": {"kept": "largest component near the origin"}})
    ext = np.asarray(cand.bounds[1] - cand.bounds[0])
    row.update(
        candidate_area_nm2=float(cand.area),
        candidate_is_largest=bool(cand is parts[0]),
        candidate_watertight=bool(report.get("is_watertight")),
        candidate_positive_volume=bool(report.get("has_positive_finite_volume")),
        candidate_volume_nm3=report.get("volume"),
        candidate_touches_bbox=report.get("surface_touches_bbox"),
        candidate_valid=bool(report.get("is_valid")),
        candidate_extent_x_nm=float(ext[0]),
        candidate_extent_y_nm=float(ext[1]),
        candidate_extent_z_nm=float(ext[2]),
        candidate_area_plausible=bool(PLAUSIBLE_AREA_NM2[0] <= cand.area <= PLAUSIBLE_AREA_NM2[1]),
        n_boundary_edges=report.get("n_boundary_edges"),
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--set", dest="set_dir", required=True, help="folder with sample_* directories, relative to --run")
    parser.add_argument("--origin-radius", type=float, default=300.0, help="nm: a component 'comes near the origin' if a vertex is closer")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite-postprocessed", action="store_true")
    args = parser.parse_args()

    run = open_run(args.run)
    cfg = run.load_config()
    bb = FrozenBBox.load(resolve_path(cfg["data"]["bbox"]))
    bbox = (bb.low, bb.high)
    root = run.root / args.set_dir
    samples = sorted(p for p in root.glob("sample_*") if p.is_dir())[: args.max_samples]
    if not samples:
        raise SystemExit(f"no sample_* folders in {root}")

    from tqdm.auto import tqdm

    rows = [analyze_sample(s, bbox, args.origin_radius, args.overwrite_postprocessed) for s in tqdm(samples, desc="meshes")]
    table = pd.DataFrame(rows)
    table.to_parquet(root / "component_analysis.parquet", index=False)

    ok = table[table["status"] == "ok"]
    print(f"\n{len(table)} samples in {root}: {len(ok)} analysed, status counts {table['status'].value_counts().to_dict()}")
    if ok.empty:
        return
    summary = {
        "n_components_median": float(ok["n_components"].median()),
        "n_components_max": int(ok["n_components"].max()),
        "total_area_median_um2": float(ok["total_area_nm2"].median() / 1e6),
        "largest_component_area_fraction_median": float(ok["largest_area_fraction"].median()),
    }
    cand = ok[ok["candidate_area_nm2"].notna()] if "candidate_area_nm2" in ok else ok.iloc[0:0]
    if len(cand):
        summary.update(
            candidate_area_median_um2=float(cand["candidate_area_nm2"].median() / 1e6),
            candidate_is_largest_rate=float(cand["candidate_is_largest"].mean()),
            candidate_watertight_rate=float(cand["candidate_watertight"].mean()),
            candidate_positive_volume_rate=float(cand["candidate_positive_volume"].mean()),
            candidate_touches_bbox_rate=float(cand["candidate_touches_bbox"].astype(float).mean()),
            candidate_area_plausible_rate=float(cand["candidate_area_plausible"].mean()),
            candidate_valid_rate=float(cand["candidate_valid"].mean()),
            candidate_extent_median_um=[float(cand[f"candidate_extent_{a}_nm"].median() / 1e3) for a in "xyz"],
        )
    print(json.dumps(summary, indent=2))
    (root / "component_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
