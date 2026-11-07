"""Are the morphometric shifts of a generator caused by its multi-component meshes?

Why (2026-11-08, chain K, evaluation 2026-11-08_011445_generator-evaluation-zopt): the VAE with a flow prior on directly
optimised codes (VAE_zopt_flow) repairs the necks (ConvexHullRatio no longer the main shift against real_mc), but only 57 %
of its meshes are a single component (raw), and its spines look SMALLER than real ones (Area -0.74 SD, Volume -0.46,
LengthAreaRatio +0.73 against real_mc). This script splits the generated meshes of each model by the number of connected
components and prints, for each group, the median shift of every metric from the real test spines in real SD (the same
measure as classifier_ablation.py), plus the area / volume share of the largest component and where the extra pieces lie.
If the small-spine shift lives only in the multi-component group, the components are the defect; if it is the same in the
single-component group, the sampled codes themselves give small spines.

    python scripts/neuron_model/diagnose_generated_components.py \
        --eval-run runs/analysis/neuron-model/2026-11-08_011445_generator-evaluation-zopt \
        --model VAE_flow=<vae run>/meshes_flow_prior --model VAE_zopt_flow=<vae run>/meshes_zopt_flow_prior \
        --model real_mc=<real controls>/mc

Writes ``<eval-run>/component_diagnosis/{per_mesh.csv, by_group.csv}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ("Length", "AverageDistance", "Area", "Volume", "ConvexHullVolume", "ConvexHullRatio",
           "LengthVolumeRatio", "LengthAreaRatio", "OpenAngle", "CVD", "JunctionArea")


def component_stats(mesh) -> dict:
    """Number of components, area / volume share of the largest, and the extra pieces' position relative to it (nm)."""
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        return {"n_components": 0}
    areas = np.array([p.area for p in parts])
    vols = np.array([abs(p.volume) for p in parts])
    main = int(np.argmax(areas))
    out = {"n_components": len(parts), "largest_area_share": float(areas[main] / areas.sum()),
           "largest_volume_share": float(vols[main] / vols.sum()) if vols.sum() > 0 else np.nan,
           "extra_area_nm2": float(areas.sum() - areas[main])}
    if len(parts) > 1:
        lo, hi = parts[main].bounds
        extra = [p for i, p in enumerate(parts) if i != main]
        biggest = max(extra, key=lambda p: p.area)
        c = biggest.centroid
        out["extra_centroid_y_nm"] = float(c[1])
        out["main_top_y_nm"] = float(hi[1])
        out["extra_above_main"] = bool(c[1] > hi[1])
        out["extra_gap_nm"] = float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - c, c - hi))))
    return out


def standardized_shift(frame: pd.DataFrame, real: pd.DataFrame, metrics=METRICS) -> dict:
    """Median (generated - real) / SD(real) per metric, as classifier_ablation.py."""
    out = {}
    for m in metrics:
        if m not in frame.columns or m not in real.columns:
            continue
        g = pd.to_numeric(frame[m], errors="coerce").dropna()
        r = pd.to_numeric(real[m], errors="coerce").dropna()
        if len(g) == 0 or len(r) < 2 or r.std() == 0:
            continue
        out[m] = float((g.median() - r.median()) / r.std())
    return out


def diagnose(eval_run: Path, models: dict, mesh_kind: str = "postprocessed", seed: int = 1) -> tuple:
    from src.neuron_model.spine_generation.evaluation.inputs import load_generated_set

    real = pd.read_parquet(eval_run / "real_test_morphometrics.parquet")
    rows, groups = [], []
    for name, root in models.items():
        meshes, _, ids = load_generated_set(Path(root) / f"seed_{seed}", mesh_kind)
        morph = pd.read_parquet(eval_run / mesh_kind / f"morphometrics_{name}_seed{seed}.parquet")
        stats = pd.DataFrame([{"sample_id": i, **(component_stats(m) if m is not None else {"n_components": 0})}
                              for i, m in zip(ids, meshes)])
        frame = morph.merge(stats, on="sample_id", how="left")
        frame.insert(0, "model", name)
        rows.append(frame)
        frame["group"] = np.where(frame["n_components"] == 1, "1 component", ">= 2 components")
        for group, part in [("all", frame)] + list(frame.groupby("group")):
            groups.append({"model": name, "group": group, "n": len(part),
                           "largest_area_share_median": float(part["largest_area_share"].median()),
                           "extra_above_main_rate": float(part["extra_above_main"].mean()) if "extra_above_main" in part and part["extra_above_main"].notna().any() else np.nan,
                           **standardized_shift(part, real)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(groups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-run", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=<generation root with seed_<k>/>")
    ap.add_argument("--mesh-kind", default="postprocessed", choices=("raw", "postprocessed"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model)
    per_mesh, by_group = diagnose(args.eval_run, models, args.mesh_kind, args.seed)
    out = args.eval_run / "component_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    per_mesh.drop(columns=[c for c in per_mesh.columns if per_mesh[c].dtype == object and c not in ("model", "sample_id", "group", "attachment_method")]).to_csv(out / "per_mesh.csv", index=False)
    by_group.to_csv(out / "by_group.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["model", "group", "n", "largest_area_share_median", "extra_above_main_rate", "Area", "Volume", "Length",
            "AverageDistance", "ConvexHullRatio", "LengthAreaRatio", "LengthVolumeRatio", "OpenAngle"]
    print(f"median shift from real test spines in real SD ({args.mesh_kind} meshes)")
    print(by_group[[c for c in show if c in by_group.columns]].round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
