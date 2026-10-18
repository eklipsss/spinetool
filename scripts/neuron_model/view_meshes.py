"""Look at a generated set in the browser: one interactive HTML page (rotate / zoom / hide by clicking the legend).

    python scripts/neuron_model/view_meshes.py --run runs/training/neuron-model/<id> --set meshes/seed_1
    python scripts/neuron_model/view_meshes.py --run runs/training/neuron-model/<id> --set meshes_reconstruction/seed_1

Every sample is drawn on a grid (the spine grows along +y, the attachment is at the origin of each cell):
pink = valid mesh, orange = invalid (hover: index, number of components, area, why invalid; for reconstructions also
Chamfer / F-score). For reconstructions the REAL spine of the same key is drawn next to it in grey - its mesh if the
manifest points at one that exists (``--manifest O:/Datasets/Minnie65/preprocessed/manifest.parquet``), otherwise
its point cloud from the run's own manifest (the training cache keeps only point clouds). ``--which postprocessed``
shows the postprocessed meshes. Output: ``<run>/report/view_<set>.html`` (open it in any browser). Only reads files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

VALID, INVALID, REAL = "#FD89C3", "#F5A623", "#B5B8BC"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _why_invalid(v: dict) -> str:
    reasons = []
    if v.get("n_connected_components", 1) != 1:
        reasons.append(f"{v.get('n_connected_components')} components")
    if not v.get("is_watertight", True):
        reasons.append("not watertight")
    if v.get("surface_touches_bbox"):
        reasons.append("touches bbox")
    if v.get("has_self_intersections"):
        reasons.append("self-intersections")
    if not v.get("has_positive_finite_volume", True):
        reasons.append("no positive volume")
    return ", ".join(reasons) or "see json"


def _real_lookup(manifests):
    from src.neuron_model.spine_generation.data.index import load_spine_index

    index = load_spine_index(manifests, check_files=(), require_train_eligible=False)
    f = index.frame
    cols = {c: dict(zip(f["spine_key"], f[c])) for c in ("local_sealed_mesh_path", "pointclouds_path", "pointcloud_8192_path") if c in f.columns}
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--set", required=True, help="folder of the set inside the run, e.g. meshes/seed_1 or meshes_reconstruction/seed_1")
    parser.add_argument("--which", choices=("raw", "postprocessed"), default="raw")
    parser.add_argument("--max", type=int, default=40, help="at most this many samples (the page gets slow beyond ~60)")
    parser.add_argument("--per-row", type=int, default=8)
    parser.add_argument("--manifest", type=Path, nargs="*", help="manifest(s) with the real meshes (default: the run's data.manifests)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import plotly.graph_objects as go
    import trimesh

    set_dir = args.run / args.set
    samples = sorted(p for p in set_dir.glob("sample_*") if p.is_dir())[: args.max]
    if not samples:
        raise SystemExit(f"no sample_* folders in {set_dir}")
    real = {}
    if any((s / "reconstruction_metrics.json").exists() for s in samples) or "reconstruction" in args.set:
        from src.neuron_model.spine_generation.experiment.config import resolve_path
        from src.neuron_model.spine_generation.experiment.run import open_run

        manifests = args.manifest or [resolve_path(p) for p in open_run(args.run).load_config()["data"]["manifests"]]
        try:
            real = _real_lookup(manifests)
        except Exception as exc:  # the page is still useful without the real spines
            print(f"[view] real spines not available: {exc!r}")

    fig = go.Figure()
    gap = 3000.0  # nm between cells (spines are ~0.5-2 um)
    n_valid = 0
    for k, d in enumerate(samples):
        mesh_path = d / f"generated_mesh_{args.which}.off"
        if not mesh_path.exists():
            continue
        v = _read_json(d / f"mesh_validation_{args.which}.json")
        meta = _read_json(d / "generation_metadata.json")
        rec = _read_json(d / "reconstruction_metrics.json")
        ok = bool(v.get("is_valid"))
        n_valid += ok
        row, col = divmod(k, args.per_row)
        width = 2 if real else 1
        x0, z0 = col * gap * width, -row * gap * 1.4
        mesh = trimesh.load(mesh_path, process=False)
        vert = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        label = f"#{meta.get('sample_index', k)} {'valid' if ok else 'INVALID: ' + _why_invalid(v)}; area {mesh.area / 1e6:.2f} um^2"
        if rec:
            label += f"; Chamfer {rec.get(f'{args.which}_chamfer_l1', float('nan')):.0f} nm, F@50 {rec.get(f'{args.which}_f_score@50', float('nan')):.2f}"
        fig.add_trace(go.Mesh3d(x=vert[:, 0] + x0, y=vert[:, 1], z=vert[:, 2] + z0, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                color=VALID if ok else INVALID, flatshading=True, name=label, showlegend=True, hovertext=label, hoverinfo="text"))
        key = meta.get("generation_config", {}).get("spine_key")
        if real and key:
            mpath = real.get("local_sealed_mesh_path", {}).get(key)
            if isinstance(mpath, str) and Path(mpath).exists():
                rm = trimesh.load(mpath, process=False)
                rv, rf = np.asarray(rm.vertices, dtype=float), np.asarray(rm.faces)
                fig.add_trace(go.Mesh3d(x=rv[:, 0] + x0 + gap, y=rv[:, 1], z=rv[:, 2] + z0, i=rf[:, 0], j=rf[:, 1], k=rf[:, 2],
                                        color=REAL, flatshading=True, name=f"#{k} real {key}", showlegend=True))
            else:
                ppath = real.get("pointclouds_path", {}).get(key) or real.get("pointcloud_8192_path", {}).get(key)
                if isinstance(ppath, str) and Path(ppath).exists():
                    from src.neuron_model.pointcloud_io import load_points

                    pts = load_points(ppath, 8192, 1)[::3]
                    fig.add_trace(go.Scatter3d(x=pts[:, 0] + x0 + gap, y=pts[:, 1], z=pts[:, 2] + z0, mode="markers",
                                               marker=dict(size=1.5, color=REAL), name=f"#{k} real {key} (points)", showlegend=True))
    title = f"{args.run.name} / {args.set} ({args.which}): {n_valid} of {len(samples)} valid - pink valid, orange invalid"
    if real:
        title += ", grey = the real spine"
    fig.update_layout(title=title, scene=dict(aspectmode="data", xaxis_title="x, nm", yaxis_title="y, nm (spine axis)", zaxis_title="z, nm"),
                      height=900, legend=dict(itemsizing="constant", font=dict(size=9)))
    out = args.out or args.run / "report" / f"view_{args.set.replace('/', '_').replace(chr(92), '_')}_{args.which}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)  # plotly.js embedded: the page works offline
    print(f"[view] {len(samples)} samples ({n_valid} valid) -> {out}")


if __name__ == "__main__":
    main()
