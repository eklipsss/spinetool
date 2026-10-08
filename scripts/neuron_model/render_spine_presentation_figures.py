"""Render presentation figures with attachment regions highlighted.

Real spines use the true attachment cap saved during preprocessing
(``attachment_region_path``). Generated spines are closed meshes, so the script
uses the same heuristic finder as evaluation (``DownFacingCapFinder``) and
colors the detected cap faces.

Examples
--------
python scripts/neuron_model/render_spine_presentation_figures.py \
  --manifest "O:/Datasets/Minnie65/preprocessed/manifest.parquet" \
  --generated-seed-dir "runs/training/neuron-model/<run_id>/meshes/seed_1" \
  --out-dir "runs/analysis/neuron-model/presentation_figures" \
  --n-real 6 --n-generated 6 --path-mode relative

Writes PNG only (static, for slides) via plotly's kaleido backend - needs
``pip install kaleido`` (now in requirements/conda_{win,macos}_requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_geometry import load_trimesh
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.evaluation.attachment import DownFacingCapFinder, _axis_seed_face, _grow, _section_contours
from src.neuron_model.visualization import ATTACHMENT_CAP_COLOR, DEFAULT_MESH_COLOR, face_color_map, mesh_trace


def _sample_evenly(items: Sequence, n: int) -> List:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n, dtype=int)
    return [items[int(i)] for i in idx]


def _translated(mesh, offset: Sequence[float]):
    copied = mesh.copy()
    copied.apply_translation(np.asarray(offset, dtype=float))
    return copied


def _mesh_width(mesh) -> float:
    extent = np.ptp(np.asarray(mesh.vertices, dtype=float), axis=0)
    return float(max(extent.max(), 1.0))


def _layout_scene(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=True,
        legend=dict(itemsizing="constant"),
    )


def _write_figure(fig: go.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(out_path, scale=2)
    except ValueError as exc:
        raise RuntimeError(
            f"PNG export failed for {out_path.name} - plotly needs 'kaleido' installed "
            f"(pip install kaleido). Original error: {exc}"
        ) from exc


def _add_label(fig: go.Figure, text: str, xyz: Sequence[float]) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[xyz[0]],
            y=[xyz[1]],
            z=[xyz[2]],
            mode="text",
            text=[text],
            textposition="top center",
            showlegend=False,
        )
    )


def real_rows(manifest: Path, *, path_mode: str, n: int) -> List[dict]:
    # check_files=() on purpose: with require_train_eligible (default True), seal/orient
    # already succeeded for every row here, so local_sealed_mesh_path/attachment_region_path
    # are guaranteed to exist - checking them would mean a NAS exists() call per row across
    # the whole manifest just to pick a handful of examples (see calibrate_reconstruction.py
    # for the same fix and why it matters at this dataset's scale).
    index = load_spine_index([manifest], path_mode=path_mode, check_files=())
    rows = index.frame.sort_values("spine_key").to_dict(orient="records")
    return _sample_evenly(rows, n)


def generated_sample_dirs(seed_dir: Path, *, mesh_kind: str, n: int) -> List[Path]:
    sample_dirs = []
    for path in sorted(Path(seed_dir).glob("sample_*")):
        if (path / f"generated_mesh_{mesh_kind}.off").exists():
            sample_dirs.append(path)
    return _sample_evenly(sample_dirs, n)


def generated_cap_faces(mesh, finder: DownFacingCapFinder) -> Tuple[np.ndarray, str]:
    """Return face indices colored as the generated attachment cap.

    The primary path mirrors DownFacingCapFinder.down_facing_cap. For the
    section fallback, color the surface below the selected section height; this
    is a visual approximation of the fallback cap used for morphometrics.
    """
    seed, seed_method = _axis_seed_face(mesh)
    facing_down = -np.asarray(mesh.face_normals)[:, 1] > math.cos(math.radians(finder.max_tilt_deg))
    if facing_down[seed]:
        return _grow(mesh, seed, facing_down), f"down_facing_cap:{seed_method}"

    if not finder.section_fallback:
        return np.empty((0,), dtype=int), "not_found"

    height = float(mesh.vertices[:, 1].max())
    if height <= 0:
        return np.empty((0,), dtype=int), "not_found"

    step = finder.section_step_frac * height
    single = []
    for y0 in np.arange(step, finder.section_band_frac * height + 0.5 * step, step):
        contours = _section_contours(mesh, float(y0))
        if len(contours) == 1:
            single.append((float(y0), contours[0]))
    if not single:
        return np.empty((0,), dtype=int), "not_found"
    y_cut, _ = max(single, key=lambda item: item[1][0])
    faces = np.flatnonzero(np.asarray(mesh.triangles_center)[:, 1] <= y_cut)
    return faces.astype(int), "section"


def render_real(rows: Sequence[dict], out_path: Path) -> None:
    fig = go.Figure()
    x_offset = 0.0
    for idx, row in enumerate(rows):
        mesh = load_trimesh(Path(row["local_sealed_mesh_path"]), process=False)
        attachment = json.loads(Path(row["attachment_region_path"]).read_text(encoding="utf-8"))
        cap_faces = attachment.get("attachment_cap_face_indices", [])
        width = _mesh_width(mesh)
        shown = _translated(mesh, (x_offset, 0.0, 0.0))
        fig.add_trace(
            mesh_trace(
                shown,
                name=f"real {idx + 1}",
                color=DEFAULT_MESH_COLOR,
                opacity=1.0,
                face_colors=face_color_map(cap_faces, ATTACHMENT_CAP_COLOR),
            )
        )
        top = shown.vertices.max(axis=0)
        _add_label(fig, f"real {idx + 1}", (x_offset, top[1] + 0.15 * width, top[2]))
        x_offset += 1.35 * width
    _layout_scene(fig, "Real spines: true attachment cap from preprocessing")
    _write_figure(fig, out_path)


def render_generated(sample_dirs: Sequence[Path], out_path: Path, *, mesh_kind: str) -> None:
    finder = DownFacingCapFinder()
    fig = go.Figure()
    x_offset = 0.0
    for idx, sample_dir in enumerate(sample_dirs):
        mesh = load_trimesh(sample_dir / f"generated_mesh_{mesh_kind}.off", process=False)
        cap_faces, method = generated_cap_faces(mesh, finder)
        width = _mesh_width(mesh)
        shown = _translated(mesh, (x_offset, 0.0, 0.0))
        fig.add_trace(
            mesh_trace(
                shown,
                name=f"generated {idx + 1}: {method}",
                color=DEFAULT_MESH_COLOR,
                opacity=1.0,
                face_colors=face_color_map(cap_faces, ATTACHMENT_CAP_COLOR),
            )
        )
        top = shown.vertices.max(axis=0)
        _add_label(fig, f"gen {idx + 1}", (x_offset, top[1] + 0.15 * width, top[2]))
        x_offset += 1.35 * width
    _layout_scene(fig, "Generated spines: heuristic attachment cap")
    _write_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, help="Preprocessing manifest.parquet for real spines")
    parser.add_argument("--generated-seed-dir", type=Path, help="Directory like <run>/meshes/seed_1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--path-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--mesh-kind", choices=("raw", "postprocessed"), default="raw")
    parser.add_argument("--n-real", type=int, default=6)
    parser.add_argument("--n-generated", type=int, default=6)
    args = parser.parse_args()

    if args.manifest:
        rows = real_rows(args.manifest, path_mode=args.path_mode, n=args.n_real)
        if not rows:
            parser.error(f"no real spines with local_sealed_mesh_path and attachment_region_path in {args.manifest}")
        out_path = args.out_dir / "real_spines_attachment.png"
        render_real(rows, out_path)
        print(f"real figure -> {out_path}")

    if args.generated_seed_dir:
        sample_dirs = generated_sample_dirs(args.generated_seed_dir, mesh_kind=args.mesh_kind, n=args.n_generated)
        if not sample_dirs:
            parser.error(f"no sample_* directories with generated_mesh_{args.mesh_kind}.off under {args.generated_seed_dir}")
        out_path = args.out_dir / "generated_spines_attachment.png"
        render_generated(sample_dirs, out_path, mesh_kind=args.mesh_kind)
        print(f"generated figure -> {out_path}")

    if not args.manifest and not args.generated_seed_dir:
        parser.error("pass --manifest, --generated-seed-dir, or both")


if __name__ == "__main__":
    main()
