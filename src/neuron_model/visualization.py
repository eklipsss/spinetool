"""Plotly 3D visual QA helpers for the spine preprocessing pipeline.

Used by ``notebooks/neuron-model/spine-preprocessing-visual-check.ipynb`` to
render, per spine:

0. the source mesh;
1. the sealed mesh, with faces added while sealing in green and the
   attachment loop (the boundary that got capped) outlined in red;
2. the sealed mesh in local coordinates, with the attachment cap faces in
   red and the local frame (origin + tangent/radial/binormal axes) drawn;
3. the source mesh in local coordinates (same frame, no cap - it does not
   exist on the unsealed mesh), with the local frame drawn;
4. the sampled point clouds, one figure per (point count, seed variant);
5. the SDF query pool, colored by signed distance (surface/near-surface/
   uniform samples as separate, semi-transparent layers) over a translucent
   copy of the local sealed mesh, for a visual sign/zero-level sanity check.

Figures 2-5 are skipped (not raised) for a spine that an earlier stage
filtered out (e.g. a false spine, or a QC-rejected mesh when
``allow_needs_review: false``) - see ``visualize_spine_stage_outputs``.

Kept separate from ``spine_geometry.py``/``spine_sampling.py`` since this is
visualization-only and pulls in plotly, which the rest of the pipeline does
not depend on.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

DEFAULT_MESH_COLOR = "lightsteelblue"
NEW_FACE_COLOR = "seagreen"
ATTACHMENT_CAP_COLOR = "crimson"
LOOP_CONTOUR_COLOR = "red"
AXIS_COLORS = ("red", "green", "blue")
AXIS_LABELS = ("tangent (+X)", "radial (+Y)", "binormal (+Z)")


def face_color_map(face_indices: Iterable[int], color: str) -> Dict[int, str]:
    """``{face_index: color}`` for :func:`mesh_trace`'s ``face_colors``."""
    return {int(i): color for i in face_indices}


def mesh_trace(
    mesh: Any,
    *,
    name: str,
    color: str = DEFAULT_MESH_COLOR,
    opacity: float = 1.0,
    face_colors: Optional[Dict[int, str]] = None,
):
    """A ``go.Mesh3d`` trace for ``mesh``, optionally with per-face override colors."""
    import plotly.graph_objects as go

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    facecolor = [face_colors.get(i, color) for i in range(len(faces))] if face_colors else None
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=None if facecolor else color,
        facecolor=facecolor,
        opacity=opacity,
        name=name,
        showscale=False,
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.6),
    )


def edge_segments_trace(
    vertices: np.ndarray,
    edges: Sequence[Sequence[int]],
    *,
    name: str,
    color: str = LOOP_CONTOUR_COLOR,
    width: int = 8,
):
    """A ``go.Scatter3d`` line trace drawing each pair in ``edges`` as its own segment.

    ``vertices`` should be the *original* (unsealed) mesh's vertices: hole
    filling does not move existing vertex positions, only adds/reindexes
    topology, so original-mesh coordinates stay valid for edges expressed in
    original-mesh vertex indices (e.g. ``attachment_loop_edges``) even when
    overlaid on the sealed mesh, without needing an index correspondence.
    """
    import plotly.graph_objects as go

    vertices = np.asarray(vertices, dtype=float)
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for a, b in edges:
        xs += [vertices[a, 0], vertices[b, 0], None]
        ys += [vertices[a, 1], vertices[b, 1], None]
        zs += [vertices[a, 2], vertices[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, line=dict(color=color, width=width))


def local_frame_traces(
    *,
    scale: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    axis_colors: Sequence[str] = AXIS_COLORS,
    axis_labels: Sequence[str] = AXIS_LABELS,
):
    """Origin marker + X/Y/Z axis traces for a local coordinate frame.

    In local coordinates the frame's own basis vectors are, by construction,
    the standard basis (tangent=+X, radial=+Y, binormal=+Z - see
    ``orient_spines_w_holes``), so only a length ``scale`` is needed, not an
    actual basis matrix.
    """
    import plotly.graph_objects as go

    origin_arr = np.asarray(origin, dtype=float)
    traces = [
        go.Scatter3d(
            x=[origin_arr[0]],
            y=[origin_arr[1]],
            z=[origin_arr[2]],
            mode="markers",
            marker=dict(size=6, color="black"),
            name="origin",
        )
    ]
    for axis_vector, color, label in zip(np.eye(3), axis_colors, axis_labels):
        end = origin_arr + axis_vector * scale
        traces.append(
            go.Scatter3d(
                x=[origin_arr[0], end[0]],
                y=[origin_arr[1], end[1]],
                z=[origin_arr[2], end[2]],
                mode="lines",
                line=dict(color=color, width=8),
                name=label,
            )
        )
    return traces


def pointcloud_trace(
    points: np.ndarray,
    *,
    is_attachment_cap: Optional[np.ndarray] = None,
    name: str = "points",
    size: float = 2.0,
):
    import plotly.graph_objects as go

    points = np.asarray(points, dtype=float)
    if is_attachment_cap is not None:
        color = np.where(np.asarray(is_attachment_cap, dtype=bool), ATTACHMENT_CAP_COLOR, "steelblue")
    else:
        color = "steelblue"
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=color),
        name=name,
    )


_SDF_SAMPLE_TYPE_STYLE = {
    # code: (label, opacity, marker size) - surface samples drawn most
    # opaque/largest since d~=0 there is the most informative check
    # (does the zero level really sit on the mesh?); uniform faintest so it
    # reads as background context rather than obscuring the other two.
    0: ("surface (d~=0)", 0.95, 3.0),
    1: ("near-surface", 0.55, 2.5),
    2: ("uniform", 0.18, 2.0),
}


def sdf_scatter_traces(
    query_points: np.ndarray,
    sdf: np.ndarray,
    sample_type: np.ndarray,
    *,
    max_points_per_type: int = 4000,
    seed: int = 0,
):
    """One ``go.Scatter3d`` trace per SDF sample type, colored by signed distance.

    All three traces share one color range (symmetric around 0, so 0 always
    maps to the colorscale's midpoint) and one colorbar. Negative values are
    inside the surface, positive outside (s-module-preprocessing.md 3.7.1);
    surface samples should look close to white/midpoint, and the uniform
    layer should visibly split into two colors either side of the mesh.

    Randomly subsamples each type to ``max_points_per_type`` for a
    responsive plot (SDF pools default to 16384 points total).
    """
    import plotly.graph_objects as go

    query_points = np.asarray(query_points, dtype=float)
    sdf = np.asarray(sdf, dtype=float)
    sample_type = np.asarray(sample_type)
    rng = np.random.default_rng(seed)
    cmax = float(np.max(np.abs(sdf))) if len(sdf) else 1.0

    traces = []
    for code, (label, opacity, marker_size) in _SDF_SAMPLE_TYPE_STYLE.items():
        indices = np.flatnonzero(sample_type == code)
        if len(indices) > max_points_per_type:
            indices = rng.choice(indices, size=max_points_per_type, replace=False)
        points = query_points[indices]
        traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=sdf[indices],
                    colorscale="RdBu_r",
                    cmin=-cmax,
                    cmax=cmax,
                    opacity=opacity,
                    showscale=(code == 0),
                    colorbar=dict(title="SDF<br>(neg=inside)") if code == 0 else None,
                ),
                name=f"{label} (n={len(indices)})",
            )
        )
    return traces


def mesh_extent(mesh: Any) -> float:
    """A representative length scale for ``mesh`` (max bounding-box extent)."""
    bounds = np.asarray(mesh.bounds, dtype=float)
    return float(np.max(bounds[1] - bounds[0]))


def make_scene_figure(traces: Sequence[Any], *, title: str, width: int = 700, height: int = 600):
    """A single-scene 3D figure with equal aspect ratio ("data")."""
    import plotly.graph_objects as go

    fig = go.Figure(data=list(traces))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def visualize_spine_stage_outputs(record: Any, cfg: Any, *, show: bool = True) -> Dict[str, Any]:
    """Build (and by default display) whichever of the 6 visual-QA figures a
    spine's output files support.

    Figures 2-5 (local sealed/original mesh, point clouds, SDF) need the
    orient/pointcloud/sdf stages' output files, which a spine does not get if
    an earlier stage filtered it out - e.g. a false spine (``false_spine_
    detection`` marks it invalid, so ``orient`` skips it via its eligibility
    check, per ``_orient_eligibility``) or one rejected by mesh QC when
    ``allow_needs_review: false``. Those figures are skipped (with a printed
    reason) rather than raising, so this still works for the "first N spines
    of a branch" case in ``spine-preprocessing-visual-check.ipynb`` where not
    every spine reaches every stage.

    Returns a dict with whichever of ``{"original", "sealed", "local_sealed",
    "local_original", "pointclouds", "sdf"}`` could be built.
    """
    import json

    from .spine_geometry import find_new_faces, load_trimesh
    from .spine_preprocessing import (
        attachment_json_path,
        local_mesh_path,
        local_sealed_mesh_path,
        pointcloud_path,
        sdf_samples_path,
        sealed_mesh_path,
    )

    spine_id = record.spine_id

    def skip(step: str, path: Any) -> None:
        print(f"spine {spine_id}: skipping {step} - file not found (not produced for this spine): {path}")

    figures: Dict[str, Any] = {}

    original = load_trimesh(record.source_path, process=False)
    figures["original"] = make_scene_figure(
        [mesh_trace(original, name="original")],
        title=f"spine {spine_id} - 0) original mesh",
    )

    sealed_path = sealed_mesh_path(record, cfg)
    attachment_path = attachment_json_path(record, cfg)
    if sealed_path.exists() and attachment_path.exists():
        sealed = load_trimesh(sealed_path, process=False)
        attachment = json.loads(attachment_path.read_text(encoding="utf-8"))
        new_faces, _, _ = find_new_faces(original, sealed)
        figures["sealed"] = make_scene_figure(
            [
                mesh_trace(sealed, name="sealed", face_colors=face_color_map(new_faces, NEW_FACE_COLOR)),
                edge_segments_trace(original.vertices, attachment["attachment_loop_edges"], name="attachment loop"),
            ],
            title=f"spine {spine_id} - 1) sealed mesh (green = new faces, red = attachment loop)",
        )
    else:
        skip("1) sealed mesh", sealed_path)
        sealed = None
        attachment = None

    local_sealed_path = local_sealed_mesh_path(record, cfg)
    if local_sealed_path.exists() and attachment is not None:
        local_sealed = load_trimesh(local_sealed_path, process=False)
        cap_colors = face_color_map(attachment["attachment_cap_face_indices"], ATTACHMENT_CAP_COLOR)
        figures["local_sealed"] = make_scene_figure(
            [
                mesh_trace(local_sealed, name="local sealed", face_colors=cap_colors),
                *local_frame_traces(scale=mesh_extent(local_sealed) * 0.3),
            ],
            title=f"spine {spine_id} - 2) local sealed mesh (red = attachment cap)",
        )
    else:
        skip("2) local sealed mesh", local_sealed_path)
        local_sealed = None

    local_mesh_path_ = local_mesh_path(record, cfg)
    if local_mesh_path_.exists():
        local_original = load_trimesh(local_mesh_path_, process=False)
        figures["local_original"] = make_scene_figure(
            [
                mesh_trace(local_original, name="local original"),
                *local_frame_traces(scale=mesh_extent(local_original) * 0.3),
            ],
            title=f"spine {spine_id} - 3) local original mesh",
        )
    else:
        skip("3) local original mesh", local_mesh_path_)

    sizes = list(cfg.pointcloud_sizes)
    n_variants = len(cfg.pointcloud_seeds)
    first_pointcloud_path = pointcloud_path(record, cfg, sizes[0], 1) if sizes else None
    if first_pointcloud_path is not None and first_pointcloud_path.exists():
        traces_by_cell: Dict[tuple, Any] = {}
        cell_titles: Dict[tuple, str] = {}
        for row, size in enumerate(sizes, start=1):
            for col in range(1, n_variants + 1):
                npz = np.load(pointcloud_path(record, cfg, size, col))
                traces_by_cell[(row, col)] = pointcloud_trace(
                    npz["points"], is_attachment_cap=npz["is_attachment_cap"], name=f"{size}/{col}"
                )
                cell_titles[(row, col)] = f"N={size}, seed variant {col}"
        figures["pointclouds"] = make_pointcloud_grid_figure(
            traces_by_cell,
            n_rows=len(sizes),
            n_cols=n_variants,
            cell_titles=cell_titles,
            title=f"spine {spine_id} - 4) point clouds (red = attachment cap)",
        )
    else:
        skip("4) point clouds", first_pointcloud_path)

    sdf_path = sdf_samples_path(record, cfg)
    if sdf_path.exists() and local_sealed is not None:
        sdf_npz = np.load(sdf_path)
        figures["sdf"] = make_scene_figure(
            [
                mesh_trace(local_sealed, name="local sealed (ref)", color="lightgray", opacity=0.12),
                *sdf_scatter_traces(sdf_npz["query_points"], sdf_npz["sdf"], sdf_npz["sample_type"]),
            ],
            title=f"spine {spine_id} - 5) SDF samples (color = signed distance, negative = inside)",
        )
    else:
        skip("5) SDF samples", sdf_path)

    if show:
        for figure in figures.values():
            figure.show()
    return figures


def make_pointcloud_grid_figure(
    traces_by_cell: Dict[tuple, Any],
    *,
    n_rows: int,
    n_cols: int,
    cell_titles: Dict[tuple, str],
    title: str,
    width: int = 1100,
    height: int = 900,
):
    """A grid of 3D scenes, one per ``(row, col)`` key in ``traces_by_cell``."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"}] * n_cols for _ in range(n_rows)],
        subplot_titles=[cell_titles.get((r, c), "") for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)],
    )
    for (row, col), trace in traces_by_cell.items():
        fig.add_trace(trace, row=row, col=col)
    scene_updates = {}
    for i in range(1, n_rows * n_cols + 1):
        key = "scene" if i == 1 else f"scene{i}"
        scene_updates[key] = dict(aspectmode="data")
    fig.update_layout(title=title, width=width, height=height, showlegend=False, **scene_updates)
    return fig
