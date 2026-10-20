"""Look at assembled branches in the browser: one interactive HTML page per branch set (rotate / zoom / hide by legend).

    python scripts/neuron_model/view_assembled_branches.py --assembly <tp-dir>/assembly_<date> [--max 6]

Each branch of ``assemble_branches.py`` is drawn in its own panel (title: index, length, spines placed / dropped,
watertight, components). Output: ``<assembly>/view_branches.html``. Only reads files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembly", type=Path, required=True, help="output directory of assemble_branches.py")
    parser.add_argument("--max", type=int, default=6, help="at most this many branches (one panel each; meshes are large)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import plotly.graph_objects as go
    import trimesh
    from plotly.subplots import make_subplots

    bt = pd.read_parquet(args.assembly / "assembly_branches.parquet")
    bt = bt[bt["mesh_path"].notna()].head(args.max) if "mesh_path" in bt.columns else bt.iloc[:0]
    if not len(bt):
        raise SystemExit("no assembled meshes")
    st = pd.read_parquet(args.assembly / "assembly_spines.parquet") if (args.assembly / "assembly_spines.parquet").exists() else pd.DataFrame()
    titles = []
    for r in bt.itertuples():
        sub = st[st["branch_index"] == r.branch_index] if len(st) else st
        placed = int((sub["status"] == "placed").sum()) if len(sub) else 0
        dropped = int((sub["status"] == "dropped").sum()) if len(sub) else 0
        length = getattr(r, "length_nm", float("nan"))
        titles.append(f"#{r.branch_index}: {length / 1000.0:.1f} um, spines {placed} placed / {dropped} dropped, "
                      f"watertight {getattr(r, 'watertight', '?')}, components {getattr(r, 'n_components', '?')}")
    fig = make_subplots(rows=len(bt), cols=1, specs=[[{"type": "scene"}]] * len(bt), subplot_titles=titles, vertical_spacing=0.02)
    for k, r in enumerate(bt.itertuples(), start=1):
        m = trimesh.load(args.assembly / r.mesh_path, process=False)
        v, f = np.asarray(m.vertices) / 1000.0, np.asarray(m.faces)
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color="#d98cb3", opacity=1.0,
                                flatshading=False, lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.2}, name=f"#{r.branch_index}"), row=k, col=1)
        fig.update_scenes(aspectmode="data", xaxis_title="x, um", yaxis_title="y, um", zaxis_title="z, um", row=k, col=1)
    fig.update_layout(height=650 * len(bt), showlegend=False, margin={"l": 0, "r": 0, "t": 40, "b": 0})
    out = args.out or args.assembly / "view_branches.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"open in a browser: {out}")


if __name__ == "__main__":
    main()
