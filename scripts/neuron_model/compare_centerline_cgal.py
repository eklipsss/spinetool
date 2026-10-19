"""Compare our centreline (section circles + smoothing spline) with the CGAL mean-curvature-flow skeleton of the SpineTool2 method.

SpineTool2 (``projects/segmentation/utils/data_processing/mesh.py::create_mesh_descriptions``): the whole closed mesh
(dendrite WITH spines, holes filled) -> ``surface_mesh_skeletonization`` -> graph -> the longest path = the dendrite skeleton
(the two Euclidean-farthest skeleton nodes + the shortest path between them), ``correspondence`` maps surface vertices to
skeleton points (the distance = a radius estimate).

    python scripts/neuron_model/compare_centerline_cgal.py --raw-root datasets/minnie65 --neuron-id <id> \
        --ours-root <output_root of run_branch_preprocessing.py> --merged-branch-id 10081 [--merged-branch-id ...]

For every branch it prints how the two lines differ (distance between them, whether they lie inside the mesh, how well
they are centred, how jagged they are) and the run times; writes ``compare_<id>.html`` (3D: mesh, our line, CGAL path)
next to ``--out``. Small checks only on the mac (single process); the mesh is closed by ``repair_mesh`` (needed by CGAL, NOT by
our method, which uses only the vertices of the shaft).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def load_members(raw_root: Path, neuron: str, limb: int, members) -> tuple:
    import trimesh

    vs, fs, off = [], [], 0
    for b in members:
        m = trimesh.load(raw_root / neuron / f"limb_{limb:03d}" / f"branch_{int(b):03d}" / "branch_mesh.off", process=False)
        vs.append(np.asarray(m.vertices, float))
        fs.append(np.asarray(m.faces) + off)
        off += len(m.vertices)
    return np.vstack(vs), np.vstack(fs)


def close_mesh(vertices, faces):
    """What SpineTool2 does before the skeleton (stitch borders + fill holes): the mesh must be closed for CGAL."""
    import trimesh

    from src.dendrite_analysis.mesh_repair import repair_mesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    repaired, _, after = repair_mesh(mesh, verbose=False)
    return repaired, bool(repaired.is_watertight)


def longest_path_spinetool2(segments):
    """find_longest_path of SpineTool2: the Euclidean-farthest pair of skeleton nodes + the shortest path between them."""
    g = nx.Graph()
    key = lambda p: tuple(np.round(np.asarray(p, float), 6))
    for a, b in segments:
        ka, kb = key(a), key(b)
        g.add_edge(ka, kb, weight=float(np.linalg.norm(np.asarray(a) - np.asarray(b))))
    # SpineTool2 stops with "Unable to reconstruct connected mesh" when the skeleton falls apart; here the component with the
    # longest extent is used and the number of components is reported
    comps = sorted(nx.connected_components(g), key=lambda c: -np.ptp(np.array(list(c)), axis=0).max())
    g1 = g.subgraph(comps[0]).copy()
    nodes = list(g1.nodes())
    xyz = np.array(nodes)
    d = squareform(pdist(xyz))
    i, j = np.unravel_index(d.argmax(), d.shape)
    path = nx.shortest_path(g1, nodes[i], nodes[j], weight="weight")
    leaves = sum(1 for n in g1.nodes() if g1.degree(n) == 1)
    return np.array(path), len(g.nodes()), leaves, len(comps)


def resample(poly: np.ndarray, step: float) -> np.ndarray:
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(t, s, poly[:, k]) for k in range(3)])


def jaggedness(poly: np.ndarray, step: float = 500.0) -> float:
    """Median deviation of a point from the midpoint of its neighbours at ``step`` spacing (nm); a straight/smooth line -> small."""
    p = resample(poly, step)
    return float(np.median(np.linalg.norm(p[1:-1] - 0.5 * (p[:-2] + p[2:]), axis=1))) if len(p) > 2 else float("nan")


def stats(x) -> str:
    x = np.asarray(x, float)
    return f"median {np.median(x):.0f}, 90 % {np.percentile(x, 90):.0f}, max {x.max():.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True, help="folder with the neurons (<root>/<neuron_id>/limb_xxx/...)")
    parser.add_argument("--neuron-id", required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument("--merged-branch-id", type=int, action="append", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/interim/neuron-model/cgal-compare")
    args = parser.parse_args()

    from src.neuron_model.spine_geometry import cgal_skeleton_segments_from_trimesh_isolated
    from src.neuron_model.spine_sampling import winding_numbers

    table = pd.read_parquet(args.ours_root / args.neuron_id / "branches.parquet")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in args.merged_branch_id:
        row = table[table["merged_branch_id"] == mid].iloc[0]
        if row["status"] != "ok":
            print(f"\n=== merged id {mid}: status {row['status']} - no centreline of ours to compare, skipped ===")
            continue
        limb = int(row["limb_id"])
        members = [int(x) for x in str(row["member_branch_ids"]).split("|")]
        print(f"\n=== limb_{limb:03d}, branch_{', '.join(f'{b:03d}' for b in members)} (merged id {mid}): L {row['length_nm'] / 1000:.1f} um, r_med {row['radius_median_nm']:.0f} nm, status {row['status']} ===")
        geo = np.load(args.ours_root / args.neuron_id / f"limb_{limb:03d}" / f"merged_{mid}" / "branch_geometry.npz")
        ours = geo["gamma"]
        v, f = load_members(args.raw_root, args.neuron_id, limb, members)
        t0 = time.perf_counter()
        closed, watertight = close_mesh(v, f)
        t_close = time.perf_counter() - t0
        print(f"mesh: {len(v)} vertices, {len(f)} faces; closed for CGAL: watertight {watertight} ({t_close:.1f} s)")
        rec = {"merged_id": mid, "r_med_nm": float(row["radius_median_nm"]), "length_um": float(row["length_nm"]) / 1000, "closed": watertight}
        rows.append(rec)
        if not watertight:
            print("  CGAL needs a closed mesh: skipped")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            off = Path(tmp) / "mesh.off"
            closed.export(off)
            t0 = time.perf_counter()
            try:
                segments = cgal_skeleton_segments_from_trimesh_isolated(off, timeout=args.timeout)
            except Exception as exc:
                print("  CGAL skeletonization failed:", str(exc)[:200])
                rec["cgal_failed"] = True
                continue
            t_cgal = time.perf_counter() - t0
        path, n_nodes, n_leaves, n_comp = longest_path_spinetool2(segments)
        print(f"CGAL MCF: {t_cgal:.1f} s, skeleton {n_nodes} nodes in {n_comp} connected component(s), {n_leaves} leaves in the main one (spines + ends), longest path {len(path)} nodes")

        ours_d, cgal_d = resample(ours, 20.0), resample(path, 20.0)
        L_ours = float(np.sum(np.linalg.norm(np.diff(ours, axis=0), axis=1)))
        L_cgal = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        print(f"length of the line: ours {L_ours / 1000:.2f} um, CGAL path {L_cgal / 1000:.2f} um; table length {row['table_length_nm'] / 1000:.2f} um")

        rec.update(n_components=n_comp, t_cgal_s=round(t_cgal, 2), len_ours_um=L_ours / 1000, len_cgal_um=L_cgal / 1000, len_table_um=float(row["table_length_nm"]) / 1000)
        d_oc = cKDTree(cgal_d).query(ours_d)[0]
        d_co = cKDTree(ours_d).query(cgal_d)[0]
        rec["dist_median_nm"] = float(np.median(d_oc))
        print(f"distance of OUR line to the CGAL path, nm:  {stats(d_oc)}")
        print(f"distance of the CGAL path to OUR line, nm:  {stats(d_co)}  (large where the CGAL path extends beyond ours: ends)")

        tri = np.asarray(closed.vertices)[np.asarray(closed.faces)]
        for name, pts in (("ours", ours_d), ("CGAL", cgal_d)):
            inside = float((winding_numbers(pts, tri) > 0.5).mean())
            rec[f"inside_{name}"] = inside
            print(f"  {name:5s} inside the closed mesh: {inside:.0%}")
        # centring: nearest-vertex distance along the line relative to our local radius (our circle fits are centred by construction)
        tree = cKDTree(np.asarray(v))
        r_local = np.interp(np.linspace(0, 1, len(ours_d)), np.linspace(0, 1, len(geo["radius"])), geo["radius"])
        r_cgal_vertex = tree.query(cgal_d)[0]
        r_ours_vertex = tree.query(ours_d)[0]
        rec["wall_over_r_ours"] = float(np.median(r_ours_vertex / r_local))
        rec["wall_over_r_cgal"] = float(np.median(r_cgal_vertex / np.interp(np.linspace(0, 1, len(cgal_d)), np.linspace(0, 1, len(geo["radius"])), geo["radius"])))
        rec["jag_ours_nm"], rec["jag_cgal_nm"], rec["jag_raw_nm"] = jaggedness(ours), jaggedness(path), jaggedness(geo["skeleton"])
        print(f"  nearest mesh vertex distance / our radius: ours median {np.median(r_ours_vertex / r_local):.2f} (10-90 %: {np.percentile(r_ours_vertex / r_local, 10):.2f}-{np.percentile(r_ours_vertex / r_local, 90):.2f}); "
              f"CGAL median {np.median(r_cgal_vertex / np.interp(np.linspace(0, 1, len(cgal_d)), np.linspace(0, 1, len(geo['radius'])), geo['radius'])):.2f}")
        print(f"  jaggedness at 500 nm (median deviation from the neighbours' midpoint), nm: ours {jaggedness(ours):.0f}, CGAL {jaggedness(path):.0f}; raw skeleton in the data: {jaggedness(geo['skeleton']):.0f}")
        # a figure
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color="lightgrey", opacity=0.2, name="mesh"))
            fig.add_trace(go.Scatter3d(x=ours[:, 0], y=ours[:, 1], z=ours[:, 2], mode="lines", line=dict(color="royalblue", width=6), name="our centreline"))
            fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines+markers", marker=dict(size=2), line=dict(color="crimson", width=4), name="CGAL longest path"))
            sk = np.asarray(segments).reshape(-1, 3)
            fig.add_trace(go.Scatter3d(x=sk[:, 0], y=sk[:, 1], z=sk[:, 2], mode="markers", marker=dict(size=1.5, color="black"), name="CGAL skeleton (all nodes)", visible="legendonly"))
            fig.update_layout(title=f"limb_{limb:03d} merged {mid}: blue = ours, red = CGAL", scene=dict(aspectmode="data"), height=800)
            out = args.out / f"compare_{args.neuron_id}_{limb:03d}_{mid}.html"
            fig.write_html(out)
            print("  figure:", out)
        except Exception as exc:
            print("  no figure:", exc)
    if rows:
        frame = pd.DataFrame(rows).round(2)
        print("\n==== summary ====")
        print(frame.to_string(index=False))
        frame.to_csv(args.out / f"summary_{args.neuron_id}.csv", index=False)


if __name__ == "__main__":
    main()
