"""Which centreline smoothing keeps the line in the tube? (input for the adaptive smoothing of the branch preprocessing)

Runs on the output of ``run_branch_preprocessing.py`` (the saved section centres ``branch_sections.npz``) + the branch
meshes; nothing is rewritten. For a stratified sample of status-ok branches (radius classes < 150, 150-250, >= 250 nm)
the centreline is rebuilt with every variant of the smoothing width h and node step:

    fixed_1000   h 1000 nm, step 500 nm  (the first version)
    fixed_500    h 500,  step 500        fixed_500_s250  h 500, step 250        fixed_250  h 250, step 250
    adapt_k2/3/4 h = clip(k r_med, 250, 1000), step = clip(h / 2, 250, 500)

and measured:

    inside      fraction of the node polyline inside the branch mesh (the problem: a 1 um smoothing leaves thin tubes)
    tube_err    median | distance of a mesh vertex to the line - r(s) | / r_med for the vertices within 2.5 r_med of the line
                (how well the round tube of THIS line describes the surface; the same vertices for every variant)
    rms         RMS of the line to the reliable centres, nm
    kappa_rms   RMS bending, 1/um (smaller h -> more of the sub-micron winding becomes bending)
    roundtrip   max error of curve -> bending -> curve, nm
    nodes       number of nodes (T representation <= 512)

    python scripts/neuron_model/study_centerline_smoothing.py --root C:/.../preprocessed_branches --raw-data-root O:/Datasets [--per-class 30]

Writes ``<root>/audit/centerline_smoothing_study.csv`` (per branch and variant) and prints the medians per class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.neuron_model import branch_geometry as bg
from src.neuron_model.branch_preprocessing.process import centerline_geometry, line_inside_fraction

CLASSES = (("r < 150 nm", 0.0, 150.0), ("150-250 nm", 150.0, 250.0), ("r >= 250 nm", 250.0, 1e9))


def variants():
    out = [("fixed_1000", lambda r: 1000.0, lambda h: 500.0), ("fixed_500", lambda r: 500.0, lambda h: 500.0),
           ("fixed_500_s250", lambda r: 500.0, lambda h: 250.0), ("fixed_250", lambda r: 250.0, lambda h: 250.0)]
    for k in (2.0, 3.0, 4.0):
        out.append((f"adapt_k{k:g}", (lambda k: lambda r: float(np.clip(k * r, 250.0, 1000.0)))(k), lambda h: float(np.clip(0.5 * h, 250.0, 500.0))))
    return out


def load_mesh(raw_root: Path, dataset: str, row) -> tuple:
    import trimesh

    vs, fs, off = [], [], 0
    for b in [int(x) for x in str(row["member_branch_ids"]).split("|") if x]:
        m = trimesh.load(raw_root / dataset / str(row["neuron_id"]) / f"limb_{int(row['limb_id']):03d}" / f"branch_{b:03d}" / "branch_mesh.off", process=False)
        vs.append(np.asarray(m.vertices, float))
        fs.append(np.asarray(m.faces) + off)
        off += len(m.vertices)
    return np.vstack(vs), np.vstack(fs)


def tube_error(vertices: np.ndarray, gamma: np.ndarray, radius: np.ndarray, r_med: float, margin: float) -> float:
    s = bg.arclength(gamma)
    dense_s = np.arange(0.0, s[-1] + 1e-9, 25.0)
    dense = np.column_stack([np.interp(dense_s, s, gamma[:, k]) for k in range(3)])
    d, idx = cKDTree(dense).query(vertices)
    s_v = dense_s[idx]
    keep = (d < 2.5 * r_med) & (s_v > margin) & (s_v < s[-1] - margin)
    if keep.sum() < 20:
        return float("nan")
    r_s = np.interp(s_v[keep], s, radius)
    return float(np.median(np.abs(d[keep] - r_s)) / r_med)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="output root of run_branch_preprocessing.py")
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--dataset", default="minnie65")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from tqdm.auto import tqdm

    table = pd.read_parquet(args.root / "branches.parquet")
    table = table[(table["status"] == "ok") & table["radius_median_nm"].notna()]
    picked = []
    for name, lo, hi in CLASSES:
        sub = table[(table["radius_median_nm"] >= lo) & (table["radius_median_nm"] < hi)]
        picked.append(sub.sample(min(args.per_class, len(sub)), random_state=args.seed).assign(radius_class=name))
    sample = pd.concat(picked)
    print(f"{len(sample)} branches: " + ", ".join(f"{n} {int((sample['radius_class'] == n).sum())}" for n, _, _ in CLASSES), flush=True)
    rows = []
    for row in tqdm(sample.to_dict("records"), desc="branches"):
        d = args.root / str(row["neuron_id"]) / f"limb_{int(row['limb_id']):03d}" / f"merged_{int(row['merged_branch_id'])}"
        try:
            sec = np.load(d / "branch_sections.npz")
            v, f = load_mesh(args.raw_data_root, args.dataset, row)
        except Exception as exc:
            print("skip", row["merged_branch_id"], repr(exc)[:120])
            continue
        ok = np.isfinite(sec["radii"]) & (sec["flags"] == 0)
        if ok.sum() < 5:
            continue
        r_med = float(row["radius_median_nm"])
        margin = 2.0 * r_med
        tri = v[f]
        for name, h_of_r, step_of_h in variants():
            h = h_of_r(r_med)
            try:
                g = centerline_geometry(sec["centers"][ok], sec["radii"][ok], smoothing_nm=h, radius_smoothing_nm=max(1000.0, h), node_step_nm=step_of_h(h), straight_fraction=0.02)
            except Exception as exc:
                print("variant failed", name, repr(exc)[:100])
                continue
            k = np.hypot(g["kappa1"], g["kappa2"])
            rows.append({
                "neuron_id": row["neuron_id"], "limb_id": row["limb_id"], "merged_branch_id": row["merged_branch_id"], "radius_class": row["radius_class"],
                "r_med_nm": r_med, "length_um": g["length"] / 1000.0, "variant": name, "h_nm": h, "step_nm": g["step"], "nodes": len(g["nodes"]),
                "inside": line_inside_fraction(g["gamma"], tri, margin=margin),
                "tube_err": tube_error(v, g["gamma"], g["radius"], r_med, margin),
                "rms_nm": g["centerline"].rms_to_centers,
                "kappa_rms_per_um": float(np.sqrt(np.mean(k ** 2))) * 1000.0 if len(k) else 0.0,
                "roundtrip_nm": float(np.max(np.linalg.norm(g["rebuilt"] - g["gamma"], axis=1))),
            })
    res = pd.DataFrame(rows)
    out = args.root / "audit" / "centerline_smoothing_study.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    order = [n for n, _, _ in variants()]
    for name, _, _ in CLASSES:
        sub = res[res["radius_class"] == name]
        if not len(sub):
            continue
        med = sub.groupby("variant")[["h_nm", "step_nm", "inside", "tube_err", "rms_nm", "kappa_rms_per_um", "roundtrip_nm", "nodes"]].median().reindex(order)
        frac90 = sub.groupby("variant")["inside"].apply(lambda x: float((x >= 0.9).mean())).reindex(order).rename("share_inside>=0.9")
        print(f"\n== {name}: {sub['merged_branch_id'].nunique()} branches (medians) ==")
        print(pd.concat([med, frac90], axis=1).round(3).to_string())
    print(f"\nper branch: {out}")


if __name__ == "__main__":
    main()
