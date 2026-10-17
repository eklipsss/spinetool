"""Are the mesh normals in sdf_samples.npz consistent with the SDF values stored next to them?

For surface samples p the true SDF gradient is estimated from the DATA alone: a local least-squares plane
``sdf(x) ~ a + g . (x - p)`` over the stored samples (surface + near-surface) within ``--radius`` nm of p. The cosine
between g and the stored normal n says whether the labels the normal loss uses agree with the SDF the model is fitted to.

    python scripts/neuron_model/check_sdf_normals.py --root <cache dir or preprocessed root> [--n-spines 40]

cos mostly > 0.9      : labels are fine, a low model cosine means the decoder has not learned the gradient
cos spread / cos < 0  : the normals (interpolated vertex normals of a rough mesh, or inconsistent orientation) disagree
                         with the SDF -> the normal loss cannot be satisfied together with the SDF loss.
Units: nm. Reads only ``sdf_samples.npz`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def local_gradient_cos(path: Path, radius: float, max_surface: int, rng: np.random.Generator):
    z = np.load(path)
    q = np.asarray(z["query_points"], dtype=float)
    s = np.asarray(z["sdf"], dtype=float)
    t = np.asarray(z["sample_type"])
    n = np.asarray(z["surface_normals"], dtype=float)
    keep = t != 2  # surface + near-surface samples (uniform ones are too sparse)
    tree = cKDTree(q[keep])
    qs, ss = q[keep], s[keep]
    surface = np.flatnonzero((t == 0) & np.isfinite(n).all(axis=1))
    surface = rng.permutation(surface)[:max_surface]
    out = []
    for i in surface:
        idx = tree.query_ball_point(q[i], radius)
        if len(idx) < 16:
            continue
        a = np.hstack([np.ones((len(idx), 1)), qs[idx] - q[i]])
        coef, *_ = np.linalg.lstsq(a, ss[idx], rcond=None)
        g = coef[1:]
        norm = np.linalg.norm(g)
        if norm < 1e-9:
            continue
        out.append((float(g @ n[i] / (norm * np.linalg.norm(n[i]))), float(norm)))
    return out


def distance_ratio(path: Path, lo: float = 30.0, hi: float = 150.0):
    """|sdf(q)| / (distance from q to the nearest stored SURFACE sample) for near-surface samples with lo <= |sdf| <= hi.

    For a true distance field the ratio is <= 1 and close to 1 (the nearest stored surface sample is a little farther
    than the true closest surface point). A ratio well below 1 means the stored SDF is not a Euclidean distance (wrong
    scale / units / sign mixing), and the eikonal term (|grad| = 1) then contradicts the SDF targets."""
    z = np.load(path)
    q = np.asarray(z["query_points"], dtype=float)
    s = np.asarray(z["sdf"], dtype=float)
    t = np.asarray(z["sample_type"])
    surf = q[t == 0]
    near = (t == 1) & (np.abs(s) >= lo) & (np.abs(s) <= hi)
    if len(surf) < 100 or not near.any():
        return np.zeros(0), np.zeros(0)
    d, _ = cKDTree(surf).query(q[near])
    return np.abs(s[near]) / np.maximum(d, 1e-9), s[near]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--n-spines", type=int, default=40)
    p.add_argument("--radius", type=float, default=80.0, help="nm")
    p.add_argument("--max-surface", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    files = sorted(args.root.rglob("sdf_samples.npz"))
    if not files:
        raise SystemExit(f"no sdf_samples.npz under {args.root}")
    files = [files[i] for i in rng.permutation(len(files))[: args.n_spines]]
    rows = []
    for f in files:
        rows += local_gradient_cos(f, args.radius, args.max_surface, rng)
    cos = np.array([r[0] for r in rows])
    gn = np.array([r[1] for r in rows])
    print(f"{len(files)} spines, {len(cos)} surface samples with a local gradient estimate (radius {args.radius:.0f} nm)")
    print("cos(data gradient, stored normal): mean %.3f; >0.9: %.1f%%, 0.5..0.9: %.1f%%, |cos|<0.5: %.1f%%, <-0.5: %.1f%%"
          % (cos.mean(), 100 * (cos > 0.9).mean(), 100 * ((cos > 0.5) & (cos <= 0.9)).mean(), 100 * (np.abs(cos) < 0.5).mean(), 100 * (cos < -0.5).mean()))
    print("|data gradient| median %.2f (1.0 for a clean distance field), 5%%-95%%: %.2f..%.2f" % (np.median(gn), np.percentile(gn, 5), np.percentile(gn, 95)))
    ratios, values = zip(*[distance_ratio(f) for f in files])
    ratio, value = np.concatenate(ratios), np.concatenate(values)
    if len(ratio):
        print("|sdf| / distance to the nearest surface sample (near samples, 30..150 nm): median %.3f, 5%%-95%%: %.3f..%.3f, >1.01: %.1f%%  (true distance: <= 1, ~0.9-1)"
              % (np.median(ratio), np.percentile(ratio, 5), np.percentile(ratio, 95), 100 * (ratio > 1.01).mean()))
        for sign, name in ((1, "outside (sdf > 0)"), (-1, "inside  (sdf < 0)")):
            m = np.sign(value) == sign
            if m.any():
                print("   %s: n %d, median ratio %.3f" % (name, int(m.sum()), np.median(ratio[m])))


if __name__ == "__main__":
    main()
