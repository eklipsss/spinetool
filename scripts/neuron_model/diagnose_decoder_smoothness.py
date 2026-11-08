"""Is the VAE decoder field smooth in the code, and do prior samples fall where it has no neck? (chain N diagnostic)

Evidence so far (chains K-M): the decoder draws necks for the code fitted to a spine (reconstruction bias 0.06-0.09 SD,
ConvexHullRatio ~ real), a flow prior reproduces the distribution of such codes (holes 1.02-1.06 vs the floor 1.00-1.03),
and still the meshes drawn from the flow have no necks (single component in 70-72 %, intact ones as convex as with encoder
codes). Midpoints of code pairs did not break (chain L). Two explanations remain and are separated here, WITHOUT training,
on the codes of the training spines fitted with the anchor (``fit_vae_codes.py --anchor-weight 1``, folder --codes-dir):

  A  decoder not smooth in the code  -> a small shift of a fitted code (c + s * sigma * eps, s = 0.25 ... 2) already fills
     or breaks the neck: single-component share and ConvexHullRatio change with s, the mesh moves by tens of nm per 0.25 sigma.
  B  samples fall BETWEEN fitted codes where the field is unconstrained -> perturbations of the fitted codes keep the
     neck, flow samples do not, and pulling a sample toward its nearest fitted code (z + lambda * (c_nn - z)) restores it.

Experiments (every mesh = decoder -> Marching Cubes on the frozen bbox -> the generation's postprocessing):
  perturb  c_i + s * sigma_i * eps_i (fixed eps per spine), s in --scales (s = 0 = the fitted code, the reference)
  lerp     (1 - t) * c_a + t * c_b for random pairs, t in --lerp-t (t = 0 = c_a, the reference)
  flow     z ~ flow prior (T = 1), z + lambda * (c_nn - z) with the nearest fitted code c_nn, lambda in --pull (1 = the code
           itself, the reference); also |z - c_nn| in units of the median nearest-neighbour distance of fitted codes

Per (experiment, condition): single-component and valid shares, median ConvexHullRatio (lower = more convex, necks filled)
and its paired change from the reference, median Chamfer-L1 (nm) to the reference mesh of the same spine. A reading:
  perturb: ConvexHullRatio / single-component fall with s already at 0.25-0.5 and Chamfer is large  -> A (decoder)
  perturb flat, flow bad, pulling toward c_nn restores the neck (lambda 0.5 and 1)                    -> B (prior / structure)
  both bad: the neck exists only at the fitted code itself (an isolated point of the field)           -> A and B

    python scripts/neuron_model/diagnose_decoder_smoothness.py --run runs/training/neuron-model/2026-10-19_003408_vae-dev-reg0-cos \
        --codes-dir latent_prior_zopt_a1 --n 150

Writes <run>/smoothness_<codes-dir>/{meshes.csv, summary.csv, report.json, summary.md}. GPU a few minutes, morphometrics ~10-20 min.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402,F401  (torch first: Windows DLL order)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRIC = "ConvexHullRatio"


def summarise(meshes: pd.DataFrame) -> pd.DataFrame:
    """Per (experiment, condition) shares and medians; paired change from the reference condition of the experiment."""
    rows = []
    for (exp, cond), g in meshes.groupby(["experiment", "condition"], sort=False):
        ref = meshes[(meshes.experiment == exp) & meshes.is_reference & (meshes.idx.isin(g.idx))].set_index("idx")
        mine = g.set_index("idx")
        paired = (mine[METRIC] - ref[METRIC].reindex(mine.index)).dropna() if METRIC in mine else pd.Series(dtype=float)
        rows.append({"experiment": exp, "condition": cond, "n": len(g), "single_component": float(g["single_component"].mean()),
                     "valid": float(g["valid"].mean()), f"median_{METRIC}": float(g[METRIC].median()),
                     f"paired_change_{METRIC}": float(paired.median()) if len(paired) else float("nan"),
                     "median_chamfer_nm": float(g["chamfer_to_reference_nm"].median()),
                     "median_dist_ratio": float(g["dist_ratio"].median()) if "dist_ratio" in g and g["dist_ratio"].notna().any() else float("nan")})
    return pd.DataFrame(rows)


def to_markdown(summary: pd.DataFrame) -> str:
    cols = ["experiment", "condition", "n", "single_component", "valid", f"median_{METRIC}", f"paired_change_{METRIC}", "median_chamfer_nm", "median_dist_ratio"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in summary[cols].to_dict("records"):
        lines.append("| " + " | ".join(str(r[c]) if isinstance(r[c], (str, int)) else ("" if not np.isfinite(r[c]) else f"{r[c]:.3f}") for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    from src.neuron_model.spine_generation.data.bbox import FrozenBBox
    from src.neuron_model.spine_generation.evaluation.attachment import finder_from_config
    from src.neuron_model.spine_generation.evaluation.morphometrics import generated_morphometrics
    from src.neuron_model.spine_generation.experiment.artifacts import write_json
    from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path
    from src.neuron_model.spine_generation.models.spine_vae.decoder_study import surface_distances
    from src.neuron_model.spine_generation.models.spine_vae.generate import latent_to_mesh
    from src.neuron_model.spine_generation.models.spine_vae.latent_prior import FlowPrior, load_vae_run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--codes-dir", default="latent_prior_zopt_a1", help="folder of the run with codes_train.npz (fit_vae_codes.py) and flow_prior.pt")
    ap.add_argument("--flow", type=Path, help="flow_prior.pt (default <run>/<codes-dir>/flow_prior.pt)")
    ap.add_argument("--n", type=int, default=150, help="spines / pairs / flow samples per experiment")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--lerp-t", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--pull", type=float, nargs="+", default=[0.0, 0.5], help="lambda; 1 (the reference, the nearest code) is always added")
    ap.add_argument("--bbox", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    started = time.time()

    model, run, cfg, ckpt, device = load_vae_run(args.run, args.checkpoint)
    codes = np.load(run.root / args.codes_dir / "codes_train.npz", allow_pickle=False)
    c = np.asarray(codes["mu"], dtype=np.float64)[:, 0]
    sigma = np.exp(0.5 * np.asarray(codes["logvar"], dtype=np.float64)[:, 0])
    n_codes, dim = c.shape
    n = min(args.n, n_codes)
    flow_path = args.flow or run.root / args.codes_dir / "flow_prior.pt"
    flow = FlowPrior.load(flow_path)
    coord_scale = float(cfg["data"]["coord_scale"])
    bbox = FrozenBBox.load(args.bbox or resolve_path(cfg["data"]["bbox"]))
    box = (tuple(bbox.low), tuple(bbox.high))
    mc_cfg = load_config(resolve_path(cfg["reconstruction"]["marching_cubes_config"]))
    post_cfg, chunk = dict(mc_cfg["postprocess"]), int(mc_cfg.get("chunk_size", 262_144))
    finder = finder_from_config(load_config(REPO_ROOT / "configs/neuron-model/evaluation/default.yaml")["morphometrics"].get("attachment_finder"))
    out = args.out or (run.root / f"smoothness_{args.codes_dir}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[smooth] {ckpt}: {n_codes} fitted codes (dim {dim}), flow {flow_path.name}, {n} per experiment, MC {args.resolution}^3", flush=True)

    rng = np.random.default_rng(args.seed)
    meshes, records = {}, []

    def decode(z):
        res = latent_to_mesh(model, np.asarray(z, dtype=np.float64), bbox=box, resolution=args.resolution, coord_scale=coord_scale, device=device,
                             chunk_size=chunk, iso_level=float(mc_cfg.get("iso_level", 0.0)), postprocess_cfg=post_cfg, check_self_intersections=False)
        return res["mesh_postprocessed"], {"single_component": bool(res["validation_raw"].get("n_connected_components") == 1),
                                           "valid": bool(res["validation_postprocessed"].get("is_valid"))}

    def add(exp, cond, idx, z, is_ref, **extra):
        mesh, flags = decode(z)
        meshes[(exp, cond, idx)] = mesh
        records.append({"experiment": exp, "condition": cond, "idx": idx, "is_reference": is_ref, **flags, **extra})

    spines = rng.choice(n_codes, n, replace=False)
    eps = rng.standard_normal((n, dim))
    for i, k in enumerate(spines):
        add("perturb", "s=0", i, c[k], True)
        for s in args.scales:
            add("perturb", f"s={s:g}", i, c[k] + s * sigma[k] * eps[i], False)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  perturb {i + 1}/{n} ({time.time() - started:.0f} s)", flush=True)
    pairs = [tuple(rng.choice(n_codes, 2, replace=False)) for _ in range(n)]
    for i, (a, b) in enumerate(pairs):
        add("lerp", "t=0", i, c[a], True)
        for t in args.lerp_t:
            add("lerp", f"t={t:g}", i, (1 - t) * c[a] + t * c[b], False)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  lerp {i + 1}/{n} ({time.time() - started:.0f} s)", flush=True)

    from scipy.spatial import cKDTree

    tree = cKDTree(c)
    probe = c[rng.choice(n_codes, min(2000, n_codes), replace=False)]
    nn_scale = float(np.median(tree.query(probe, k=2)[0][:, 1]))  # distance to the nearest OTHER fitted code
    z_flow = flow.sample(n, temperature=1.0, seed=args.seed)
    d_nn, nn = tree.query(z_flow)
    print(f"[smooth] median nearest-neighbour distance of fitted codes {nn_scale:.3f}; flow samples are {np.median(d_nn) / nn_scale:.2f} of it from their nearest code", flush=True)
    for i in range(n):
        ratio = float(d_nn[i] / nn_scale)
        add("flow", "nearest code (lambda=1)", i, c[nn[i]], True, dist_ratio=ratio)
        for lam in args.pull:
            add("flow", "sample (lambda=0)" if lam == 0 else f"lambda={lam:g}", i, z_flow[i] + lam * (c[nn[i]] - z_flow[i]), False, dist_ratio=ratio)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  flow {i + 1}/{n} ({time.time() - started:.0f} s)", flush=True)

    frame = pd.DataFrame(records)
    ref_cond = {"perturb": "s=0", "lerp": "t=0", "flow": "nearest code (lambda=1)"}
    chamfer = []
    for r in records:
        ref = meshes[(r["experiment"], ref_cond[r["experiment"]], r["idx"])]
        chamfer.append(0.0 if r["is_reference"] else surface_distances(meshes[(r["experiment"], r["condition"], r["idx"])], ref, seed=r["idx"])["chamfer_l1"])
    frame["chamfer_to_reference_nm"] = chamfer
    keys = [(r["experiment"], r["condition"], r["idx"]) for r in records]
    morph = generated_morphometrics([meshes[k] for k in keys], attachment_finder=finder, compute_chords=False, sample_ids=[str(i) for i in range(len(keys))])
    frame[METRIC] = pd.to_numeric(morph.set_index("sample_id")[METRIC].reindex([str(i) for i in range(len(keys))]).to_numpy(), errors="coerce")
    frame.to_csv(out / "meshes.csv", index=False)
    summary = summarise(frame)
    summary.to_csv(out / "summary.csv", index=False)
    md = to_markdown(summary)
    (out / "summary.md").write_text(f"# Decoder smoothness diagnostic ({ckpt}, codes {args.codes_dir})\n\nmedian nearest-neighbour distance of fitted codes {nn_scale:.3f}\n\n{md}\n", encoding="utf-8")
    write_json(out / "report.json", {"run": str(run.root), "checkpoint": ckpt, "codes_dir": args.codes_dir, "n": n, "seed": args.seed, "resolution": args.resolution,
                                     "nn_scale": nn_scale, "summary": json.loads(summary.to_json(orient="records")), "seconds": time.time() - started})
    print("\n" + md)
    print(f"-> {out}  ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
