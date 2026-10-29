"""Interactive atlas of spine generators: one self-contained HTML (opens offline, no server).

    python scripts/neuron_model/build_spine_atlas.py --vae-run runs/training/neuron-model/<vae_run> \
        --manifest O:/Datasets/Minnie65/preprocessed/manifest.parquet \
        --model VAE=runs/training/neuron-model/<vae_run>/meshes/seed_1 \
        --model MoGen=runs/training/neuron-model/<mogen_run>/meshes/seed_1 \
        [--n-real 1500 --n-gen 600 --n-train 1000 --traversal-spines 4]

What is inside (data: src/neuron_model/spine_generation/atlas.py; page: src/neuron_model/reports/assets/spine_atlas.js;
style: docs/neuron-model/html-materials-style-guide.md):

1. summary tiles and a comparison table of every generated set against the real reference;
2. a map of real and generated spines in the latent space of the VAE encoder (UMAP fitted on real spines) and in the
   space of standardised morphometrics (PCA), coloured by source, cell type, compartment, any morphometric or surface
   quality; a click opens the 3-D mesh and the morphometric profile (percentiles among real spines), a second spine can
   be pinned for side-by-side comparison;
3. distributions of every morphometric per source, the W1 / KS table, correlation matrices (real and the difference);
4. surface quality: validity rates, roughness, needle-like triangles, area, components;
5. latent traversals: meshes along the first principal directions of the real latent codes for a few real spines,
   with the chosen morphometric along the path against the real 10-90 % band.

The real reference is the ``--split`` (default test) of the VAE run's split; ``--n-train`` train spines are encoded only
for the memorisation ratio. Real meshes and morphometrics are read from ``--manifest`` (the NAS manifest: the training
cache holds point clouds only). Windows + GPU: ~10-20 min for the defaults (CGAL morphometrics of generated meshes and
the traversal meshes dominate). Output: ``runs/analysis/neuron-model/atlas/<date>/spine_atlas.html`` (10-20 MB).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  (first on Windows, see fit_t0.py)

import numpy as np
import pandas as pd

from src.neuron_model.reports import html_kit as hk
from src.neuron_model.reports.atlas_page import DEFAULT_TITLE, render_page
from src.neuron_model.spine_generation import atlas as A
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.config import resolve_path



def _model_arg(value: str):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("--model expects NAME=<seed directory with sample_* folders>")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae-run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--manifest", type=Path, nargs="+", required=True, help="manifest(s) with real meshes and morphometrics (NAS)")
    parser.add_argument("--path-mode", choices=("relative", "absolute"), help="default: data.path_mode of the VAE run; 'absolute' for a LOCAL COPY of the NAS manifest (its paths are O:\\...)")
    parser.add_argument("--splits", type=Path, help="default: data.splits of the VAE run")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-real", type=int, default=1500)
    parser.add_argument("--real-meshes", type=int, default=150)
    parser.add_argument("--n-train", type=int, default=1000, help="train spines encoded for the memorisation ratio (0 = off)")
    parser.add_argument("--model", type=_model_arg, action="append", default=[], help="NAME=<seed dir with sample_*> (repeatable, up to 3)")
    parser.add_argument("--n-gen", type=int, default=600)
    parser.add_argument("--gen-meshes", type=int, default=120)
    parser.add_argument("--mesh-kind", default="postprocessed", choices=["raw", "postprocessed"])
    parser.add_argument("--labid-manifest", type=Path, nargs="+", help="manifest(s) of the LabID S preprocessing (<converted tree>/labid/preprocessed/manifest.parquet): its spines are encoded by the same VAE and shown on both maps (not part of any comparison)")
    parser.add_argument("--labid-path-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--labid-groups", nargs="+", help="LabID groups to show (default: all present)")
    parser.add_argument("--labid-n", type=int, default=150, help="spines per LabID group")
    parser.add_argument("--labid-meshes", type=int, default=15, help="meshes per LabID group available in the viewer")
    parser.add_argument("--traversal-spines", type=int, default=4)
    parser.add_argument("--traversal-dirs", type=int, default=4)
    parser.add_argument("--traversal-steps", default="-2,-1,0,1,2")
    parser.add_argument("--traversal-resolution", type=int, default=96)
    parser.add_argument("--max-faces", type=int, default=2500, help="display meshes are decimated to at most this many faces")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.model) > 3:
        parser.error("at most 3 generated sets (the map palette validates three series for scatter plots)")

    vae = A.load_vae(args.vae_run, args.checkpoint)
    d = vae.cfg["data"]
    index = load_spine_index(list(args.manifest), path_mode=args.path_mode or d.get("path_mode", "relative"), check_files=())
    splits = load_splits(args.splits or resolve_path(d["splits"]))
    real_index = index.with_split(splits, args.split, drop_unassigned=True)
    if not len(real_index):
        raise SystemExit(f"no spines of split {args.split!r} in the manifest")
    print(f"[atlas] reference: {len(real_index)} spines of split {args.split}; VAE {vae.checkpoint_id}", flush=True)
    real = A.real_records(vae, real_index, args.n_real, seed=args.seed, n_meshes=args.real_meshes, source="real")
    train = []
    if args.n_train > 0:
        train_index = index.with_split(splits, "train", drop_unassigned=True)
        train = A.real_records(vae, train_index, args.n_train, seed=args.seed + 1, n_meshes=0, source="train") if len(train_index) else []
    gens = {name: A.generated_records(vae, path, name, n=args.n_gen, n_meshes=args.gen_meshes, mesh_kind=args.mesh_kind, seed=args.seed) for name, path in args.model}

    labid = {}
    if args.labid_manifest:
        labid_index = load_spine_index(list(args.labid_manifest), path_mode=args.labid_path_mode, check_files=())
        labid = A.labid_records(vae, labid_index, groups=args.labid_groups, n_per_group=args.labid_n, n_meshes=args.labid_meshes, seed=args.seed)
        print(f"[atlas] LabID: " + ", ".join(f"{g} {len(rs)}" for g, rs in labid.items()), flush=True)

    # ---- sources and points (real first, then the generated sets, then the LabID groups)
    colors = list(hk.SERIES)
    sources = [{"id": "real", "name": f"Реальные ({args.split})", "kind": "real", "color": hk.PALETTE["reference"], "n": len(real)}]
    for k, name in enumerate(gens):
        sources.append({"id": name, "name": name, "kind": "generated", "color": colors[k], "n": len(gens[name])})
    # LabID groups: own colours (the 3-colour validated palette + two more, each group also by the diamond marker: Minnie = circles)
    labid_colors = {"Wt": hk.PALETTE["series_2"], "Ab": hk.PALETTE["series_1"], "Wt+9009": "#2a6fb0", "Ab+9009": "#c46a00", "Wt_old": hk.PALETTE["series_3"]}
    for g, rs in labid.items():
        sources.append({"id": f"labid_{g}", "name": f"LabID {g}", "kind": "external", "symbol": "diamond", "color": labid_colors.get(g, "#555555"), "n": len(rs)})
    records = list(real) + [r for name in gens for r in gens[name]] + [r for g in labid for r in labid[g]]
    src_index = {s["id"]: i for i, s in enumerate(sources)}

    lat = lambda rs: np.array([r.latent for r in rs if r.latent is not None]) if any(r.latent is not None for r in rs) else np.zeros((0, 1))
    real_lat = lat(real)
    gen_lat = {name: lat(rs) for name, rs in gens.items()}
    lab_lat = {g: lat(rs) for g, rs in labid.items()}
    xy_real, xy_all, method = A.fit_map(real_lat, list(gen_lat.values()) + list(lab_lat.values()), method="umap", seed=args.seed)
    xy_gens, xy_lab = xy_all[: len(gens)], xy_all[len(gens):]
    xr, ref_stats = A.morph_matrix(real)
    xg = {name: A.morph_matrix(rs, ref_stats)[0] for name, rs in gens.items()}
    xl = {g: A.morph_matrix(rs, ref_stats)[0] for g, rs in labid.items()}
    pr, pall, _ = A.fit_map(xr, list(xg.values()) + list(xl.values()), method="pca", seed=args.seed)
    pg = pall[: len(gens)]
    pl = pall[len(gens):]

    def place(rs, coords):
        out, k = [], 0
        for r in rs:
            if r.latent is None:
                out.append(None)
            else:
                out.append([round(float(coords[k][0]), 4), round(float(coords[k][1]), 4)])
                k += 1
        return out

    latent_xy = place(real, xy_real)
    for (name, rs), c in zip(gens.items(), xy_gens):
        latent_xy += place(rs, c)
    for (g, rs), c in zip(labid.items(), xy_lab):
        latent_xy += place(rs, c)
    morph_xy = [[round(float(a), 4), round(float(b), 4)] for a, b in pr]
    for c in list(pg) + list(pl):
        morph_xy += [[round(float(a), 4), round(float(b), 4)] for a, b in c]

    meshes, mesh_ref = [], []
    for r in records:
        if r.mesh is not None and len(r.mesh.faces):
            mesh_ref.append(len(meshes))
            meshes.append(hk.pack_mesh(np.asarray(r.mesh.vertices), np.asarray(r.mesh.faces), max_faces=args.max_faces))
        else:
            mesh_ref.append(-1)

    metric_cols = list(A.METRIC_INFO)
    qual_cols = list(A.QUALITY_INFO)
    points = {
        "source": [src_index[r.source] for r in records],
        "key": [r.key for r in records],
        "cell_type": [r.meta.get("cell_type") for r in records],
        "ctb": [r.meta.get("cell_type_binary") for r in records],
        "compartment": [r.meta.get("compartment") for r in records],
        "valid": [r.meta.get("valid") for r in records],
        "group": [r.meta.get("labid_group") for r in records],
        "mesh": mesh_ref,
        "maps": {"latent": {"name": f"Латентное пространство VAE ({method})", "axis": method, "xy": latent_xy},
                 "morph": {"name": "Морфометрические признаки (PCA)", "axis": "ГК", "xy": morph_xy}},
        "metrics": {m: hk.clean_floats(r.metrics.get(m) for r in records) for m in metric_cols},
        "quality": {q: hk.clean_floats(r.quality.get(q) for r in records) for q in qual_cols},
    }

    # ---- comparison numbers
    comparison, summary, quality_rates = {}, [], {}
    real_vals = {m: np.array([np.nan if r.metrics.get(m) is None else r.metrics[m] for r in real], float) for m in metric_cols}
    train_lat = lat(train) if train else None
    for k, (name, rs) in enumerate(gens.items()):
        mets = {m: A.compare_metric(np.array([np.nan if r.metrics.get(m) is None else r.metrics[m] for r in rs], float), real_vals[m]) for m in metric_cols}
        corr = A.correlation_gap(np.column_stack([np.array([np.nan if r.metrics.get(m) is None else r.metrics[m] for r in rs], float) for m in metric_cols]),
                                 np.column_stack([real_vals[m] for m in metric_cols]))
        latent = A.latent_scores(gen_lat[name], real_lat, train_lat if train_lat is not None and len(train_lat) else None)
        comparison[name] = {"metrics": mets, "corr": corr, "latent": latent}
        w = [(v["w1_rel"], m) for m, v in mets.items() if v["w1_rel"] is not None]
        valid = np.mean([bool(r.meta.get("valid")) for r in rs]) if rs else None
        summary.append({"id": name, "name": name, "color": colors[k], "valid": None if valid is None else float(valid),
                        "w1": float(np.mean([x for x, _ in w])) if w else None, "worst": A.METRIC_INFO[max(w)[1]]["name"] if w else None,
                        "max_gap": corr["max_gap"], **latent})
        present = [r for r in rs if r.quality.get("mesh_present")]
        quality_rates[name] = {"valid": None if valid is None else float(valid),
                               "watertight": float(np.mean([r.quality["watertight"] for r in present])) if present else None,
                               "single": float(np.mean([r.quality.get("n_components") == 1 for r in present])) if present else None,
                               "attachment": float(np.mean([bool(r.meta.get("attachment_found")) for r in present])) if present else None}
    rq = [r for r in real if r.quality.get("mesh_present")]
    quality_rates["real"] = {"valid": float(np.mean([r.quality["watertight"] and r.quality.get("n_components") == 1 for r in rq])) if rq else None,
                             "watertight": float(np.mean([r.quality["watertight"] for r in rq])) if rq else None,
                             "single": float(np.mean([r.quality.get("n_components") == 1 for r in rq])) if rq else None, "attachment": None}

    # ---- latent traversal (meshes appended to the same mesh list)
    steps = [float(x) for x in args.traversal_steps.split(",")]
    traversal_payload = None
    if args.traversal_spines > 0 and len(real_lat) > 2:
        rng = np.random.default_rng(args.seed + 2)
        cand = [r for r in real if r.latent is not None]
        base = [(cand[i].key, cand[i].latent) for i in rng.permutation(len(cand))[: args.traversal_spines]]
        tr = A.traversal(vae, base, real_lat, n_directions=args.traversal_dirs, steps=steps, resolution=args.traversal_resolution)
        frames = []
        for b in range(len(base)):
            per_dir = []
            for dd in range(len(tr["directions"])):
                row = []
                for t in steps:
                    f = tr["frames"][(b, dd, t)]
                    idx = -1
                    if f["mesh"] is not None and len(f["mesh"].faces):
                        idx = len(meshes)
                        meshes.append(hk.pack_mesh(np.asarray(f["mesh"].vertices), np.asarray(f["mesh"].faces), max_faces=args.max_faces))
                    row.append({"mesh": idx, "valid": f["valid"], "metrics": {m: hk.clean_floats([f["metrics"].get(m)])[0] for m in metric_cols}})
                per_dir.append(row)
            frames.append(per_dir)
        traversal_payload = {"directions": tr["directions"], "steps": steps, "base": tr["base"], "frames": frames}

    neurons = {r.meta.get("neuron_id") for r in real if r.meta.get("neuron_id")}
    meta = {
        "title": args.title,
        "subtitle": "Реальные и сгенерированные шипики в общем пространстве форм, их морфометрические признаки, качество поверхности и латентные направления.",
        "meta_lines": [f"VAE: {vae.checkpoint_id}", f"Эталон: {len(real)} шипиков, сплит {args.split}", f"Построено {_dt.datetime.now():%d.%m.%Y %H:%M}"],
        "split": args.split, "n_neurons": len(neurons),
        "footer": (f"Эталон — {len(real)} случайных шипиков сплита {args.split} из {len(neurons)} нейронов. Для оценки копирования закодировано "
                   f"{len(train)} обучающих шипиков. Сгенерированные наборы: " + ", ".join(f"{n} — {len(rs)} образцов ({p})" for (n, p), rs in zip(args.model, gens.values()))
                   + (". Шипики LabID (ромбы; " + ", ".join(f"{g} — {len(rs)}" for g, rs in labid.items()) + ") закодированы тем же VAE и лишь проецируются на карты, построенные по шипикам Minnie65; в сравнения они не входят" if labid else "")
                   + f". Сетки в атласе упрощены для показа до {args.max_faces} треугольников, а все числа посчитаны по полным сеткам. Длины в мкм, объёмы в мкм³ (пересчёт из нм)."),
    }
    payload = {
        "meta": meta, "sources": sources, "points": points, "meshes": meshes, "summary": summary, "comparison": comparison, "quality_rates": quality_rates,
        "metric_info": A.METRIC_INFO, "quality_info": A.QUALITY_INFO, "traversal": traversal_payload,
        "categories": {"cell_type": sorted({v for v in points["cell_type"] if v}), "compartment": sorted({v for v in points["compartment"] if v})},
    }
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model/atlas" / f"{_dt.datetime.now():%Y-%m-%d_%H%M%S}" / "spine_atlas.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(args.title, meta, sources, summary, payload), encoding="utf-8")
    (out.parent / "atlas_summary.json").write_text(json.dumps({"summary": summary, "quality_rates": quality_rates}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[atlas] {len(records)} spines, {len(meshes)} meshes, {out.stat().st_size / 1e6:.1f} MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
