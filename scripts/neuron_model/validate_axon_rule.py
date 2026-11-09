"""Check the axon labels of Minnie65 limbs against synapses (``src/neuron_model/axon_validation.py``).

Question: is the "thin and long" rule (limb radius < --radius-max-nm and cable > --cable-min-nm) a valid way to find the
axon of INHIBITORY neurons, whose dendrites have almost no spines? The labels of the colleague's table and of the branch
rule (thin + no spines) are checked the same way. Ground truth independent of geometry: presynaptic points of synapses
made by the neuron lie on its axon, postsynaptic shaft / spine points of synapses onto it lie on dendrites.

    python scripts/neuron_model/validate_axon_rule.py ^
        --limbs runs\\analysis\\neuron-model\\2026-11-03_005059_apical-by-direction-minnie65\\limbs.csv ^
        --dataset-root O:\\Datasets\\Minnie65 --metadata-root datasets

Reads only limb skeletons and soma meshes of the neurons that have annotated points (~100 MB from the NAS). Writes
``runs/analysis/neuron-model/<time>_axon-rule-check/{points.parquet, limb_evidence.csv, neurons.csv, summary.json,
summary.md}``. Statistics are per neuron (the unit), with a bootstrap over neurons.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INHIBITORY = {"BC", "MC", "BPC", "NGC", "Unsure I"}
LABELS = ("rule_axon", "axon_like_table", "axon_like_branches", "axon_like")


def main() -> None:
    from src.neuron_model.axon_validation import (assign_points, bootstrap_mean, branch_spine_counts, limb_evidence, load_synapse_points,
                                                  neuron_index, per_neuron)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limbs", type=Path, required=True, help="limbs.csv of apical_by_direction.py (limb_radius_nm, cable_nm, axon labels)")
    ap.add_argument("--dataset-root", type=Path, required=True, help="folder with the Minnie65 neuron folders")
    ap.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    ap.add_argument("--synapse-dir", type=Path, help="folder with vortex_compartment_targets.parquet (default <metadata-root>/minnie65/minnie65_metadata)")
    ap.add_argument("--radius-max-nm", type=float, default=180.0)
    ap.add_argument("--cable-min-nm", type=float, default=1.5e6)
    ap.add_argument("--max-dist-nm", type=float, default=3000.0, help="points farther from every limb / the soma are not assigned")
    ap.add_argument("--min-pre", type=int, default=3, help="neurons with at least this many assigned presynaptic points enter the per-neuron statistics")
    ap.add_argument("--min-dendrite-points", type=int, default=3)
    ap.add_argument("--neurons", nargs="*", type=int)
    ap.add_argument("--no-branches", action="store_true", help="limb skeletons only (one file per limb); default: branch skeletons, one file per branch, "
                    "to find the branch of every point and its spine count")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.time()

    limbs = pd.read_csv(args.limbs)
    limbs["neuron_id"] = limbs["neuron_id"].astype("int64")
    limbs["rule_axon"] = (limbs["limb_radius_nm"] < args.radius_max_nm) & (limbs["cable_nm"] > args.cable_min_nm)
    for c in LABELS:
        if c not in limbs.columns:
            limbs[c] = False
        limbs[c] = limbs[c].fillna(False).astype(bool)
    types = pd.read_csv(args.metadata_root / "minnie65" / "cell_types_with_nucleus.csv")
    cell_type = dict(zip(types["segment_id"].astype("int64"), types["cell_type"]))
    neurons = sorted(set(limbs["neuron_id"]) & set(args.neurons or limbs["neuron_id"]))
    cls = {n: ("inhibitory" if cell_type.get(n) in INHIBITORY else "excitatory" if isinstance(cell_type.get(n), str) else "unknown") for n in neurons}
    meta_dir = args.synapse_dir or args.metadata_root / "minnie65" / "minnie65_metadata"
    if not (meta_dir / "vortex_compartment_targets.parquet").exists():
        raise SystemExit(f"[axon] no vortex_compartment_targets.parquet in {meta_dir}: give --synapse-dir (folder minnie65_metadata of the Minnie65 tables)")
    points = load_synapse_points(meta_dir, neurons)
    with_points = sorted(set(points["neuron_id"]))
    print(f"[axon] {len(neurons)} neurons in {args.limbs.name}; annotated points {len(points)} on {len(with_points)} neurons: "
          f"{points['kind'].value_counts().to_dict()}", flush=True)

    def work(nid):
        d = args.dataset_root / str(nid)
        pts, lab, br = neuron_index(d, branches=not args.no_branches)
        p = points[points["neuron_id"] == nid]
        limb_id, branch_id, dist = assign_points(p[["x", "y", "z"]].to_numpy(), pts, lab, br)
        spines = branch_spine_counts(d)
        n_sp = [spines.get((int(a), int(b)), -1) if b >= 0 else -1 for a, b in zip(limb_id, branch_id)]
        return p.assign(limb_id=limb_id, branch_id=branch_id, branch_spines=n_sp, dist_nm=dist)

    with ThreadPoolExecutor(args.threads) as pool:
        parts = list(pool.map(work, with_points))
    assigned = pd.concat(parts, ignore_index=True) if parts else points.assign(limb_id=[], dist_nm=[])
    assigned["cell_class"] = assigned["neuron_id"].map(cls)
    missing = assigned["limb_id"].eq(-2).groupby(assigned["neuron_id"]).all()
    if missing.any():
        print(f"[axon] no skeleton found for {int(missing.sum())} neurons (e.g. {missing[missing].index[:3].tolist()})", flush=True)
    print(f"[axon] read {len(with_points)} neurons ({time.time() - t0:.0f} s)", flush=True)

    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "voxel_nm": [4, 4, 40]}
    dq = assigned[assigned["limb_id"] > -2].groupby("kind")["dist_nm"].describe(percentiles=[0.5, 0.9, 0.99]).round(0)
    summary["distance_to_nearest_limb_or_soma_nm"] = dq.to_dict(orient="index")
    within = assigned["dist_nm"] <= args.max_dist_nm
    summary["share_within_max_dist"] = assigned.groupby("kind").apply(lambda g: float((g["dist_nm"] <= args.max_dist_nm).mean())).to_dict()
    on_soma = assigned[within].groupby("kind").apply(lambda g: float((g["limb_id"] == -1).mean())).to_dict()
    summary["share_on_soma"] = on_soma
    print("[axon] distance of the points to the nearest limb skeleton / soma (nm):\n" + dq.to_string(), flush=True)
    print(f"[axon] within {args.max_dist_nm:.0f} nm: {summary['share_within_max_dist']}; on the soma: {on_soma}", flush=True)

    ev = limb_evidence(assigned, max_dist_nm=args.max_dist_nm)
    ev = ev.merge(limbs[["neuron_id", "limb_id", "limb_radius_nm", "cable_nm", "limb_density_per_um", *LABELS]], on=["neuron_id", "limb_id"], how="left")
    ev["cell_class"] = ev["neuron_id"].map(cls)
    nn = per_neuron(assigned, limbs, LABELS, max_dist_nm=args.max_dist_nm)
    nn["cell_class"] = nn["neuron_id"].map(cls)
    nn["cell_type"] = nn["neuron_id"].map(cell_type)

    md = [f"# Axon labels against synapses\n", f"Rule: radius < {args.radius_max_nm:.0f} nm and cable > {args.cable_min_nm / 1e6:.2f} mm; "
          f"points within {args.max_dist_nm:.0f} nm of a limb skeleton; neuron = unit (mean over neurons, bootstrap 95 % CI).\n"]
    summary["by_class"] = {}
    for c in ("inhibitory", "excitatory"):
        n_all = sum(1 for v in cls.values() if v == c)
        k = limbs[limbs["neuron_id"].map(cls) == c].groupby("neuron_id")["rule_axon"].sum()
        s = {"neurons": n_all, "rule_axon_limbs_per_neuron": k.value_counts().sort_index().to_dict()}
        pre_n = nn[(nn["cell_class"] == c) & (nn["n_pre"] >= args.min_pre)]
        dn = nn[(nn["cell_class"] == c) & (nn["n_dendrite_points"] >= args.min_dendrite_points)]
        s["neurons_with_pre"] = int(len(pre_n))
        s["neurons_with_dendrite_points"] = int(len(dn))
        s["presynaptic_points_on_one_limb"] = bootstrap_mean((pre_n["n_limbs_with_pre"] == 1).to_numpy(float))
        md.append(f"\n## {c}: {n_all} neurons, rule-axon limbs per neuron {s['rule_axon_limbs_per_neuron']}\n")
        md.append(f"{len(pre_n)} neurons with >= {args.min_pre} presynaptic points, {len(dn)} with >= {args.min_dendrite_points} dendritic points; "
                  f"all presynaptic points on one limb in {s['presynaptic_points_on_one_limb'][0]:.2f} of the neurons.\n")
        md.append("| label | presynaptic points on it [CI] | limb with most presynaptic points is it [CI] | dendritic points on it [CI] |\n| --- | --- | --- | --- |")
        for lab in LABELS:
            a = bootstrap_mean(pre_n[f"pre_on_{lab}"].to_numpy(float))
            b = bootstrap_mean(pre_n[f"top_pre_limb_is_{lab}"].astype(float).to_numpy())
            d = bootstrap_mean(dn[f"dendrite_points_on_{lab}"].to_numpy(float))
            s[lab] = {"pre_on_label": a, "top_pre_limb_is_label": b, "dendrite_points_on_label": d}
            f = lambda t: "n/a" if not np.isfinite(t[0]) else f"{t[0]:.2f} [{t[1]:.2f}; {t[2]:.2f}]"
            md.append(f"| {lab} | {f(a)} | {f(b)} | {f(d)} |")
        evc = ev[(ev["cell_class"] == c) & ev["by_synapses"].isin(["axon", "dendrite", "mixed"])]
        s["limb_confusion"] = {lab: pd.crosstab(evc["by_synapses"], evc[lab]).to_dict() for lab in LABELS}
        md.append(f"\nLimbs with >= 3 points, label by synapses (rows) against the geometric label (columns False / True):\n")
        for lab in LABELS:
            md.append(f"`{lab}`\n```\n{pd.crosstab(evc['by_synapses'], evc[lab]).to_string()}\n```")
        geo = evc[evc["by_synapses"].isin(["axon", "dendrite"])].groupby("by_synapses")[["limb_radius_nm", "cable_nm", "limb_density_per_um"]]
        if len(evc):
            desc = geo.quantile([0.1, 0.5, 0.9]).round(3)
            s["geometry_of_limbs_by_synapses"] = {str(k): v for k, v in desc.to_dict(orient="index").items()}
            md.append(f"Geometry of limbs labelled by synapses (10 / 50 / 90 %):\n```\n{desc.to_string()}\n```")
        # threshold grid on the limbs labelled by synapses: sensitivity to axon limbs, false positives on dendrite limbs
        grid = []
        lab_ax = evc["by_synapses"] == "axon"
        lab_de = evc["by_synapses"] == "dendrite"
        for r in (150.0, 180.0, 220.0, 260.0):
            for cab in (0.25e6, 0.5e6, 1.0e6, 1.5e6, 2.0e6):
                pred = (evc["limb_radius_nm"] < r) & (evc["cable_nm"] > cab)
                grid.append({"radius_max_nm": r, "cable_min_mm": cab / 1e6, "axon_limbs_found": f"{int((pred & lab_ax).sum())}/{int(lab_ax.sum())}",
                             "dendrite_limbs_flagged": f"{int((pred & lab_de).sum())}/{int(lab_de.sum())}"})
        s["threshold_grid"] = grid
        md.append(f"Threshold grid on these limbs:\n```\n{pd.DataFrame(grid).to_string(index=False)}\n```")
        if not args.no_branches and (assigned["cell_class"] == c).any():
            # point level: inside limbs the rule calls axon, do presynaptic points sit on branches without spines and
            # dendritic points on branches with spines? (an axon and a spiny dendrite can share one limb)
            pa = assigned[(assigned["cell_class"] == c) & (assigned["dist_nm"] <= args.max_dist_nm) & (assigned["limb_id"] >= 0)].copy()
            rule_key = set(map(tuple, limbs.loc[limbs["rule_axon"], ["neuron_id", "limb_id"]].to_numpy()))
            pa["on_rule_limb"] = [(n, l) in rule_key for n, l in zip(pa["neuron_id"], pa["limb_id"])]
            pa["spiny_branch"] = pa["branch_spines"] > 0
            pa["rule_axon_spineless_branch"] = pa["on_rule_limb"] & ~pa["spiny_branch"]
            pts_rule = pa[pa["on_rule_limb"]]
            tab = pd.crosstab(pts_rule["kind"], pts_rule["spiny_branch"].map({True: "branch with spines", False: "branch without spines"}))
            md.append(f"Points on rule-axon limbs by the spines of their branch:\n```\n{tab.to_string()}\n```")
            s["points_on_rule_limbs_by_branch_spines"] = tab.to_dict()
            per = []
            for nid, g in pa.groupby("neuron_id"):
                pre, dend = g["kind"] == "pre", g["kind"].isin(["post_shaft", "post_spine"])
                per.append({"neuron_id": nid, "n_pre": int(pre.sum()), "n_dend": int(dend.sum()),
                            "pre_on": g.loc[pre, "rule_axon_spineless_branch"].mean() if pre.any() else np.nan,
                            "dend_on": g.loc[dend, "rule_axon_spineless_branch"].mean() if dend.any() else np.nan,
                            "rule_limb_has_dend": bool((dend & g["on_rule_limb"]).sum() >= args.min_dendrite_points)})
            per = pd.DataFrame(per)
            if len(per):
                a = bootstrap_mean(per.loc[per["n_pre"] >= args.min_pre, "pre_on"].to_numpy(float))
                d = bootstrap_mean(per.loc[per["n_dend"] >= args.min_dendrite_points, "dend_on"].to_numpy(float))
                s["rule_axon_spineless_branch"] = {"pre_on_label": a, "dendrite_points_on_label": d,
                                                   "neurons_with_dendritic_points_on_rule_limb": int(per["rule_limb_has_dend"].sum())}
                f = lambda t: "n/a" if not np.isfinite(t[0]) else f"{t[0]:.2f} [{t[1]:.2f}; {t[2]:.2f}]"
                md.append(f"Refined label `rule_axon_spineless_branch` (rule-axon limb AND branch without spines): presynaptic points on it {f(a)}, "
                          f"dendritic points on it {f(d)}; neurons with >= {args.min_dendrite_points} dendritic points on a rule-axon limb: "
                          f"{int(per['rule_limb_has_dend'].sum())}")
        summary["by_class"][c] = s
        print("\n".join(md[-(len(LABELS) * 2 + 10):]), flush=True)

    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_axon-rule-check"
    out.mkdir(parents=True, exist_ok=True)
    assigned.to_parquet(out / "points.parquet", index=False)
    ev.to_csv(out / "limb_evidence.csv", index=False)
    nn.to_csv(out / "neurons.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
