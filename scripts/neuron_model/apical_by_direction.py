"""Apical / basal limbs by the direction towards the pia (src/neuron_model/compartments.py), with the check against the
cable-length labels of the Minnie65 table, and the same labelling for H01.

1. soma centroid of every neuron (``soma/soma_mesh.off``) and the pia axis estimated from the cortical layers of the
   neurons (H01: h01_layer.csv; Minnie65: the layer in the cell type name) - or ``--pia-axis x,y,z``;
2. per limb: cable length, initial direction of the dendrite (10-30 um along the limb skeleton), cos to the pia;
3. axon-like limbs: Minnie65 - the flag of the existing table (so that only the apical rule is compared); with
   ``--branches-root`` also the same caliber / spine-density rule as the table (median diameter < 0.35 um and density
   < 0.1 spines/um over >= 3 merged branches of the limb) computed from the branch preprocessing (the only option for H01);
4. labels: apical / basal / uncertain / axon_like / not_applicable (cell types without an apical dendrite);
5. Minnie65: agreement with the table - for the neurons the table calls confident, is our apical limb the same? how
   many table-uncertain neurons get a confident label? cos of the table's apical vs basal limbs.

    python scripts/neuron_model/apical_by_direction.py --dataset-root O:/Datasets/Minnie65 --dataset minnie65
    python scripts/neuron_model/apical_by_direction.py --dataset-root O:/Datasets/H01/H01_Seg --dataset h01 \
        --branches-root C:/Users/Student/LipsEkaterina/data/h01/preprocessed_branches

Writes ``runs/analysis/neuron-model/<time>_apical-by-direction-<dataset>/{limbs.csv, neurons.csv, summary.json,
limb_compartments_direction.parquet}`` (the last one in the format of minnie65_limb_compartments_length_rule.parquet).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.compartments import classify_by_direction, estimate_pia_axis, layer_value, limb_features, soma_center  # noqa: E402
from src.neuron_model.metadata import cell_type_binary, load_metadata_tables  # noqa: E402

NO_APICAL_TYPES = {"INTERNEURON", "BC", "BPC", "MC", "NGC", "Unsure I", "OLIGO"}


def axon_like_from_branches(branches_root: Path, density_max: float = 0.10, diameter_max_um: float = 0.35, min_branches: int = 3):
    """{(neuron_id, limb_id)} matching the caliber / density rule of the Minnie65 table, from branches.parquet."""
    b = pd.read_parquet(branches_root / "branches.parquet", columns=["neuron_id", "limb_id", "status", "length_nm", "radius_median_nm", "n_spines"])
    b = b[b["status"].astype(str) == "ok"]
    g = b.groupby(["neuron_id", "limb_id"]).agg(n=("n_spines", "size"), spines=("n_spines", "sum"), length=("length_nm", "sum"), radius=("radius_median_nm", "median"))
    g = g[g["n"] >= min_branches]
    density = g["spines"] / (g["length"] / 1000.0)
    mask = (density < density_max) & (2.0 * g["radius"] / 1000.0 < diameter_max_um)
    return {(str(n), int(l)) for n, l in g.index[mask]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True, help="folder with the neuron folders")
    parser.add_argument("--dataset", choices=("minnie65", "h01"), required=True)
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--branches-root", type=Path, help="branch preprocessing output (axon-like rule from the branches)")
    parser.add_argument("--pia-axis", help="x,y,z of the direction to the pia (default: estimated from the layers)")
    parser.add_argument("--neurons", nargs="*", help="only these neuron folders")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-cos", type=float, default=0.6)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    t0 = time.time()
    meta = load_metadata_tables(args.metadata_root.resolve())
    types = meta.get("cell_types", {}).get(args.dataset, {})
    layers = meta.get("layers", {}).get(args.dataset, {})
    pattern = "neuron_*" if args.dataset == "h01" else "[0-9]*"
    dirs = [d for d in sorted(args.dataset_root.glob(pattern)) if d.is_dir() and (d / "soma" / "soma_mesh.off").exists()]
    if args.neurons:
        dirs = [d for d in dirs if d.name in set(args.neurons)]
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"[apical] {len(dirs)} neurons with a soma mesh under {args.dataset_root}", flush=True)

    with ThreadPoolExecutor(args.threads) as pool:
        centers = dict(zip([d.name for d in dirs], pool.map(soma_center, dirs)))
    # pia axis: layers (H01 table, or the Minnie65 type name) regressed on the soma position
    lay = {n: layer_value(layers.get(n)) if args.dataset == "h01" else layer_value(types.get(n)) for n in centers}
    if args.pia_axis:
        pia = np.array([float(v) for v in args.pia_axis.split(",")])
        pia /= np.linalg.norm(pia)
        r2 = None
    else:
        ids = [n for n in centers if lay.get(n) is not None and centers[n] is not None]
        if len(ids) < 10 or len({lay[n] for n in ids}) < 3:
            raise SystemExit(f"[apical] only {len(ids)} neurons with a layer ({len({lay[n] for n in ids})} distinct layers): cannot estimate the pia "
                             "axis, give --pia-axis x,y,z")
        pia, r2 = estimate_pia_axis(np.array([centers[n] for n in ids]), np.array([lay[n] for n in ids]))
        print(f"[apical] pia axis from the layers of {len(ids)} neurons: {np.round(pia, 3).tolist()} (R^2 of layer ~ soma position {r2:.2f})", flush=True)

    def features(d):
        c = centers[d.name]
        if c is None:
            return []
        rows = limb_features(d, c)
        for r in rows:
            r["neuron_id"] = d.name
        return rows

    with ThreadPoolExecutor(args.threads) as pool:
        limbs = pd.DataFrame([r for rows in pool.map(features, dirs) for r in rows])
    print(f"[apical] {len(limbs)} limbs ({time.time() - t0:.0f} s)", flush=True)

    # axon-like limbs
    limbs["limb"] = limbs["neuron_id"] + "_limb_" + limbs["limb_id"].map("{:03d}".format)
    table = None
    if args.dataset == "minnie65":
        path = args.metadata_root / "minnie65" / "minnie65_limb_compartments_length_rule.parquet"
        if path.exists():
            table = pd.read_parquet(path)[["limb", "compartment", "axon_like", "classification_confident"]].rename(
                columns={"compartment": "compartment_length", "axon_like": "axon_like_table", "classification_confident": "confident_length"})
            limbs = limbs.merge(table, on="limb", how="left")
    branch_axon = axon_like_from_branches(args.branches_root) if args.branches_root else None
    if branch_axon is not None:
        limbs["axon_like_branches"] = [(n, int(l)) in branch_axon for n, l in zip(limbs["neuron_id"], limbs["limb_id"])]
    if "axon_like_table" in limbs.columns and limbs["axon_like_table"].notna().any():
        limbs["axon_like"] = limbs["axon_like_table"].fillna(False).astype(bool)
    elif branch_axon is not None:
        limbs["axon_like"] = limbs["axon_like_branches"]
    else:
        limbs["axon_like"] = False
        print("[apical] WARNING: no axon-like flags (no table, no --branches-root): the axon limb competes as a dendrite", flush=True)

    applicable = {n: (types.get(n) not in NO_APICAL_TYPES) for n in limbs["neuron_id"].unique()}
    labelled, neurons = classify_by_direction(limbs, pia, min_cos=args.min_cos, applicable=applicable)
    neurons["cell_type"] = neurons["neuron_id"].map(types)
    neurons["cell_type_binary"] = neurons["cell_type"].map(lambda t: cell_type_binary(t) if isinstance(t, str) else None)

    summary = {"dataset": args.dataset, "n_neurons": int(len(neurons)), "n_limbs": int(len(labelled)), "pia_axis": pia.tolist(), "pia_r2": r2,
               "confident_share": float(neurons["confident"].mean()),
               "confident_by_type": neurons.groupby("cell_type", dropna=False)["confident"].mean().round(3).to_dict(),
               "reasons": neurons["reason"].value_counts().to_dict()}
    print(f"[apical] confident {summary['confident_share']:.1%} of the neurons; reasons {summary['reasons']}")
    if "compartment_length" in labelled.columns:
        lt = labelled[labelled["compartment_length"].notna()]
        ap_len = lt[lt["compartment_length"] == "apical"].set_index("neuron_id")["limb_id"]
        ours = neurons.set_index("neuron_id")["apical_limb"]
        both = [n for n in ap_len.index if n in ours.index and pd.notna(ours[n])]
        agree = float(np.mean([int(ours[n]) == int(ap_len[n]) for n in both])) if both else float("nan")
        unc_neurons = set(lt.loc[lt["compartment_length"] == "uncertain", "neuron_id"])
        resolved = float(neurons[neurons["neuron_id"].isin(unc_neurons)]["confident"].mean()) if unc_neurons else float("nan")
        cos_apical = lt.loc[lt["compartment_length"] == "apical", "cos_pia"].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_dict()
        cos_basal = lt.loc[lt["compartment_length"] == "basal", "cos_pia"].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_dict()
        lost = [n for n in ap_len.index if n in ours.index and pd.isna(ours[n])]
        summary.update({"length_rule_confident_neurons": int(len(ap_len)), "both_confident": int(len(both)), "agreement_on_the_apical_limb": agree,
                        "length_confident_but_direction_uncertain": int(len(lost)),
                        "length_uncertain_neurons": int(len(unc_neurons)), "length_uncertain_resolved_by_direction": resolved,
                        "cos_of_length_apical": cos_apical, "cos_of_length_basal": cos_basal,
                        "limb_confusion": pd.crosstab(lt["compartment_length"], lt["compartment_dir"]).to_dict()})
        print(f"[apical] vs the length rule: {len(ap_len)} neurons confident there; both confident {len(both)}, same apical limb in {agree:.1%}; "
              f"confident there but uncertain here {len(lost)}; table-uncertain neurons {len(unc_neurons)}, resolved here {resolved:.1%}")
        print(f"[apical] cos to the pia of the table's apical limbs {cos_apical}\n[apical]                         basal limbs  {cos_basal}")
        print(pd.crosstab(lt["compartment_length"], lt["compartment_dir"]).to_string())
        if "axon_like_branches" in labelled.columns:
            print("[apical] axon-like: table vs the rule from branches\n" + pd.crosstab(lt["axon_like_table"], lt["axon_like_branches"]).to_string())

    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_apical-by-direction-{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    keep = [c for c in labelled.columns if c not in ("root", "dir_init", "dir_limb")]
    labelled[keep].to_csv(out / "limbs.csv", index=False)
    neurons.to_csv(out / "neurons.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    conf = labelled.merge(neurons[["neuron_id", "confident"]], on="neuron_id", how="left")
    pd.DataFrame({"limb_id": conf["limb_id"], "cable_um": conf["cable_nm"] / 1000.0, "neuron": conf["neuron_id"], "limb": conf["limb"],
                  "axon_like": conf["axon_like"], "compartment": conf["compartment_dir"], "cos_pia": conf["cos_pia"],
                  "classification_confident": conf["confident"]}).to_parquet(out / "limb_compartments_direction.parquet", index=False)
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
