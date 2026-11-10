"""Second-stage analysis of a finished synapse check (validate_axon_rule.py) - the numbers of
docs/neuron-model/compartment-by-direction.md, section 9.6. Tables only, no NAS (runs on the mac).

Why a second stage: the summary of validate_axon_rule.py labels a limb "axon" / "dendrite" by >= 3 synapse points, which
misleads when the coverage is uneven (presynapses are annotated in detail for a handful of inhibitory neurons only: a
thick dendrite with 3 presynapses of a passing collateral becomes "axon", an axon without presynapses with 3 stray
postsynapses becomes "dendrite"). Here:
1. well-covered neurons (>= --min-pre presynaptic points within 3 um of a skeleton): per limb the presynapses /
   postsynapses, radius, cable, branches, and whether it is the rule limb (radius < --radius-max-nm, cable > --cable-min-nm);
2. inside the rule limbs: the median radius of the branch carrying every pre / post point (branches.parquet,
   original branch -> merged branch by member_branch_ids), and the branch-level rule "axon = branch thinner than T"
   for a grid of T (share of presynapses kept on the axon, per well-covered neuron; postsynapses put on it);
3. thick branches (radius >= 200 nm) inside rule limbs: share of the cable and spine density;
4. excitatory limbs that the rule flags and the synapses call "dendrite": their geometry and table label.

    python scripts/neuron_model/analyze_axon_rule_check.py --run runs/analysis/neuron-model/2026-11-10_030133_axon-rule-check ^
        --limbs runs/analysis/neuron-model/2026-11-03_005059_apical-by-direction-minnie65/limbs.csv ^
        --branches C:/Users/Student/LipsEkaterina/data/minnie65/preprocessed_branches/branches.parquet

Writes ``<run>/branch_analysis/{limbs_well_covered.csv, point_branch_radius.csv, branch_threshold.csv, summary.md}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INHIBITORY = {"BC", "MC", "BPC", "NGC", "Unsure I"}
DEND = ("post_shaft", "post_spine")


def member_map(branches: pd.DataFrame) -> pd.DataFrame:
    """One row per ORIGINAL branch: neuron_id (int), limb_id, branch_id, merged radius / length / spines."""
    rows = []
    for r in branches.itertuples(index=False):
        for m in str(r.member_branch_ids).split("|"):
            if m.strip().lstrip("-").isdigit():
                rows.append((int(r.neuron_id), int(r.limb_id), int(m), int(r.merged_branch_id), r.radius_median_nm, r.length_nm, r.n_spines))
    return pd.DataFrame(rows, columns=["neuron_id", "limb_id", "branch_id", "merged_branch_id", "radius", "length", "n_spines"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="output folder of validate_axon_rule.py")
    ap.add_argument("--limbs", type=Path, required=True, help="limbs.csv of apical_by_direction.py")
    ap.add_argument("--branches", type=Path, required=True, help="branches.parquet of the Minnie65 branch preprocessing")
    ap.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    ap.add_argument("--radius-max-nm", type=float, default=180.0)
    ap.add_argument("--cable-min-nm", type=float, default=1.5e6)
    ap.add_argument("--max-dist-nm", type=float, default=3000.0)
    ap.add_argument("--min-pre", type=int, default=100)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[180.0, 200.0, 220.0, 250.0])
    ap.add_argument("--thick-nm", type=float, default=200.0)
    args = ap.parse_args()

    p = pd.read_parquet(args.run / "points.parquet")
    limbs = pd.read_csv(args.limbs)
    limbs["neuron_id"] = limbs["neuron_id"].astype("int64")
    limbs["rule"] = (limbs["limb_radius_nm"] < args.radius_max_nm) & (limbs["cable_nm"] > args.cable_min_nm)
    types = pd.read_csv(args.metadata_root / "minnie65" / "cell_types_with_nucleus.csv")
    inh_ids = set(types.loc[types["cell_type"].isin(INHIBITORY), "segment_id"].astype("int64"))
    b = pd.read_parquet(args.branches, columns=["neuron_id", "limb_id", "merged_branch_id", "member_branch_ids", "radius_median_nm", "length_nm", "n_spines"])
    b["neuron_id"] = b["neuron_id"].astype(str).str.replace("neuron_", "", regex=False).astype("int64")
    mm = member_map(b)
    out = args.run / "branch_analysis"
    out.mkdir(exist_ok=True)
    md = ["# Synapse check: limb and branch analysis\n",
          f"Rule limb: median branch radius < {args.radius_max_nm:.0f} nm and cable > {args.cable_min_nm / 1e6:.2f} mm; points within "
          f"{args.max_dist_nm:.0f} nm of a skeleton; well-covered neuron: >= {args.min_pre} presynaptic points.\n"]

    near = p[(p["dist_nm"] <= args.max_dist_nm) & (p["limb_id"] >= 0)].copy()
    near["is_pre"] = near["kind"] == "pre"
    near["is_post"] = near["kind"].isin(DEND)

    # 1. well-covered neurons, per limb
    npre = near[near["is_pre"]].groupby("neuron_id").size()
    good = sorted(npre[npre >= args.min_pre].index)
    per = near[near["neuron_id"].isin(good)].groupby(["neuron_id", "limb_id"]).agg(pre=("is_pre", "sum"), post=("is_post", "sum")).reset_index()
    lw = limbs[limbs["neuron_id"].isin(good)][["neuron_id", "limb_id", "rule", "limb_radius_nm", "cable_nm", "limb_n_branches", "limb_density_per_um"]]
    lw = lw.merge(per, on=["neuron_id", "limb_id"], how="left").fillna({"pre": 0, "post": 0})
    lw["inhibitory"] = lw["neuron_id"].isin(inh_ids)
    lw["pre_share"] = lw["pre"] / lw.groupby("neuron_id")["pre"].transform("sum")
    lw.to_csv(out / "limbs_well_covered.csv", index=False)
    md.append(f"\n## 1. Well-covered neurons ({len(good)}; inhibitory {sum(n in inh_ids for n in good)})\n")
    rows = []
    for n, g in lw.groupby("neuron_id"):
        r, o = g[g["rule"]], g[~g["rule"]]
        rows.append({"neuron": n, "inhibitory": n in inh_ids, "pre": int(g["pre"].sum()), "rule_limbs": int(len(r)),
                     "pre_share_on_rule": round(float(r["pre"].sum() / max(g["pre"].sum(), 1)), 3),
                     "rule_radius_nm": round(float(r["limb_radius_nm"].median()), 0) if len(r) else np.nan,
                     "rule_cable_mm": round(float(r["cable_nm"].median() / 1e6), 2) if len(r) else np.nan,
                     "rule_branches": float(r["limb_n_branches"].median()) if len(r) else np.nan, "post_on_rule": int(r["post"].sum()),
                     "other_radius_nm": f"{o['limb_radius_nm'].min():.0f}-{o['limb_radius_nm'].max():.0f}" if len(o) else "",
                     "other_cable_max_mm": round(float(o["cable_nm"].max() / 1e6), 2) if len(o) else np.nan,
                     "other_pre_max": int(o["pre"].max()) if len(o) else 0, "other_post": int(o["post"].sum())})
    md.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```")

    # 2. branch radius of the points inside rule limbs + branch-level threshold
    rk = limbs.loc[limbs["rule"], ["neuron_id", "limb_id"]].assign(on_rule=True)
    q = near[near["branch_id"] >= 0].merge(mm, on=["neuron_id", "limb_id", "branch_id"], how="left").merge(rk, on=["neuron_id", "limb_id"], how="left")
    q["on_rule"] = q["on_rule"].astype("boolean").fillna(False).astype(bool)
    q["inhibitory"] = q["neuron_id"].isin(inh_ids)
    q = q[q["is_pre"] | q["is_post"]]
    q["grp"] = np.where(q["is_pre"], "pre", "post")
    q[["neuron_id", "limb_id", "branch_id", "kind", "grp", "on_rule", "inhibitory", "radius", "dist_nm"]].to_csv(out / "point_branch_radius.csv", index=False)
    md.append("\n## 2. Median radius of the branch carrying the point (nm)\n")
    desc = q.groupby(["inhibitory", "on_rule", "grp"])["radius"].describe(percentiles=[0.1, 0.5, 0.9]).round(0)
    md.append("```\n" + desc.to_string() + "\n```")
    thr = []
    for inh in (True, False):
        r = q[(q["inhibitory"] == inh) & q["on_rule"] & q["radius"].notna()]
        pre, post = r[r["grp"] == "pre"], r[r["grp"] == "post"]
        cov = pre.groupby("neuron_id").size()
        for t in args.thresholds:
            per_n = pre[pre["neuron_id"].isin(cov[cov >= args.min_pre].index)].groupby("neuron_id").apply(lambda h: (h["radius"] < t).mean())
            thr.append({"class": "inhibitory" if inh else "excitatory", "T_nm": t, "pre_on_axon_pooled": round(float((pre["radius"] < t).mean()), 3) if len(pre) else np.nan,
                        "pre_on_axon_per_neuron": f"{per_n.min():.2f}-{per_n.max():.2f}" if len(per_n) else "",
                        "post_on_axon": f"{int((post['radius'] < t).sum())}/{len(post)}"})
    thr = pd.DataFrame(thr)
    thr.to_csv(out / "branch_threshold.csv", index=False)
    md.append("\nBranch rule inside rule limbs: axon = branch thinner than T\n```\n" + thr.to_string(index=False) + "\n```")
    post_rule = q[q["inhibitory"] & q["on_rule"] & (q["grp"] == "post") & q["radius"].notna()]
    t0 = 200.0 if 200.0 in args.thresholds else args.thresholds[0]
    md.append(f"Inhibitory postsynapses on branches thinner than {t0:.0f} nm inside rule limbs: distance to the skeleton "
              f"{post_rule.loc[post_rule['radius'] < t0, 'dist_nm'].round(0).tolist()} nm")

    # 3. thick branches inside rule limbs
    mb = mm.drop_duplicates(["neuron_id", "merged_branch_id"]).merge(rk, on=["neuron_id", "limb_id"])
    mb = mb[mb["radius"].notna()]
    mb["inhibitory"] = mb["neuron_id"].isin(inh_ids)
    mb["thick"] = mb["radius"] >= args.thick_nm
    md.append(f"\n## 3. Branches of rule limbs: thick (>= {args.thick_nm:.0f} nm) vs thin\n")
    for inh, g in mb.groupby("inhibitory"):
        share = g.loc[g["thick"], "length"].sum() / g["length"].sum()
        dens = g.groupby("thick").apply(lambda h: h["n_spines"].sum() / (h["length"].sum() / 1000.0))
        md.append(f"- {'inhibitory' if inh else 'excitatory'}: {len(g)} merged branches, thick cable share {share:.3f}, "
                  f"spines per um thick {dens.get(True, np.nan):.3f} / thin {dens.get(False, np.nan):.3f}")

    # 4. excitatory rule limbs called "dendrite" by the synapse labels
    ev = pd.read_csv(args.run / "limb_evidence.csv")
    lab = limbs[["neuron_id", "limb_id", "cos_pia", "compartment_length"]]
    x = ev[(ev["cell_class"] == "excitatory") & ev["rule_axon"].astype(bool) & (ev["by_synapses"] == "dendrite")].merge(lab, on=["neuron_id", "limb_id"], how="left")
    md.append(f"\n## 4. Excitatory rule limbs labelled 'dendrite' by >= 3 synapse points: {len(x)}\n")
    if len(x):
        md.append(f"radius {x['limb_radius_nm'].min():.0f}-{x['limb_radius_nm'].max():.0f} nm, cable {x['cable_nm'].min() / 1e6:.1f}-{x['cable_nm'].max() / 1e6:.1f} mm, "
                  f"median cos to the pia {x['cos_pia'].median():.2f}, presynapses {int(x['n_pre'].sum())}, postsynapses per limb "
                  f"{int(x['n_dendrite_points'].min())}-{int(x['n_dendrite_points'].max())}; colleague's label {x['compartment_length'].value_counts(dropna=False).to_dict()}")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
