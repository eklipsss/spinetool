"""Where do the extra spines per branch of a fully generated T+P set come from? (tables only, minutes)

Chain AH: with a generated trunk the median number of spines per branch is 11 against 7 (significant), with the REAL
trunk and generated spines 5.5 against 5 (not significant). So P reproduces the count once the trunk is right, and the
excess must come from the trunk T0: its radius distribution (thin branches carry almost no spines). This script pairs
every generated branch with the real branch whose conditions and length it was generated from and splits the
excess by branch group and by the thin / thick status of both members of the pair.

    python scripts/neuron_model/diagnose_tp_counts.py --tp-dir <generate_tp set> [--evaluation <its evaluation folder>]

Inputs: ``<tp-dir>/branches.parquet`` (request conditions, ``source_branch``), ``<evaluation>/evaluation_branches.csv``
(``kind`` generated / real, one row per branch in request order; the pairing is checked by equal lengths).
Writes ``<evaluation>/count_diagnosis/{pairs.csv, by_group.csv, thin_crosstab.csv, radius_quantiles.csv, summary.md}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THIN_NM = 200.0
SOMA_BINS = [-np.inf, 50e3, 100e3, 200e3, np.inf]
SOMA_LABELS = ["< 50 мкм", "50–100 мкм", "100–200 мкм", ">= 200 мкм"]


def pair_frame(tp_dir: Path, evaluation: Path) -> pd.DataFrame:
    req = pd.read_parquet(tp_dir / "branches.parquet").sort_values("branch_index").reset_index(drop=True)
    ev = pd.read_csv(evaluation / "evaluation_branches.csv")
    gen = ev[ev["kind"] == "generated"].reset_index(drop=True)
    real = ev[ev["kind"] == "real"].reset_index(drop=True)
    if not (len(gen) == len(real) == len(req)):
        raise SystemExit(f"cannot pair: {len(req)} requests, {len(gen)} generated, {len(real)} real rows")
    rel = np.abs(gen["length_nm"].to_numpy() - real["length_nm"].to_numpy()) / real["length_nm"].to_numpy()
    if np.nanmedian(rel) > 0.05:
        print(f"warning: generated and real lengths differ (median relative difference {np.nanmedian(rel):.2f}); pairing by order may be wrong")
    out = pd.DataFrame({
        "branch_index": req["branch_index"], "source_branch": req.get("source_branch"),
        "species": req.get("species"), "cell_type_binary": req.get("cell_type_binary"), "compartment": req.get("compartment"),
        "is_terminal": req.get("is_terminal"), "soma_distance_nm": req.get("soma_distance_start_nm"), "length_nm": real["length_nm"],
        "r_gen": gen["mean_radius_nm"], "r_real": real["mean_radius_nm"], "n_gen": gen["n_spines"], "n_real": real["n_spines"],
    })
    out["thin_gen"], out["thin_real"] = out["r_gen"] < THIN_NM, out["r_real"] < THIN_NM
    out["n_excess"] = out["n_gen"] - out["n_real"]
    out["soma_class"] = pd.cut(out["soma_distance_nm"], SOMA_BINS, labels=SOMA_LABELS) if out["soma_distance_nm"].notna().any() else "unknown"
    return out


def by_group(p: pd.DataFrame, keys) -> pd.DataFrame:
    g = p.groupby(keys, dropna=False, observed=True)
    t = g.agg(n=("n_gen", "size"), thin_real=("thin_real", "mean"), thin_gen=("thin_gen", "mean"),
              r_real=("r_real", "median"), r_gen=("r_gen", "median"), n_real=("n_real", "median"), n_gen=("n_gen", "median"),
              excess_sum=("n_excess", "sum"))
    t["excess_share"] = t["excess_sum"] / p["n_excess"].sum()
    return t.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp-dir", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, help="default: newest <tp-dir>/evaluation* folder with evaluation_branches.csv")
    args = ap.parse_args()
    ev = args.evaluation
    if ev is None:
        cands = sorted((d for d in args.tp_dir.glob("evaluation*") if (d / "evaluation_branches.csv").exists()), key=lambda d: d.stat().st_mtime)
        if not cands:
            raise SystemExit(f"no evaluation*/evaluation_branches.csv in {args.tp_dir}")
        ev = cands[-1]
    p_all = pair_frame(args.tp_dir, ev)
    out = ev / "count_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    p_all.to_csv(out / "pairs.csv", index=False)
    # a request whose generated trunk was rejected (or whose evaluation row is missing) has NaN count / radius: every
    # statistic below uses the SAME complete pairs, otherwise the means and the summed excess refer to different branches
    complete = p_all[["n_gen", "n_real", "r_gen", "r_real"]].notna().all(axis=1)
    inc = p_all[~complete]
    p = p_all[complete].reset_index(drop=True)

    cross = p.groupby(["thin_real", "thin_gen"]).agg(n=("n_gen", "size"), n_real_mean=("n_real", "mean"), n_gen_mean=("n_gen", "mean"),
                                                      excess_sum=("n_excess", "sum")).reset_index()
    cross["excess_share"] = cross["excess_sum"] / p["n_excess"].sum()
    cross.to_csv(out / "thin_crosstab.csv", index=False)
    groupings = [("cell_type_binary",), ("compartment",), ("is_terminal",), ("soma_class",), ("cell_type_binary", "compartment", "is_terminal")]
    groups = {k: by_group(p, list(k)) for k in groupings}
    pd.concat([g.assign(grouping="+".join(k)) for k, g in groups.items()]).to_csv(out / "by_group.csv", index=False)
    q = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    rq = pd.DataFrame({"quantile": q, "r_real": np.nanquantile(p["r_real"], q), "r_gen": np.nanquantile(p["r_gen"], q)})
    rq.to_csv(out / "radius_quantiles.csv", index=False)

    lines = ["# Spines per branch: where the excess comes from", "",
             f"Set `{args.tp_dir.name}`, evaluation `{ev.name}`: {len(p_all)} pairs (generated branch, its source real branch), "
             f"{len(p)} complete; all numbers below on the complete pairs.",
             (f"Incomplete pairs: {len(inc)} (generated missing {int(inc['n_gen'].isna().sum())}, real missing {int(inc['n_real'].isna().sum())}); "
              f"their real branches: mean {inc['n_real'].mean():.1f} spines, median length {inc['length_nm'].median() / 1000:.1f} um "
              f"(complete: {p['n_real'].mean():.1f}, {p['length_nm'].median() / 1000:.1f} um)." if len(inc) else "Incomplete pairs: 0."),
             f"Spines per branch: mean {p['n_gen'].mean():.2f} generated against {p['n_real'].mean():.2f} real, median {p['n_gen'].median():g} against {p['n_real'].median():g}; "
             f"thin (r < {THIN_NM:g} nm) {p['thin_gen'].mean():.3f} against {p['thin_real'].mean():.3f}; total excess {p['n_excess'].sum():.0f} spines "
             f"({p['n_excess'].mean():+.2f} per branch).", "",
             "## Thin status of the pair", "", "| real thin | generated thin | n | spines real (mean) | generated (mean) | share of the excess |", "| --- | --- | --- | --- | --- | --- |"]
    for _, r in cross.iterrows():
        lines.append(f"| {r.thin_real} | {r.thin_gen} | {r.n} | {r.n_real_mean:.1f} | {r.n_gen_mean:.1f} | {r.excess_share:.2f} |")
    lines += ["", "## Radius quantiles (nm)", "", "| quantile | real | generated |", "| --- | --- | --- |"]
    for _, r in rq.iterrows():
        lines.append(f"| {r['quantile']:.2f} | {r.r_real:.0f} | {r.r_gen:.0f} |")
    for k, g in groups.items():
        lines += ["", f"## By {' + '.join(k)}", "", "| group | n | thin real | thin gen | r real | r gen | spines real | gen | share of excess |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in g.iterrows():
            name = " / ".join(str(r[c]) for c in k)
            lines.append(f"| {name} | {r.n} | {r.thin_real:.2f} | {r.thin_gen:.2f} | {r.r_real:.0f} | {r.r_gen:.0f} | {r.n_real:g} | {r.n_gen:g} | {r.excess_share:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
