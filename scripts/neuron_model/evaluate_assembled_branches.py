"""Generated / assembled branches vs the real branches they were requested from (module-interfaces-spec.md, section 10).

    # T + P level: every branch of a generate_tp.py set
    python scripts/neuron_model/evaluate_assembled_branches.py --tp-dir runs/analysis/neuron-model/tp_generation/<id>

    # assembled level: only the assembled branches, only the spines that were actually placed
    python scripts/neuron_model/evaluate_assembled_branches.py --tp-dir <tp> --assembly <tp>/assembly_<date>

Real reference = the source branch of every request (same conditions and length); its tables and geometry come from
the P0 run of the set (``data.branches_root`` / ``data.pack_dir``: local disk, no NAS). Compared: spines per branch,
density, share of branches without spines (overall and per length class), trunk functionals, position along the
branch, nearest-neighbour distance on the trunk surface, gaps along the branch, angle around the trunk; noise floor =
two halves of the real reference. Writes ``<out>/evaluation.json``, ``<out>/evaluation_branches.csv`` and ``<out>/evaluation_spines.csv``
(out = the assembly directory, else ``<tp-dir>/evaluation``). CPU, seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch FIRST on Windows (see fit_t0.py); the T0 modules import it
import torch  # noqa: F401,E402

import numpy as np
import pandas as pd

from src.neuron_model.branch_assembly import load_tp_set
from src.neuron_model.branch_assembly.evaluate import collect, compare_sets, spine_geometry_stats, split_halves, summarize, trunk_functionals
from src.neuron_model.branch_data import key_series, prepare_training_data, trunk_record
from src.neuron_model.spine_generation.experiment.config import resolve_path
from src.neuron_model.spine_generation.experiment.run import open_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp-dir", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, help="output of assemble_branches.py on this set (assembled level)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamplings of NEURONS for the interval of the median difference (0 = off)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    info = json.loads((args.tp_dir / "generation_info.json").read_text(encoding="utf-8"))
    d = open_run(Path(info["p0_run"])).load_config()["data"]
    table, spines, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]),
                                                splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    branches, trunks, placements = load_tp_set(args.tp_dir)
    indices = list(range(len(trunks)))
    placed = None
    asm_info = None
    if args.assembly:
        ab = pd.read_parquet(args.assembly / "assembly_branches.parquet")
        indices = sorted(int(i) for i in ab.loc[ab["status"].isin(["ok", "invalid"]), "branch_index"])
        st = pd.read_parquet(args.assembly / "assembly_spines.parquet")
        placed = {int(k): set(g.loc[g["status"] == "placed", "spine_index"].astype(int)) for k, g in st.groupby("branch_index")}
        asm_info = {"branches": int(len(ab)), "branch_status": ab["status"].value_counts().to_dict(),
                    "spines_placed": int((st["status"] == "placed").sum()), "spines_dropped": int((st["status"] == "dropped").sum()),
                    "immersion_depth_nm_median": float(st["delta_nm"].median()) if "delta_nm" in st.columns else None,
                    "immersion_depth_nm_p90": float(st["delta_nm"].quantile(0.9)) if "delta_nm" in st.columns else None}

    # ---- generated
    gen_rows, gen_stats, sources = [], [], []
    for i in indices:
        t = trunks[i]
        p = placements.get(i)
        if p is not None and placed is not None:
            p = p[p["spine_index"].astype(int).isin(placed.get(i, set()))]
        s = p["s"].to_numpy(float) if p is not None else np.zeros(0)
        th = p["theta"].to_numpy(float) if p is not None else np.zeros(0)
        r = np.interp(s, t["s"], t["radius"])
        gen_rows.append({"branch_index": i, "length_nm": float(t["length"]), "n_spines": int(len(s)), **trunk_functionals(t)})
        gen_stats.append(spine_geometry_stats(s, th, r, float(t["length"])))
        sources.append(str(branches.loc[branches["branch_index"] == i, "source_branch"].iloc[0]))
    gen_neurons = [k.split("/")[1] for k in sources]  # a generated branch shares the neuron of the request it copies (same conditions)

    # ---- real reference: the source branch of every request (paired list, duplicates kept)
    keys = key_series(table)
    rows_by_key = dict(zip(keys, table.to_dict("records")))
    sp = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].fillna(False).astype(bool)] if len(spines) else spines
    sp_by_key = {k: g for k, g in sp.groupby(key_series(sp))} if len(sp) else {}
    real_rows, real_stats, real_neurons = [], [], []
    for key in sources:
        if key not in pack or key not in rows_by_key:
            continue
        row = rows_by_key[key]
        rec = trunk_record(pack.get(key))
        eligible_p = bool(row.get("train_eligible_P"))
        n = int(row["n_spines_in_range"]) if eligible_p and pd.notna(row.get("n_spines_in_range")) else np.nan
        real_rows.append({"source_branch": key, "length_nm": float(rec["length"]), "n_spines": n, **trunk_functionals(rec)})
        g = sp_by_key.get(key)
        if eligible_p and g is not None:
            real_stats.append(spine_geometry_stats(g["s"].to_numpy(float), g["theta"].to_numpy(float), g["radius_at_s"].to_numpy(float), float(rec["length"])))
        else:
            real_stats.append(spine_geometry_stats(np.zeros(0), np.zeros(0), np.zeros(0), float(rec["length"])))
        real_neurons.append(key.split("/")[1])

    gen_b, gen_s = collect(gen_rows, gen_stats, gen_neurons)
    real_b, real_s = collect(real_rows, real_stats, real_neurons)
    result = compare_sets(gen_b, real_b, gen_s, real_s, n_boot=args.bootstrap)
    rng = np.random.default_rng(args.seed)
    half = split_halves(real_neurons, rng)
    idx_a, idx_b = np.flatnonzero(half), np.flatnonzero(~half)
    floor = None
    if len(idx_a) >= 3 and len(idx_b) >= 3:
        ba, sa = collect([real_rows[i] for i in idx_a], [real_stats[i] for i in idx_a])
        bb, sb = collect([real_rows[i] for i in idx_b], [real_stats[i] for i in idx_b])
        floor = compare_sets(ba, bb, sa, sb)

    out = args.out or (args.assembly if args.assembly else args.tp_dir / "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    report = {"tp_dir": str(args.tp_dir), "assembly": str(args.assembly) if args.assembly else None, "level": "assembled" if args.assembly else "tp",
              "n_generated": int(len(gen_b)), "n_real": int(len(real_b)), "n_real_neurons": int(len(set(real_neurons))),
              "noise_floor_by": "neuron" if len(set(real_neurons)) >= 2 else "branch", "result": result, "noise_floor": floor, "assembly_info": asm_info}
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    pd.concat([gen_b.assign(kind="generated"), real_b.assign(kind="real")], ignore_index=True).to_csv(out / "evaluation_branches.csv", index=False)
    pd.concat([gen_s.assign(kind="generated"), real_s.assign(kind="real")], ignore_index=True).to_csv(out / "evaluation_spines.csv", index=False)

    print(f"[evaluate] {report['level']} level: {len(gen_b)} generated branches vs {len(real_b)} real source branches "
          f"({report['n_real_neurons']} neurons; noise floor by {report['noise_floor_by']})")
    if asm_info:
        print(f"[evaluate] assembly: {asm_info['branch_status']}, spines placed {asm_info['spines_placed']}, dropped {asm_info['spines_dropped']}, "
              f"immersion depth median {asm_info['immersion_depth_nm_median']:.0f} nm (90 % {asm_info['immersion_depth_nm_p90']:.0f})")
    for line in summarize(result, floor):
        print(line)
    print(f"[evaluate] -> {out / 'evaluation.json'}")


if __name__ == "__main__":
    main()
