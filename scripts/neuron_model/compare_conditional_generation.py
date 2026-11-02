"""Does a conditional generator reproduce the DIFFERENCE between conditions? (morphometrics of generated vs real spines)

The loss-based check (check_condition_use.py) only says whether the model reacts to the condition. This one asks what
matters for the branch model: spines generated with "inhibitory" should differ from those generated with "excitatory"
the way real inhibitory spines differ from real excitatory ones (real: Volume -28 %, Length -19 %, base area +28 %, ...,
runs/analysis/neuron-model/2026-11-01_184009_morphology-by-condition).

Input: generated sets (``generate_mogen.py --run <run> --condition cell_type_binary=<value> [--guidance g]``, folder
``<run>/meshes/cond_..._seed_<s>``), each given as ``--gen <value>[@<label>]=<folder>``; the real spines of every value
from the stage-8 morphometrics table + the metadata tables (as morphology_by_condition.py). Per metric and generated
set:
    median generated / median real of the same value;
    W1/sd to the real spines of the SAME value and of every OTHER value (sd = pooled real sd): a set that follows its
        condition is closer to its own value (``closer_to_own``);
and per pair of values with the same label (e.g. inhibitory@g3 vs excitatory@g3): the generated ratio of medians
against the real one (``ratio_gen`` vs ``ratio_real``) - the base finder bias of generated meshes (JunctionArea, ...)
cancels in the ratio. Base of generated meshes: DownFacingCapFinder, as in evaluate_generators.py.

    python scripts/neuron_model/compare_conditional_generation.py --block cell_type_binary ^
        --gen excitatory@g1=<run>\\meshes\\cond_cell_type_binary-excitatory_g1_seed_1 ^
        --gen inhibitory@g1=<run>\\meshes\\cond_cell_type_binary-inhibitory_g1_seed_1 ^
        --morphometrics-table cache\\morphometrics_minnie65_full.parquet

Writes runs/analysis/neuron-model/<time>_conditional-generation/{per_set.csv, pairs.csv, generated_metrics.parquet, summary.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402


def wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else float("nan")


def compare(generated: dict, real: dict, metrics, sd: dict) -> tuple:
    """``generated`` {(value, label): frame}, ``real`` {value: frame}. Returns (per_set, pairs) frames."""
    rows = []
    for (value, label), g in generated.items():
        for m in metrics:
            gv = g[m].to_numpy(float)
            row = {"value": value, "label": label, "metric": m, "n_generated": int(np.isfinite(gv).sum()),
                   "median_generated": float(np.nanmedian(gv)) if np.isfinite(gv).any() else np.nan}
            if value in real:
                rv = real[value][m].to_numpy(float)
                row["median_real"] = float(np.nanmedian(rv))
                row["gen_over_real"] = row["median_generated"] / row["median_real"] if row["median_real"] else np.nan
            dists = {v: wasserstein(gv, r[m].to_numpy(float)) / sd[m] for v, r in real.items()}
            for v, d in dists.items():
                row[f"w1sd_to_{v}"] = d
            others = [d for v, d in dists.items() if v != value and np.isfinite(d)]
            row["closer_to_own"] = bool(value in dists and others and dists[value] < min(others))
            rows.append(row)
    per_set = pd.DataFrame(rows)
    pairs = []
    labels = sorted({lab for _, lab in generated})
    values = sorted({v for v, _ in generated})
    for lab in labels:
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                if (a, lab) not in generated or (b, lab) not in generated:
                    continue
                for m in metrics:
                    ga, gb = generated[(a, lab)][m], generated[(b, lab)][m]
                    ra, rb = (real[a][m], real[b][m]) if a in real and b in real else (None, None)
                    row = {"label": lab, "a": a, "b": b, "metric": m,
                           "ratio_gen": float(np.nanmedian(ga) / np.nanmedian(gb)) if np.nanmedian(gb) else np.nan}
                    if ra is not None:
                        row["ratio_real"] = float(np.nanmedian(ra) / np.nanmedian(rb)) if np.nanmedian(rb) else np.nan
                        # how much of the real difference the generator reproduces (1 = all, 0 = none, < 0 = wrong sign)
                        real_log, gen_log = np.log(row["ratio_real"]), np.log(row["ratio_gen"])
                        row["share_of_real_difference"] = float(gen_log / real_log) if abs(real_log) > 1e-3 else np.nan
                    from scipy.stats import mannwhitneyu

                    x, y = ga.to_numpy(float), gb.to_numpy(float)
                    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
                    row["p_gen_a_vs_b"] = float(mannwhitneyu(x, y).pvalue) if len(x) > 2 and len(y) > 2 else np.nan
                    pairs.append(row)
    return per_set, pd.DataFrame(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", action="append", required=True, metavar="VALUE[@LABEL]=FOLDER", help="a generated set (repeatable)")
    parser.add_argument("--block", default="cell_type_binary", choices=("cell_type_binary", "cell_type"))
    parser.add_argument("--morphometrics-table", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--splits", type=Path, help="real spines only of the neurons of --split (default: all neurons)")
    parser.add_argument("--split")
    parser.add_argument("--mesh-kind", default="postprocessed", choices=("raw", "postprocessed"))
    parser.add_argument("--max-real", type=int, default=20000, help="real spines per value (random subset)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import importlib.util

    from src.neuron_model.spine_generation.evaluation.attachment import DownFacingCapFinder
    from src.neuron_model.spine_generation.evaluation.inputs import load_generated_set
    from src.neuron_model.spine_generation.evaluation.morphometrics import generated_morphometrics

    t0 = time.time()
    spec = importlib.util.spec_from_file_location("morphology_by_condition", REPO_ROOT / "scripts/neuron_model/morphology_by_condition.py")
    mbc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mbc)
    frame = mbc.load_spines(args.morphometrics_table, args.metadata_root.resolve(), args.splits, args.split)
    rng = np.random.default_rng(args.seed)
    metrics = [m for m in SCALAR_METRICS if m in frame.columns]

    finder = DownFacingCapFinder()
    generated, gen_frames = {}, []
    for item in args.gen:
        key, folder = item.split("=", 1)
        value, _, label = key.partition("@")
        meshes, reports, ids = load_generated_set(Path(folder), args.mesh_kind)
        g = generated_morphometrics(meshes, attachment_finder=finder, compute_chords=False, sample_ids=ids)
        g["valid"] = [bool(r.get("is_valid")) for r in reports]
        generated[(value, label or "default")] = g
        gen_frames.append(g.assign(value=value, label=label or "default", folder=folder))
        print(f"[cond-gen] {value}@{label or 'default'}: {len(g)} samples, {int(g['mesh_present'].sum())} meshes, "
              f"{int(g['attachment_found'].sum())} with a base, {int(g['valid'].sum())} valid ({folder})", flush=True)

    real = {}
    for value in sorted({v for v, _ in generated}):
        r = frame[frame[args.block] == value]
        if len(r) > args.max_real:
            r = r.iloc[rng.choice(len(r), args.max_real, replace=False)]
        if len(r):
            real[value] = r
        print(f"[cond-gen] real {args.block}={value}: {len(r)} spines of {frame.loc[frame[args.block] == value, 'neuron_id'].nunique()} neurons", flush=True)
    pooled = pd.concat(real.values()) if real else frame
    sd = {m: float(np.nanstd(pooled[m].to_numpy(float))) or 1.0 for m in metrics}

    per_set, pairs = compare(generated, real, metrics, sd)
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_conditional-generation"
    out.mkdir(parents=True, exist_ok=True)
    per_set.to_csv(out / "per_set.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    pd.concat(gen_frames, ignore_index=True).to_parquet(out / "generated_metrics.parquet", index=False)
    (out / "run_info.json").write_text(json.dumps({"gen": args.gen, "block": args.block, "mesh_kind": args.mesh_kind,
                                                   "morphometrics_table": str(args.morphometrics_table), "created": time.strftime("%Y-%m-%d %H:%M")},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Conditional generation vs real ({args.block}, {args.mesh_kind} meshes)", ""]
    if len(pairs) and "ratio_real" in pairs.columns:
        lines += ["## Ratio of medians a / b: generated vs real", "", "| label | a / b | metric | real | generated | share of the real difference | p (gen a vs b) |", "|---|---|---|---|---|---|---|"]
        for _, r in pairs.iterrows():
            lines.append(f"| {r['label']} | {r['a']} / {r['b']} | {r['metric']} | {r['ratio_real']:.3f} | {r['ratio_gen']:.3f} | {r.get('share_of_real_difference', np.nan):.2f} | {r['p_gen_a_vs_b']:.2g} |")
        lines.append("")
    own = per_set.groupby(["value", "label"])["closer_to_own"].mean()
    lines += ["## Share of metrics where the generated set is closer (W1/sd) to the real spines of its OWN value", ""]
    lines += [f"- {v}@{lab}: {s:.0%}" for (v, lab), s in own.items()]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
