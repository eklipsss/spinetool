"""Kolmogorov-Smirnov with the neuron as the unit for every spine morphometric: real after the processing chain vs real,
model vs real, model vs real after the same chain (``significance.ks_sample_vs_neurons``).

    # unconditional evaluation (evaluate_generators run): model:floor pairs
    python scripts/neuron_model/ks_spine_models.py --eval-run runs/analysis/neuron-model/<id>_generator-evaluation-... ^
        --model mogen_v3:real_poisson --model mogen_A:real_poisson --out <dir>

    # conditional v4 by organism x binary cell type (compare_species_generation report; real spines re-read from the tables)
    python scripts/neuron_model/ks_spine_models.py --species-report runs/analysis/neuron-model/<id>_species-generation-v4-valfloor --out <dir>

Three comparisons per set and metric, all with the neuron as the unit and equal sample sizes (see the function):
  floor vs real   what the processing chain alone changes (real spines through the generator's reconstruction chain;
                  compared only with OTHER neurons, the neuron of every processed spine is read from its generation_metadata.json)
  model vs real   the whole difference seen on the slides
  model vs floor  what the GENERATOR changes beyond its chain (the reference is the processed real spines, clustered by
                  their source neuron)
p < 0.05 after Benjamini-Hochberg over the 11 metrics of one comparison of one set = significant (*). KS is invariant to
monotone transforms, so no logarithms are needed. Writes <out>/ks_<set>.csv and <out>/ks_summary.md. CPU, minutes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.quality_report import S_METRICS, S_NAMES  # noqa: E402
from src.neuron_model.significance import ks_sample_vs_neurons  # noqa: E402
from src.neuron_model.spine_generation.evaluation.morphology_by_condition import bh_qvalues  # noqa: E402


def _neuron(key) -> str | None:
    parts = str(key).split("/")
    return parts[1] if len(parts) > 1 else None


ALT_ROOTS: list = []  # --alt-root: where old runs were moved (e.g. O:\neuron-model, same runs\... structure)


def _resolve(p) -> Path:
    p = Path(str(p))
    if p.is_absolute():
        return p
    for root in [REPO_ROOT, *ALT_ROOTS]:
        if (root / p).exists():
            return root / p
    return REPO_ROOT / p


def sample_neurons(folder: Path, sample_ids, seeds=None) -> list:
    """Source neuron of every processed real sample (``generation_metadata.json`` of the sample: spine_key)."""
    out = []
    for k, sid in enumerate(sample_ids):
        cands = [folder / str(sid)]
        if seeds is not None:
            cands.insert(0, folder / f"seed_{int(seeds[k])}" / str(sid))
        key = None
        for c in cands:
            f = c / "generation_metadata.json"
            if f.exists():
                key = json.loads(f.read_text(encoding="utf-8")).get("spine_key")
                break
        out.append(_neuron(key) if key else None)
    return out


def compare(label: str, real: pd.DataFrame, gen: pd.DataFrame, floor: pd.DataFrame | None, floor_neurons, n_splits: int, seed: int) -> pd.DataFrame:
    rn = real["neuron_id"].astype(str).to_numpy()
    fn = None if floor_neurons is None or any(x is None for x in floor_neurons) else np.asarray(floor_neurons, str)
    rows = []
    for m in S_METRICS:
        if m not in real.columns or m not in gen.columns:
            continue
        r = real[m].to_numpy(float)
        g = gen[m].to_numpy(float)
        row = {"set": label, "metric": m}
        res = {"model_vs_real": ks_sample_vs_neurons(g, None, r, rn, n_splits=n_splits, seed=seed)}
        if floor is not None and m in floor.columns:
            f = floor[m].to_numpy(float)
            res["floor_vs_real"] = ks_sample_vs_neurons(f, fn, r, rn, n_splits=n_splits, seed=seed + 1)
            if fn is not None:
                res["model_vs_floor"] = ks_sample_vs_neurons(g, None, f, fn, n_splits=n_splits, seed=seed + 2)
        for k, v in res.items():
            row.update({f"{k}_{x}": v[x] for x in ("d_model", "d_null_median", "d_null_p95", "p", "n_compared", "n_neurons")})
        rows.append(row)
    t = pd.DataFrame(rows)
    for k in ("floor_vs_real", "model_vs_real", "model_vs_floor"):
        if f"{k}_p" in t.columns:
            ok = t[f"{k}_p"].notna()
            t[f"{k}_q"] = np.nan
            t.loc[ok, f"{k}_q"] = bh_qvalues(t.loc[ok, f"{k}_p"].to_numpy(float))
            t[f"{k}_significant"] = t[f"{k}_q"] < 0.05
    return t


def table_md(t: pd.DataFrame, title: str, n_info: str) -> list:
    cols = [("floor_vs_real", "постобработка против реальных"), ("model_vs_real", "модель против реальных"),
            ("model_vs_floor", "модель против реальных после постобработки")]
    cols = [(k, n) for k, n in cols if f"{k}_p" in t.columns]
    lines = [f"## {title}", "", n_info, "", "| метрика | " + " | ".join(f"{n}: D (пол p95), p" for _, n in cols) + " |",
             "| --- |" + " --- |" * len(cols)]
    for _, r in t.iterrows():
        cells = []
        for k, _ in cols:
            if not np.isfinite(r.get(f"{k}_p", np.nan)):
                cells.append("—")
                continue
            star = "*" if r[f"{k}_significant"] else ""
            cells.append(f"{r[f'{k}_d_model']:.3f} ({r[f'{k}_d_null_p95']:.3f}), {r[f'{k}_p']:.3f}{star}")
        lines.append(f"| {S_NAMES.get(r['metric'], r['metric'])} | " + " | ".join(cells) + " |")
    sig = {n: int(t[f"{k}_significant"].sum()) for k, n in cols}
    lines += ["", "Значимых отличий из " + str(len(t)) + ": " + "; ".join(f"{n} {v}" for n, v in sig.items()) + ".", ""]
    return lines


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts/neuron_model" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-run", type=Path, help="evaluate_generators run (real_test_morphometrics + postprocessed/morphometrics_*)")
    parser.add_argument("--model", action="append", default=[], metavar="MODEL[:FLOOR]", help="model of the run and the real-control set of its chain")
    parser.add_argument("--species-report", type=Path, help="compare_species_generation report (v4 by organism x binary cell type)")
    parser.add_argument("--gen-label", default="v4-pw8")
    parser.add_argument("--floor-label", default="real-pw8val")
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--splits", type=int, default=1000, help="random halves of the reference neurons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alt-root", type=Path, action="append", default=[],
                        help="extra root for relative run paths that are not under the repository (runs moved to the NAS)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ALT_ROOTS.extend(args.alt_root)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# КС с нейроном как единицей: морфометрики шипиков", "",
             "D — статистика Колмогорова — Смирнова модели (среднее по разбиениям), в скобках 95-й процентиль D реальной выборки "
             "того же размера из других нейронов (пол); p — доля разбиений, где пол не меньше D модели; * — значимо после поправки "
             "Бенджамини — Хохберга по метрикам одного сравнения. Тест консервативен (при верной выборке p около 0,5).", ""]

    if args.eval_run:
        from src.neuron_model.quality_report import _read_gen_sets

        info = json.loads((args.eval_run / "run_info.json").read_text(encoding="utf-8"))
        real = pd.read_parquet(args.eval_run / "real_test_morphometrics.parquet", columns=["spine_key", *S_METRICS])
        real["neuron_id"] = real["spine_key"].map(_neuron)
        sets = _read_gen_sets(args.eval_run / "postprocessed")
        for item in args.model:
            name, _, floor_name = item.partition(":")
            if name not in sets:
                print(f"[ks] no {name} in {args.eval_run} ({sorted(sets)})")
                continue
            floor = sets.get(floor_name) if floor_name else None
            fn = None
            if floor is not None:
                folder = _resolve(info.get("models", {}).get(floor_name, ""))
                fn = sample_neurons(folder, floor["sample_id"].tolist(), floor["generation_seed"].tolist())
                n_known = sum(x is not None for x in fn)
                print(f"[ks] {floor_name}: source neuron known for {n_known} of {len(fn)} samples ({folder})", flush=True)
                if n_known < len(fn):
                    print("[ks]   -> the floor is compared as a sample without neurons; model vs floor is not computed", flush=True)
            t = compare(name, real, sets[name], floor, fn, args.splits, args.seed)
            t.to_csv(args.out / f"ks_{name}.csv", index=False)
            lines += table_md(t, f"{name} (оценка `{args.eval_run.name}`, пол {floor_name or '—'})",
                              f"Реальные: {len(real)} шипиков {real['neuron_id'].nunique()} нейронов; модель {len(sets[name])}; "
                              f"пол {0 if floor is None else len(floor)}.")
            print(f"[ks] {name}: done", flush=True)

    if args.species_report:
        rep = args.species_report
        info = json.loads((rep / "run_info.json").read_text(encoding="utf-8"))
        gen = pd.read_parquet(rep / "generated_metrics.parquet")
        csg, mbc = _load_script("compare_species_generation"), _load_script("morphology_by_condition")
        splits = Path(info["splits"]) if info.get("splits") not in (None, "None") else None
        root = args.metadata_root.resolve()
        frames = {"mouse": mbc.load_spines(Path(info["minnie_table"]), root, splits, info.get("split")),
                  "human": csg.load_h01_spines(Path(info["h01_table"]), root,
                                               Path(info["h01_detached"]) if info.get("h01_detached") not in (None, "None") else None, splits, info.get("split"))}
        for species, frame in frames.items():
            for ctype in ("excitatory", "inhibitory"):
                group = f"{species}/{ctype}"
                real = frame[frame["cell_type_binary"] == ctype].copy()
                g = gen[(gen["value"] == group) & (gen["label"] == args.gen_label)]
                fl = gen[(gen["value"] == group) & (gen["label"] == args.floor_label)]
                if not len(real) or not len(g):
                    continue
                real["neuron_id"] = real["neuron_id"].astype(str)
                fn = None
                if len(fl):
                    fn = [None] * len(fl)
                    for folder, idx in fl.groupby("folder").groups.items():
                        got = sample_neurons(_resolve(folder), fl.loc[idx, "sample_id"].tolist())
                        for i, v in zip(idx, got):
                            fn[fl.index.get_loc(i)] = v
                    print(f"[ks] {group} floor: source neuron known for {sum(x is not None for x in fn)} of {len(fn)}", flush=True)
                label = f"v4_{species}_{ctype}"
                t = compare(label, real, g, fl if len(fl) else None, fn, args.splits, args.seed)
                t.to_csv(args.out / f"ks_{label}.csv", index=False)
                lines += table_md(t, f"v4: {group} (отчёт `{rep.name}`)",
                                  f"Реальные проверочные: {len(real)} шипиков {real['neuron_id'].nunique()} нейронов; v4 {len(g)}; пол {len(fl)}.")
                print(f"[ks] {group}: done", flush=True)

    (args.out / "ks_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ks] -> {args.out / 'ks_summary.md'}")


if __name__ == "__main__":
    main()
