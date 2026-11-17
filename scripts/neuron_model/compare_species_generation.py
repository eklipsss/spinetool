"""Does the species-conditional MoGen (v4) reproduce the difference between species and cell types? (morphometrics)

Extension of compare_conditional_generation.py to the groups ``species/cell_type_binary`` (``mouse/excitatory``,
``mouse/inhibitory``, ``human/excitatory``, ``human/inhibitory``): generated sets of ``generate_mogen.py --run <v4>
--condition species=<s>,cell_type_binary=<c>`` against the REAL spines of the same group (mouse = Minnie65, human = H01,
stage-8 morphometrics tables of the two datasets). Per metric: median generated / median real, W1/sd to the real spines of
the own group and of the others (``closer_to_own``) and, per pair of groups, the generated ratio of medians against the real
one (``share_of_real_difference``: 1 = the generator reproduces the real difference, 0 = none). Pairs of interest: the
species effect inside a type (human/excitatory against mouse/excitatory) and the type effect inside a species.

H01 spines whose neck stayed in the trunk (``detached_spine_keys.py``) are left out of the real H01 spines, as in the
training of v4. Real spines of all neurons are used unless ``--splits`` / ``--split`` select neurons (after a split the real
side is the validation part; the generator never saw it).

    python scripts\\neuron_model\\compare_species_generation.py ^
        --gen mouse/excitatory@g1.25=<v4>\\meshes_poisson_d8_pw8\\v4_mouse_excitatory_g1.25_seed_1 ^
        --gen human/excitatory@g1.25=<v4>\\meshes_poisson_d8_pw8\\v4_human_excitatory_g1.25_seed_1 ^
        --minnie-table cache\\morphometrics_minnie65_full.parquet ^
        --h01-table <h01 preprocessed_v2>\\H01_Seg\\morphometrics.parquet ^
        --h01-detached data\\processed\\neuron-model\\splits\\h01_detached_spines.parquet

Writes runs/analysis/neuron-model/<time>_species-generation/{per_set.csv, pairs.csv, generated_metrics.parquet, summary.md}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.metadata import cell_type_binary, dataset_kind, load_metadata_tables  # noqa: E402
from src.neuron_model.spine_generation.evaluation.morphology_by_condition import KEYS  # noqa: E402
from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts/neuron_model" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spine_key(frame: pd.DataFrame) -> pd.Series:
    """``dataset/neuron/limb/branch/spine`` as in the split and exclusion tables."""
    return frame[KEYS].astype(str).agg("/".join, axis=1)


def load_h01_spines(table_path: Path, metadata_root: Path, detached: Path | None = None, splits: Path | None = None,
                    split: str | None = None) -> pd.DataFrame:
    """Stage-8 morphometrics of H01 with the binary / detailed cell type of the neuron (metadata tables), without the spines
    whose neck stayed in the trunk (``detached``)."""
    import pyarrow.parquet as pq

    table_path = Path(table_path)
    parts = Path(f"{table_path}.parts")
    paths = [table_path] + sorted(parts.glob("*.parquet")) if parts.exists() else [table_path]
    frames = []
    for path in paths:
        names = set(pq.read_schema(path).names)
        frames.append(pq.read_table(path, columns=[c for c in KEYS + ["status"] + list(SCALAR_METRICS) if c in names]).to_pandas())
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEYS, keep="last")
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str) == "success"]
    frame = frame[frame["dataset"].astype(str).map(dataset_kind) == "h01"].copy()
    for key in ("neuron_id", "limb_id"):
        frame[key] = frame[key].astype(str)
    keys = spine_key(frame)
    if detached is not None:
        drop = set(pd.read_parquet(detached)["spine_key"].astype(str))
        n_before = len(frame)
        frame = frame[~keys.isin(drop)]
        print(f"[species-gen] H01: {n_before - len(frame)} of {n_before} spines left out as detached (list of {len(drop)} keys)", flush=True)
    if splits is not None:
        s = pd.read_parquet(splits)
        neuron_split = dict(zip(s["spine_key"].astype(str).str.split("/").str[1], s["split"].astype(str)))
        frame["split"] = frame["neuron_id"].map(neuron_split)
        frame = frame[frame["split"] == split] if split else frame[frame["split"].notna()]
    types = load_metadata_tables(metadata_root)["cell_types"].get("h01", {})
    frame["cell_type"] = frame["neuron_id"].map(types)
    frame["cell_type_binary"] = frame["cell_type"].map(lambda t: cell_type_binary(t) if isinstance(t, str) else None)
    for m in SCALAR_METRICS:
        if m in frame.columns:
            frame[m] = pd.to_numeric(frame[m], errors="coerce")
    return frame


def real_groups(frames: dict, max_real: int, rng: np.random.Generator) -> dict:
    """{``species/type``: frame} from {species: frame with ``cell_type_binary``}; groups without spines are left out."""
    out = {}
    for species, frame in frames.items():
        for ctype in ("excitatory", "inhibitory"):
            r = frame[frame["cell_type_binary"] == ctype]
            if len(r) > max_real:
                r = r.iloc[rng.choice(len(r), max_real, replace=False)]
            if len(r):
                out[f"{species}/{ctype}"] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", action="append", required=True, metavar="SPECIES/TYPE[@LABEL]=FOLDER", help="a generated set (repeatable)")
    parser.add_argument("--minnie-table", type=Path, required=True)
    parser.add_argument("--h01-table", type=Path, required=True)
    parser.add_argument("--h01-detached", type=Path)
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--splits", type=Path, help="real spines only of the neurons of --split (default: all neurons)")
    parser.add_argument("--split")
    parser.add_argument("--mesh-kind", default="postprocessed", choices=("raw", "postprocessed"))
    parser.add_argument("--max-real", type=int, default=20000, help="real spines per group (random subset)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from src.neuron_model.spine_generation.evaluation.attachment import DownFacingCapFinder
    from src.neuron_model.spine_generation.evaluation.morphometrics import generated_set_metrics

    t0 = time.time()
    ccg = _load_script("compare_conditional_generation")
    mbc = _load_script("morphology_by_condition")
    metadata_root = args.metadata_root.resolve()
    frames = {"mouse": mbc.load_spines(args.minnie_table, metadata_root, args.splits, args.split),
              "human": load_h01_spines(args.h01_table, metadata_root, args.h01_detached, args.splits, args.split)}
    metrics = [m for m in SCALAR_METRICS if all(m in f.columns for f in frames.values())]
    rng = np.random.default_rng(args.seed)

    finder = DownFacingCapFinder()
    generated, gen_frames = {}, []
    for item in args.gen:
        key, folder = item.split("=", 1)
        value, _, label = key.partition("@")
        g = generated_set_metrics(Path(folder), args.mesh_kind, attachment_finder=finder)
        generated[(value, label or "default")] = g
        gen_frames.append(g.assign(value=value, label=label or "default", folder=folder))
        print(f"[species-gen] {value}@{label or 'default'}: {len(g)} samples, {int(g['mesh_present'].sum())} meshes, "
              f"{int(g['attachment_found'].sum())} with a base, {int(g['valid'].sum())} valid ({folder})", flush=True)

    real = real_groups(frames, args.max_real, rng)
    for group, r in real.items():
        print(f"[species-gen] real {group}: {len(r)} spines of {r['neuron_id'].nunique()} neurons", flush=True)
    pooled = pd.concat(real.values())
    sd = {m: float(np.nanstd(pooled[m].to_numpy(float))) or 1.0 for m in metrics}

    per_set, pairs = ccg.compare(generated, real, metrics, sd)
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_species-generation"
    out.mkdir(parents=True, exist_ok=True)
    per_set.to_csv(out / "per_set.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    pd.concat(gen_frames, ignore_index=True).to_parquet(out / "generated_metrics.parquet", index=False)
    (out / "run_info.json").write_text(json.dumps({"gen": args.gen, "mesh_kind": args.mesh_kind, "minnie_table": str(args.minnie_table),
                                                   "h01_table": str(args.h01_table), "h01_detached": str(args.h01_detached),
                                                   "splits": str(args.splits), "split": args.split,
                                                   "real_spines": {g: len(r) for g, r in real.items()},
                                                   "created": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Species and cell type: generated vs real ({args.mesh_kind} meshes)", ""]
    lines += ["Real spines: " + ", ".join(f"{g} {len(r)} ({r['neuron_id'].nunique()} neurons)" for g, r in real.items()), ""]
    if len(pairs) and "ratio_real" in pairs.columns:
        lines += ["## Ratio of medians a / b: generated vs real", "",
                  "| label | a / b | metric | real | generated | share of the real difference | p (gen a vs b) |", "|---|---|---|---|---|---|---|"]
        for _, r in pairs.iterrows():
            lines.append(f"| {r['label']} | {r['a']} / {r['b']} | {r['metric']} | {r['ratio_real']:.3f} | {r['ratio_gen']:.3f} | "
                         f"{r.get('share_of_real_difference', np.nan):.2f} | {r['p_gen_a_vs_b']:.2g} |")
        lines.append("")
    own = per_set.groupby(["value", "label"])["closer_to_own"].mean()
    lines += ["## Share of metrics where the generated set is closer (W1/sd) to the real spines of its OWN group", ""]
    lines += [f"- {v}@{lab}: {s:.0%}" for (v, lab), s in own.items()]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
