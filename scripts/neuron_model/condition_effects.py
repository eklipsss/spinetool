"""Step 0 of the conditional spine generator: which condition blocks change the real morphology more than the
variation between neurons? (src/neuron_model/spine_generation/evaluation/condition_effects.py)

Windows, the same data arguments as the MoGen training (local cache manifest + split; morphometrics from the stage-8
table; CPU, minutes):

    python scripts/neuron_model/condition_effects.py \
        "data.manifests=['C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet']" data.path_mode=absolute \
        data.splits=data/processed/neuron-model/splits/splits_dev_100n.parquet data.drop_unassigned=true \
        real.morphometrics_table=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

Writes ``runs/analysis/neuron-model/<id>_condition-effects/condition_effects.json``; pass it to the training as
``condition.effects_report=<that file>`` (``condition.mode: auto`` enables only the blocks marked enabled).
Uses the TRAIN split by default (the test split stays untouched for the evaluation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.conditioning import DEFAULT_BLOCKS, spine_conditions  # noqa: E402
from src.neuron_model.spine_generation.data.index import load_spine_index  # noqa: E402
from src.neuron_model.spine_generation.data.splits import load_splits  # noqa: E402
from src.neuron_model.spine_generation.evaluation.condition_effects import condition_effects  # noqa: E402
from src.neuron_model.spine_generation.evaluation.morphometrics import real_morphometrics  # noqa: E402
from src.neuron_model.spine_generation.experiment.artifacts import write_json  # noqa: E402
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path  # noqa: E402
from src.neuron_model.spine_generation.experiment.run import create_run  # noqa: E402
from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/neuron-model/evaluation/condition_effects.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    if not d["manifests"]:
        parser.error("set data.manifests=[...]")
    index = load_spine_index([resolve_path(p) for p in d["manifests"]], path_mode=d.get("path_mode", "relative"), check_files=())
    if d.get("splits"):
        index = index.with_split(load_splits(resolve_path(d["splits"])), split=d.get("split") or None, drop_unassigned=bool(d.get("drop_unassigned")))
    print(f"[effects] {len(index)} spines ({d.get('split') or 'all splits'})", flush=True)
    conditions = spine_conditions(index.frame, resolve_path(cfg["condition"]["metadata_root"]))
    keep = ~conditions["compartment"].isin(set(cfg["condition"].get("exclude_compartments") or []))
    table = resolve_path(cfg["real"]["morphometrics_table"]) if cfg["real"].get("morphometrics_table") else None
    morph = real_morphometrics(index, table=table, threads=int(cfg["real"]["io_threads"]), label="real")
    features = morph[list(SCALAR_METRICS)].apply(lambda c: c.astype(float))
    e = cfg["effects"]
    report = condition_effects(features[keep.to_numpy()].reset_index(drop=True), conditions[keep].reset_index(drop=True),
                               index.frame.loc[keep, "neuron_id"].astype(str).reset_index(drop=True), DEFAULT_BLOCKS,
                               min_neurons=int(e["min_neurons"]), alpha=float(e["alpha"]), n_permutations=int(e["n_permutations"]),
                               max_per_neuron=int(e["max_per_neuron"]), seed=int(cfg["seed"]))
    report["n_spines"] = int(keep.sum())
    run = create_run(cfg, runs_root=resolve_path(cfg["runs_root"]), extra_info={"n_spines": int(keep.sum())})
    write_json(run.root / "condition_effects.json", report)
    print("\n[effects] blocks:")
    for name, block in report["blocks"].items():
        counts = {k: v.get("neurons", v.get("spines")) for k, v in block.get("counts", {}).items()}
        print(f"  {name:18s} enabled={block['enabled']!s:5s}  {block['reason']}  (neurons per value: {counts})")
    print(f"-> {run.root / 'condition_effects.json'}  (enabled: {[n for n, b in report['blocks'].items() if b['enabled']]})")


if __name__ == "__main__":
    main()
