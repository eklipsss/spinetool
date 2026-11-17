"""How much of the difference between Minnie65 (EM) and LabID (confocal) spines does the confocal imaging model explain?
(src/neuron_model/spine_generation/evaluation/domain_shift.py, docs/neuron-model/confocal-imaging-model.md)

Inputs are sets in the generator layout written by make_real_controls.py, all measured identically here:

    # EM reference and EM through the imaging model (several parameter choices = sensitivity)
    python scripts\\neuron_model\\make_real_controls.py --manifest cache\\minnie65_v1_60k\\manifest.parquet --splits <splits_v1> --split train ^
        --controls asis confocal --n-samples 500 --workers 8 --out <out>\\minnie
    python scripts\\neuron_model\\make_real_controls.py ... --controls confocal --confocal-param threshold=0.3 --out <out>\\minnie_t03
    # LabID healthy spines as they are (group Wt; the LabID manifest, its own split file with every folder in "test")
    python scripts\\neuron_model\\make_real_controls.py --manifest <labid_raw_z07>\\labid\\preprocessed\\manifest.parquet --splits <labid split> ^
        --where "neuron_id~^Wt_[0-9]" --controls asis --n-samples 400 --out <out>\\labid_wt

    python scripts\\neuron_model\\confocal_domain_shift.py --reference minnie=<out>\\minnie\\asis\\seed_1 ^
        --target labid_wt=<out>\\labid_wt\\asis\\seed_1 --imaged t05=<out>\\minnie\\confocal\\seed_1 --imaged t03=<out>\\minnie_t03\\confocal\\seed_1

Unit of inference: the source neuron (Minnie65) / the branch folder (LabID), from ``spine_key`` in generation_metadata.json.
Writes ``runs/analysis/neuron-model/<time>_confocal-shift/{shift.csv, summary.csv, imaging.csv, run_info.json}``.
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

from src.neuron_model.spine_generation.evaluation.domain_shift import shift_table, summarize  # noqa: E402
from src.neuron_model.spine_morphometrics import SCALAR_METRICS  # noqa: E402


def measure(folder: Path, mesh_kind: str) -> tuple:
    """Metrics of one set + per-sample unit (neuron / folder of the source spine) + imaging diagnostics of a confocal set."""
    from src.neuron_model.spine_generation.evaluation.morphometrics import generated_set_metrics

    if not Path(folder).is_dir() or not any(Path(folder).glob("sample_*")):
        raise SystemExit(f"{folder}: no samples (did make_real_controls.py fail? see its error messages)")
    frame = generated_set_metrics(folder, mesh_kind)
    units, imaging = [], []
    for sample in frame["sample_id"].astype(str):
        meta = folder / sample / "generation_metadata.json"
        key = json.loads(meta.read_text(encoding="utf-8")).get("spine_key") if meta.exists() else None
        units.append(str(key).split("/")[1] if key else sample)
        rep = folder / sample / "mesh_validation_raw.json"
        conf = json.loads(rep.read_text(encoding="utf-8")).get("confocal") if rep.exists() else None
        imaging.append(conf or {})
    frame["unit"] = units
    frame["detected"] = [c.get("detected", True) for c in imaging]
    frame["attached_to_trunk"] = [c.get("attached_to_trunk") for c in imaging]
    frame["base_gap_nm"] = [c.get("base_gap_nm") for c in imaging]
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, metavar="NAME=FOLDER", help="EM spines as they are (control asis)")
    ap.add_argument("--target", required=True, metavar="NAME=FOLDER", help="confocal spines (LabID healthy, control asis)")
    ap.add_argument("--imaged", action="append", required=True, metavar="NAME=FOLDER", help="EM spines through the imaging model (repeatable)")
    ap.add_argument("--mesh-kind", default="postprocessed", choices=("raw", "postprocessed"))
    ap.add_argument("--attached-only", action="store_true", help="imaged sets: only spines whose neck stays visible (attached to the trunk)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.time()

    def split(item):
        name, _, folder = item.partition("=")
        return name, Path(folder)

    ref_name, ref_dir = split(args.reference)
    tgt_name, tgt_dir = split(args.target)
    sets, imaging_rows = {}, []
    for name, folder in [(ref_name, ref_dir), (tgt_name, tgt_dir)] + [split(i) for i in args.imaged]:
        print(f"[shift] measuring {name}: {folder}", flush=True)
        frame = measure(folder, args.mesh_kind)
        present = frame["mesh_present"].astype(bool) if "mesh_present" in frame.columns else pd.Series(True, index=frame.index)
        row = {"set": name, "n": int(len(frame)), "n_units": int(frame["unit"].nunique()), "mesh_present": float(present.mean()),
               "detected": float(np.mean(frame["detected"].astype(bool)))}
        att = frame["attached_to_trunk"].dropna()
        if len(att):
            row.update(attached_to_trunk=float(att.astype(bool).mean()), base_gap_median_nm=float(np.nanmedian(frame["base_gap_nm"].astype(float))))
        imaging_rows.append(row)
        print(f"[shift]   {row}", flush=True)
        frame = frame[present]
        if args.attached_only and name not in (ref_name, tgt_name) and len(att):
            frame = frame[frame["attached_to_trunk"].eq(True)]
        sets[name] = frame
    metrics = [m for m in SCALAR_METRICS if all(m in f.columns for f in sets.values())]
    table = shift_table(sets, ref_name, tgt_name, metrics, n_boot=args.n_boot, seed=args.seed)
    if table.empty:
        raise SystemExit("no metric has >= 5 positive values in every set (sets too small or empty)")
    summary = summarize(table, sets, ref_name)

    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_confocal-shift"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "shift.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(imaging_rows).to_csv(out / "imaging.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print("\nper metric (log scale; * = 95 % interval excludes 0):")
    for r in table.itertuples():
        star = lambda s: "*" if s else " "  # noqa: E731
        print(f"  {r.set:12s} {r.metric:18s}{'' if r.base_independent else ' (base)':7s} total {r.total:+.3f}{star(r.total_sig)}  "
              f"optics {r.optics:+.3f}{star(r.optics_sig)}  rest {r.rest:+.3f}{star(r.rest_sig)}  share {r.share:.2f}")
    print("\nsummary (base-independent metrics, in reference sd):")
    print(summary.to_string(index=False))
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
