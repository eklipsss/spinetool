"""Quality of the generation per module: tables + distribution figures from the outputs of the existing evaluations.

    python scripts/neuron_model/make_quality_report.py --out <dir> \
        [--s-eval <evaluate_generators run>] [--s-conditional <conditional report dir> ...] \
        [--tp <generate_tp set>[@label] ...] [--cell-types datasets/minnie65/cell_types_with_nucleus.csv] [--n-boot 1000]

Protocol and meaning of every number: docs/neuron-model/generation-quality-protocol.md (src/neuron_model/quality_report.py).
S per type: the real spines of a type are the rows of the real table of --s-eval (or --real-table) whose neuron has that
type in --cell-types (binary or detailed, whichever the report compares). T and P: the newest of <set>/evaluation_paired,
<set>/evaluation (evaluate_assembled_branches.py). Writes <out>/quality_report.md, CSV tables and PNG figures. CPU, minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.neuron_model import quality_report as Q  # noqa: E402


def neuron_types(path: Path, detailed: bool) -> dict:
    from src.neuron_model.metadata import cell_type_binary

    t = pd.read_csv(path)
    col = "segment_id" if "segment_id" in t.columns else t.columns[0]
    out = {}
    for seg, ct in zip(t[col].astype(str), t["cell_type"].astype(str)):
        out[seg] = ct if detailed else cell_type_binary(ct)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s-eval", type=Path)
    ap.add_argument("--s-conditional", type=Path, nargs="*", default=[])
    ap.add_argument("--real-table", type=Path, help="real spines (spine_key + metrics) for the per-type S figures; default: of --s-eval")
    ap.add_argument("--cell-types", type=Path, default=REPO_ROOT / "datasets/minnie65/cell_types_with_nucleus.csv")
    ap.add_argument("--tp", nargs="*", default=[], help="generate_tp set folder[@label]")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    lines = ["# Качество генерации: таблицы и рисунки", "",
             "Протокол и смысл каждой величины — `docs/neuron-model/generation-quality-protocol.md`. Создано `scripts/neuron_model/make_quality_report.py`.", ""]
    if args.s_eval:
        lines += Q.s_overall(args.s_eval, args.out, n_boot=args.n_boot)
    if args.s_conditional:
        real_path = args.real_table or (args.s_eval / "real_test_morphometrics.parquet" if args.s_eval else None)
        if real_path is None:
            ap.error("--s-conditional needs --real-table or --s-eval")
        real = pd.read_parquet(real_path, columns=["spine_key", *Q.S_METRICS])
        for d in args.s_conditional:
            values = set(pd.read_parquet(d / "generated_metrics.parquet", columns=["value"])["value"].astype(str))
            detailed = not values <= {"excitatory", "inhibitory"}
            lines += Q.s_by_type(d, real, neuron_types(args.cell_types, detailed), args.out / d.name)
    for spec in args.tp:
        path, _, label = spec.partition("@")
        lines += Q.tp_report(Path(path), args.out, label=label, n_boot=args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {args.out / 'quality_report.md'}")


if __name__ == "__main__":
    main()
