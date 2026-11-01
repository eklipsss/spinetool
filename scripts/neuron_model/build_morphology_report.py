"""Rebuild the HTML report of morphology_by_condition.py from its output folder (the analysis writes it itself).

    python scripts/neuron_model/build_morphology_report.py --dir runs/analysis/neuron-model/<time>_morphology-by-condition
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.reports.morphology_report import build_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(f"-> {build_report(args.dir, args.out)}")


if __name__ == "__main__":
    main()
