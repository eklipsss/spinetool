"""Build the global HTML report from the folder written by build_global_tables.py.

    python scripts/neuron_model/build_global_report.py --dir runs\\analysis\\neuron-model\\<time>_global-tables

Writes ``<date>_global-report.html`` next to the tables (or ``--out``). Self-contained, opens offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.reports.global_report import build_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    print(f"-> {build_report(args.dir, args.out)}")


if __name__ == "__main__":
    main()
