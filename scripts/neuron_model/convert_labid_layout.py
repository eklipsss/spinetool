"""Convert LabID into the raw layout of Minnie65 so that the S preprocessing runs on it unchanged
(src/neuron_model/labid_layout.py: units um -> nm, the lab's seal removed from every spine, axis skeleton from CGAL).

    python scripts/neuron_model/convert_labid_layout.py --labid-root datasets/labid --out data/interim/neuron-model/labid_raw
    # then the S preprocessing on the converted tree, with the LabID config:
    python scripts/run_spine_preprocessing.py --config data/processed/config/spine_preprocessing_labid.yaml \
        --raw-data-root data/interim/neuron-model/labid_raw --dataset labid --metadata-root datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.labid_layout import convert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labid-root", type=Path, required=True, help="the LabID folder (<group>/<dendrite>/...)")
    parser.add_argument("--out", type=Path, required=True, help="output root; the tree goes to <out>/labid/")
    parser.add_argument("--branch", action="append", dest="units", help="only these branch ids (e.g. Ab+9009_24_2); repeatable")
    parser.add_argument("--unit-nm", type=float, default=1000.0, help="mesh unit in nm (default 1000: the meshes are in micrometres)")
    parser.add_argument("--z-scale", action="append", default=[], metavar="GROUP=FACTOR",
                        help="multiply z of every mesh of a group (Wt_old=0.35) or of all groups (all=0.4); off by default. "
                             "Another scale needs a NEW --out folder")
    parser.add_argument("--overwrite", action="store_true", help="redo dendrites already converted")
    args = parser.parse_args()
    z = {}
    for item in args.z_scale:
        group, value = item.split("=", 1)
        z[group] = float(value)
    convert(args.labid_root.resolve(), args.out.resolve(), units=args.units, unit_nm=args.unit_nm, z_scale=z or None, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
