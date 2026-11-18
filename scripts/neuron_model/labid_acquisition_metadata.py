"""Acquisition metadata (Leica LAS X ``*_Properties.xml``) of the LabID source stacks -> one table, one row per source series.

Layout of a group folder (``datasets/labid/<group>/``): ``Исходники/<N>/MetaData/Project_Series<NNN>_Properties.xml`` (source image N,
from which several dendrite crops were cut) and ``output/<name>-<k>[_processed]/`` (the crops; ``<name>`` = group prefix + N, ``k`` = crop
of the same source image). The crops of one source image come from ONE image of one cell: not independent units for statistics.

Columns: voxel size (xy, z), field, planes, objective, NA, refractive index, pinhole (AU), excitation lines on (laser %), detector band,
gain, frame average, zoom, scan speed, date, plus ``crops`` (output folders cut from this image) and, if the stack folders exist,
``deconvolved_crops`` (``_processed`` suffix). Compare the groups with ``--compare`` (median per group of the numeric columns).

    python scripts/neuron_model/labid_acquisition_metadata.py --labid-root datasets/labid --groups Ab Wt Ab+9009 Wt+9009 \
        --out data/interim/neuron-model/labid_acquisition.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

SOURCE_DIR = "Исходники"


def _f(value: Optional[str]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_properties(path: Path) -> Dict[str, object]:
    """One ``*_Properties.xml`` -> flat dict (None where the field is absent)."""
    root = ET.parse(path).getroot()
    row: Dict[str, object] = {"series": re.search(r"Series(\d+)", path.name).group(1) if re.search(r"Series(\d+)", path.name) else None}
    dims = {d.get("DimID"): d for d in root.iter("DimensionDescription")}
    for axis, name in (("X", "x"), ("Y", "y"), ("Z", "z")):
        d = dims.get(axis)
        row[f"n_{name}"] = _f(d.get("NumberOfElements")) if d is not None else None
        row[f"voxel_{name}_um"] = _f(d.get("Voxel")) if d is not None else None
        row[f"length_{name}_um"] = _f(d.get("Length")) if d is not None else None
    if dims.get("X") is not None and row["n_x"] and row["length_x_um"]:
        row["voxel_x_nm"] = 1000.0 * row["length_x_um"] / row["n_x"]
    if dims.get("Z") is not None and row["n_z"] and row["length_z_um"] and row["n_z"] > 1:
        row["voxel_z_nm"] = 1000.0 * row["length_z_um"] / (row["n_z"] - 1)
    start = root.findtext(".//StartTime")
    row["start_time"] = start.strip() if start else None
    hw = next(iter(root.iter("ATLConfocalSettingDefinition")), None)
    if hw is not None:
        for key, col in (("ObjectiveName", "objective"), ("Immersion", "immersion"), ("ScanSpeed", "scan_speed"), ("Pinhole", "pinhole"),
                         ("PinholeAiry", "pinhole_au"), ("UserSettingName", "setting_name")):
            row[col] = (hw.get(key) or "").strip() or None
        for key, col in (("NumericalAperture", "na"), ("RefractionIndex", "refractive_index"), ("Zoom", "zoom"), ("FrameAverage", "frame_average"),
                         ("LineAverage", "line_average"), ("Magnification", "magnification"), ("BitSize", "bits"), ("Sections", "sections")):
            row[col] = _f(hw.get(key))
        row["pinhole_au"] = _f(str(row["pinhole_au"]).replace("AU", "").strip()) if row.get("pinhole_au") else None
        lines = {}
        for line in hw.iter("LaserLineSetting"):
            if _f(line.get("IntensityDev")):
                lines[int(float(line.get("LaserLine")))] = round(float(line.get("IntensityDev")), 3)
        row["laser_lines_percent"] = ";".join(f"{k}:{v}" for k, v in sorted(lines.items())) or None
        row["laser_488_percent"] = lines.get(488)
        for det in hw.iter("Detector"):
            if det.get("IsActive") == "1" and det.get("Type") == "PMT" and det.get("ScanType") == "Internal":
                row["detector_gain"] = _f(det.get("Gain"))
                row["detector_band"] = det.get("Band")
                break
    return row


def group_table(group_dir: Path) -> pd.DataFrame:
    """Rows of one group folder, with the crops cut from every source image."""
    rows: List[Dict[str, object]] = []
    out_dir = group_dir / "output"
    crops = sorted(p.name for p in out_dir.iterdir() if p.is_dir()) if out_dir.exists() else []
    for src in sorted((p for p in (group_dir / SOURCE_DIR).iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)) if (group_dir / SOURCE_DIR).exists() else []:
        props = sorted((src / "MetaData").glob("*_Properties.xml"))
        row: Dict[str, object] = {"group": group_dir.name, "source": src.name}
        if props:
            row.update(parse_properties(props[0]))
            if len(props) > 1:
                row["note"] = f"{len(props)} properties files, the first used"
        else:
            row["note"] = "no Properties.xml"
        mine = [c for c in crops if re.match(rf"^[A-Za-z+_]*{re.escape(src.name)}-\d", c)]
        row["crops"] = ";".join(mine)
        row["n_crops"] = len(mine)
        row["deconvolved_crops"] = sum(c.endswith("_processed") for c in mine)
        row["z_planes_in_folder"] = len(list(src.glob("*.tif")))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labid-root", type=Path, default=REPO_ROOT / "datasets/labid")
    parser.add_argument("--groups", nargs="+", default=["Ab"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", action="store_true", help="print the median per group of the numeric columns")
    args = parser.parse_args()

    tables = [group_table(args.labid_root / g) for g in args.groups if (args.labid_root / g).exists()]
    table = pd.concat([t for t in tables if len(t)], ignore_index=True) if tables else pd.DataFrame()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = [c for c in ("group", "source", "series", "voxel_x_nm", "voxel_z_nm", "n_z", "objective", "na", "refractive_index", "pinhole_au", "zoom",
                        "frame_average", "detector_gain", "laser_488_percent", "start_time", "n_crops", "deconvolved_crops") if c in table.columns]
    print(table[cols].to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"-> {args.out}")
    if args.compare and len(table):
        num = table.select_dtypes("number").columns.drop(["n_crops"], errors="ignore")
        print(table.groupby("group")[list(num)].median().T.to_string())


if __name__ == "__main__":
    main()
