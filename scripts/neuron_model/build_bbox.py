"""Build only the frozen train bbox for an already written split.

For when ``build_splits.py`` saved the split but the bbox step was interrupted (the split
is frozen, so ``build_splits.py`` refuses to run again). Same config as build_splits:

    python scripts/neuron_model/build_bbox.py --config configs/neuron-model/data/splits_v1.yaml \
        "manifests=['O:/Datasets/Minnie65/preprocessed/manifest.parquet']"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.data.bbox import compute_frozen_bbox
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if not cfg.get("manifests"):
        parser.error("config has no manifests (set manifests=[...])")
    bbox_cfg = cfg["bbox"]
    output = resolve_path(bbox_cfg["output"])
    if output.exists():
        parser.error(f"{output} exists - the bbox is frozen once written")

    index = load_spine_index(
        [resolve_path(p) for p in cfg["manifests"]],
        require_train_eligible=cfg.get("require_train_eligible", True),
        path_mode=cfg.get("path_mode", "relative"),
        check_files=(),
    )
    train = index.with_split(load_splits(resolve_path(cfg["output"])), "train")
    print(f"train spines: {len(train)}")
    bbox = compute_frozen_bbox(
        train,
        margin=float(bbox_cfg["margin"]),
        n_points=int(bbox_cfg["n_points"]),
        split_version=cfg["version"],
        source=bbox_cfg.get("source", "pointcloud"),
        sdf_bbox_margin=float(bbox_cfg.get("sdf_bbox_margin", 100.0)),
        io_threads=int(bbox_cfg.get("io_threads", 1)),
        progress=True,
    )
    bbox.save(output)
    print(f"bbox -> {output}: low={bbox.low} high={bbox.high}")


if __name__ == "__main__":
    main()
