"""Build the frozen group split (splits_v1.parquet) and the frozen train bbox.

Run once on the Windows workstation after the full preprocessing:

    python scripts/neuron_model/build_splits.py --config configs/neuron-model/data/splits_v1.yaml \
        "manifests=['O:/Datasets/Minnie65/preprocessed/manifest.parquet']"

Refuses to overwrite an existing split/bbox (a split is frozen once written;
bump ``version`` for a new one).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.data.bbox import compute_frozen_bbox
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import build_group_split, choose_group_level, save_splits, split_summary
from src.neuron_model.spine_generation.experiment.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if not cfg.get("manifests"):
        parser.error("config has no manifests (set manifests=[...])")

    index = load_spine_index(
        [resolve_path(p) for p in cfg["manifests"]],
        require_train_eligible=cfg.get("require_train_eligible", True),
        path_mode=cfg.get("path_mode", "relative"),
        # splits only need manifest columns (train_eligible + group keys), not the actual
        # pointcloud/sdf files - skip load_spine_index's default per-file exists() check,
        # which is ~2 NAS stat() calls per row and dominates runtime on a slow NAS share.
        check_files=(),
    )
    print(f"spines: manifest={index.frame.attrs['n_manifest']} train_eligible={index.frame.attrs['n_train_eligible']} with_files={len(index)}")
    level = cfg.get("group_level") or choose_group_level(index.frame)
    splits = build_group_split(index.frame, fractions=cfg["fractions"], seed=cfg["seed"], group_level=level, stratify=cfg["stratify"])
    summary = split_summary(splits, index.frame, cfg["stratify"])
    output = resolve_path(cfg["output"])
    save_splits(
        splits,
        output,
        version=cfg["version"],
        info={
            "seed": cfg["seed"],
            "fractions": cfg["fractions"],
            "group_level": level,
            "stratify": cfg["stratify"],
            "dataset_version": index.dataset_version,
            "manifests": [str(p) for p in cfg["manifests"]],
            "summary": summary,
        },
    )
    print(f"split -> {output}")
    print(json.dumps({k: summary[k] for k in ("n_spines", "n_groups")}, indent=2))

    bbox_cfg = cfg.get("bbox")
    if bbox_cfg:
        train = index.with_split(splits, "train")
        bbox = compute_frozen_bbox(train, margin=float(bbox_cfg["margin"]), n_points=int(bbox_cfg["n_points"]), split_version=cfg["version"])
        bbox.save(resolve_path(bbox_cfg["output"]))
        print(f"bbox -> {resolve_path(bbox_cfg['output'])}: low={bbox.low} high={bbox.high}")


if __name__ == "__main__":
    main()
