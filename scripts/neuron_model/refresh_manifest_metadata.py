"""Write a copy of an S-module manifest with the metadata columns recomputed by the FIXED lookup
(cell type from ``cell_types_with_nucleus.csv``, compartment from the limb table, health / treatment / species by dataset).

Why: ``manifest.parquet`` of the full Minnie65 run has ``compartment`` and ``cell_type_binary`` = None for every spine (stage 9 ran with
the old compartment lookup and without the cell type table), and ``build_splits.py`` stratifies by exactly these columns. The copy is
only a table (no stage is re-run, no per-spine file is written); the original manifest is not modified.

    python scripts/neuron_model/refresh_manifest_metadata.py --manifest O:/Datasets/Minnie65/preprocessed/manifest.parquet \
        --metadata-root datasets --out cache/manifest_minnie65_meta.parquet
    python scripts/neuron_model/build_splits.py --config configs/neuron-model/data/splits_v1.yaml \
        "manifests=['cache/manifest_minnie65_meta.parquet']" bbox=null
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.neuron_model.spine_generation.conditioning import spine_conditions  # noqa: E402

COLUMNS = ("species", "health", "treatment", "cell_type_binary", "compartment")


def refresh_metadata(manifest: pd.DataFrame, metadata_root: Path) -> pd.DataFrame:
    """Copy of ``manifest`` with ``COLUMNS`` from ``spine_conditions`` (needs dataset, neuron_id, limb_id)."""
    conditions = spine_conditions(manifest, metadata_root)
    out = manifest.copy()
    for column in COLUMNS:
        out[column] = conditions[column].to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True, help="folder with <dataset>/cell_types*.csv and the compartment table (datasets/)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.manifest)
    print(f"[manifest] {len(frame)} spines, {frame['neuron_id'].nunique()} neurons", flush=True)
    out = refresh_metadata(frame, args.metadata_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    neurons = out.drop_duplicates("neuron_id")
    print("cell_type_binary per neuron:", neurons["cell_type_binary"].fillna("unknown").value_counts().to_dict(), flush=True)
    print("compartment per spine:", out["compartment"].fillna("unknown").value_counts().to_dict(), flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
