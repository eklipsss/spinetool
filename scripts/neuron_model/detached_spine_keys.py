"""Spines whose attachment centre lies far OUTSIDE the trunk surface: in H01 (~21 %) only the head is segmented and the neck
stayed in the trunk mesh, so the spine mesh has no neck (CLAUDE.md, H01 section). They are left out of the training of S
(decision 2026-11-17); in P they stay (their position along the branch is right).

    python scripts\\neuron_model\\detached_spine_keys.py --branches-root C:\\...\\data\\h01\\preprocessed_branches ^
        --out data\\processed\\neuron-model\\splits\\h01_detached_spines.parquet

A spine is detached when the branch preprocessing measured its offset ``dist_perp_minus_radius`` and flagged it
``position_outlier`` on the OUTSIDE (d > 0; centres deeper inside the trunk are a different defect and are kept).
Spines without a measurement (no branch row, beyond the end of the centre line) are kept. Output: ``spine_key`` (as in the S
manifest: ``<dataset>/<neuron>/<limb:03d>/<branch:03d>/<spine>``) + the offset, for ``cache_subset.py --exclude-keys``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402


def detached_keys(branch_spines: pd.DataFrame) -> pd.DataFrame:
    b = branch_spines
    d = pd.to_numeric(b["dist_perp_minus_radius"], errors="coerce")
    outlier = b["position_outlier"].fillna(False).astype(bool)
    end = b["projects_to_end"].fillna(False).astype(bool) if "projects_to_end" in b.columns else pd.Series(False, index=b.index)
    sel = outlier & (d > 0) & ~end
    keys = (b["dataset"].astype(str) + "/" + b["neuron_id"].astype(str) + "/" + pd.to_numeric(b["limb_id"]).astype(int).map("{:03d}".format)
            + "/" + pd.to_numeric(b["original_branch_id"]).astype(int).map("{:03d}".format) + "/" + b["spine_id"].astype(str))
    return pd.DataFrame({"spine_key": keys[sel], "dist_perp_minus_radius_nm": d[sel]}).drop_duplicates("spine_key").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cols = ["dataset", "neuron_id", "limb_id", "original_branch_id", "spine_id", "position_outlier", "dist_perp_minus_radius", "projects_to_end"]
    import pyarrow.parquet as pq

    path = args.branches_root / "branch_spines.parquet"
    names = set(pq.read_schema(path).names)
    b = pq.read_table(path, columns=[c for c in cols if c in names]).to_pandas()
    measured = int(pd.to_numeric(b["dist_perp_minus_radius"], errors="coerce").notna().sum())
    out = detached_keys(b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out)} detached spines of {measured} with a measured offset ({len(out) / max(measured, 1):.1%}); "
          f"median offset {out['dist_perp_minus_radius_nm'].median():.0f} nm -> {args.out}")


if __name__ == "__main__":
    main()
